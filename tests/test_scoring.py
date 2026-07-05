import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

import evaluate_predictions
import monitor
from monitor import Signal, score_signal


def make_signal(**overrides):
    base = dict(
        ticker="TEST",
        close=100.0,
        ret_5d=0.0,
        ret_20d=0.0,
        ret_60d=0.0,
        drawdown_60d=-0.05,
        rsi_14=50.0,
        ma20_gap=0.01,
        ma50_gap=0.02,
        volume_ratio_5_20=1.0,
        news_score=0.0,
        negative_news_ratio=0.0,
    )
    base.update(overrides)
    return Signal(**base)


def make_price_frame(end="2026-07-02", periods=100):
    dates = pd.bdate_range(end=end, periods=periods)
    close = pd.Series(np.linspace(90.0, 120.0, len(dates)), index=dates)
    return pd.DataFrame(
        {
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(1_000_000, 1_300_000, len(dates)),
        },
        index=dates,
    )


def test_probabilities_sum_to_one():
    result = score_signal(make_signal(), portfolio_breadth=0.5)
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-9


def test_score_signal_records_logits_and_contributions():
    result = score_signal(make_signal(), portfolio_breadth=0.5)

    assert result.raw_logits
    assert result.score_components["portfolio_breadth"] == 0.5
    assert set(result.indicator_contributions) >= {
        "ma20_below_average",
        "news_sentiment_risk",
        "portfolio_breadth_risk",
        "keep_intercept",
    }

    sell_sum = sum(
        row["sell_all"] for row in result.indicator_contributions.values()
    )
    reduce_sum = sum(
        row["reduce_on_rebound"] for row in result.indicator_contributions.values()
    )
    keep_sum = sum(row["keep"] for row in result.indicator_contributions.values())

    assert abs(sell_sum - result.raw_logits[monitor.LABEL_SELL_ALL]) < 1e-9
    assert abs(reduce_sum - result.raw_logits[monitor.LABEL_REDUCE_ON_REBOUND]) < 1e-9
    assert abs(keep_sum - result.raw_logits[monitor.LABEL_KEEP]) < 1e-9


def test_oversold_case_prefers_waiting_over_holding():
    result = score_signal(
        make_signal(
            ret_5d=-0.16,
            ret_20d=-0.20,
            drawdown_60d=-0.27,
            rsi_14=27.0,
            ma20_gap=-0.16,
            ma50_gap=-0.08,
            volume_ratio_5_20=1.7,
            news_score=-0.05,
            negative_news_ratio=0.25,
        ),
        portfolio_breadth=0.25,
    )
    assert result.probabilities["等待回弹后卖出或减仓"] > result.probabilities["留下"]


def test_positive_case_prefers_holding():
    result = score_signal(
        make_signal(
            ret_5d=0.04,
            ret_20d=0.10,
            ret_60d=0.28,
            drawdown_60d=-0.04,
            rsi_14=61.0,
            ma20_gap=0.08,
            ma50_gap=0.16,
            volume_ratio_5_20=1.1,
            news_score=0.25,
            negative_news_ratio=0.05,
        ),
        portfolio_breadth=0.75,
    )
    assert result.recommendation == "留下"


def test_empty_news_feed_uses_neutral_scores(monkeypatch):
    monkeypatch.setattr(monitor, "alpha_get", lambda params, api_key: {"feed": []})

    summary, top_news = monitor.fetch_news(["MU", "WDC"], "dummy")

    assert top_news == []
    assert summary["MU"]["mean_score"] == 0.0
    assert summary["MU"]["negative_ratio"] == 0.0
    assert summary["MU"]["article_count"] == 0


def test_fetch_daily_keeps_adjusted_close_separate(monkeypatch):
    dates = pd.bdate_range(end="2026-07-02", periods=70)
    payload = {
        "Time Series (Daily)": {
            date.strftime("%Y-%m-%d"): {
                "1. open": "100",
                "2. high": "101",
                "3. low": "99",
                "4. close": "100",
                "5. adjusted close": "10",
                "6. volume": "123456",
            }
            for date in dates
        }
    }

    def fake_alpha_get(params, api_key):
        assert params["function"] == "TIME_SERIES_DAILY"
        return payload

    monkeypatch.setattr(monitor, "alpha_get", fake_alpha_get)

    frame = monitor.fetch_daily("MU", "dummy")

    assert frame["close"].iloc[-1] == 100.0
    assert frame["adjusted_close"].iloc[-1] == 10.0
    assert frame["volume"].iloc[-1] == 123456.0


def test_fetch_daily_uses_close_when_adjusted_close_is_unavailable(monkeypatch):
    dates = pd.bdate_range(end="2026-07-02", periods=70)
    payload = {
        "Time Series (Daily)": {
            date.strftime("%Y-%m-%d"): {
                "1. open": "100",
                "2. high": "101",
                "3. low": "99",
                "4. close": "100",
                "5. volume": "123456",
            }
            for date in dates
        }
    }
    monkeypatch.setattr(monitor, "alpha_get", lambda params, api_key: payload)

    frame = monitor.fetch_daily("MU", "dummy")

    assert frame["adjusted_close"].iloc[-1] == frame["close"].iloc[-1]


def test_long_telegram_line_is_split():
    chunks = monitor.split_telegram_text("A" * 5000, limit=3900)

    assert [len(chunk) for chunk in chunks] == [3900, 1100]


def test_nonfinite_signal_does_not_create_nan_probabilities():
    result = score_signal(make_signal(ma20_gap=float("nan")), portfolio_breadth=0.5)

    assert all(math.isfinite(value) for value in result.probabilities.values())
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-9


def test_report_does_not_link_invalid_news_url():
    signal = score_signal(make_signal(), portfolio_breadth=0.5)

    report, _ = monitor.build_report(
        [signal],
        [{"title": "Bad <url>", "url": "javascript:alert(1)", "source": "Src"}],
        "2026-07-02",
    )

    assert "<a href=" not in report
    assert "Bad &lt;url&gt;" in report


def test_report_includes_asset_translation_and_note():
    signal = score_signal(
        make_signal(
            name_zh="美光科技",
            name_en="Micron Technology",
            note="存储芯片厂商，主要看 DRAM/NAND 周期。",
            price_as_of="2026-07-02",
        ),
        portfolio_breadth=0.5,
    )

    report, _ = monitor.build_report([signal], [], "2026-07-02")

    assert "美光科技 (TEST, Micron Technology)" in report
    assert "注：存储芯片厂商" in report
    assert "未校准的相对倾向" in report
    assert "不代表真实发生概率" in report


def test_risk_alert_includes_cvar_when_history_is_sufficient():
    dates = pd.bdate_range(end="2026-07-02", periods=180)
    close = pd.Series(np.linspace(1.2, 1.0, len(dates)), index=dates)
    close.iloc[40] *= 0.90
    close.iloc[90] *= 0.88
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(100_000, 220_000, len(dates)),
        },
        index=dates,
    )
    asset = monitor.Asset(symbol="512480", asset_type="CN_ETF", name_zh="半导体ETF")
    signal = score_signal(
        monitor.calculate_signal("512480", frame, monitor.neutral_news_summary(["512480"])["512480"], asset),
        portfolio_breadth=0.5,
    )

    alert = monitor.calculate_risk_alert(signal, frame)

    assert alert.cvar_95_1d is not None
    assert alert.cvar_95_5d is not None
    assert alert.cvar_95_1d < 0
    assert alert.level in {"低", "中", "中高", "高"}


def test_report_includes_cn_risk_alert_and_key_links():
    signal = score_signal(
        make_signal(
            ticker="512480",
            asset_type="CN_ETF",
            name_zh="半导体ETF国联安",
            name_en="GTJA-Allianz CSI Semiconductor ETF",
            ret_5d=-0.06,
            ret_20d=-0.08,
            ma20_gap=-0.04,
            price_as_of="2026-07-02",
        ),
        portfolio_breadth=0.5,
    )
    alert = monitor.RiskAlert(
        ticker="512480",
        level="中高",
        direction="转弱",
        score=0.62,
        current_drawdown_60d=-0.12,
        annual_vol_20d=0.32,
        annual_vol_60d=0.24,
        vol_ratio_20_60=1.33,
        downside_vol_20d=0.20,
        var_95_1d=-0.03,
        cvar_95_1d=-0.04,
        var_95_5d=-0.08,
        cvar_95_5d=-0.10,
        consecutive_down_days=3,
        sample_days=160,
        warnings=["近5日跌幅进入历史尾部区间"],
    )
    stories = [
        monitor.MarketStory(
            symbol="512480",
            title="半导体板块反弹",
            url="https://example.com/news",
            source="测试新闻",
            kind="news",
            sentiment=0.4,
        )
    ]

    report, _ = monitor.build_report(
        [signal],
        [],
        "2026-07-02",
        risk_alerts={"512480": alert},
        market_stories=stories,
    )

    assert "中国ETF/基金风险预警" in report
    assert "CVaR95(5日)" in report
    assert "关键链接" in report
    assert "fund.eastmoney.com/512480.html" in report
    assert "半导体板块反弹" in report


def test_load_assets_reads_enabled_config(tmp_path):
    config = tmp_path / "assets.json"
    config.write_text(
        json.dumps(
            {
                "assets": [
                    {"symbol": "MU", "type": "US_STOCK", "enabled": True},
                    {"symbol": "SMH", "type": "US_ETF", "enabled": True},
                    {"symbol": "161725", "type": "CN_FUND", "enabled": False},
                ]
            }
        ),
        encoding="utf-8",
    )

    assets = monitor.load_assets(str(config))

    assert [asset.symbol for asset in assets] == ["MU", "SMH"]
    assert assets[0].asset_type == "US_STOCK"
    assert assets[1].asset_type == "US_ETF"


def test_cn_fund_daily_normalizes_akshare_nav(monkeypatch):
    dates = pd.bdate_range(end="2026-07-02", periods=70)
    raw = pd.DataFrame(
        {
            "净值日期": dates.strftime("%Y-%m-%d"),
            "单位净值": np.linspace(1.0, 1.2, len(dates)),
        }
    )
    fake_akshare = SimpleNamespace(
        fund_open_fund_info_em=lambda symbol, indicator: raw
    )
    monkeypatch.setattr(monitor, "import_akshare", lambda: fake_akshare)

    frame = monitor.fetch_cn_fund_daily("161725")

    assert len(frame) == 70
    assert frame["close"].iloc[-1] == 1.2
    assert frame["open"].iloc[-1] == 1.2
    assert frame["volume"].sum() == 0.0


def test_cn_etf_daily_falls_back_to_eastmoney(monkeypatch):
    class FakeResponse:
        status_code = 200

        def json(self):
            dates = pd.bdate_range(end="2026-07-02", periods=70)
            klines = [
                f"{date.strftime('%Y-%m-%d')},{1 + idx / 100:.3f},{1 + idx / 90:.3f},"
                f"{1 + idx / 80:.3f},{1 + idx / 110:.3f},{100000 + idx}"
                for idx, date in enumerate(dates)
            ]
            return {"data": {"klines": klines}}

    fake_akshare = SimpleNamespace(
        fund_etf_hist_em=lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("akshare disconnected")
        )
    )
    monkeypatch.setattr(monitor, "import_akshare", lambda: fake_akshare)
    monkeypatch.setattr(monitor.requests, "get", lambda *args, **kwargs: FakeResponse())

    frame = monitor.fetch_cn_etf_daily("512480")

    assert len(frame) == 70
    assert frame["close"].iloc[-1] > frame["close"].iloc[0]
    assert frame["volume"].iloc[-1] > 100000


def test_cn_etf_daily_falls_back_to_sina(monkeypatch):
    class FakeEastMoneyResponse:
        status_code = 500

        def json(self):
            return {}

    class FakeSinaResponse:
        status_code = 200

        def json(self):
            dates = pd.bdate_range(end="2026-07-02", periods=70)
            return [
                {
                    "day": date.strftime("%Y-%m-%d"),
                    "open": f"{1 + idx / 100:.3f}",
                    "high": f"{1 + idx / 80:.3f}",
                    "low": f"{1 + idx / 110:.3f}",
                    "close": f"{1 + idx / 90:.3f}",
                    "volume": str(100000 + idx),
                }
                for idx, date in enumerate(dates)
            ]

    fake_akshare = SimpleNamespace(
        fund_etf_hist_em=lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("akshare disconnected")
        )
    )

    def fake_get(url, *args, **kwargs):
        if url == monitor.EASTMONEY_KLINE_URL:
            return FakeEastMoneyResponse()
        return FakeSinaResponse()

    monkeypatch.setattr(monitor, "import_akshare", lambda: fake_akshare)
    monkeypatch.setattr(monitor.requests, "get", fake_get)
    monkeypatch.setattr(monitor.time, "sleep", lambda seconds: None)

    frame = monitor.fetch_cn_etf_daily("159770")

    assert len(frame) == 70
    assert frame["close"].iloc[-1] > frame["close"].iloc[0]
    assert frame["volume"].iloc[-1] > 100000


def test_sina_fallback_back_adjusts_split_like_jumps(monkeypatch):
    class FakeEastMoneyResponse:
        status_code = 500

        def json(self):
            return {}

    class FakeSinaResponse:
        status_code = 200

        def json(self):
            dates = pd.bdate_range(end="2026-07-02", periods=70)
            rows = []
            for idx, date in enumerate(dates):
                base_close = 2.0 + idx * 0.002
                split_factor = 1.0 if idx < 60 else 0.5
                close = base_close * split_factor
                rows.append(
                    {
                        "day": date.strftime("%Y-%m-%d"),
                        "open": f"{close * 0.998:.3f}",
                        "high": f"{close * 1.004:.3f}",
                        "low": f"{close * 0.996:.3f}",
                        "close": f"{close:.3f}",
                        "volume": str(int((100000 + idx) / split_factor)),
                    }
                )
            return rows

    fake_akshare = SimpleNamespace(
        fund_etf_hist_em=lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("akshare disconnected")
        )
    )

    def fake_get(url, *args, **kwargs):
        if url == monitor.EASTMONEY_KLINE_URL:
            return FakeEastMoneyResponse()
        return FakeSinaResponse()

    monkeypatch.setattr(monitor, "import_akshare", lambda: fake_akshare)
    monkeypatch.setattr(monitor.requests, "get", fake_get)
    monkeypatch.setattr(monitor.time, "sleep", lambda seconds: None)

    frame = monitor.fetch_cn_etf_daily("512480")
    returns = frame["close"].pct_change().dropna()

    assert returns.min() > -0.05
    assert abs(frame["close"].iloc[60] / frame["close"].iloc[59] - 1.0) < 0.01


def test_prediction_snapshot_and_forward_outcome_are_structured():
    full_frame = make_price_frame(periods=120)
    prediction_frame = full_frame.iloc[:90]
    asset = monitor.Asset(symbol="MU", asset_type="US_STOCK", name_zh="美光科技")
    signal = score_signal(
        monitor.calculate_signal(
            "MU",
            prediction_frame,
            {"mean_score": 0.2, "negative_ratio": 0.1, "article_count": 4},
            asset,
        ),
        portfolio_breadth=0.75,
    )
    timestamp = pd.Timestamp("2026-07-04T00:30:00Z").to_pydatetime()

    snapshots = monitor.build_prediction_snapshots(
        [signal],
        timestamp,
        portfolio_breadth=0.75,
        market_stories=[],
    )
    snapshot = snapshots[0]

    assert snapshot["symbol"] == "MU"
    assert snapshot["sell_all_logit"] == signal.raw_logits[monitor.LABEL_SELL_ALL]
    assert snapshot["sell_all_softmax"] == signal.probabilities[monitor.LABEL_SELL_ALL]
    assert snapshot["raw_indicators"]["news_count"] == 4
    assert "indicator_contributions" in snapshot

    audit_config = {
        "observation_windows": [5, 10, 20],
        "clear_rebound_threshold_pct": 0.02,
        "clear_drawdown_threshold_pct": -0.02,
        "clear_rebound_threshold_atr": 1.0,
        "clear_drawdown_threshold_atr": -1.0,
        "severe_drawdown_threshold_pct": -0.08,
        "min_bucket_samples": 2,
    }
    outcome = monitor.compute_forward_outcome(
        snapshot,
        full_frame,
        audit_config,
        timestamp,
    )

    assert outcome["forward_return_5d"] > 0
    assert outcome["forward_max_drawdown_20d"] == 0.0
    assert outcome["forward_max_rebound_20d"] > 0
    assert outcome["forward_max_price_date_20d"] >= outcome["price_date"]
    assert outcome["first_clear_move_pct"] in {"rebound", "drawdown", "none"}


def test_save_history_writes_prediction_jsonl(monkeypatch, tmp_path):
    frame = make_price_frame(periods=90)
    signal = score_signal(
        monitor.calculate_signal(
            "MU",
            frame,
            {"mean_score": 0.1, "negative_ratio": 0.0, "article_count": 2},
            monitor.Asset(symbol="MU", asset_type="US_STOCK"),
        ),
        portfolio_breadth=1.0,
    )
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))

    output_path = monitor.save_history(
        [signal],
        signal.probabilities,
        [],
        signal.price_as_of,
        frames={"MU": frame},
        portfolio_breadth=1.0,
    )

    snapshots_path = tmp_path / "prediction_snapshots.jsonl"
    assert output_path.exists()
    assert snapshots_path.exists()
    rows = monitor.read_jsonl(snapshots_path)
    assert len(rows) == 1
    assert rows[0]["portfolio_breadth"] == 1.0
    assert rows[0]["keep_softmax"] == signal.probabilities[monitor.LABEL_KEEP]


def test_evaluate_predictions_builds_bucket_and_correlation_reports(tmp_path):
    snapshot_path = tmp_path / "prediction_snapshots.jsonl"
    outcome_path = tmp_path / "prediction_outcomes.jsonl"
    snapshots = []
    outcomes = []
    for idx in range(6):
        timestamp = f"2026-01-{idx + 1:02d}T00:30:00+00:00"
        snapshots.append(
            {
                "symbol": "MU" if idx < 3 else "SMH",
                "prediction_date": f"2026-01-{idx + 1:02d}",
                "prediction_timestamp_utc": timestamp,
                "price_date": f"2026-01-{idx + 1:02d}",
                "sell_all_softmax": idx / 5,
                "reduce_on_rebound_softmax": 0.5,
                "keep_softmax": 1 - idx / 5,
            }
        )
        outcomes.append(
            {
                "symbol": "MU" if idx < 3 else "SMH",
                "prediction_date": f"2026-01-{idx + 1:02d}",
                "prediction_timestamp_utc": timestamp,
                "price_date": f"2026-01-{idx + 1:02d}",
                "forward_return_5d": 0.02 - idx * 0.01,
                "forward_return_10d": 0.03 - idx * 0.01,
                "forward_return_20d": 0.04 - idx * 0.01,
                "forward_max_drawdown_5d": -idx * 0.01,
                "forward_max_drawdown_10d": -idx * 0.012,
                "forward_max_drawdown_20d": -idx * 0.015,
                "forward_max_rebound_5d": 0.02,
                "forward_max_rebound_10d": 0.03,
                "forward_max_rebound_20d": 0.04,
                "first_clear_move_pct": "rebound" if idx % 2 == 0 else "drawdown",
            }
        )
    monitor.append_jsonl(snapshot_path, snapshots)
    monitor.append_jsonl(outcome_path, outcomes)

    frame = evaluate_predictions.load_audit_frame(snapshot_path, outcome_path)
    bucket_report, corr_report = evaluate_predictions.build_reports(
        frame,
        {
            "observation_windows": [5, 10, 20],
            "severe_drawdown_threshold_pct": -0.05,
            "min_bucket_samples": 2,
        },
    )

    assert not bucket_report.empty
    assert not corr_report.empty
    assert set(bucket_report["scope_type"]) >= {"all_assets", "symbol", "year"}
    assert "pearson_corr" in corr_report.columns


def test_main_continues_when_one_ticker_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "dummy-alpha")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy-telegram")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "dummy-chat")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    config = tmp_path / "assets.json"
    config.write_text(
        json.dumps(
            {
                "assets": [
                    {"symbol": "MU", "type": "US_STOCK", "enabled": True},
                    {"symbol": "WDC", "type": "US_STOCK", "enabled": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ASSET_CONFIG", str(config))

    monkeypatch.setattr(
        monitor,
        "fetch_news",
        lambda tickers, api_key: (
            monitor.neutral_news_summary(tickers),
            [],
        ),
    )

    def fake_fetch_daily(ticker, api_key):
        if ticker == "WDC":
            raise RuntimeError("simulated WDC failure")
        dates = pd.bdate_range(end="2026-07-02", periods=70)
        return pd.DataFrame(
            {
                "open": np.linspace(90, 100, len(dates)),
                "high": np.linspace(91, 101, len(dates)),
                "low": np.linspace(89, 99, len(dates)),
                "close": np.linspace(90, 100, len(dates)),
                "volume": np.linspace(1_000_000, 1_200_000, len(dates)),
            },
            index=dates,
        )

    sent = []
    monkeypatch.setattr(monitor, "fetch_daily", fake_fetch_daily)
    monkeypatch.setattr(
        monitor, "send_telegram", lambda token, chat_id, text: sent.append(text)
    )
    monkeypatch.setattr(monitor.time, "sleep", lambda seconds: None)

    assert monitor.main() == 0
    assert len(sent) == 1
    assert "WDC daily: simulated WDC failure" in sent[0]
    assert len(list(tmp_path.glob("result_*.json"))) == 1
