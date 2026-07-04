import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

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


def test_probabilities_sum_to_one():
    result = score_signal(make_signal(), breadth=0.5)
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-9


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
        breadth=0.25,
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
        breadth=0.75,
    )
    assert result.recommendation == "留下"


def test_empty_news_feed_uses_neutral_scores(monkeypatch):
    monkeypatch.setattr(monitor, "alpha_get", lambda params, api_key: {"feed": []})

    summary, top_news = monitor.fetch_news(["MU", "WDC"], "dummy")

    assert top_news == []
    assert summary["MU"]["mean_score"] == 0.0
    assert summary["MU"]["negative_ratio"] == 0.0
    assert summary["MU"]["article_count"] == 0


def test_long_telegram_line_is_split():
    chunks = monitor.split_telegram_text("A" * 5000, limit=3900)

    assert [len(chunk) for chunk in chunks] == [3900, 1100]


def test_nonfinite_signal_does_not_create_nan_probabilities():
    result = score_signal(make_signal(ma20_gap=float("nan")), breadth=0.5)

    assert all(math.isfinite(value) for value in result.probabilities.values())
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-9


def test_report_does_not_link_invalid_news_url():
    signal = score_signal(make_signal(), breadth=0.5)

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
        breadth=0.5,
    )

    report, _ = monitor.build_report([signal], [], "2026-07-02")

    assert "美光科技 (TEST, Micron Technology)" in report
    assert "注：存储芯片厂商" in report


def test_load_assets_reads_enabled_config(tmp_path):
    config = tmp_path / "assets.json"
    config.write_text(
        json.dumps(
            {
                "assets": [
                    {"symbol": "MU", "type": "US_STOCK", "enabled": True},
                    {"symbol": "161725", "type": "CN_FUND", "enabled": False},
                ]
            }
        ),
        encoding="utf-8",
    )

    assets = monitor.load_assets(str(config))

    assert [asset.symbol for asset in assets] == ["MU"]
    assert assets[0].asset_type == "US_STOCK"


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
