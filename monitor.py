from __future__ import annotations

import html
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests


ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
TELEGRAM_URL = "https://api.telegram.org/bot{token}/{method}"
TIMEOUT = 30
TELEGRAM_MESSAGE_LIMIT = 4096
TELEGRAM_SAFE_LIMIT = 3900
STALE_PRICE_DAYS = 5
DEFAULT_TICKERS = "MU,SNDK,WDC,STX"
DEFAULT_ASSET_CONFIG = "assets.json"
SUPPORTED_ASSET_TYPES = {"US_STOCK", "CN_FUND", "CN_ETF"}


@dataclass
class Asset:
    symbol: str
    asset_type: str = "US_STOCK"
    name_zh: str = ""
    name_en: str = ""
    note: str = ""
    enabled: bool = True
    news_symbol: str = ""


@dataclass
class Signal:
    ticker: str
    close: float
    ret_5d: float
    ret_20d: float
    ret_60d: float
    drawdown_60d: float
    rsi_14: float
    ma20_gap: float
    ma50_gap: float
    volume_ratio_5_20: float
    news_score: float
    negative_news_ratio: float
    asset_type: str = "US_STOCK"
    name_zh: str = ""
    name_en: str = ""
    note: str = ""
    price_as_of: str = ""
    breakdown_score: float = 0.0
    oversold_score: float = 0.0
    news_risk_score: float = 0.0
    probabilities: dict[str, float] | None = None
    recommendation: str = ""

    def display_name(self) -> str:
        if self.name_zh and self.name_en:
            return f"{self.name_zh} ({self.ticker}, {self.name_en})"
        if self.name_zh:
            return f"{self.name_zh} ({self.ticker})"
        if self.name_en:
            return f"{self.name_en} ({self.ticker})"
        return self.ticker


@dataclass
class FetchFailure:
    ticker: str
    stage: str
    error: str


def env_required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def redact_secret(text: str, secret: str | None) -> str:
    if not secret:
        return text
    return text.replace(secret, "***")


def sanitize_error_message(error: Exception, *secrets: str | None) -> str:
    message = str(error)
    for secret in secrets:
        message = redact_secret(message, secret)
    return message


def clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return default
    result = numerator / denominator
    return float(result) if np.isfinite(result) else default


def neutral_news_summary(tickers: list[str]) -> dict[str, dict[str, float]]:
    return {
        ticker: {
            "mean_score": 0.0,
            "negative_ratio": 0.0,
            "article_count": 0,
        }
        for ticker in tickers
    }


def parse_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def normalize_asset_type(value: Any) -> str:
    asset_type = str(value or "US_STOCK").strip().upper().replace("-", "_")
    if asset_type not in SUPPORTED_ASSET_TYPES:
        supported = ", ".join(sorted(SUPPORTED_ASSET_TYPES))
        raise RuntimeError(f"Unsupported asset type {asset_type}. Supported: {supported}")
    return asset_type


def asset_from_config(item: dict[str, Any]) -> Asset:
    symbol = str(item.get("symbol", "")).strip().upper()
    if not symbol:
        raise RuntimeError("Asset config item is missing symbol")

    asset_type = normalize_asset_type(item.get("type", item.get("asset_type")))
    news_symbol = str(item.get("news_symbol", "")).strip().upper()
    return Asset(
        symbol=symbol,
        asset_type=asset_type,
        name_zh=str(item.get("name_zh", "")).strip(),
        name_en=str(item.get("name_en", "")).strip(),
        note=str(item.get("note", "")).strip(),
        enabled=parse_enabled(item.get("enabled", True)),
        news_symbol=news_symbol,
    )


def assets_from_tickers(tickers_value: str) -> list[Asset]:
    tickers = [
        ticker.strip().upper()
        for ticker in tickers_value.split(",")
        if ticker.strip()
    ]
    return [Asset(symbol=ticker) for ticker in tickers]


def load_assets(config_path: str | None = None) -> list[Asset]:
    path_value = config_path or os.getenv("ASSET_CONFIG", DEFAULT_ASSET_CONFIG)
    path = Path(path_value)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_assets = data.get("assets", data) if isinstance(data, dict) else data
        if not isinstance(raw_assets, list):
            raise RuntimeError("Asset config must contain an assets list")
        assets = [
            asset
            for asset in (asset_from_config(item) for item in raw_assets)
            if asset.enabled
        ]
    else:
        assets = assets_from_tickers(os.getenv("TICKERS", DEFAULT_TICKERS))

    if not assets:
        raise RuntimeError("No enabled assets configured")
    return assets


def asset_type_label(asset_type: str) -> str:
    return {
        "US_STOCK": "美股",
        "CN_FUND": "中国基金",
        "CN_ETF": "中国ETF",
    }.get(asset_type, asset_type)


def is_safe_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def alpha_get(params: dict[str, Any], api_key: str) -> dict[str, Any]:
    payload = dict(params)
    payload["apikey"] = api_key
    try:
        response = requests.get(ALPHA_VANTAGE_URL, params=payload, timeout=TIMEOUT)
    except requests.RequestException as exc:
        function_name = payload.get("function", "unknown")
        raise RuntimeError(f"Alpha Vantage request failed for {function_name}") from exc

    if response.status_code >= 400:
        function_name = payload.get("function", "unknown")
        raise RuntimeError(
            f"Alpha Vantage HTTP {response.status_code} for {function_name}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        function_name = payload.get("function", "unknown")
        raise RuntimeError(f"Alpha Vantage returned invalid JSON for {function_name}") from exc

    error = data.get("Error Message") or data.get("Note") or data.get("Information")
    if error:
        function_name = payload.get("function", "unknown")
        raise RuntimeError(f"Alpha Vantage response for {function_name}: {error}")
    return data


def fetch_daily(ticker: str, api_key: str) -> pd.DataFrame:
    data = alpha_get(
        {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": "compact",
        },
        api_key,
    )
    series = data.get("Time Series (Daily)")
    if not series:
        raise RuntimeError(f"No daily time series returned for {ticker}")

    rows = []
    for date_str, values in series.items():
        rows.append(
            {
                "date": pd.Timestamp(date_str),
                "open": safe_float(values.get("1. open")),
                "high": safe_float(values.get("2. high")),
                "low": safe_float(values.get("3. low")),
                "close": safe_float(values.get("4. close")),
                "volume": safe_float(values.get("5. volume")),
            }
        )

    frame = pd.DataFrame(rows).sort_values("date").set_index("date")
    if len(frame) < 65:
        raise RuntimeError(f"Insufficient price history for {ticker}: {len(frame)} rows")
    return frame


def import_akshare():
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError(
            "akshare is required for CN_FUND/CN_ETF assets. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return ak


def find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def numeric_series(frame: pd.DataFrame, column: str | None, default: float = 0.0) -> pd.Series:
    if column is None:
        return pd.Series(default, index=frame.index, dtype=float)
    values = frame[column].astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(values, errors="coerce")


def normalize_price_frame(
    raw: pd.DataFrame,
    symbol: str,
    date_candidates: tuple[str, ...],
    close_candidates: tuple[str, ...],
    open_candidates: tuple[str, ...] = ("开盘", "open"),
    high_candidates: tuple[str, ...] = ("最高", "high"),
    low_candidates: tuple[str, ...] = ("最低", "low"),
    volume_candidates: tuple[str, ...] = ("成交量", "volume"),
) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise RuntimeError(f"No price history returned for {symbol}")

    date_col = find_column(raw, date_candidates)
    close_col = find_column(raw, close_candidates)
    if date_col is None or close_col is None:
        columns = ", ".join(str(column) for column in raw.columns)
        raise RuntimeError(f"Unexpected price columns for {symbol}: {columns}")

    close = numeric_series(raw, close_col)
    open_col = find_column(raw, open_candidates)
    high_col = find_column(raw, high_candidates)
    low_col = find_column(raw, low_candidates)
    volume_col = find_column(raw, volume_candidates)
    open_values = numeric_series(raw, open_col).fillna(close) if open_col else close
    high_values = numeric_series(raw, high_col).fillna(close) if high_col else close
    low_values = numeric_series(raw, low_col).fillna(close) if low_col else close
    volume_values = (
        numeric_series(raw, volume_col)
        if volume_col
        else pd.Series(0.0, index=raw.index, dtype=float)
    )

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col], errors="coerce"),
            "open": open_values,
            "high": high_values,
            "low": low_values,
            "close": close,
            "volume": volume_values,
        }
    )
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").set_index("date")
    if len(frame) < 65:
        raise RuntimeError(f"Insufficient price history for {symbol}: {len(frame)} rows")
    return frame


def fetch_cn_fund_daily(symbol: str) -> pd.DataFrame:
    ak = import_akshare()
    raw = ak.fund_open_fund_info_em(symbol=symbol, indicator="单位净值走势")
    return normalize_price_frame(
        raw,
        symbol,
        date_candidates=("净值日期", "日期", "date"),
        close_candidates=("单位净值", "净值", "累计净值", "close"),
    )


def fetch_cn_etf_daily(symbol: str) -> pd.DataFrame:
    ak = import_akshare()
    end_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    raw = ak.fund_etf_hist_em(
        symbol=symbol,
        period="daily",
        start_date="20000101",
        end_date=end_date,
        adjust="qfq",
    )
    return normalize_price_frame(
        raw,
        symbol,
        date_candidates=("日期", "date"),
        close_candidates=("收盘", "close"),
    )


def fetch_asset_daily(asset: Asset, api_key: str | None) -> pd.DataFrame:
    if asset.asset_type == "US_STOCK":
        if not api_key:
            raise RuntimeError("ALPHAVANTAGE_API_KEY is required for US_STOCK assets")
        return fetch_daily(asset.symbol, api_key)
    if asset.asset_type == "CN_FUND":
        return fetch_cn_fund_daily(asset.symbol)
    if asset.asset_type == "CN_ETF":
        return fetch_cn_etf_daily(asset.symbol)
    raise RuntimeError(f"Unsupported asset type {asset.asset_type}")


def fetch_news(
    tickers: list[str], api_key: str, lookback_days: int = 7
) -> tuple[dict[str, dict[str, float]], list[dict[str, str]]]:
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    data = alpha_get(
        {
            "function": "NEWS_SENTIMENT",
            "tickers": ",".join(tickers),
            "time_from": since.strftime("%Y%m%dT%H%M"),
            "sort": "LATEST",
            "limit": 200,
        },
        api_key,
    )

    scores: dict[str, list[float]] = {ticker: [] for ticker in tickers}
    negative_counts = {ticker: 0 for ticker in tickers}
    total_counts = {ticker: 0 for ticker in tickers}
    top_news: list[dict[str, str]] = []

    for article in data.get("feed", []):
        title = str(article.get("title", "")).strip()
        url = str(article.get("url", "")).strip()
        source = str(article.get("source", "")).strip()
        published = str(article.get("time_published", "")).strip()

        matched_any = False
        for item in article.get("ticker_sentiment", []):
            ticker = str(item.get("ticker", "")).upper()
            if ticker not in scores:
                continue

            relevance = safe_float(item.get("relevance_score"))
            if relevance < 0.10:
                continue

            score = safe_float(
                item.get("ticker_sentiment_score"),
                safe_float(article.get("overall_sentiment_score")),
            )
            scores[ticker].append(score)
            total_counts[ticker] += 1
            if score <= -0.15:
                negative_counts[ticker] += 1
            matched_any = True

        if matched_any and title and len(top_news) < 5:
            top_news.append(
                {
                    "title": title,
                    "url": url,
                    "source": source,
                    "published": published,
                }
            )

    summary = {}
    for ticker in tickers:
        values = scores[ticker]
        summary[ticker] = {
            "mean_score": float(np.mean(values)) if values else 0.0,
            "negative_ratio": (
                negative_counts[ticker] / total_counts[ticker]
                if total_counts[ticker]
                else 0.0
            ),
            "article_count": total_counts[ticker],
        }
    return summary, top_news


def rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gains = delta.clip(lower=0).rolling(period).mean()
    losses = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gains / losses.replace(0, np.nan)
    value = 100 - 100 / (1 + rs)
    latest = value.iloc[-1]
    if pd.isna(latest):
        latest_gain = gains.iloc[-1]
        latest_loss = losses.iloc[-1]
        if pd.isna(latest_gain) or pd.isna(latest_loss):
            return 50.0
        if latest_gain == 0 and latest_loss == 0:
            return 50.0
        if latest_loss == 0:
            return 100.0
        if latest_gain == 0:
            return 0.0
        return 50.0
    return safe_float(latest, 50.0)


def calculate_signal(
    ticker: str,
    frame: pd.DataFrame,
    news: dict[str, float],
    asset: Asset | None = None,
) -> Signal:
    close = frame["close"]
    volume = frame["volume"]

    current = float(close.iloc[-1])
    ret_5d = safe_ratio(current, float(close.iloc[-6]), 1.0) - 1
    ret_20d = safe_ratio(current, float(close.iloc[-21]), 1.0) - 1
    ret_60d = safe_ratio(current, float(close.iloc[-61]), 1.0) - 1

    rolling_peak = close.iloc[-60:].cummax()
    drawdown = close.iloc[-60:] / rolling_peak.replace(0, np.nan) - 1
    drawdown_60d = safe_float(drawdown.min())

    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    ma20_gap = safe_ratio(current, ma20, 1.0) - 1
    ma50_gap = safe_ratio(current, ma50, 1.0) - 1

    volume_5 = float(volume.iloc[-5:].mean())
    volume_20 = float(volume.iloc[-20:].mean())
    volume_ratio = safe_ratio(volume_5, volume_20, 1.0) if volume_20 > 0 else 1.0

    news_score = safe_float(news.get("mean_score"))
    negative_ratio = safe_float(news.get("negative_ratio"))

    return Signal(
        ticker=ticker,
        close=current,
        ret_5d=ret_5d,
        ret_20d=ret_20d,
        ret_60d=ret_60d,
        drawdown_60d=drawdown_60d,
        rsi_14=rsi(close),
        ma20_gap=ma20_gap,
        ma50_gap=ma50_gap,
        volume_ratio_5_20=volume_ratio,
        news_score=news_score,
        negative_news_ratio=negative_ratio,
        asset_type=asset.asset_type if asset else "US_STOCK",
        name_zh=asset.name_zh if asset else "",
        name_en=asset.name_en if asset else "",
        note=asset.note if asset else "",
        price_as_of=frame.index.max().strftime("%Y-%m-%d"),
    )


def softmax(scores: dict[str, float], temperature: float = 0.85) -> dict[str, float]:
    keys = list(scores)
    values = np.array(
        [safe_float(scores[key]) / temperature for key in keys], dtype=float
    )
    if len(values) == 0:
        return {}
    if not np.all(np.isfinite(values)):
        return {key: 1.0 / len(keys) for key in keys}
    values -= values.max()
    exp_values = np.exp(values)
    total = exp_values.sum()
    if not np.isfinite(total) or total <= 0:
        return {key: 1.0 / len(keys) for key in keys}
    probabilities = exp_values / total
    return {key: float(value) for key, value in zip(keys, probabilities)}


def score_signal(signal: Signal, breadth: float) -> Signal:
    breadth = clip01(breadth)
    # Each component is intentionally transparent and bounded to [0, 1].
    breakdown = np.mean(
        [
            float(signal.ma20_gap < 0),
            float(signal.ma50_gap < 0),
            clip01(-signal.ret_20d / 0.20),
            clip01(-signal.drawdown_60d / 0.35),
            clip01((signal.volume_ratio_5_20 - 1.0) / 1.5),
        ]
    )

    oversold = np.mean(
        [
            clip01((42.0 - signal.rsi_14) / 22.0),
            clip01(-signal.ret_5d / 0.15),
            clip01(-signal.ma20_gap / 0.15),
        ]
    )

    news_risk = np.mean(
        [
            clip01((-signal.news_score + 0.10) / 0.60),
            clip01(signal.negative_news_ratio / 0.60),
        ]
    )

    positive_momentum = np.mean(
        [
            clip01(signal.ret_60d / 0.30),
            clip01(signal.ma20_gap / 0.15),
            clip01(signal.ma50_gap / 0.25),
        ]
    )
    breadth_risk = 1.0 - breadth

    raw_scores = {
        "全部卖出": (
            1.95 * breakdown
            + 1.15 * news_risk
            + 0.75 * breadth_risk
            - 0.65 * oversold
        ),
        "等待回弹后卖出或减仓": (
            1.55 * oversold
            + 0.85 * breakdown
            + 0.50 * news_risk
            + 0.35 * breadth_risk
        ),
        "留下": (
            1.35 * (1.0 - breakdown)
            + 0.95 * (1.0 - news_risk)
            + 0.75 * positive_momentum
            + 0.45 * breadth
        ),
    }

    signal.breakdown_score = float(breakdown)
    signal.oversold_score = float(oversold)
    signal.news_risk_score = float(news_risk)
    signal.probabilities = softmax(raw_scores)
    signal.recommendation = max(signal.probabilities, key=signal.probabilities.get)
    return signal


def pct(value: float) -> str:
    return f"{value * 100:+.1f}%"


def probability_pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def price_age_days(as_of: str) -> int | None:
    try:
        as_of_date = datetime.strptime(as_of, "%Y-%m-%d").date()
    except ValueError:
        return None
    return (datetime.now(timezone.utc).date() - as_of_date).days


def build_report(
    signals: list[Signal],
    top_news: list[dict[str, str]],
    as_of: str,
    failures: list[FetchFailure] | None = None,
) -> tuple[str, dict[str, float]]:
    if not signals:
        raise RuntimeError("No valid ticker signals to report")

    aggregate = {
        label: float(np.mean([s.probabilities[label] for s in signals]))
        for label in signals[0].probabilities
    }
    aggregate = {
        key: value / sum(aggregate.values()) for key, value in aggregate.items()
    }
    overall = max(aggregate, key=aggregate.get)

    lines = [
        "<b>投资风险周报</b>",
        f"数据截至：{html.escape(as_of)}",
        "",
        f"<b>总体判断：{html.escape(overall)}</b>",
        " / ".join(
            f"{html.escape(label)} {probability_pct(value)}"
            for label, value in aggregate.items()
        ),
        "",
        "<b>资产信号</b>",
    ]

    age_days = price_age_days(as_of)
    if age_days is not None and age_days > STALE_PRICE_DAYS:
        lines.insert(3, f"注意：最新价格数据距今天 {age_days} 天，可能包含非交易日延迟。")
        lines.insert(4, "")

    for signal in signals:
        probabilities = signal.probabilities or {}
        lines.extend(
            [
                (
                    f"<b>{html.escape(signal.display_name())}</b>  "
                    f"{html.escape(signal.recommendation)}"
                ),
                (
                    f"{html.escape(asset_type_label(signal.asset_type))} / "
                    f"最新价 {signal.close:.2f} / "
                    f"数据日 {html.escape(signal.price_as_of or as_of)}"
                ),
                (
                    f"5日 {pct(signal.ret_5d)} / 20日 {pct(signal.ret_20d)} / "
                    f"60日 {pct(signal.ret_60d)} / RSI {signal.rsi_14:.1f}"
                ),
                (
                    f"MA20偏离 {pct(signal.ma20_gap)} / "
                    f"60日最大回撤 {pct(signal.drawdown_60d)} / "
                    f"新闻情绪 {signal.news_score:+.2f}"
                ),
                " / ".join(
                    f"{html.escape(label)} {probability_pct(value)}"
                    for label, value in probabilities.items()
                ),
            ]
        )
        if signal.note:
            lines.append(f"注：{html.escape(truncate_text(signal.note, 160))}")
        lines.append("")

    if top_news:
        lines.append("<b>本周主要新闻</b>")
        for item in top_news[:4]:
            title = html.escape(truncate_text(item["title"], 240))
            source = html.escape(truncate_text(item["source"], 80))
            raw_url = item["url"].strip()
            url = html.escape(raw_url, quote=True) if is_safe_http_url(raw_url) else ""
            if url:
                lines.append(f'- <a href="{url}">{title}</a>  {source}')
            else:
                lines.append(f"- {title}  {source}")
        lines.append("")

    if failures:
        lines.append("<b>未纳入计算的数据</b>")
        for failure in failures[:8]:
            lines.append(
                "- "
                f"{html.escape(failure.ticker)} "
                f"{html.escape(failure.stage)}: "
                f"{html.escape(truncate_text(failure.error, 180))}"
            )
        if len(failures) > 8:
            lines.append(f"- 其余 {len(failures) - 8} 项略")
        lines.append("")

    lines.extend(
        [
            "<b>解释</b>",
            "概率来自透明规则评分和 Softmax 转换，尚未经过个人持仓约束和历史概率校准。",
            "基金/ETF 暂沿用同一价格趋势框架；新闻情绪仅对支持的美股启用。",
            "用于风险监测，不构成自动交易指令。",
        ]
    )
    return "\n".join(lines), aggregate


def split_telegram_text(text: str, limit: int = TELEGRAM_SAFE_LIMIT) -> list[str]:
    limit = min(limit, TELEGRAM_MESSAGE_LIMIT)
    chunks = []
    current = ""
    for line in text.splitlines():
        if len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            while len(line) > limit:
                chunks.append(line[:limit])
                line = line[limit:]
            if line:
                current = line
            continue

        candidate = f"{current}\n{line}".strip()
        if len(candidate) > limit and current:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def send_telegram(token: str, chat_id: str, text: str) -> None:
    url = TELEGRAM_URL.format(token=token, method="sendMessage")
    for chunk in split_telegram_text(text):
        try:
            response = requests.post(
                url,
                json={
                    "chat_id": chat_id,
                    "text": chunk,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=TIMEOUT,
            )
        except requests.RequestException as exc:
            raise RuntimeError("Telegram request failed") from exc

        if response.status_code >= 400:
            raise RuntimeError(f"Telegram HTTP {response.status_code}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("Telegram returned invalid JSON") from exc
        if not payload.get("ok"):
            description = truncate_text(str(payload.get("description", "unknown")), 200)
            raise RuntimeError(f"Telegram error: {description}")


def save_history(
    signals: list[Signal],
    aggregate: dict[str, float],
    top_news: list[dict[str, str]],
    as_of: str,
    failures: list[FetchFailure] | None = None,
) -> Path:
    output_dir = Path(os.getenv("OUTPUT_DIR", "history"))
    output_dir.mkdir(parents=True, exist_ok=True)
    date_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"result_{date_stamp}.json"
    payload = {
        "as_of": as_of,
        "aggregate_probabilities": aggregate,
        "signals": [asdict(signal) for signal in signals],
        "top_news": top_news,
        "failures": [asdict(failure) for failure in (failures or [])],
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path


def main() -> int:
    telegram_token = env_required("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = env_required("TELEGRAM_CHAT_ID")

    assets = load_assets()
    needs_alpha = any(asset.asset_type == "US_STOCK" for asset in assets)
    api_key = env_required("ALPHAVANTAGE_API_KEY") if needs_alpha else os.getenv(
        "ALPHAVANTAGE_API_KEY", ""
    ).strip()

    failures: list[FetchFailure] = []
    news_symbols = [
        asset.news_symbol or asset.symbol
        for asset in assets
        if asset.asset_type == "US_STOCK"
    ]
    try:
        if news_symbols:
            news_summary, top_news = fetch_news(news_symbols, api_key)
        else:
            news_summary, top_news = {}, []
    except Exception as exc:
        failures.append(
            FetchFailure(
                ticker="ALL",
                stage="news",
                error=sanitize_error_message(exc, api_key),
            )
        )
        news_summary = neutral_news_summary(news_symbols)
        top_news = []

    signals: list[Signal] = []
    frames: dict[str, pd.DataFrame] = {}
    us_stock_symbols = {asset.symbol for asset in assets if asset.asset_type == "US_STOCK"}
    fetched_us_stocks = 0

    for asset in assets:
        try:
            frames[asset.symbol] = fetch_asset_daily(asset, api_key)
            news_key = asset.news_symbol or asset.symbol
            ticker_news = news_summary.get(
                news_key,
                neutral_news_summary([news_key])[news_key],
            )
            signals.append(
                calculate_signal(asset.symbol, frames[asset.symbol], ticker_news, asset)
            )
        except Exception as exc:
            failures.append(
                FetchFailure(
                    ticker=asset.symbol,
                    stage="daily",
                    error=sanitize_error_message(exc, api_key),
                )
            )
        # Avoid bursting the free API tier.
        if asset.asset_type == "US_STOCK":
            fetched_us_stocks += 1
        if fetched_us_stocks < len(us_stock_symbols) and asset.asset_type == "US_STOCK":
            time.sleep(13)

    if not signals:
        raise RuntimeError("No valid ticker data available")

    breadth = float(
        np.mean(
            [
                signal.ma20_gap > 0
                for signal in signals
            ]
        )
    )
    signals = [score_signal(signal, breadth) for signal in signals]

    as_of = max(frame.index.max() for frame in frames.values()).strftime("%Y-%m-%d")
    report, aggregate = build_report(signals, top_news, as_of, failures)
    output_path = save_history(signals, aggregate, top_news, as_of, failures)
    send_telegram(telegram_token, telegram_chat_id, report)

    print(report)
    print(f"\nSaved: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            "ERROR: "
            + sanitize_error_message(
                exc,
                os.getenv("ALPHAVANTAGE_API_KEY"),
                os.getenv("TELEGRAM_BOT_TOKEN"),
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)
