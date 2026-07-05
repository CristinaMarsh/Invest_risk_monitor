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
from urllib.parse import quote, urlparse

import numpy as np
import pandas as pd
import requests


ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
SINA_KLINE_URL = "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketData.getKLineData"
TELEGRAM_URL = "https://api.telegram.org/bot{token}/{method}"
TIMEOUT = 30
TELEGRAM_MESSAGE_LIMIT = 4096
TELEGRAM_SAFE_LIMIT = 3900
STALE_PRICE_DAYS = 5
DEFAULT_TICKERS = "MU,SNDK,WDC,STX"
DEFAULT_ASSET_CONFIG = "assets.json"
DEFAULT_AUDIT_CONFIG = "audit_config.json"
SUPPORTED_ASSET_TYPES = {"US_STOCK", "US_ETF", "CN_FUND", "CN_ETF"}
TRADING_DAYS_PER_YEAR = 252
RISK_LOOKBACK_DAYS = 120
SPLIT_LIKE_JUMP_THRESHOLD = 0.30
LABEL_SELL_ALL = "全部卖出"
LABEL_REDUCE_ON_REBOUND = "等待回弹后卖出或减仓"
LABEL_KEEP = "留下"
LOGIT_FIELD_BY_LABEL = {
    LABEL_SELL_ALL: "sell_all",
    LABEL_REDUCE_ON_REBOUND: "reduce_on_rebound",
    LABEL_KEEP: "keep",
}

NEGATIVE_KEYWORDS = (
    "下跌",
    "回调",
    "调整",
    "风险",
    "利空",
    "减持",
    "亏损",
    "放缓",
    "承压",
    "监管",
    "制裁",
    "限制",
    "大跌",
    "跳水",
)
POSITIVE_KEYWORDS = (
    "上涨",
    "反弹",
    "利好",
    "增长",
    "突破",
    "创新高",
    "扩产",
    "订单",
    "景气",
    "回暖",
    "超预期",
    "修复",
)


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
    news_count: int = 0
    atr_14_pct: float = 0.0
    adjusted_close: float = 0.0
    asset_type: str = "US_STOCK"
    name_zh: str = ""
    name_en: str = ""
    note: str = ""
    price_as_of: str = ""
    breakdown_score: float = 0.0
    oversold_score: float = 0.0
    news_risk_score: float = 0.0
    positive_momentum_score: float = 0.0
    portfolio_breadth: float = 0.0
    raw_logits: dict[str, float] | None = None
    score_components: dict[str, float] | None = None
    indicator_contributions: dict[str, dict[str, float]] | None = None
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


@dataclass
class RiskAlert:
    ticker: str
    level: str
    direction: str
    score: float
    current_drawdown_60d: float
    annual_vol_20d: float
    annual_vol_60d: float
    vol_ratio_20_60: float
    downside_vol_20d: float
    var_95_1d: float | None
    cvar_95_1d: float | None
    var_95_5d: float | None
    cvar_95_5d: float | None
    consecutive_down_days: int
    sample_days: int
    warnings: list[str]


@dataclass
class MarketStory:
    symbol: str
    title: str
    url: str = ""
    source: str = ""
    published: str = ""
    kind: str = "news"
    sentiment: float = 0.0


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
        "US_ETF": "美股ETF",
        "CN_FUND": "中国基金",
        "CN_ETF": "中国ETF",
    }.get(asset_type, asset_type)


def uses_alpha_vantage(asset_type: str) -> bool:
    return asset_type in {"US_STOCK", "US_ETF"}


def is_safe_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def safe_link(url: str, label: str) -> str:
    if not is_safe_http_url(url):
        return html.escape(label)
    return f'<a href="{html.escape(url, quote=True)}">{html.escape(label)}</a>'


def truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def headline_sentiment(text: str) -> float:
    positive = sum(1 for keyword in POSITIVE_KEYWORDS if keyword in text)
    negative = sum(1 for keyword in NEGATIVE_KEYWORDS if keyword in text)
    if positive == negative:
        return 0.0
    return float(np.clip((positive - negative) / 3.0, -1.0, 1.0))


def sentiment_label(score: float) -> str:
    if score >= 0.20:
        return "偏正"
    if score <= -0.20:
        return "偏负"
    return "中性"


def theme_query(asset: Asset | Signal) -> str:
    base = getattr(asset, "name_zh", "") or getattr(asset, "ticker", "") or getattr(
        asset, "symbol", ""
    )
    symbol = getattr(asset, "symbol", getattr(asset, "ticker", ""))
    if base and symbol and symbol not in base:
        return f"{base} {symbol}"
    return base or symbol


def key_links_for_asset(asset: Asset | Signal) -> list[tuple[str, str]]:
    symbol = getattr(asset, "symbol", getattr(asset, "ticker", ""))
    asset_type = getattr(asset, "asset_type", "")
    query = quote(theme_query(asset))
    links: list[tuple[str, str]] = []
    if asset_type in {"CN_ETF", "CN_FUND"}:
        links.extend(
            [
                ("天天基金", f"https://fund.eastmoney.com/{symbol}.html"),
                ("新闻", f"https://so.eastmoney.com/news/s?keyword={query}"),
                ("股吧", f"https://guba.eastmoney.com/list,of{symbol}.html"),
                ("雪球", f"https://xueqiu.com/k?q={query}"),
            ]
        )
    elif uses_alpha_vantage(asset_type):
        links.extend(
            [
                ("Yahoo", f"https://finance.yahoo.com/quote/{symbol}"),
                ("新闻", f"https://www.google.com/search?q={quote(symbol + ' stock news')}"),
            ]
        )
    return links


def annualized_volatility(returns: pd.Series) -> float:
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    return safe_float(clean.std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR))


def historical_var_cvar(returns: pd.Series, confidence: float = 0.95) -> tuple[float | None, float | None]:
    clean = returns.dropna()
    if len(clean) < RISK_LOOKBACK_DAYS:
        return None, None
    var_value = safe_float(clean.quantile(1.0 - confidence))
    tail = clean[clean <= var_value]
    if tail.empty:
        return var_value, None
    return var_value, safe_float(tail.mean())


def consecutive_down_days(close: pd.Series) -> int:
    diffs = close.diff().dropna()
    count = 0
    for value in reversed(diffs.tolist()):
        if safe_float(value) < 0:
            count += 1
        else:
            break
    return count


def risk_level(score: float) -> str:
    if score >= 0.72:
        return "高"
    if score >= 0.55:
        return "中高"
    if score >= 0.38:
        return "中"
    return "低"


def direction_label(signal: Signal) -> str:
    if signal.ret_20d <= -0.08 and signal.ma20_gap <= -0.03:
        return "明显偏弱"
    if signal.ret_20d < 0 or signal.ma20_gap < 0:
        return "转弱"
    if signal.ret_20d > 0.05 and signal.ma20_gap > 0 and signal.rsi_14 >= 52:
        return "偏强"
    return "震荡"


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
            "function": "TIME_SERIES_DAILY_ADJUSTED",
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
                "adjusted_close": safe_float(
                    values.get("5. adjusted close"),
                    safe_float(values.get("4. close")),
                ),
                "volume": safe_float(
                    values.get("6. volume"),
                    safe_float(values.get("5. volume")),
                ),
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
            "adjusted_close": close,
            "volume": volume_values,
        }
    )
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").set_index("date")
    if len(frame) < 65:
        raise RuntimeError(f"Insufficient price history for {symbol}: {len(frame)} rows")
    return frame


def back_adjust_split_like_jumps(
    frame: pd.DataFrame,
    threshold: float = SPLIT_LIKE_JUMP_THRESHOLD,
) -> pd.DataFrame:
    if frame.empty or "close" not in frame:
        return frame

    adjusted = frame.copy()
    price_columns = [
        column for column in ("open", "high", "low", "close") if column in adjusted.columns
    ]
    volume_columns = [column for column in ("volume",) if column in adjusted.columns]
    if not price_columns:
        return adjusted

    for position in range(1, len(adjusted)):
        previous_close = float(adjusted["close"].iloc[position - 1])
        current_close = float(adjusted["close"].iloc[position])
        if previous_close <= 0 or current_close <= 0:
            continue
        ratio = current_close / previous_close
        if not math.isfinite(ratio) or ratio <= 0:
            continue
        if abs(ratio - 1.0) < threshold:
            continue

        adjusted.iloc[:position, adjusted.columns.get_indexer(price_columns)] *= ratio
        if volume_columns:
            adjusted.iloc[:position, adjusted.columns.get_indexer(volume_columns)] /= ratio

    return adjusted


def fetch_cn_fund_daily(symbol: str) -> pd.DataFrame:
    ak = import_akshare()
    raw = ak.fund_open_fund_info_em(symbol=symbol, indicator="单位净值走势")
    return normalize_price_frame(
        raw,
        symbol,
        date_candidates=("净值日期", "日期", "date"),
        close_candidates=("单位净值", "净值", "累计净值", "close"),
    )


def eastmoney_market_prefix(symbol: str) -> str:
    if symbol.startswith(("5", "6", "9")):
        return "1"
    return "0"


def sina_market_symbol(symbol: str) -> str:
    prefix = "sh" if symbol.startswith(("5", "6", "9")) else "sz"
    return f"{prefix}{symbol}"


def fetch_cn_etf_daily_sina(symbol: str) -> pd.DataFrame:
    params = {
        "symbol": sina_market_symbol(symbol),
        "scale": "240",
        "ma": "no",
        "datalen": "1500",
    }
    last_error: Exception | None = None
    response = None
    for attempt in range(3):
        try:
            response = requests.get(SINA_KLINE_URL, params=params, timeout=TIMEOUT)
            if response.status_code < 500:
                break
            last_error = RuntimeError(f"Sina HTTP {response.status_code}")
        except requests.RequestException as exc:
            last_error = exc
        if attempt < 2:
            time.sleep(1.5 * (attempt + 1))

    if response is None:
        raise RuntimeError(f"Sina request failed for {symbol}: {last_error}")
    if response.status_code >= 400:
        raise RuntimeError(f"Sina HTTP {response.status_code} for {symbol}")
    try:
        raw = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Sina returned invalid JSON for {symbol}") from exc
    if not raw:
        raise RuntimeError(f"No Sina ETF daily history returned for {symbol}")
    frame = pd.DataFrame(raw).rename(
        columns={
            "day": "日期",
            "open": "开盘",
            "high": "最高",
            "low": "最低",
            "close": "收盘",
            "volume": "成交量",
        }
    )
    normalized = normalize_price_frame(
        frame,
        symbol,
        date_candidates=("日期", "date"),
        close_candidates=("收盘", "close"),
    )
    return back_adjust_split_like_jumps(normalized)


def fetch_cn_etf_daily_eastmoney(symbol: str) -> pd.DataFrame:
    end_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    params = {
        "secid": f"{eastmoney_market_prefix(symbol)}.{symbol}",
        "klt": "101",
        "fqt": "1",
        "beg": "20000101",
        "end": end_date,
        "lmt": "1000",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
        ),
        "Referer": "https://quote.eastmoney.com/",
    }
    last_error: Exception | None = None
    response = None
    for attempt in range(3):
        try:
            response = requests.get(
                EASTMONEY_KLINE_URL,
                params=params,
                headers=headers,
                timeout=TIMEOUT,
            )
            if response.status_code < 500:
                break
            last_error = RuntimeError(f"EastMoney HTTP {response.status_code}")
        except requests.RequestException as exc:
            last_error = exc
        if attempt < 2:
            time.sleep(1.5 * (attempt + 1))

    if response is None:
        raise RuntimeError(f"EastMoney request failed for {symbol}: {last_error}")
    if response.status_code >= 400:
        raise RuntimeError(f"EastMoney HTTP {response.status_code} for {symbol}")
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(f"EastMoney returned invalid JSON for {symbol}") from exc

    klines = (data.get("data") or {}).get("klines") or []
    if not klines:
        raise RuntimeError(f"No EastMoney ETF daily history returned for {symbol}")

    rows = []
    for line in klines:
        parts = str(line).split(",")
        if len(parts) < 6:
            continue
        rows.append(
            {
                "日期": parts[0],
                "开盘": parts[1],
                "收盘": parts[2],
                "最高": parts[3],
                "最低": parts[4],
                "成交量": parts[5],
            }
        )
    return normalize_price_frame(
        pd.DataFrame(rows),
        symbol,
        date_candidates=("日期", "date"),
        close_candidates=("收盘", "close"),
    )


def fetch_cn_etf_daily(symbol: str) -> pd.DataFrame:
    errors: list[str] = []
    try:
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
    except Exception as akshare_error:
        errors.append(f"akshare failed: {akshare_error}")
    try:
        return fetch_cn_etf_daily_eastmoney(symbol)
    except Exception as eastmoney_error:
        errors.append(f"EastMoney failed: {eastmoney_error}")
    try:
        return fetch_cn_etf_daily_sina(symbol)
    except Exception as sina_error:
        errors.append(f"Sina failed: {sina_error}")
        raise RuntimeError("; ".join(errors)) from sina_error


def fetch_asset_daily(asset: Asset, api_key: str | None) -> pd.DataFrame:
    if uses_alpha_vantage(asset.asset_type):
        if not api_key:
            raise RuntimeError(
                f"ALPHAVANTAGE_API_KEY is required for {asset.asset_type} assets"
            )
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


def story_from_row(
    row: pd.Series,
    symbol: str,
    kind: str,
    title_candidates: tuple[str, ...],
    url_candidates: tuple[str, ...],
    source_candidates: tuple[str, ...],
    published_candidates: tuple[str, ...],
) -> MarketStory | None:
    title_col = find_column(pd.DataFrame([row]), title_candidates)
    if title_col is None:
        return None
    title = str(row.get(title_col, "")).strip()
    if not title:
        return None

    url_col = find_column(pd.DataFrame([row]), url_candidates)
    source_col = find_column(pd.DataFrame([row]), source_candidates)
    published_col = find_column(pd.DataFrame([row]), published_candidates)
    url = str(row.get(url_col, "")).strip() if url_col else ""
    source = str(row.get(source_col, "")).strip() if source_col else ""
    published = str(row.get(published_col, "")).strip() if published_col else ""
    return MarketStory(
        symbol=symbol,
        title=title,
        url=url if is_safe_http_url(url) else "",
        source=source,
        published=published,
        kind=kind,
        sentiment=headline_sentiment(title),
    )


def fetch_cn_asset_news(ak: Any, asset: Asset, limit: int = 2) -> list[MarketStory]:
    if not hasattr(ak, "stock_news_em"):
        return []
    raw = ak.stock_news_em(symbol=asset.symbol)
    if raw is None or raw.empty:
        return []

    stories: list[MarketStory] = []
    for _, row in raw.head(max(limit * 3, limit)).iterrows():
        story = story_from_row(
            row,
            asset.symbol,
            "news",
            title_candidates=("新闻标题", "标题", "title", "Title"),
            url_candidates=("新闻链接", "链接", "url", "URL"),
            source_candidates=("文章来源", "来源", "source", "Source"),
            published_candidates=("发布时间", "时间", "日期", "date", "Date"),
        )
        if story:
            stories.append(story)
        if len(stories) >= limit:
            break
    return stories


def row_matches_asset(row: pd.Series, asset: Asset) -> bool:
    text = " ".join(str(value) for value in row.values if pd.notna(value))
    candidates = {asset.symbol, asset.name_zh, asset.name_en}
    candidates.update(part for part in asset.note.replace("，", " ").split() if len(part) >= 2)
    return any(candidate and candidate in text for candidate in candidates)


def fetch_cn_social_mentions(ak: Any, assets: list[Asset], limit: int = 6) -> list[MarketStory]:
    stories: list[MarketStory] = []
    for function_name in (
        "stock_hot_rank_em",
        "stock_hot_tweet_xq",
        "stock_hot_follow_xq",
        "stock_hot_deal_xq",
    ):
        function = getattr(ak, function_name, None)
        if function is None:
            continue

        raw = None
        for kwargs in ({}, {"symbol": "最热门"}):
            try:
                raw = function(**kwargs)
                break
            except TypeError:
                continue
            except Exception:
                raw = None
                break
        if raw is None or getattr(raw, "empty", True):
            continue

        for _, row in raw.head(80).iterrows():
            matched = next((asset for asset in assets if row_matches_asset(row, asset)), None)
            if matched is None:
                continue
            story = story_from_row(
                row,
                matched.symbol,
                "social",
                title_candidates=("股票简称", "简称", "名称", "标题", "内容", "关注", "讨论", "symbol"),
                url_candidates=("链接", "url", "URL"),
                source_candidates=("来源", "平台", "source", "Source"),
                published_candidates=("时间", "日期", "发布时间", "date", "Date"),
            )
            if story is None:
                title = truncate_text(" ".join(str(value) for value in row.values if pd.notna(value)), 120)
                story = MarketStory(
                    symbol=matched.symbol,
                    title=title,
                    source=function_name,
                    kind="social",
                    sentiment=headline_sentiment(title),
                )
            stories.append(story)
            if len(stories) >= limit:
                return stories
    return stories


def fetch_cn_market_stories(assets: list[Asset]) -> list[MarketStory]:
    cn_assets = [asset for asset in assets if asset.asset_type in {"CN_ETF", "CN_FUND"}]
    if not cn_assets:
        return []
    ak = import_akshare()

    stories: list[MarketStory] = []
    for asset in cn_assets:
        try:
            stories.extend(fetch_cn_asset_news(ak, asset, limit=2))
        except Exception:
            continue
    try:
        stories.extend(fetch_cn_social_mentions(ak, cn_assets, limit=6))
    except Exception:
        pass
    return stories[:16]


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


def atr_pct(frame: pd.DataFrame, period: int = 14) -> float:
    if not {"high", "low", "close"}.issubset(frame.columns):
        return 0.0

    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    latest_atr = true_range.rolling(period).mean().iloc[-1]
    latest_close = close.iloc[-1]
    return safe_ratio(safe_float(latest_atr), safe_float(latest_close), 0.0)


def calculate_signal(
    ticker: str,
    frame: pd.DataFrame,
    news: dict[str, float],
    asset: Asset | None = None,
) -> Signal:
    close = frame["close"]
    volume = frame["volume"]
    adjusted_close = frame["adjusted_close"] if "adjusted_close" in frame else close

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
    news_count = int(safe_float(news.get("article_count")))

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
        news_count=news_count,
        atr_14_pct=atr_pct(frame),
        adjusted_close=float(adjusted_close.iloc[-1]),
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


def contribution_row(
    sell_all: float = 0.0,
    reduce_on_rebound: float = 0.0,
    keep: float = 0.0,
) -> dict[str, float]:
    return {
        "sell_all": safe_float(sell_all),
        "reduce_on_rebound": safe_float(reduce_on_rebound),
        "keep": safe_float(keep),
    }


def score_signal(signal: Signal, portfolio_breadth: float) -> Signal:
    portfolio_breadth = clip01(portfolio_breadth)
    # Each component is intentionally transparent and bounded to [0, 1].
    breakdown_components = {
        "ma20_below_average": float(signal.ma20_gap < 0),
        "ma50_below_average": float(signal.ma50_gap < 0),
        "ret_20d_negative": clip01(-signal.ret_20d / 0.20),
        "drawdown_60d_depth": clip01(-signal.drawdown_60d / 0.35),
        "volume_expansion": clip01((signal.volume_ratio_5_20 - 1.0) / 1.5),
    }
    oversold_components = {
        "rsi_14_oversold": clip01((42.0 - signal.rsi_14) / 22.0),
        "ret_5d_negative": clip01(-signal.ret_5d / 0.15),
        "ma20_gap_negative": clip01(-signal.ma20_gap / 0.15),
    }
    news_components = {
        "news_sentiment_risk": clip01((-signal.news_score + 0.10) / 0.60),
        "negative_news_ratio": clip01(signal.negative_news_ratio / 0.60),
    }
    positive_momentum_components = {
        "ret_60d_positive": clip01(signal.ret_60d / 0.30),
        "ma20_gap_positive": clip01(signal.ma20_gap / 0.15),
        "ma50_gap_positive": clip01(signal.ma50_gap / 0.25),
    }

    breakdown = np.mean(list(breakdown_components.values()))
    oversold = np.mean(list(oversold_components.values()))
    news_risk = np.mean(list(news_components.values()))
    positive_momentum = np.mean(list(positive_momentum_components.values()))
    breadth_risk = 1.0 - portfolio_breadth

    raw_scores = {
        LABEL_SELL_ALL: (
            1.95 * breakdown
            + 1.15 * news_risk
            + 0.75 * breadth_risk
            - 0.65 * oversold
        ),
        LABEL_REDUCE_ON_REBOUND: (
            1.55 * oversold
            + 0.85 * breakdown
            + 0.50 * news_risk
            + 0.35 * breadth_risk
        ),
        LABEL_KEEP: (
            1.35 * (1.0 - breakdown)
            + 0.95 * (1.0 - news_risk)
            + 0.75 * positive_momentum
            + 0.45 * portfolio_breadth
        ),
    }

    indicator_contributions: dict[str, dict[str, float]] = {}
    for name, value in breakdown_components.items():
        indicator_contributions[name] = contribution_row(
            sell_all=(1.95 / len(breakdown_components)) * value,
            reduce_on_rebound=(0.85 / len(breakdown_components)) * value,
            keep=-(1.35 / len(breakdown_components)) * value,
        )
    for name, value in oversold_components.items():
        indicator_contributions[name] = contribution_row(
            sell_all=-(0.65 / len(oversold_components)) * value,
            reduce_on_rebound=(1.55 / len(oversold_components)) * value,
        )
    for name, value in news_components.items():
        indicator_contributions[name] = contribution_row(
            sell_all=(1.15 / len(news_components)) * value,
            reduce_on_rebound=(0.50 / len(news_components)) * value,
            keep=-(0.95 / len(news_components)) * value,
        )
    for name, value in positive_momentum_components.items():
        indicator_contributions[name] = contribution_row(
            keep=(0.75 / len(positive_momentum_components)) * value,
        )
    indicator_contributions["portfolio_breadth_risk"] = contribution_row(
        sell_all=0.75 * breadth_risk,
        reduce_on_rebound=0.35 * breadth_risk,
    )
    indicator_contributions["portfolio_breadth_positive"] = contribution_row(
        keep=0.45 * portfolio_breadth,
    )
    indicator_contributions["keep_intercept"] = contribution_row(keep=1.35 + 0.95)

    signal.breakdown_score = float(breakdown)
    signal.oversold_score = float(oversold)
    signal.news_risk_score = float(news_risk)
    signal.positive_momentum_score = float(positive_momentum)
    signal.portfolio_breadth = float(portfolio_breadth)
    signal.raw_logits = {key: safe_float(value) for key, value in raw_scores.items()}
    signal.score_components = {
        "breakdown": float(breakdown),
        "oversold": float(oversold),
        "news_risk": float(news_risk),
        "positive_momentum": float(positive_momentum),
        "portfolio_breadth": float(portfolio_breadth),
        "portfolio_breadth_risk": float(breadth_risk),
    }
    signal.indicator_contributions = indicator_contributions
    signal.probabilities = softmax(raw_scores)
    signal.recommendation = max(signal.probabilities, key=signal.probabilities.get)
    return signal


def calculate_risk_alert(signal: Signal, frame: pd.DataFrame) -> RiskAlert:
    close = frame["close"]
    returns_1d = close.pct_change().dropna()
    returns_5d = close.pct_change(5).dropna()
    sample_days = len(returns_1d)

    recent = close.iloc[-60:]
    current_peak = float(recent.max())
    current_drawdown = safe_ratio(float(close.iloc[-1]), current_peak, 1.0) - 1.0

    vol_20 = annualized_volatility(returns_1d.iloc[-20:])
    vol_60 = annualized_volatility(returns_1d.iloc[-60:])
    downside_vol_20 = annualized_volatility(returns_1d.iloc[-20:].clip(upper=0))
    vol_ratio = safe_ratio(vol_20, vol_60, 1.0) if vol_60 > 0 else 1.0

    var_1d, cvar_1d = historical_var_cvar(returns_1d.iloc[-RISK_LOOKBACK_DAYS:])
    var_5d, cvar_5d = historical_var_cvar(returns_5d.iloc[-RISK_LOOKBACK_DAYS:])

    warnings: list[str] = []
    if sample_days < RISK_LOOKBACK_DAYS:
        warnings.append("样本不足，暂不估计CVaR")
    if cvar_5d is not None and signal.ret_5d <= cvar_5d:
        warnings.append("近5日跌幅进入历史尾部区间")
    if signal.ma20_gap < 0 and signal.ret_20d < 0:
        warnings.append("短期趋势转弱")
    if vol_ratio >= 1.35:
        warnings.append("20日波动率明显高于60日")
    if signal.volume_ratio_5_20 >= 1.8:
        warnings.append("成交量显著放大")
    down_days = consecutive_down_days(close)
    if down_days >= 3:
        warnings.append(f"连续下跌{down_days}日")
    if current_drawdown <= -0.12:
        warnings.append("距离60日高点回撤较深")

    cvar_component = (
        clip01((-(cvar_5d or 0.0) - 0.04) / 0.10) if cvar_5d is not None else 0.35
    )
    score = float(
        np.mean(
            [
                clip01(-signal.ret_20d / 0.12),
                clip01(-current_drawdown / 0.18),
                clip01((vol_ratio - 1.0) / 0.75),
                clip01(downside_vol_20 / 0.45),
                clip01((signal.volume_ratio_5_20 - 1.0) / 1.5),
                cvar_component,
            ]
        )
    )

    return RiskAlert(
        ticker=signal.ticker,
        level=risk_level(score),
        direction=direction_label(signal),
        score=score,
        current_drawdown_60d=current_drawdown,
        annual_vol_20d=vol_20,
        annual_vol_60d=vol_60,
        vol_ratio_20_60=vol_ratio,
        downside_vol_20d=downside_vol_20,
        var_95_1d=var_1d,
        cvar_95_1d=cvar_1d,
        var_95_5d=var_5d,
        cvar_95_5d=cvar_5d,
        consecutive_down_days=down_days,
        sample_days=sample_days,
        warnings=warnings,
    )


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


def short_probability_line(probabilities: dict[str, float]) -> str:
    labels = {
        LABEL_SELL_ALL: "卖出",
        LABEL_REDUCE_ON_REBOUND: "等待",
        LABEL_KEEP: "留下",
    }
    return " / ".join(
        f"{labels.get(label, label)} {probability_pct(value)}"
        for label, value in probabilities.items()
    )


def optional_pct(value: float | None) -> str:
    if value is None:
        return "样本不足"
    return pct(value)


def stories_by_symbol(stories: list[MarketStory]) -> dict[str, list[MarketStory]]:
    grouped: dict[str, list[MarketStory]] = {}
    for story in stories:
        grouped.setdefault(story.symbol, []).append(story)
    return grouped


def story_sentiment_summary(stories: list[MarketStory]) -> str:
    if not stories:
        return "暂无可用新闻/热度数据"
    score = float(np.mean([story.sentiment for story in stories]))
    news_count = sum(1 for story in stories if story.kind == "news")
    social_count = sum(1 for story in stories if story.kind == "social")
    return (
        f"{sentiment_label(score)} "
        f"(新闻{news_count}条，热度{social_count}条，标题情绪{score:+.2f})"
    )


def us_market_summary(signals: list[Signal]) -> list[str]:
    us_signals = [signal for signal in signals if uses_alpha_vantage(signal.asset_type)]
    if not us_signals:
        return []

    growth_symbols = {"AIQ", "BOTZ", "SMH", "LIT", "MU", "SNDK", "WDC", "STX"}
    defensive_symbols = {"SCHD"}
    growth = [signal for signal in us_signals if signal.ticker in growth_symbols]
    defensive = [signal for signal in us_signals if signal.ticker in defensive_symbols]
    growth_ret20 = float(np.mean([signal.ret_20d for signal in growth])) if growth else 0.0
    defensive_ret20 = (
        float(np.mean([signal.ret_20d for signal in defensive])) if defensive else 0.0
    )
    monitored_us_breadth = float(np.mean([signal.ma20_gap > 0 for signal in us_signals]))
    news_score = float(np.mean([signal.news_score for signal in us_signals]))
    negative_ratio = float(np.mean([signal.negative_news_ratio for signal in us_signals]))

    spread = growth_ret20 - defensive_ret20
    if spread >= 0.03 and monitored_us_breadth >= 0.55:
        appetite = "偏强"
    elif spread <= -0.03 or monitored_us_breadth <= 0.35:
        appetite = "偏弱"
    else:
        appetite = "中性"

    return [
        f"风险偏好：{appetite}",
        f"成长/科技20日 {pct(growth_ret20)}，红利防御20日 {pct(defensive_ret20)}，相对强弱 {pct(spread)}",
        f"美股监控资产组宽度 {monitored_us_breadth * 100:.0f}%，新闻情绪 {news_score:+.2f}，负面新闻占比 {negative_ratio * 100:.0f}%",
    ]


def build_report(
    signals: list[Signal],
    top_news: list[dict[str, str]],
    as_of: str,
    failures: list[FetchFailure] | None = None,
    risk_alerts: dict[str, RiskAlert] | None = None,
    market_stories: list[MarketStory] | None = None,
) -> tuple[str, dict[str, float]]:
    if not signals:
        raise RuntimeError("No valid ticker signals to report")

    risk_alerts = risk_alerts or {}
    market_stories = market_stories or []
    grouped_stories = stories_by_symbol(market_stories)
    aggregate = {
        label: float(np.mean([s.probabilities[label] for s in signals]))
        for label in signals[0].probabilities
    }
    aggregate = {
        key: value / sum(aggregate.values()) for key, value in aggregate.items()
    }
    overall = max(aggregate, key=aggregate.get)

    lines = [
        "<b>投资风险日报</b>",
        f"数据截至：{html.escape(as_of)}",
        "",
        f"<b>模型倾向：{html.escape(overall)}</b>",
        "三分类未校准的相对倾向：" + short_probability_line(aggregate),
        "说明：三类数值由规则分数经过 softmax 归一化得到，合计为100%，尚未经过历史概率校准，不代表真实发生概率。",
        "这是风险监测和信息整理，不是自动交易指令。",
        "",
    ]

    age_days = price_age_days(as_of)
    if age_days is not None and age_days > STALE_PRICE_DAYS:
        lines.insert(3, f"注意：最新价格数据距今天 {age_days} 天，可能包含非交易日延迟。")
        lines.insert(4, "")

    us_summary = us_market_summary(signals)
    if us_summary:
        lines.append("<b>美股大方向</b>")
        lines.extend(us_summary)
        lines.append("")

    cn_signals = [signal for signal in signals if signal.asset_type in {"CN_ETF", "CN_FUND"}]
    if cn_signals:
        lines.append("<b>中国ETF/基金风险预警</b>")
    for signal in cn_signals:
        alert = risk_alerts.get(signal.ticker)
        stories = grouped_stories.get(signal.ticker, [])
        links = " / ".join(
            safe_link(url, label) for label, url in key_links_for_asset(signal)
        )
        if alert:
            warning_text = "；".join(alert.warnings[:3]) if alert.warnings else "暂无明显极端预警"
            lines.extend(
                [
                    (
                        f"<b>{html.escape(signal.display_name())}</b> "
                        f"预警 {html.escape(alert.level)} / 走向 {html.escape(alert.direction)}"
                    ),
                    (
                        f"价格：5日 {pct(signal.ret_5d)} / 20日 {pct(signal.ret_20d)} / "
                        f"当前回撤 {pct(alert.current_drawdown_60d)} / RSI {signal.rsi_14:.1f}"
                    ),
                    (
                        f"风险：CVaR95(5日) {optional_pct(alert.cvar_95_5d)} / "
                        f"20日波动 {pct(alert.annual_vol_20d)} / "
                        f"波动放大 {alert.vol_ratio_20_60:.2f}x"
                    ),
                    f"新闻/热度：{html.escape(story_sentiment_summary(stories))}",
                    f"预警依据：{html.escape(warning_text)}",
                    f"关键链接：{links}",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    f"<b>{html.escape(signal.display_name())}</b>",
                    f"价格：5日 {pct(signal.ret_5d)} / 20日 {pct(signal.ret_20d)} / RSI {signal.rsi_14:.1f}",
                    f"新闻/热度：{html.escape(story_sentiment_summary(stories))}",
                    f"关键链接：{links}",
                    "",
                ]
            )

    if market_stories:
        lines.append("<b>中国新闻/热度摘录</b>")
        for story in market_stories[:8]:
            title = html.escape(truncate_text(story.title, 180))
            source = html.escape(truncate_text(story.source or story.kind, 60))
            label = "新闻" if story.kind == "news" else "热度"
            if story.url and is_safe_http_url(story.url):
                lines.append(
                    f'- [{html.escape(story.symbol)} {label}] '
                    f'<a href="{html.escape(story.url, quote=True)}">{title}</a>  {source}'
                )
            else:
                lines.append(f"- [{html.escape(story.symbol)} {label}] {title}  {source}")
        lines.append("")

    lines.append("<b>单资产未校准相对倾向</b>")
    for signal in signals:
        probabilities = signal.probabilities or {}
        lines.extend(
            [
                (
                    f"<b>{html.escape(signal.display_name())}</b>  "
                    f"倾向 {html.escape(signal.recommendation)}"
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
                short_probability_line(probabilities),
            ]
        )
        if signal.note:
            lines.append(f"注：{html.escape(truncate_text(signal.note, 160))}")
        lines.append("")

    if top_news:
        lines.append("<b>美股最新新闻</b>")
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
            "三类数值由规则分数经过 softmax 归一化得到，合计为100%，尚未经过历史概率校准，不代表真实发生概率。",
            "中国ETF/基金预警使用趋势、回撤、波动、VaR/CVaR、成交量和新闻/热度链接做辅助判断。",
            "新闻和社交热度只作为解释层，暂不改变三分类评分权重和阈值。",
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


def load_audit_config(path: str | None = None) -> dict[str, Any]:
    config_path = Path(path or os.getenv("AUDIT_CONFIG") or DEFAULT_AUDIT_CONFIG)
    if not config_path.exists():
        raise RuntimeError(f"Missing audit config file: {config_path}")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "observation_windows",
        "clear_rebound_threshold_pct",
        "clear_drawdown_threshold_pct",
        "clear_rebound_threshold_atr",
        "clear_drawdown_threshold_atr",
        "severe_drawdown_threshold_pct",
        "min_bucket_samples",
    }
    missing = sorted(required - set(data))
    if missing:
        raise RuntimeError(f"Audit config missing keys: {', '.join(missing)}")

    windows = sorted({int(value) for value in data["observation_windows"]})
    if not windows or min(windows) <= 0:
        raise RuntimeError("Audit config observation_windows must contain positive integers")
    data["observation_windows"] = windows
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def compact_label_values(values: dict[str, float] | None) -> dict[str, float]:
    values = values or {}
    return {
        field_name: safe_float(values.get(label))
        for label, field_name in LOGIT_FIELD_BY_LABEL.items()
    }


def raw_indicator_snapshot(
    signal: Signal,
    portfolio_breadth: float,
    market_story_count: int = 0,
    market_story_sentiment: float = 0.0,
) -> dict[str, float | int]:
    return {
        "ret_5d": signal.ret_5d,
        "ret_20d": signal.ret_20d,
        "ret_60d": signal.ret_60d,
        "drawdown_60d": signal.drawdown_60d,
        "rsi_14": signal.rsi_14,
        "ma20_gap": signal.ma20_gap,
        "ma50_gap": signal.ma50_gap,
        "volume_ratio_5_20": signal.volume_ratio_5_20,
        "news_score": signal.news_score,
        "negative_news_ratio": signal.negative_news_ratio,
        "news_count": signal.news_count,
        "atr_14_pct": signal.atr_14_pct,
        "portfolio_breadth": portfolio_breadth,
        "market_story_count": market_story_count,
        "market_story_sentiment": market_story_sentiment,
    }


def audit_record_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("symbol", "")),
        str(row.get("prediction_timestamp_utc") or row.get("prediction_date", "")),
        str(row.get("price_date", "")),
    )


def build_prediction_snapshots(
    signals: list[Signal],
    prediction_timestamp: datetime,
    portfolio_breadth: float,
    market_stories: list[MarketStory] | None = None,
) -> list[dict[str, Any]]:
    grouped_stories = stories_by_symbol(market_stories or [])
    snapshots = []
    for signal in signals:
        stories = grouped_stories.get(signal.ticker, [])
        story_sentiment = (
            float(np.mean([story.sentiment for story in stories])) if stories else 0.0
        )
        logits = compact_label_values(signal.raw_logits)
        probabilities = compact_label_values(signal.probabilities)
        news_count = signal.news_count if signal.news_count else len(stories)
        news_sentiment_mean = signal.news_score if signal.news_count else story_sentiment
        snapshots.append(
            {
                "schema_version": 1,
                "symbol": signal.ticker,
                "asset_type": signal.asset_type,
                "prediction_date": prediction_timestamp.date().isoformat(),
                "prediction_timestamp_utc": prediction_timestamp.isoformat(),
                "price_date": signal.price_as_of,
                "adjusted_close": signal.adjusted_close or signal.close,
                "sell_all_logit": logits["sell_all"],
                "reduce_on_rebound_logit": logits["reduce_on_rebound"],
                "keep_logit": logits["keep"],
                "sell_all_softmax": probabilities["sell_all"],
                "reduce_on_rebound_softmax": probabilities["reduce_on_rebound"],
                "keep_softmax": probabilities["keep"],
                "predicted_class": signal.recommendation,
                "raw_indicators": raw_indicator_snapshot(
                    signal,
                    portfolio_breadth,
                    market_story_count=len(stories),
                    market_story_sentiment=story_sentiment,
                ),
                "score_components": signal.score_components or {},
                "indicator_contributions": signal.indicator_contributions or {},
                "news_count": news_count,
                "news_sentiment_mean": news_sentiment_mean,
                "negative_news_ratio": signal.negative_news_ratio,
                "portfolio_breadth": portfolio_breadth,
            }
        )
    return snapshots


def first_threshold_cross(
    relative_moves: pd.Series,
    rebound_threshold: float,
    drawdown_threshold: float,
) -> tuple[str, str | None, float | None]:
    for date_value, move in relative_moves.iloc[1:].items():
        move_value = safe_float(move)
        if move_value >= rebound_threshold:
            return "rebound", pd.Timestamp(date_value).strftime("%Y-%m-%d"), move_value
        if move_value <= drawdown_threshold:
            return "drawdown", pd.Timestamp(date_value).strftime("%Y-%m-%d"), move_value
    return "none", None, None


def compute_forward_outcome(
    snapshot: dict[str, Any],
    frame: pd.DataFrame,
    audit_config: dict[str, Any],
    evaluated_at: datetime,
) -> dict[str, Any] | None:
    close = frame["close"].dropna().sort_index()
    price_date = pd.Timestamp(snapshot.get("price_date"))
    matches = np.flatnonzero(close.index == price_date)
    if len(matches) == 0:
        return None

    windows = [int(value) for value in audit_config["observation_windows"]]
    max_window = max(windows)
    start_position = int(matches[-1])
    if start_position + max_window >= len(close):
        return None

    full_window = close.iloc[start_position : start_position + max_window + 1]
    start_price = safe_float(full_window.iloc[0])
    if start_price <= 0:
        return None
    relative_moves = full_window / start_price - 1.0

    outcome: dict[str, Any] = {
        "schema_version": 1,
        "symbol": snapshot.get("symbol"),
        "asset_type": snapshot.get("asset_type"),
        "prediction_date": snapshot.get("prediction_date"),
        "prediction_timestamp_utc": snapshot.get("prediction_timestamp_utc"),
        "price_date": snapshot.get("price_date"),
        "evaluated_at_utc": evaluated_at.isoformat(),
        "evaluation_price_date": close.index.max().strftime("%Y-%m-%d"),
        "start_price": start_price,
        "forward_path_days_available": max_window,
    }

    atr_14_pct = safe_float(
        (snapshot.get("raw_indicators") or {}).get("atr_14_pct"),
        safe_float(snapshot.get("atr_14_pct")),
    )
    for window in windows:
        window_moves = relative_moves.iloc[: window + 1]
        forward_return = safe_float(window_moves.iloc[-1])
        max_drawdown = safe_float(window_moves.min())
        max_rebound = safe_float(window_moves.max())
        outcome[f"forward_return_{window}d"] = forward_return
        outcome[f"forward_max_drawdown_{window}d"] = max_drawdown
        outcome[f"forward_max_rebound_{window}d"] = max_rebound
        outcome[f"forward_max_drawdown_{window}d_atr"] = (
            safe_ratio(max_drawdown, atr_14_pct, 0.0) if atr_14_pct > 0 else None
        )
        outcome[f"forward_max_rebound_{window}d_atr"] = (
            safe_ratio(max_rebound, atr_14_pct, 0.0) if atr_14_pct > 0 else None
        )

    min_date = relative_moves.idxmin()
    max_date = relative_moves.idxmax()
    outcome.update(
        {
            "forward_min_price_20d": safe_float(full_window.loc[min_date]),
            "forward_max_price_20d": safe_float(full_window.loc[max_date]),
            "forward_min_price_date_20d": pd.Timestamp(min_date).strftime("%Y-%m-%d"),
            "forward_max_price_date_20d": pd.Timestamp(max_date).strftime("%Y-%m-%d"),
            "atr_14_pct_at_prediction": atr_14_pct,
        }
    )

    fixed_direction, fixed_date, fixed_value = first_threshold_cross(
        relative_moves,
        safe_float(audit_config["clear_rebound_threshold_pct"]),
        safe_float(audit_config["clear_drawdown_threshold_pct"]),
    )
    outcome.update(
        {
            "first_clear_move_pct": fixed_direction,
            "first_clear_move_pct_date": fixed_date,
            "first_clear_move_pct_value": fixed_value,
            "clear_rebound_threshold_pct": safe_float(
                audit_config["clear_rebound_threshold_pct"]
            ),
            "clear_drawdown_threshold_pct": safe_float(
                audit_config["clear_drawdown_threshold_pct"]
            ),
        }
    )

    if atr_14_pct > 0:
        atr_moves = relative_moves / atr_14_pct
        atr_direction, atr_date, atr_value = first_threshold_cross(
            atr_moves,
            safe_float(audit_config["clear_rebound_threshold_atr"]),
            safe_float(audit_config["clear_drawdown_threshold_atr"]),
        )
    else:
        atr_direction, atr_date, atr_value = "none", None, None
    outcome.update(
        {
            "first_clear_move_atr": atr_direction,
            "first_clear_move_atr_date": atr_date,
            "first_clear_move_atr_value": atr_value,
            "clear_rebound_threshold_atr": safe_float(
                audit_config["clear_rebound_threshold_atr"]
            ),
            "clear_drawdown_threshold_atr": safe_float(
                audit_config["clear_drawdown_threshold_atr"]
            ),
        }
    )
    return outcome


def update_prediction_outcomes(
    snapshots_path: Path,
    outcomes_path: Path,
    frames: dict[str, pd.DataFrame],
    audit_config: dict[str, Any],
    evaluated_at: datetime,
) -> list[dict[str, Any]]:
    existing_keys = {audit_record_key(row) for row in read_jsonl(outcomes_path)}
    new_outcomes = []
    for snapshot in read_jsonl(snapshots_path):
        key = audit_record_key(snapshot)
        if key in existing_keys:
            continue
        symbol = str(snapshot.get("symbol", ""))
        frame = frames.get(symbol)
        if frame is None:
            continue
        outcome = compute_forward_outcome(snapshot, frame, audit_config, evaluated_at)
        if outcome is None:
            continue
        new_outcomes.append(outcome)
        existing_keys.add(key)
    append_jsonl(outcomes_path, new_outcomes)
    return new_outcomes


def save_history(
    signals: list[Signal],
    aggregate: dict[str, float],
    top_news: list[dict[str, str]],
    as_of: str,
    failures: list[FetchFailure] | None = None,
    risk_alerts: dict[str, RiskAlert] | None = None,
    market_stories: list[MarketStory] | None = None,
    frames: dict[str, pd.DataFrame] | None = None,
    portfolio_breadth: float = 0.0,
) -> Path:
    output_dir = Path(os.getenv("OUTPUT_DIR", "history"))
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_timestamp = datetime.now(timezone.utc)
    date_stamp = prediction_timestamp.strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"result_{date_stamp}.json"
    snapshots_path = output_dir / "prediction_snapshots.jsonl"
    outcomes_path = output_dir / "prediction_outcomes.jsonl"
    audit_config = load_audit_config()
    snapshots = build_prediction_snapshots(
        signals,
        prediction_timestamp,
        portfolio_breadth,
        market_stories=market_stories,
    )
    append_jsonl(snapshots_path, snapshots)
    new_outcomes = update_prediction_outcomes(
        snapshots_path,
        outcomes_path,
        frames or {},
        audit_config,
        prediction_timestamp,
    )
    payload = {
        "as_of": as_of,
        "aggregate_probabilities": aggregate,
        "signals": [asdict(signal) for signal in signals],
        "portfolio_breadth": portfolio_breadth,
        "prediction_snapshots_file": str(snapshots_path),
        "prediction_outcomes_file": str(outcomes_path),
        "new_prediction_snapshot_count": len(snapshots),
        "new_prediction_outcome_count": len(new_outcomes),
        "risk_alerts": {
            ticker: asdict(alert) for ticker, alert in (risk_alerts or {}).items()
        },
        "top_news": top_news,
        "market_stories": [asdict(story) for story in (market_stories or [])],
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
    needs_alpha = any(uses_alpha_vantage(asset.asset_type) for asset in assets)
    api_key = env_required("ALPHAVANTAGE_API_KEY") if needs_alpha else os.getenv(
        "ALPHAVANTAGE_API_KEY", ""
    ).strip()

    failures: list[FetchFailure] = []
    news_symbols = [
        asset.news_symbol or asset.symbol
        for asset in assets
        if uses_alpha_vantage(asset.asset_type)
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

    try:
        market_stories = fetch_cn_market_stories(assets)
    except Exception as exc:
        failures.append(
            FetchFailure(
                ticker="CN",
                stage="market_context",
                error=sanitize_error_message(exc, api_key),
            )
        )
        market_stories = []

    signals: list[Signal] = []
    frames: dict[str, pd.DataFrame] = {}
    alpha_symbols = {
        asset.symbol for asset in assets if uses_alpha_vantage(asset.asset_type)
    }
    fetched_alpha_assets = 0

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
        if uses_alpha_vantage(asset.asset_type):
            fetched_alpha_assets += 1
        if fetched_alpha_assets < len(alpha_symbols) and uses_alpha_vantage(
            asset.asset_type
        ):
            time.sleep(13)

    if not signals:
        raise RuntimeError("No valid ticker data available")

    portfolio_breadth = float(
        np.mean(
            [
                signal.ma20_gap > 0
                for signal in signals
            ]
        )
    )
    signals = [score_signal(signal, portfolio_breadth) for signal in signals]
    risk_alerts = {
        signal.ticker: calculate_risk_alert(signal, frames[signal.ticker])
        for signal in signals
        if signal.ticker in frames and signal.asset_type in {"CN_ETF", "CN_FUND"}
    }

    as_of = max(frame.index.max() for frame in frames.values()).strftime("%Y-%m-%d")
    report, aggregate = build_report(
        signals,
        top_news,
        as_of,
        failures,
        risk_alerts=risk_alerts,
        market_stories=market_stories,
    )
    output_path = save_history(
        signals,
        aggregate,
        top_news,
        as_of,
        failures,
        risk_alerts=risk_alerts,
        market_stories=market_stories,
        frames=frames,
        portfolio_breadth=portfolio_breadth,
    )
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
