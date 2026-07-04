# Invest Risk Monitor

每周自动抓取资产价格和新闻情绪，输出三类风险判断，并通过 Telegram 推送：

1. 全部卖出
2. 等待回弹后卖出或减仓
3. 留下

当前版本是透明规则原型：它把价格趋势、回撤、均线、RSI、成交量和新闻情绪映射为相对概率。概率尚未经过历史样本校准，不构成自动交易指令。

## 数据流

1. `assets.json` 定义监控资产、资产类型、中文名、英文名和给非专业用户看的备注。
2. 美股和美股 ETF 价格来自 Alpha Vantage `TIME_SERIES_DAILY`。
3. 美股和美股 ETF 新闻情绪来自 Alpha Vantage `NEWS_SENTIMENT`。
4. 中国开放式基金和中国 ETF 价格来自 akshare。
5. `monitor.py` 计算三分类概率，生成 Telegram HTML 报告。
6. GitHub Actions 每周运行一次，并把快照写入 `history/`。

## 默认资产

默认启用四只存储相关美股：

- `MU`：美光科技，Micron Technology
- `SNDK`：闪迪，SanDisk
- `WDC`：西部数据，Western Digital
- `STX`：希捷科技，Seagate Technology

同时默认启用一组美股和中国 ETF，用于覆盖 AI、人工智能、半导体、红利、CPO/通信、机器人和储能电池方向：

- 美股 ETF：`AIQ`、`BOTZ`、`SMH`、`SCHD`、`LIT`
- 中国 ETF：`515070`、`512480`、`510880`、`515880`、`159770`、`159566`

需要删减监控范围时，把对应条目的 `enabled` 改为 `false`。需要新增基金或 ETF 时，复制一条后替换成自己的代码。

## 本地运行

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

$env:ALPHAVANTAGE_API_KEY="..."
$env:TELEGRAM_BOT_TOKEN="..."
$env:TELEGRAM_CHAT_ID="..."
$env:ASSET_CONFIG="assets.json"

python monitor.py
```

只监控美股或美股 ETF 时必须配置 `ALPHAVANTAGE_API_KEY`。如果只监控中国基金或 ETF，可以不使用 Alpha Vantage，但 GitHub Actions 默认仍会校验该 Secret。

## GitHub Actions 部署

在仓库中进入：

```text
Settings -> Secrets and variables -> Actions -> New repository secret
```

创建三个 Repository secrets：

- `ALPHAVANTAGE_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

然后到 `Actions` 页面运行：

- `Test Telegram delivery`：只测试 Telegram 是否能收到消息
- `Weekly memory-stock risk monitor`：正式运行监控

默认计划任务是每周一 `08:37 America/Los_Angeles` 运行。GitHub 计划任务可能延迟几分钟，这是正常现象。

## 资产配置格式

```json
{
  "assets": [
    {
      "symbol": "MU",
      "type": "US_STOCK",
      "name_zh": "美光科技",
      "name_en": "Micron Technology",
      "note": "存储芯片厂商，主要看 DRAM/NAND 价格周期、AI 服务器需求和库存变化。",
      "enabled": true
    }
  ]
}
```

支持的 `type`：

- `US_STOCK`：美股，价格和新闻情绪来自 Alpha Vantage
- `US_ETF`：美股 ETF，价格和新闻情绪来自 Alpha Vantage
- `CN_FUND`：中国开放式基金，净值来自 akshare
- `CN_ETF`：中国 ETF，日行情来自 akshare

不建议继续把基金硬塞进 `TICKERS`。`TICKERS` 只保留为没有 `assets.json` 时的兼容回退。

## 评分说明

评分逻辑位于 `monitor.py` 的 `score_signal()`：

- `全部卖出` 更重视趋势破位、回撤、放量下跌、负面新闻和市场宽度恶化。
- `等待回弹后卖出或减仓` 更重视短期超跌和回弹可能。
- `留下` 更重视趋势仍在均线上方、新闻风险较低和正向动量。

当前没有自动交易功能，也不会读取个人持仓、成本价、税务或资金约束。后续若要调整权重和阈值，应先做回测和概率校准。

更多 Telegram 和资产扩展说明见 `docs/telegram_and_assets.md`。
