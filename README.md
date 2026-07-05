# Invest Risk Monitor

工作日自动抓取资产价格、新闻情绪和中国 ETF 风险因子，输出三类未校准的相对倾向，并通过 Telegram 推送：

1. 全部卖出
2. 等待回弹后卖出或减仓
3. 留下

当前版本是透明规则原型：它把价格趋势、回撤、均线、RSI、成交量和新闻情绪映射为三类规则分数，再经 softmax 归一化为合计 100% 的相对倾向，并额外为中国 ETF/基金生成 VaR/CVaR、波动和回撤预警。这些数值尚未经过历史概率校准，不代表真实发生概率，也不构成自动交易指令。

## 数据流

1. `assets.json` 定义监控资产、资产类型、中文名、英文名和给非专业用户看的备注。
2. 美股和美股 ETF 价格来自 Alpha Vantage `TIME_SERIES_DAILY`。
3. 美股和美股 ETF 新闻情绪来自 Alpha Vantage `NEWS_SENTIMENT`。
4. 中国开放式基金和中国 ETF 价格来自 akshare。
5. 中国 ETF/基金额外计算 20/60 日波动、VaR/CVaR、连续下跌、成交量放大和当前回撤。
6. `monitor.py` 生成 Telegram HTML 日报，包含美股大方向、中国 ETF 风险预警、关键链接和三分类未校准相对倾向。
7. 每次运行会把逐资产预测快照追加到 `history/prediction_snapshots.jsonl`，并在未来价格窗口成熟后追加 `history/prediction_outcomes.jsonl`。
8. GitHub Actions 在北京时间周一到周六早上运行，成功后会把两份 JSONL 审计文件提交回仓库；完整单次报告仍会随 `history/` artifact 上传。

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
- `Daily investment risk monitor`：正式运行监控

默认计划任务是 UTC 周一到周六 `00:30` 运行，也就是北京时间周一到周六 `08:30` 左右。周二到周六覆盖上一美股交易日，周一用于 A 股开盘前检查周末新闻和中国 ETF 风险。GitHub 计划任务可能延迟几分钟，这是正常现象。

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

- `全部卖出` 更重视趋势破位、回撤、放量下跌、负面新闻和监控资产组宽度恶化。
- `等待回弹后卖出或减仓` 更重视短期超跌和回弹可能。
- `留下` 更重视趋势仍在均线上方、新闻风险较低和正向动量。

当前没有自动交易功能，也不会读取个人持仓、成本价、税务或资金约束。后续若要调整权重和阈值，应先做回测和概率校准。

## 模型审计与历史验证

本项目会额外保存 append-only JSONL，方便后续用 pandas 读取：

- `history/prediction_snapshots.jsonl`：逐资产预测快照，包括 softmax 前 logits、softmax 后倾向、预测类别、原始指标、分项贡献、新闻数量、新闻情绪和监控资产组宽度。
- `history/prediction_outcomes.jsonl`：当未来价格窗口足够时，补充未来 5/10/20 日收益、最大回撤、最大反弹、20 日内最高/最低价格及先反弹还是先下跌。
- `audit_config.json`：仅保存验证用阈值，例如明显反弹、明显下跌和严重回撤阈值。这些是初始审计参数，不是历史优化后的最优参数。

生成验证报告：

```powershell
python evaluate_predictions.py --history-dir history
```

脚本会输出：

- 三个倾向分别按 0%-20%、20%-40%、40%-60%、60%-80%、80%-100% 分桶的未来收益、回撤和反弹统计。
- 三个倾向与未来收益、未来最大回撤的相关性。
- 全部资产合并、各资产分别、各年份分别统计。
- 样本数量不足提示。

本轮只做审计和历史验证，不执行 temperature scaling、isotonic regression 或其他概率校准，也不根据验证结果自动改权重。

## 中国 ETF 风险预警

中国 ETF/基金会额外生成一个不改变三分类未校准相对倾向的风险预警层：

- `VaR95` / `CVaR95`：基于历史收益的 95% 分位尾部风险，样本不足时不会强行估计。
- `20日/60日年化波动率`：观察短期波动是否明显放大。
- `当前60日回撤`：衡量距离近期高点的跌幅。
- `连续下跌天数` 和 `成交量放大`：用于识别短期风险释放或情绪拥挤。
- `关键链接`：天天基金、东方财富新闻、股吧和雪球搜索，方便人工复核。

这些预警只作为解释层，暂时不改变 `全部卖出`、`等待回弹后卖出或减仓`、`留下` 的定义、权重和阈值。

更多 Telegram 和资产扩展说明见 `docs/telegram_and_assets.md`。
