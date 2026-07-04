# Telegram 与资产配置说明

## 密钥安全

不要把 `TELEGRAM_BOT_TOKEN`、`ALPHAVANTAGE_API_KEY` 或其他 API key 发到聊天窗口、README、issue、commit message 或日志里。它们只应放在 GitHub Actions Repository secrets 中。

如果 token 或 key 已经发出来，按泄露处理：重新生成 token/key，并更新 GitHub Secrets。

## 创建 Telegram Bot

1. 在 Telegram 搜索 `@BotFather`。
2. 发送 `/newbot`。
3. 按提示设置机器人名称和用户名。
4. 保存 BotFather 返回的 token，这就是 `TELEGRAM_BOT_TOKEN`。
5. 打开新建的机器人，发送 `/start`。

获取私人聊天的 `TELEGRAM_CHAT_ID`：

```text
https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/getUpdates
```

使用时不要保留尖括号。例如真实地址应类似：

```text
https://api.telegram.org/bot123456:ABC/getUpdates
```

返回 JSON 里的 `message.chat.id` 就是私人聊天 ID。

## 配置 GitHub Secrets

进入仓库：

```text
Settings -> Secrets and variables -> Actions -> New repository secret
```

创建三个 Repository secrets：

- `ALPHAVANTAGE_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

注意：

- `Name` 字段只填 secret 名称，例如 `TELEGRAM_CHAT_ID`。
- `Secret` 字段只填值本身，例如聊天 ID 数字。
- 不要把 `Name:`、`Value:`、引号或尖括号填进去。
- 不要填到 Repository variables；本项目读取的是 Secrets。

配置后先运行 `Test Telegram delivery`。成功后再运行 `Daily investment risk monitor`。

## 推送时间

正式任务默认使用 GitHub Actions cron：

```text
30 0 * * 2-6
```

这是 UTC 周二到周六 00:30，约等于美股前一交易日晚间，也通常是北京时间早上。这样美股日线和新闻更可能已经更新完，同时方便早上阅读中国 ETF 风险日报。

## 资产配置文件

资产统一写在仓库根目录的 `assets.json` 中。每个资产可以带中文名、英文名和备注，Telegram 报告会直接展示这些信息，方便不熟悉美股代码的人阅读。

示例：

```json
{
  "symbol": "MU",
  "type": "US_STOCK",
  "name_zh": "美光科技",
  "name_en": "Micron Technology",
  "note": "存储芯片厂商，主要看 DRAM/NAND 价格周期、AI 服务器需求和库存变化。",
  "enabled": true
}
```

支持的资产类型：

- `US_STOCK`：美股。价格和新闻情绪来自 Alpha Vantage。
- `US_ETF`：美股 ETF。价格和新闻情绪来自 Alpha Vantage。
- `CN_FUND`：中国开放式基金。净值走势来自 akshare。
- `CN_ETF`：中国 ETF。日行情来自 akshare。

默认美股中文名：

- `MU`：美光科技，Micron Technology
- `SNDK`：闪迪，SanDisk
- `WDC`：西部数据，Western Digital
- `STX`：希捷科技，Seagate Technology

默认主题 ETF：

- `AIQ`：Global X 人工智能与科技ETF，美股AI主题。
- `BOTZ`：Global X 机器人与人工智能ETF，美股机器人主题。
- `SMH`：VanEck 半导体ETF，美股半导体主题。
- `SCHD`：Schwab 美国红利股票ETF，美股红利主题。
- `LIT`：Global X 锂电池科技ETF，海外电池和储能产业链。
- `515070`：人工智能ETF华夏，中国人工智能主题。
- `512480`：半导体ETF国联安，中国半导体主题。
- `510880`：红利ETF华泰柏瑞，中国红利主题。
- `515880`：通信ETF国泰，覆盖通信、光模块和CPO相关链条。
- `159770`：机器人ETF天弘，中国机器人主题。
- `159566`：储能电池ETF易方达，中国储能电池主题。

## 添加中国基金

复制 `assets.json` 中的基金示例，替换基金代码和名称：

```json
{
  "symbol": "161725",
  "type": "CN_FUND",
  "name_zh": "招商中证白酒指数(LOF)A",
  "name_en": "China Merchants CSI Baijiu Index Fund",
  "note": "中国开放式基金，使用单位净值走势计算趋势、回撤和 RSI。",
  "enabled": true
}
```

开放式基金通常没有成交量，程序会把成交量按 0 处理，成交量相关风险项不会被放大。

## 添加中国 ETF

复制 ETF 示例，替换 ETF 代码和名称：

```json
{
  "symbol": "510300",
  "type": "CN_ETF",
  "name_zh": "华泰柏瑞沪深300ETF",
  "name_en": "CSI 300 ETF",
  "note": "中国 ETF，使用 akshare 日行情计算价格趋势。",
  "enabled": true
}
```

ETF 有开盘、最高、最低、收盘和成交量字段，程序会按现有价格评分框架计算，并额外生成风险预警。

## 中国 ETF 风险预警

中国 ETF/基金会额外计算以下指标：

- 5日、20日、60日收益。
- 当前60日回撤和近60日最大回撤。
- 20日、60日年化波动率和波动放大倍数。
- 20日下行波动率。
- 95% VaR 和 95% CVaR，包含单日和5日窗口。
- 连续下跌天数和成交量相对20日均量。

如果历史样本少于 120 个交易日，程序会显示“样本不足”，不会强行给出 CVaR。风险预警只用于解释市场状态，不改变三分类评分权重。

## 新闻、热度和关键链接

美股和美股 ETF 的新闻情绪来自 Alpha Vantage。中国 ETF/基金会尝试通过 akshare 抓取东方财富新闻和可用的热度接口；如果第三方接口临时不可用，程序仍会继续运行，并在报告中保留关键链接供人工判断：

- 天天基金页面。
- 东方财富新闻搜索。
- 东方财富股吧或基金吧。
- 雪球搜索。

新闻和社交热度只作为解释层。它们会影响报告文字和人工判断线索，暂时不会改变三分类概率。

## 常见失败原因

- Telegram 测试通过但主监控失败：通常是 Alpha Vantage key、资产代码或行情源问题。
- `Missing ... secret`：Secret 名称拼错，或填到了 Variables 而不是 Secrets。
- `Telegram HTTP 401`：bot token 无效或已撤销。
- `Telegram error: chat not found`：chat_id 错误，或机器人还没有收到 `/start`。
- `akshare is required`：没有安装依赖，运行 `pip install -r requirements.txt`。
- `Insufficient price history`：行情源返回的数据少于 65 条，暂时不能计算 60 日指标。

## 关于评分

新增基金和 ETF 后，三分类定义、权重和阈值没有改变。基金/ETF 暂时沿用同一套价格趋势评分；新闻情绪仅对支持的美股和美股 ETF 启用。中国 ETF/基金的 CVaR、波动、新闻和热度目前只作为预警解释层。

后续若要调整评分逻辑，建议先做三件事：

1. 保存足够长的 `history/` 快照。
2. 定义未来 20 到 60 个交易日的验证标签。
3. 用历史回测和概率校准检查当前规则是否可靠。
