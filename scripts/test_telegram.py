from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitor import env_required, send_telegram


def main() -> int:
    token = env_required("TELEGRAM_BOT_TOKEN")
    chat_id = env_required("TELEGRAM_CHAT_ID")
    sent_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    send_telegram(
        token,
        chat_id,
        f"<b>Invest Risk Monitor Telegram test</b>\nSent at: {sent_at}",
    )
    print("Telegram test message sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
