"""Predictable parent, unique run names, and no filesystem writes on validation."""
from datetime import datetime, timezone
from pathlib import Path
import re
from uuid import uuid4


def new_run(tool, manifest):
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(manifest).stem).strip("-")[:48] or "input"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"{tool}-{stem}-{stamp}-{uuid4().hex[:8]}"
