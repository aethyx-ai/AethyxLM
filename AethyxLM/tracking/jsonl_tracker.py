"""Append-only JSONL experiment tracking with no hosted-service dependency."""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class JsonlExperimentTracker:
    """Write durable, machine-readable run events to a local JSONL file."""

    def __init__(
        self,
        path: str | Path,
        run_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex
        self._lock = threading.Lock()
        if metadata:
            self.log("run_started", **metadata)

    def log(self, event: str, step: Optional[int] = None, **values: Any):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": event,
            **({"step": int(step)} if step is not None else {}),
            **values,
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

