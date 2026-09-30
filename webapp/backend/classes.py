"""The label set from an uploaded classes.txt, persisted across restarts.

One list per install (this is a single-user box). Line N of the file is class
id N, matching the YOLO convention, so ids stay stable for downstream tooling.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Optional

from . import config

log = logging.getLogger("sam3webapp.classes")

MAX_CLASSES = 1000
MAX_NAME_LEN = 80


def parse(text: str) -> list[str]:
    """classes.txt -> names, in file order. Blank lines and `#` comments drop out.

    A leading `0 person` / `0: person` index is tolerated and ignored — some
    exports carry one, and the line's position is what defines the id either way.
    """
    names: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        head, _, rest = line.partition(" ")
        if rest and head.rstrip(":").isdigit():
            line = rest.strip()
        names.append(line[:MAX_NAME_LEN])
        if len(names) >= MAX_CLASSES:
            log.warning("classes.txt truncated at %d entries", MAX_CLASSES)
            break
    return names


class ClassStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._names: list[str] = []
        self._source: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not config.CLASSES_PATH.is_file():
            return
        try:
            doc = json.loads(config.CLASSES_PATH.read_text())
            self._names = [str(n) for n in doc.get("names", [])]
            self._source = doc.get("source")
            log.info("loaded %d classes from %s", len(self._names), config.CLASSES_PATH)
        except Exception:  # noqa: BLE001
            log.warning("could not read %s — starting with no classes",
                        config.CLASSES_PATH, exc_info=True)

    def replace(self, names: list[str], source: Optional[str]) -> dict:
        with self._lock:
            self._names = names
            self._source = source
            config.CLASSES_PATH.write_text(json.dumps({"names": names, "source": source}, indent=1))
        return self.public()

    def clear(self) -> dict:
        with self._lock:
            self._names = []
            self._source = None
            config.CLASSES_PATH.unlink(missing_ok=True)
        return self.public()

    def name(self, cls_id: Optional[int]) -> Optional[str]:
        if cls_id is None:
            return None
        with self._lock:
            return self._names[cls_id] if 0 <= cls_id < len(self._names) else None

    def count(self) -> int:
        with self._lock:
            return len(self._names)

    def public(self) -> dict:
        with self._lock:
            return {
                "classes": [{"id": i, "name": n} for i, n in enumerate(self._names)],
                "count": len(self._names),
                "source": self._source,
            }
