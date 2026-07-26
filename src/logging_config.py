"""Central logging setup for the assistant.

One call to `configure_logging()` wires up the whole system: a console handler and a
rotating file handler (`logs/app.log`), both driven by env vars so behaviour can change
without touching code:

- LOG_LEVEL   : root level (DEBUG / INFO / WARNING / ...). Default INFO.
- LOG_FILE    : path to the log file, or "" / "none" to disable file logging. Default logs/app.log.
- LOG_TO_STDERR: "0" to silence the console handler (e.g. when only a file is wanted). Default on.

Every module logs through `logging.getLogger(__name__)`, so the emitted name shows exactly
which part of the pipeline a line came from (e.g. `src.agents.sql_analyst`). Setup is
idempotent - calling it more than once (Streamlit reruns, tests) does not duplicate handlers.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_CONFIGURED = False

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def _clean(value: str) -> str:
    return (value or "").strip().strip("'\"")


def configure_logging() -> None:
    """Set up console + rotating-file logging once, from environment variables."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = _clean(os.getenv("LOG_LEVEL")).upper() or "INFO"
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    if _clean(os.getenv("LOG_TO_STDERR")) != "0":
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

    log_file = _clean(os.getenv("LOG_FILE"))
    if log_file.lower() != "none":
        path = Path(log_file) if log_file else ROOT / "logs" / "app.log"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3,
                                               encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError:
            # A read-only filesystem (e.g. some hosted runtimes) should not crash the app;
            # console logging still works.
            root.warning("File logging disabled - could not open %s", path)

    # Third-party libraries are noisy at INFO; keep them at WARNING unless we're debugging.
    for noisy in ("httpx", "httpcore", "openai", "anthropic", "urllib3", "langsmith"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def snippet(text: str, limit: int = 200) -> str:
    """Collapse a value to a single short line for a log message."""
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit] + "…"
