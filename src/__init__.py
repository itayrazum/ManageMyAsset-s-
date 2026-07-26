"""ManageMyAsset(s) - a multi-agent assistant over a property financial ledger.

Importing the package configures logging once, so every module can log through
`logging.getLogger(__name__)` without any per-entrypoint setup.
"""

from .logging_config import configure_logging

configure_logging()
