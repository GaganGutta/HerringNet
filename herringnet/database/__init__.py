"""Relational image + labeling database for HerringNet.

SQLite is the single source of truth for camera-trap imagery and labels.
FiftyOne (see ``fiftyone_io``) is a working surface layered on top for
visual review; it reads from and writes back to this database.
"""

from herringnet.database.db import DEFAULT_DB_PATH, Database

__all__ = ["Database", "DEFAULT_DB_PATH"]
