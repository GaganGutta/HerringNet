"""SQLite schema for the HerringNet image and labeling database.

This is the single source of truth for camera-trap imagery and its
labels. It replaces the per-week Excel sheets with one relational
store that:

  - keys every image to an actual file (via relative path + content hash),
  - enforces foreign keys so categorical fields cannot drift,
  - separates human observations, expert reviews, and model predictions
    so the same image can carry all three without overwriting each other.

FiftyOne reads from and writes back to this database; it is a working
surface, not a second copy of the data.
"""

from __future__ import annotations

# Schema version is bumped when the DDL changes in a breaking way.
SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS sites (
    site_id              TEXT PRIMARY KEY,
    site_name            TEXT,
    location_description TEXT,
    camera_type          TEXT,
    installation_date    TEXT,
    notes                TEXT
);

CREATE TABLE IF NOT EXISTS observers (
    observer_id   TEXT PRIMARY KEY,
    observer_name TEXT,
    role          TEXT,
    email         TEXT,
    training_date TEXT,
    active        INTEGER,
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS count_bins (
    bin_code  INTEGER PRIMARY KEY,
    label     TEXT,
    min_count INTEGER,
    max_count INTEGER          -- NULL means open-ended (e.g. 500+)
);

CREATE TABLE IF NOT EXISTS uncountable_categories (
    category_id   INTEGER PRIMARY KEY,
    category      TEXT,
    description   TEXT,
    common_causes TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    site_id                 TEXT REFERENCES sites(site_id),
    label                   TEXT UNIQUE,   -- original sheet / folder name
    start_date              TEXT,
    end_date                TEXT,
    sd_card_collection_date TEXT,
    upload_date             TEXT,
    notes                   TEXT
);

CREATE TABLE IF NOT EXISTS images (
    image_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    original_name TEXT,                       -- GoPro id from sheet / filename stem
    site_id       TEXT REFERENCES sites(site_id),
    session_id    INTEGER REFERENCES sessions(session_id),
    camera_id     TEXT,
    date_captured TEXT,
    time_captured TEXT,
    file_path     TEXT,                       -- relative to the image root
    content_hash  TEXT,                       -- sha256 of file bytes when available
    file_exists   INTEGER DEFAULT 0,
    width         INTEGER,
    height        INTEGER,
    file_size     INTEGER,
    added_at      TEXT,
    UNIQUE(session_id, original_name)
);

CREATE TABLE IF NOT EXISTS observations (
    obs_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id               INTEGER REFERENCES images(image_id),
    observer_id            TEXT REFERENCES observers(observer_id),
    countable              INTEGER,   -- 1 / 0 / NULL
    uncountable_reason     INTEGER REFERENCES uncountable_categories(category_id),
    rh_present             INTEGER,   -- 1 / 0 / NULL
    rh_count               INTEGER,
    count_bin              INTEGER REFERENCES count_bins(bin_code),
    general_count_estimate INTEGER,
    bycatch_present        INTEGER,
    bycatch_id             TEXT,
    bycatch_count          INTEGER,
    count_method           TEXT,
    confidence_level       TEXT,
    observation_date       TEXT,
    source                 TEXT,      -- 'excel_migration' | 'fiftyone' | 'manual'
    notes                  TEXT
);

CREATE TABLE IF NOT EXISTS expert_reviews (
    review_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id           INTEGER REFERENCES images(image_id),
    expert_id          TEXT REFERENCES observers(observer_id),
    expert_reviewed    INTEGER,
    expert_rh_count    INTEGER,
    expert_review_date TEXT,
    notes              TEXT
);

CREATE TABLE IF NOT EXISTS model_predictions (
    pred_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id        INTEGER REFERENCES images(image_id),
    model_name      TEXT,
    fish_count      INTEGER,
    max_confidence  REAL,
    detections_json TEXT,            -- [{bbox:[x1,y1,x2,y2], confidence, class_name}]
    run_at          TEXT
);

CREATE INDEX IF NOT EXISTS idx_images_session ON images(session_id);
CREATE INDEX IF NOT EXISTS idx_images_hash    ON images(content_hash);
CREATE INDEX IF NOT EXISTS idx_obs_image      ON observations(image_id);
CREATE INDEX IF NOT EXISTS idx_review_image   ON expert_reviews(image_id);
CREATE INDEX IF NOT EXISTS idx_pred_image     ON model_predictions(image_id);
"""
