#!/usr/bin/env bash
# One-shot server setup for the HerringNet hosted stack (bare-metal, no Docker).
# Installs systemd services for the upload page, the folder watcher, and the
# Datasette viewer, plus a Caddy reverse proxy with a shared-password gate.
# Idempotent: safe to re-run after code updates.
set -euo pipefail

APP=/opt/herringnet
DATA=/srv/herring-data
PY=$APP/.venv/bin

# --- systemd: upload page -------------------------------------------------
cat > /etc/systemd/system/herringnet-upload.service <<EOF
[Unit]
Description=HerringNet upload page
After=network.target

[Service]
WorkingDirectory=$APP
Environment=HN_INCOMING_DIR=$DATA/incoming
ExecStart=$PY/uvicorn herringnet.database.upload_server:app --host 127.0.0.1 --port 8002
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# --- systemd: folder watcher ------------------------------------------------
cat > /etc/systemd/system/herringnet-watch.service <<EOF
[Unit]
Description=HerringNet incoming-folder watcher
After=network.target

[Service]
WorkingDirectory=$APP
ExecStart=$PY/python -m cli.main db --db $DATA/herringnet.db watch $DATA/incoming --archive $DATA/archive --interval 10 --no-label-studio
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# --- systemd: Datasette viewer ----------------------------------------------
cat > /etc/systemd/system/herringnet-datasette.service <<EOF
[Unit]
Description=HerringNet Datasette database viewer
After=network.target

[Service]
WorkingDirectory=$DATA
ExecStart=$PY/datasette serve $DATA/herringnet.db --host 127.0.0.1 --port 8010 --setting base_url /db/ --setting sql_time_limit_ms 5000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# --- systemd: FiftyOne app (requires `pip install fiftyone` in the venv) ----
cat > /etc/systemd/system/herringnet-fiftyone.service <<EOF
[Unit]
Description=HerringNet FiftyOne app (synced with the database)
After=network.target

[Service]
WorkingDirectory=$APP
Environment=HN_DB=$DATA/herringnet.db
Environment=HN_ARCHIVE=$DATA/archive
Environment=HN_FO_PORT=5152
Environment=FIFTYONE_DATABASE_DIR=$DATA/fiftyone
ExecStart=$PY/python -m herringnet.database.fiftyone_server
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# --- Caddy: password-gated HTTP (viewer/upload on :80, FiftyOne on :5151) ---
# TEAM_PASSWORD must be exported by the caller.
HASH=$(caddy hash-password --plaintext "$TEAM_PASSWORD")
cat > /etc/caddy/Caddyfile <<EOF
:80 {
	encode gzip
	basic_auth {
		team $HASH
	}
	handle_path /upload/* {
		reverse_proxy 127.0.0.1:8002
	}
	handle /db/* {
		reverse_proxy 127.0.0.1:8010
	}
	redir / /db/ 302
}

:5151 {
	basic_auth {
		team $HASH
	}
	reverse_proxy 127.0.0.1:5152
}
EOF

systemctl daemon-reload
systemctl enable --now herringnet-upload herringnet-watch herringnet-datasette
systemctl enable --now herringnet-fiftyone || true  # needs fiftyone installed
systemctl restart caddy

sleep 2
systemctl is-active herringnet-upload herringnet-watch herringnet-datasette caddy
echo "SETUP COMPLETE"
