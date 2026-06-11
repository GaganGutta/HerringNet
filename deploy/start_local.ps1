# Start the HerringNet local hosting stack (Windows, no Docker needed).
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File deploy\start_local.ps1
#
#   http://localhost:8002  upload page (drop a folder of images)
#   http://localhost:8010  database viewer (Datasette)
#
# The watcher ingests anything dropped into data\incoming (via the upload
# page or by copying a folder there directly) into data\archive + the DB.

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

New-Item -ItemType Directory -Force data\incoming, data\archive | Out-Null

$env:HN_INCOMING_DIR = "data/incoming"
Start-Process python -ArgumentList "-m", "uvicorn",
    "herringnet.database.upload_server:app", "--host", "0.0.0.0", "--port", "8002"
Start-Process python -ArgumentList "-m", "cli.main", "db",
    "--db", "data/herringnet.db", "watch", "data/incoming",
    "--archive", "data/archive", "--interval", "10", "--no-label-studio"
Start-Process python -ArgumentList "-m", "datasette", "serve",
    "data/herringnet.db", "--host", "0.0.0.0", "--port", "8010"

Write-Host "HerringNet local stack starting:"
Write-Host "  Upload:   http://localhost:8002"
Write-Host "  Database: http://localhost:8010"
Write-Host "Drop folders into data\incoming or use the upload page."
