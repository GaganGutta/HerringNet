# Hosting HerringNet on a cloud VM

This runs the whole system on one small always-on VM, reachable by URL:

| URL | What it is |
|-----|------------|
| `https://DOMAIN/`        | Database viewer (Datasette): browse, filter, run SQL. Read-only. |
| `https://DOMAIN/upload/` | Upload page: drop a folder of images, it gets ingested automatically. |
| `https://DOMAIN/label/`  | Label Studio: the team reviews and labels images. |

A single shared password protects the viewer and uploader. Label Studio uses its own (shared) account. This is intentionally not a full per-user login system.

## What you need

- A small VM (2 vCPU / 4 GB RAM is enough), e.g. Hetzner CX22 or DigitalOcean basic. About $5-6/month.
- A storage volume for the images and database (size for ~4 MB per image; 100-200 GB covers 10-50k images). A few dollars/month.
- A domain name (or subdomain) with an `A` record pointing at the VM's IP.
- Docker with the Compose plugin installed on the VM.

Rough total: $9-12/month.

## One-time setup

1. **Provision the VM and mount the volume**, for example at `/mnt/herring-data`. Create the working folders:
   ```bash
   sudo mkdir -p /mnt/herring-data/{archive,incoming}
   ```

2. **Point your domain** (`A` record) at the VM's public IP and wait for it to resolve.

3. **Install Docker** (Ubuntu): follow the official `get.docker.com` script, then confirm `docker compose version` works.

4. **Get the code and configure:**
   ```bash
   git clone https://github.com/GaganGutta/HerringNet.git
   cd HerringNet/deploy
   cp .env.example .env
   ```
   Edit `.env`:
   - `DOMAIN` = your domain.
   - `DATA_DIR` = `/mnt/herring-data`.
   - `BASIC_AUTH_HASH` = generate with
     `docker run --rm caddy caddy hash-password --plaintext 'YOUR-PASSWORD'`.
   - `LS_USERNAME`, `LS_PASSWORD`, `LS_TOKEN`, `LS_DB_PASSWORD` = pick values
     (`LS_TOKEN` should be a long random string).

5. **Load your existing data** onto the volume:
   - Copy your database file to `/mnt/herring-data/herringnet.db` (or build it
     fresh, see below).
   - Put your image folders under `/mnt/herring-data/archive/` (one folder per
     site-week), or upload them later through the upload page.

6. **Start everything:**
   ```bash
   docker compose up -d --build
   ```

7. **Set up Label Studio (first time only):**
   - Open `https://DOMAIN/label/`, log in with `LS_USERNAME` / `LS_PASSWORD`.
   - Create a project. In **Settings -> Labeling Interface**, paste
     `deploy/labeling_config.xml`.
   - In **Settings -> Cloud Storage**, add a **Local files** source with the
     path `/label-studio/files` (the image archive is mounted there).
   - Note the project's numeric id (in its URL). Put it in `.env` as
     `LS_PROJECT_ID`, then `docker compose up -d` again so the watcher will
     auto-create tasks for new uploads.

## Building the database from your Excel + images (if starting fresh)

From `deploy/`:
```bash
# put the workbook somewhere on the VM, then:
docker compose run --rm watcher \
  python -m cli.main db --db /data/herringnet.db import-excel /data/Workbook.xlsx
docker compose run --rm watcher \
  python -m cli.main db --db /data/herringnet.db scan /data/archive
```

## Day-to-day use

- **Add images:** go to `https://DOMAIN/upload/`, pick a folder, upload. Within a minute the watcher ingests it and (if Label Studio is configured) creates tasks. Or `rsync` folders into `/mnt/herring-data/incoming/`.
- **View / query data:** `https://DOMAIN/`.
- **Label:** `https://DOMAIN/label/`.
- **Pull finished labels into the database:**
  ```bash
  docker compose run --rm watcher \
    python -m cli.main db --db /data/herringnet.db ls-pull
  ```

## Backups

The only state that matters is on the volume: `herringnet.db`, the `archive/`
images, and the Label Studio Docker volumes. Snapshot the volume on a schedule,
and copy `herringnet.db` somewhere off-box regularly (it is small).

## Honest caveats

- The Label Studio push/pull (`ls-push` / `ls-pull`) talks to the live server's
  API and should be verified on the first deploy; if task import or pull behaves
  unexpectedly, check `docker compose logs watcher`.
- "No login system" means a shared password and a shared Label Studio account.
  Anyone with the URL and password can view and label. For a small trusted lab
  this is fine; do not post the link publicly.
- This is a single-VM setup. It is simple and cheap, not highly available. Keep
  backups and you can rebuild it in minutes.
