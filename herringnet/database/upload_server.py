"""Minimal web upload page: drop a folder of images, it lands in the drop zone.

A tiny FastAPI app with one page. The user picks a folder of images in the
browser; the files are saved into ``incoming/<session>/`` on the server, where
the watcher (``herringnet.database.watch``) picks them up and ingests them.

Run via uvicorn (see deploy/docker-compose.yml):
    uvicorn herringnet.database.upload_server:app --host 0.0.0.0 --port 8002

The incoming directory is taken from the HN_INCOMING_DIR environment variable
(default: data/incoming).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

INCOMING_DIR = Path(os.environ.get("HN_INCOMING_DIR", "data/incoming"))
DB_PATH = os.environ.get("HN_DB", "data/herringnet.db")
ARCHIVE_DIR = os.environ.get("HN_ARCHIVE", "data/archive")

app = FastAPI(title="HerringNet Upload")

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>HerringNet Upload</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:640px;margin:3rem auto;padding:0 1rem}
 h1{font-size:1.4rem} label{display:block;margin:1rem 0 .3rem}
 input[type=text]{width:100%;padding:.5rem;font-size:1rem}
 button{margin-top:1.2rem;padding:.6rem 1.2rem;font-size:1rem;cursor:pointer}
 #status{margin-top:1rem;white-space:pre-wrap}
</style></head><body>
<h1>Upload camera-trap images</h1>
<p>Pick a folder of images (named for the site and week). It will be added to
the database automatically within a minute.</p>
<label>Session name (defaults to the folder name)</label>
<input id="session" type="text" placeholder="e.g. BRIDE_week5">
<label>Folder of images</label>
<input id="folder" type="file" webkitdirectory directory multiple>
<button id="go">Upload</button>
<div id="status"></div>

<h1 style="margin-top:2.5rem">Manage sessions</h1>
<p>Each row is one uploaded folder (session). Deleting removes its images
and database records permanently.</p>
<table id="sessions" style="width:100%;border-collapse:collapse"></table>

<script>
const $=id=>document.getElementById(id);
const folder=$('folder'), session=$('session'), status=$('status'), go=$('go');
folder.addEventListener('change',()=>{
  if(!session.value && folder.files.length){
    const rel=folder.files[0].webkitRelativePath||'';
    if(rel.includes('/')) session.value=rel.split('/')[0];
  }
});
go.addEventListener('click',async()=>{
  if(!folder.files.length){status.textContent='Pick a folder first.';return;}
  const files=[...folder.files], total=files.length, sess=session.value||'';
  let done=0, failed=0;
  go.disabled=true;
  // Parallel workers: several concurrent streams use a slow uplink far
  // better than one big sequential request, and one failed file no
  // longer kills the whole batch.
  const WORKERS=4;
  const queue=files.slice();
  async function worker(){
    while(queue.length){
      const f=queue.shift();
      const fd=new FormData();
      fd.append('session', sess);
      fd.append('files', f, f.name);
      try{
        const r=await fetch('upload',{method:'POST',body:fd});
        if(r.ok) done++; else failed++;
      }catch(e){failed++;}
      status.textContent='Uploaded '+done+'/'+total+(failed?' ('+failed+' failed)':'');
    }
  }
  status.textContent='Uploading '+total+' files...';
  await Promise.all(Array.from({length:WORKERS},worker));
  status.textContent='Done: '+done+' / '+total+' uploaded'+
    (failed?', '+failed+' FAILED (re-run to retry)':'')+'. Ingest runs shortly.';
  go.disabled=false;
  loadSessions();
});

async function loadSessions(){
  const tbl=$('sessions');
  try{
    const rows=await (await fetch('sessions')).json();
    tbl.innerHTML='<tr><th align="left">Session</th><th>Images</th>'+
      '<th>With files</th><th></th></tr>'+
      rows.map(r=>'<tr><td>'+r.label+'</td><td align="center">'+r.images+
        '</td><td align="center">'+(r.with_files||0)+
        '</td><td><button onclick="delSession(\\''+r.label.replace(/'/g,"\\\\'")+
        '\\')">Delete</button></td></tr>').join('');
  }catch(e){tbl.innerHTML='<tr><td>Could not load sessions: '+e+'</td></tr>';}
}

async function delSession(label){
  const typed=prompt('This permanently deletes ALL images and records of "'+
    label+'".\\n\\nType the session name to confirm:');
  if(typed!==label){
    if(typed!==null)alert('Name did not match; nothing deleted.');
    return;
  }
  const fd=new FormData(); fd.append('label', label);
  const r=await fetch('delete-session',{method:'POST',body:fd});
  const j=await r.json();
  alert(r.ok?('Deleted: '+j.images+' images, '+j.files_removed+' files.')
            :('Error: '+(j.detail||r.status)));
  loadSessions();
}
loadSessions();
</script></body></html>"""


def _slug(name: str, fallback: str = "upload") -> str:
    """Make a filesystem-safe folder name."""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", (name or "").strip()).strip("_.")
    return name or fallback


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE


@app.get("/sessions")
def sessions() -> JSONResponse:
    from herringnet.database.admin import list_sessions
    from herringnet.database.db import Database

    db = Database(DB_PATH)
    rows = list_sessions(db)
    db.close()
    return JSONResponse(rows)


@app.post("/delete-session")
async def delete_session_endpoint(label: str = Form(...)) -> JSONResponse:
    from herringnet.database.admin import delete_session
    from herringnet.database.db import Database

    db = Database(DB_PATH)
    summary = delete_session(db, label.strip(), archive_dir=ARCHIVE_DIR)
    db.close()
    if summary["sessions"] == 0:
        return JSONResponse({"detail": f"No session named {label!r}"}, status_code=404)
    return JSONResponse(summary)


@app.post("/upload")
async def upload(files: list[UploadFile], session: str = Form("")) -> JSONResponse:
    session_dir = INCOMING_DIR / _slug(session)
    session_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for f in files:
        safe_name = Path(f.filename or "").name  # strip any path components
        if not safe_name:
            continue
        dest = session_dir / safe_name
        with open(dest, "wb") as out:
            while chunk := await f.read(1 << 20):
                out.write(chunk)
        saved += 1

    return JSONResponse({"saved": saved, "session": session_dir.name})
