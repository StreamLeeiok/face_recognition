import json
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from scripts.face_embeddings import (
    EMBEDDINGS_DIR,
    KNOWN_FACES_DIR,
    ROOT,
    delete_registered_face,
    list_registered_faces,
    save_registered_face,
)

UPLOADS_DIR = ROOT / "uploads"
WEB_DIR = ROOT / "web"
IMAGES_DIR = ROOT / "images"
RECEIVED_DIR = ROOT / "received"
LATEST_RESULT_PATH = RECEIVED_DIR / "latest.json"

for runtime_dir in (
    KNOWN_FACES_DIR,
    EMBEDDINGS_DIR,
    IMAGES_DIR,
    UPLOADS_DIR,
    RECEIVED_DIR,
    ROOT / "outputs",
):
    runtime_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="YOLO + InsightFace Attendance API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/known_faces", StaticFiles(directory=str(KNOWN_FACES_DIR)), name="known_faces")
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


def save_upload(file: UploadFile) -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "").suffix or ".jpg"
    path = UPLOADS_DIR / f"{uuid.uuid4().hex}{suffix}"
    with path.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    return path


def json_error(error: Exception, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "ok": False,
            "error": str(error),
        },
    )


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/manage", response_class=HTMLResponse)
def manage() -> FileResponse:
    return FileResponse(WEB_DIR / "manage.html", headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "ok": True,
        "registered_count": len(list_registered_faces()),
    }


@app.get("/api/latest")
def latest() -> dict[str, object]:
    if not LATEST_RESULT_PATH.exists():
        return {
            "ok": True,
            "has_latest": False,
            "message": "No photo has been received yet.",
        }

    data = json.loads(LATEST_RESULT_PATH.read_text(encoding="utf-8"))
    return {
        "ok": True,
        "has_latest": True,
        "latest": data,
    }


@app.get("/api/faces")
def faces() -> dict[str, object]:
    return {
        "ok": True,
        "faces": list_registered_faces(),
    }


@app.post("/api/register")
def register(name: str = Form(...), file: UploadFile = File(...)) -> JSONResponse:
    try:
        original_filename = file.filename or ""
        image_path = save_upload(file)
        result = save_registered_face(
            name,
            image_path,
            registration_id=Path(original_filename).stem,
        )
        result["ok"] = True
        result["registered"] = True
        result["image_url"] = f"/known_faces/{Path(result['registered_image']).name}"
        return JSONResponse(content=result)
    except Exception as error:
        return json_error(error)


@app.delete("/api/faces/{person_id}")
def delete_face(person_id: str) -> JSONResponse:
    try:
        result = delete_registered_face(person_id)
        result["ok"] = True
        return JSONResponse(content=result)
    except Exception as error:
        return json_error(error, status_code=404)
