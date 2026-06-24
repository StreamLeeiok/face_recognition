from datetime import datetime
import json
from pathlib import Path
import sys

import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.face_embeddings import DEFAULT_THRESHOLD, recognize_face  # noqa: E402
from scripts.firestore_attendance import mark_present  # noqa: E402


SAVE_FOLDER = ROOT / "images"
RECEIVED_FOLDER = ROOT / "received"
LATEST_RESULT_PATH = RECEIVED_FOLDER / "latest.json"

app = FastAPI(title="Raspberry Pi Photo Receiver")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
SAVE_FOLDER.mkdir(parents=True, exist_ok=True)
RECEIVED_FOLDER.mkdir(parents=True, exist_ok=True)


def image_url_for(path: Path) -> str:
    return f"/images/{path.name}"


@app.post("/receive_photo")
async def receive_photo(photo: UploadFile = File(...)):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"attendance_{timestamp}.jpg"
    filepath = SAVE_FOLDER / filename

    contents = await photo.read()
    filepath.write_bytes(contents)

    try:
        # Face recognition is intentionally triggered only when a new Raspberry Pi photo arrives.
        recognition = recognize_face(filepath, threshold=DEFAULT_THRESHOLD)
        database_result = {
            "saved": False,
            "reason": "face_not_recognized",
        }
        if recognition["recognized"]:
            best_match = recognition["best_match"]
            try:
                database_result = mark_present(
                    student_id=best_match["student_id"],
                )
            except Exception as database_error:
                database_result = {
                    "saved": False,
                    "error": str(database_error),
                }

        result = {
            "ok": True,
            "status": "recognized" if recognition["recognized"] else "not_recognized",
            "filename": filename,
            "image": str(filepath),
            "image_url": image_url_for(filepath),
            "received_at": datetime.now().isoformat(timespec="seconds"),
            "recognition": recognition,
            "database": database_result,
        }
    except Exception as error:
        result = {
            "ok": False,
            "status": "error",
            "filename": filename,
            "image": str(filepath),
            "image_url": image_url_for(filepath),
            "received_at": datetime.now().isoformat(timespec="seconds"),
            "error": str(error),
        }

    LATEST_RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {filepath}")
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
