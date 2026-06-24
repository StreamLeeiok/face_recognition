import json
import shutil
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from scripts.detect_faces import detect_faces


ROOT = Path(__file__).resolve().parents[1]
KNOWN_FACES_DIR = ROOT / "known_faces"
EMBEDDINGS_DIR = ROOT / "face_embeddings"
INSIGHTFACE_ROOT = ROOT / "models" / "insightface"
DEFAULT_THRESHOLD = 0.5

_FACE_APP: FaceAnalysis | None = None

warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\..*")


def safe_person_id(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    return cleaned.strip("_") or "person"


def parse_registration_identity(
    filename_or_id: str,
    fallback_name: str,
) -> tuple[str, str, str]:
    registration_id = safe_person_id(Path(filename_or_id).stem)
    if "_" in registration_id:
        student_id, parsed_name = registration_id.split("_", 1)
        if student_id and parsed_name:
            return registration_id, student_id, parsed_name

    person_id = safe_person_id(fallback_name)
    return person_id, person_id, fallback_name.strip()


def get_face_app() -> FaceAnalysis:
    global _FACE_APP
    if _FACE_APP is None:
        INSIGHTFACE_ROOT.mkdir(parents=True, exist_ok=True)
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            _FACE_APP = FaceAnalysis(
                name="buffalo_l",
                root=str(INSIGHTFACE_ROOT),
                providers=["CPUExecutionProvider"],
            )
            _FACE_APP.prepare(ctx_id=-1, det_size=(640, 640))
    return _FACE_APP


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    embedding = embedding.astype(np.float32)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        raise ValueError("InsightFace returned an empty embedding.")
    return embedding / norm


def extract_embedding(image_path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    image_path = Path(image_path)
    yolo_result = detect_faces(image_path)
    if not yolo_result["valid_for_assessment"]:
        raise ValueError(
            f"Image is not valid for assessment: {yolo_result['status']} "
            f"({yolo_result['face_count']} faces)"
        )

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    with redirect_stderr(StringIO()):
        faces = get_face_app().get(image)
    if len(faces) != 1:
        raise ValueError(f"InsightFace expected 1 face, found {len(faces)} faces.")

    face = faces[0]
    embedding = getattr(face, "normed_embedding", None)
    if embedding is None:
        embedding = normalize_embedding(face.embedding)

    return normalize_embedding(np.asarray(embedding)), yolo_result


def save_registered_face(
    name: str,
    image_path: str | Path,
    registration_id: str | None = None,
) -> dict[str, Any]:
    image_path = Path(image_path)
    person_id, student_id, registered_name = parse_registration_identity(
        registration_id or image_path.name,
        name,
    )
    embedding, yolo_result = extract_embedding(image_path)

    KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    image_suffix = image_path.suffix or ".jpg"
    registered_image_path = KNOWN_FACES_DIR / f"{person_id}{image_suffix}"
    embedding_path = EMBEDDINGS_DIR / f"{person_id}.npy"
    metadata_path = EMBEDDINGS_DIR / f"{person_id}.json"

    shutil.copy2(image_path, registered_image_path)
    np.save(embedding_path, embedding)

    metadata = {
        "person_id": person_id,
        "student_id": student_id,
        "name": registered_name,
        "registered_image": str(registered_image_path),
        "embedding": str(embedding_path),
        "yolo": yolo_result,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_known_embeddings() -> list[dict[str, Any]]:
    records = []
    if not EMBEDDINGS_DIR.exists():
        return records

    for metadata_path in sorted(EMBEDDINGS_DIR.glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        embedding_path = Path(metadata["embedding"])
        if not embedding_path.exists():
            continue
        records.append(
            {
                "person_id": metadata["person_id"],
                "student_id": metadata.get("student_id", metadata["person_id"]),
                "name": metadata["name"],
                "embedding": normalize_embedding(np.load(embedding_path)),
                "metadata": metadata,
            }
        )
    return records


def list_registered_faces() -> list[dict[str, Any]]:
    faces = []
    if not EMBEDDINGS_DIR.exists():
        return faces

    for metadata_path in sorted(EMBEDDINGS_DIR.glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        registered_image = Path(metadata.get("registered_image", ""))
        image_name = registered_image.name if registered_image.exists() else None
        faces.append(
            {
                "person_id": metadata["person_id"],
                "student_id": metadata.get("student_id", metadata["person_id"]),
                "name": metadata["name"],
                "image_url": f"/known_faces/{image_name}" if image_name else None,
                "embedding": metadata.get("embedding"),
                "registered_image": metadata.get("registered_image"),
            }
        )
    return faces


def delete_registered_face(person_id: str) -> dict[str, Any]:
    person_id = safe_person_id(person_id)
    deleted = []

    for path in EMBEDDINGS_DIR.glob(f"{person_id}.*"):
        if path.is_file():
            path.unlink()
            deleted.append(str(path))

    for path in KNOWN_FACES_DIR.glob(f"{person_id}.*"):
        if path.is_file():
            path.unlink()
            deleted.append(str(path))

    if not deleted:
        raise FileNotFoundError(f"No registered face found for: {person_id}")

    return {
        "deleted": True,
        "person_id": person_id,
        "files": deleted,
    }


def recognize_face(image_path: str | Path, threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    embedding, yolo_result = extract_embedding(image_path)
    known_faces = load_known_embeddings()
    if not known_faces:
        raise FileNotFoundError("No registered face embeddings found. Run register_face.py first.")

    matches = []
    for known in known_faces:
        similarity = float(np.dot(embedding, known["embedding"]))
        matches.append(
            {
                "person_id": known["person_id"],
                "student_id": known["student_id"],
                "name": known["name"],
                "similarity": round(similarity, 4),
            }
        )

    matches.sort(key=lambda item: item["similarity"], reverse=True)
    best_match = matches[0]
    recognized = best_match["similarity"] >= threshold

    return {
        "image": str(image_path),
        "recognized": recognized,
        "threshold": threshold,
        "best_match": best_match if recognized else None,
        "best_candidate": best_match,
        "matches": matches,
        "yolo": yolo_result,
    }
