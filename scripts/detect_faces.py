import argparse
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "yolov11s-face.pt"
INPUT_PATH = ROOT / "images" / "test.jpg"
OUTPUT_PATH = ROOT / "outputs" / "result.jpg"
YOLO_CONFIG_DIR = ROOT / ".ultralytics"
MPL_CONFIG_DIR = ROOT / ".matplotlib"

YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
    from ultralytics import YOLO  # noqa: E402

_MODEL_CACHE = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect faces in one image with YOLO.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="YOLO face model path.")
    parser.add_argument("--image", type=Path, default=INPUT_PATH, help="Input image path.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Annotated output image path.")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum confidence threshold.")
    return parser.parse_args()


def load_model(model_path: Path = MODEL_PATH) -> YOLO:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Face model not found: {model_path}\n"
            "Put a YOLO face model file named yolov11s-face.pt into the models folder."
        )

    cache_key = str(model_path.resolve())
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = YOLO(cache_key)
    return _MODEL_CACHE[cache_key]


def detect_faces(
    image_path: str | Path,
    model_path: str | Path = MODEL_PATH,
    output_path: str | Path | None = None,
    conf: float = 0.25,
) -> dict[str, Any]:
    input_path = Path(image_path)
    model_path = Path(model_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Test image not found: {input_path}\n"
            "Put a test image named test.jpg into the images folder."
        )

    model = load_model(model_path)
    image = cv2.imread(str(input_path))

    if image is None:
        raise ValueError(f"OpenCV could not read image: {input_path}")

    results = model(str(input_path), conf=conf, verbose=False)
    faces = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            faces.append(
                {
                    "box": [x1, y1, x2, y2],
                    "confidence": round(confidence, 4),
                }
            )

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"face {confidence:.2f}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    face_count = len(faces)
    result = {
        "image": str(input_path),
        "has_face": face_count > 0,
        "face_count": face_count,
        "status": "no_face" if face_count == 0 else "single_face" if face_count == 1 else "multiple_faces",
        "valid_for_assessment": face_count == 1,
        "faces": faces,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        result["output"] = str(output_path)

    return result


def main() -> None:
    args = parse_args()
    result = detect_faces(
        image_path=args.image,
        model_path=args.model,
        output_path=args.output,
        conf=args.conf,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
