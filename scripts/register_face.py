import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.face_embeddings import save_registered_face


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register one known face with InsightFace.")
    parser.add_argument("--name", required=True, help="Person name, for example Alice or ZhangSan.")
    parser.add_argument("--image", type=Path, required=True, help="Single-face photo path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = save_registered_face(
            args.name,
            args.image,
            registration_id=Path(args.image).stem,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as error:
        print(
            json.dumps(
                {
                    "registered": False,
                    "error": str(error),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
