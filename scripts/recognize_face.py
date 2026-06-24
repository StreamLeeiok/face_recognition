import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.face_embeddings import DEFAULT_THRESHOLD, recognize_face


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recognize one face with InsightFace embeddings.")
    parser.add_argument("--image", type=Path, required=True, help="New single-face photo path.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum cosine similarity to accept a match.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = recognize_face(args.image, threshold=args.threshold)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as error:
        print(
            json.dumps(
                {
                    "recognized": False,
                    "error": str(error),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
