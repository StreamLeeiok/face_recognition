from pathlib import Path
from typing import Any

import firebase_admin
from firebase_admin import credentials, firestore


SERVICE_ACCOUNT_PATH = Path(__file__).resolve().parent / "serviceAccountKey.json"


def get_database():
    if not SERVICE_ACCOUNT_PATH.exists():
        raise FileNotFoundError(
            f"Firebase service account key was not found: {SERVICE_ACCOUNT_PATH}"
        )

    try:
        app = firebase_admin.get_app()
    except ValueError:
        credential = credentials.Certificate(str(SERVICE_ACCOUNT_PATH))
        app = firebase_admin.initialize_app(credential)

    return firestore.client(app=app)


def mark_present(student_id: str) -> dict[str, Any]:
    database = get_database()
    database.collection("members").document(student_id).update(
        {
            "present": True,
            "lastSeen": firestore.SERVER_TIMESTAMP,
        }
    )
    return {
        "saved": True,
        "collection": "members",
        "student_id": student_id,
        "updated_fields": ["present", "lastSeen"],
    }
