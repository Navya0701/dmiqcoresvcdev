from google.cloud import firestore
from datetime import datetime
import os

# Use Google credentials from environment
db = firestore.Client()

def create_thread(user_id):
    if not user_id:
        raise ValueError("user_id is required")

    threads_ref = db.collection("users").document(user_id).collection("threads")
    thread_doc = threads_ref.document()  # Auto ID

    thread_id = thread_doc.id

    thread_doc.set({
        "title": f"Session - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
        "createdAt": firestore.SERVER_TIMESTAMP,
        "lastUpdated": firestore.SERVER_TIMESTAMP
    })

    return thread_id


def save_message(user_id, thread_id, role, content):
    if not all([user_id, thread_id, role, content]):
        return

    messages_ref = (
        db.collection("users")
        .document(user_id)
        .collection("threads")
        .document(thread_id)
        .collection("messages")
    )

    messages_ref.add({
        "role": role,
        "content": content,
        "timestamp": firestore.SERVER_TIMESTAMP,
    })

    # Update thread timestamp
    db.collection("users").document(user_id).collection("threads").document(thread_id).update({
        "lastUpdated": firestore.SERVER_TIMESTAMP
    })


def get_threads(user_id):
    threads_ref = db.collection("users").document(user_id).collection("threads")
    docs = threads_ref.order_by("lastUpdated", direction=firestore.Query.DESCENDING).stream()
    return [{"id": d.id, **d.to_dict()} for d in docs]


def get_messages(user_id, thread_id):
    msgs_ref = (
        db.collection("users")
        .document(user_id)
        .collection("threads")
        .document(thread_id)
        .collection("messages")
    )
    docs = msgs_ref.order_by("timestamp").stream()
    return [{"id": d.id, **d.to_dict()} for d in docs]
