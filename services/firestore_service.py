import os
from google.cloud import firestore

# Fix Firestore Mode Issue: Explicit project parameter
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
db = firestore.Client(project=project_id)

def save_chat(user_id, question, answer):
    try:
        doc_ref = db.collection("chat_history").add({
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        print(f"🔥 Saved chat document: {doc_ref}")
    except Exception as e:
        print(f"❌ Firestore Save Error: {e}")
