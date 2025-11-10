from pathlib import Path
from typing import List
import os
import PyPDF2


def read_pdf(path: str) -> str:
    """Extract text from a PDF using PyPDF2.
    - If the file is encrypted, tries to decrypt with an empty password or PDF_PASSWORD env.
    - Requires 'pycryptodome' for AES-encrypted PDFs with PyPDF2.
    Returns empty string on failure.
    """
    path = Path(path)
    if not path.exists():
        return ""
    text_pages: List[str] = []
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            # Attempt decryption if needed
            if getattr(reader, "is_encrypted", False):
                passwords = []
                env_pw = os.environ.get("PDF_PASSWORD")
                if env_pw:
                    passwords.append(env_pw)
                # Always try empty password (many PDFs are user-unlocked with empty pwd)
                passwords.append("")
                for pw in passwords:
                    try:
                        res = reader.decrypt(pw)
                        # PyPDF2 returns 0/1/2; success if > 0
                        if isinstance(res, int) and res > 0:
                            break
                    except Exception:
                        # try next password
                        continue

            for page in reader.pages:
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""
                text_pages.append(page_text)
    except Exception:
        return ""
    return "\n".join(text_pages)


def read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""
def read_pdf(file_path):
    import PyPDF2

    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_pdfs(folder_path):
    import os

    pdf_texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            pdf_texts[filename] = read_pdf(file_path)
    return pdf_texts