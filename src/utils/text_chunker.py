from typing import List
import re


def chunk_text(text: str, chunk_chars: int = 1500, overlap_chars: int = 200) -> List[str]:
    """Simple chunker: prefer splitting on paragraph/sentence boundaries when possible,
    otherwise fall back to character windows with overlap.
    Returns a list of cleaned chunks.
    """
    if not text:
        return []
    # Normalize newlines
    text = text.replace("\r\n", "\n").strip()
    # Try to split into paragraphs first
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []

    for para in paragraphs:
        if len(para) <= chunk_chars:
            chunks.append(para)
            continue
        # break long paragraph into sentence-like pieces
        sentences = re.split(r'(?<=[.?!])\s+', para)
        buf = ""
        for sent in sentences:
            if not sent:
                continue
            if len(buf) + len(sent) + 1 <= chunk_chars:
                buf = (buf + " " + sent).strip()
            else:
                if buf:
                    chunks.append(buf.strip())
                # if single sentence is too long, chunk by characters
                if len(sent) > chunk_chars:
                    start = 0
                    while start < len(sent):
                        end = start + chunk_chars
                        chunks.append(sent[start:end].strip())
                        start = max(end - overlap_chars, end) if end < len(sent) else end
                    buf = ""
                else:
                    buf = sent.strip()
        if buf:
            chunks.append(buf.strip())

    # Apply a final sliding-window to enforce overlap between chunks
    final_chunks: List[str] = []
    for c in chunks:
        if len(c) <= chunk_chars:
            final_chunks.append(c)
            continue
        start = 0
        L = len(c)
        while start < L:
            end = start + chunk_chars
            final_chunks.append(c[start:end].strip())
            if end >= L:
                break
            start = end - overlap_chars

    # Remove empties and strip
    return [c for c in final_chunks if c]
def split_text(text, max_length=1000):
    """
    Splits the input text into smaller chunks based on the specified maximum length.

    Parameters:
    - text (str): The text to be split into chunks.
    - max_length (int): The maximum length of each chunk.

    Returns:
    - list: A list of text chunks.
    """
    chunks = []
    while len(text) > max_length:
        # Find the last space within the max_length limit
        split_index = text.rfind(' ', 0, max_length)
        if split_index == -1:  # No space found, split at max_length
            split_index = max_length
        chunks.append(text[:split_index])
        text = text[split_index:].lstrip()  # Remove leading spaces
    chunks.append(text)  # Add the remaining text as the last chunk
    return chunks