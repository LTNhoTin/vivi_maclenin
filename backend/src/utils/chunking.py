import re
from typing import List, Dict


def normalize_text(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"\t+", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_sentences(paragraph: str) -> List[str]:
    parts = re.split(r"(?<=[\.!\?])\s+", paragraph)
    return [p.strip() for p in parts if p and p.strip()]


def word_count(s: str) -> int:
    return len(re.findall(r"\w+", s))


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    separators: List[str] = None,
    source: str = "unknown"
) -> List[Dict]:
    """
    Chia nhỏ văn bản theo "điểm a)" (đề xuất hợp lý):
    - Tôn trọng ranh giới câu/đoạn.
    - Kích thước chunk theo số từ (mặc định ~800 từ), chồng lấn 10-20% (mặc định 120 từ).
    - Không cắt giữa câu, giữ cấu trúc bullet/đoạn.
    - Gán metadata nguồn để trích dẫn.

    Có thể điều chỉnh bằng config.
    """
    if separators is None:
        separators = ["\n\n", "\n", ".", "?", "!", ";", ","]

    text = normalize_text(text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: List[Dict] = []
    current_words: List[str] = []
    current_sentences: List[str] = []

    def flush_chunk():
        if not current_sentences:
            return
        chunk_text_str = " ".join(current_sentences).strip()
        chunks.append({
            "id": len(chunks),
            "text": chunk_text_str,
            "source": source,
        })

    for para in paragraphs:
        sentences = split_into_sentences(para)
        for sent in sentences:
            sent_words = re.findall(r"\w+", sent)
            if len(current_words) + len(sent_words) > chunk_size:
                flush_chunk()
                if chunk_overlap > 0:
                    overlap_words = current_words[-chunk_overlap:] if len(current_words) > chunk_overlap else current_words
                    if current_sentences:
                        overlap_tail = " ".join(overlap_words)
                        current_sentences = [overlap_tail]
                        current_words = overlap_words.copy()
                    else:
                        current_sentences = []
                        current_words = []
                else:
                    current_sentences = []
                    current_words = []

            current_sentences.append(sent)
            current_words.extend(sent_words)

    flush_chunk()

    cleaned = [c for c in chunks if word_count(c["text"]) >= 40]
    return cleaned if cleaned else chunks


def preview_chunks(chunks: List[Dict], n: int = 3) -> List[Dict]:
    return chunks[:n]
