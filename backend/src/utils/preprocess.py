import re
from typing import List


def strip_bom(text: str) -> str:
    return text.lstrip("\ufeff")


def normalize_newlines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def unify_bullets(text: str) -> str:
    bullets = ["•", "▪", "‣", "·", "○", "●", "–", "—"]
    for b in bullets:
        text = text.replace(b, "-")
    return text


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s) > 120:
        return False
    upper_ratio = sum(1 for ch in s if ch.isupper()) / max(1, len(s))
    if upper_ratio > 0.6:
        return True
    if re.match(r"^(Chương|CHƯƠNG|Mục|MỤC|A\.|B\.|C\.|I\.|II\.|III\.|IV\.|V\.)", s):
        return True
    return False


def is_bullet(line: str) -> bool:
    return bool(re.match(r"^[\-\*\+]\s+", line.strip()))


def reflow_paragraph(paragraph: str) -> str:
    lines = [l.strip() for l in paragraph.split("\n") if l.strip()]
    new_lines: List[str] = []
    current = ""
    for i, line in enumerate(lines):
        if is_heading(line) or is_bullet(line):
            if current:
                new_lines.append(current.strip())
                current = ""
            new_lines.append(line)
            continue
        if not current:
            current = line
        else:
            if current.endswith("-"):
                current = current[:-1] + line
            else:
                current = current + " " + line
    if current:
        new_lines.append(current.strip())
    return "\n".join(new_lines)


def merge_short_paragraphs(paragraphs: List[str], min_words: int = 20) -> List[str]:
    merged: List[str] = []
    for p in paragraphs:
        wc = len(re.findall(r"\w+", p))
        if wc < min_words and merged and not is_heading(p):
            merged[-1] = merged[-1].rstrip() + "\n" + p.strip()
        else:
            merged.append(p)
    return merged


def preprocess_text(text: str) -> str:
    text = strip_bom(text)
    text = normalize_newlines(text)
    text = unify_bullets(text)

    paragraphs = [p for p in text.split("\n\n")]
    paragraphs = [reflow_paragraph(p) for p in paragraphs if p and p.strip()]

    paragraphs = merge_short_paragraphs(paragraphs, min_words=20)

    processed = "\n\n".join(paragraphs)
    processed = re.sub(r"[ \t]+", " ", processed)
    processed = re.sub(r"\n{3,}", "\n\n", processed)
    return processed.strip()
