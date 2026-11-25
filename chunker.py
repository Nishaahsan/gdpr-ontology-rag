"""GDPR-aware chunking: text ko Article ke hisab se todta hai, taake har chunk ka citation sahi ho.

Is file me koi bahari library nahi chahiye, isliye ise akele test kar sakte ho:  python chunker.py
"""
import re
from dataclasses import dataclass
from typing import List, Optional

MAX_WORDS = 220      # is se lamba article ho to chhote chunks me todo
OVERLAP_WORDS = 30

ARTICLE_RE = re.compile(r"^\s*Article\s+(\d{1,3})\s*$", re.MULTILINE)
# EUR-Lex PDF ke har page ka header/footer, jo text me ghus jata hai
NOISE_RE = re.compile(
    r"^\s*(\d{1,2}\.\d{1,2}\.\d{4}\s+L\s*\d+/\d+.*|EN\s+Official Journal.*|L\s*\d+/\d+\s+EN.*|"
    r"Official Journal of the European Union.*|\d{1,3}\s*)$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class Chunk:
    text: str
    article: Optional[int]      # None = recitals / preamble
    title: str
    part: int                   # article ke andar chunk number

    @property
    def label(self) -> str:
        return f"Art. {self.article}" if self.article else "Recitals"


def clean(text: str) -> str:
    text = NOISE_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _split_words(text: str):
    words = text.split()
    if len(words) <= MAX_WORDS:
        yield " ".join(words)
        return
    step = MAX_WORDS - OVERLAP_WORDS
    for start in range(0, len(words), step):
        piece = words[start:start + MAX_WORDS]
        if piece:
            yield " ".join(piece)
        if start + MAX_WORDS >= len(words):
            break


def chunk_gdpr(raw_text: str) -> List[Chunk]:
    text = clean(raw_text)
    matches = list(ARTICLE_RE.finditer(text))
    chunks: List[Chunk] = []

    # Article 1 se pehle ka hissa = recitals/preamble
    preamble = text[: matches[0].start()] if matches else text
    for i, piece in enumerate(_split_words(preamble)):
        if piece.strip():
            chunks.append(Chunk(piece, None, "Recitals / Preamble", i))

    # EUR-Lex me "Article N" do dafa aa sakta hai (jaise table of contents); aakhri nahi, pehla poora wala rakho
    for idx, m in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if not body:
            continue
        lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
        title = lines[0] if lines else ""
        content = " ".join(lines[1:]) if len(lines) > 1 else ""
        if len(content.split()) < 8:        # table of contents wali khali entry
            continue
        number = int(m.group(1))
        for part, piece in enumerate(_split_words(content)):
            header = f"Article {number} - {title}: "
            chunks.append(Chunk(header + piece, number, title, part))
    return chunks


if __name__ == "__main__":
    sample = """
(1) The protection of natural persons is a fundamental right. This is recital text for the test, it is long enough.
(2) The principles of and rules on the protection of natural persons should respect their fundamental rights.

Article 1
Subject-matter and objectives
1. This Regulation lays down rules relating to the protection of natural persons with regard to the processing of personal data.
2. This Regulation protects fundamental rights and freedoms of natural persons.

4.5.2016 L 119/1 Official Journal of the European Union EN

Article 17
Right to erasure ('right to be forgotten')
1. The data subject shall have the right to obtain from the controller the erasure of personal data concerning him or her
without undue delay and the controller shall have the obligation to erase personal data without undue delay where one of the following grounds applies.
"""
    for c in chunk_gdpr(sample):
        print(f"[{c.label}] part {c.part} | {c.title} | {len(c.text.split())} words")
    arts = [c.article for c in chunk_gdpr(sample)]
    assert arts == [None, 1, 17], arts
    print("chunker OK")
