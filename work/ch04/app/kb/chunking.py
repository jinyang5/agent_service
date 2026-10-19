import re

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
# 中文无词边界,分隔符优先段落/换行,再句末标点,最后逐字
CJK_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "!", "?", ";", "，", " ", ""]


def split_sections(md: str) -> list[Document]:
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=True)
    return splitter.split_text(md)


def recursive_split(text: str, chunk_size: int, chunk_overlap: int = 0) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=CJK_SEPARATORS, is_separator_regex=False, length_function=len,
    )
    return splitter.split_text(text)


_SENT_RE = re.compile(r"[^。！？!?…\n]*[。！？!?…\n]|[^。！？!?…\n]+$")


def _split_sentences(text: str) -> list[str]:
    return [m for m in _SENT_RE.findall(text) if m]


def _trailing_sentences(text: str, max_chars: int) -> str:
    """取 text 结尾若干**完整句**作为重叠,总长尽量不超过 max_chars;
    单句超长则整句保留(优先「不留半截话」)。"""
    out: list[str] = []
    total = 0
    for s in reversed(_split_sentences(text)):
        if out and total + len(s) > max_chars:
            break
        out.insert(0, s)
        total += len(s)
    return "".join(out)


def apply_sentence_overlap(chunks: list[str], overlap: int) -> list[str]:
    if not chunks:
        return []
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        ov = _trailing_sentences(chunks[i - 1], overlap)
        out.append(ov + chunks[i] if ov else chunks[i])
    return out


_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def is_table_block(text: str) -> bool:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return (
        len(lines) >= 2
        and lines[0].lstrip().startswith("|")
        and bool(_TABLE_SEP_RE.match(lines[1])) and "-" in lines[1]
    )


def split_table_rows(table_md: str, max_rows: int) -> list[str]:
    lines = [ln for ln in table_md.strip().splitlines() if ln.strip()]
    header, sep, rows = lines[0], lines[1], lines[2:]
    if len(rows) <= max_rows:
        return [table_md.strip()]   # 0 数据行也走这里:整块原样返回
    out: list[str] = []
    for i in range(0, len(rows), max_rows):
        out.append("\n".join([header, sep, *rows[i:i + max_rows]]))
    return out
