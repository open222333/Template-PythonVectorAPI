"""切塊：先依段落（空行）切，過長段落再依句號 / 換行 / 固定長度切；相鄰塊保留重疊。"""
import re

_SENTENCE = re.compile(r'(?<=[。！？!?；;.\n])')


def _split_long(text: str, size: int) -> list[str]:
    parts, buf = [], ''
    for sent in (s for s in _SENTENCE.split(text) if s):
        while len(sent) > size:                     # 單句超長：硬切
            if buf:
                parts.append(buf)
                buf = ''
            parts.append(sent[:size])
            sent = sent[size:]
        if len(buf) + len(sent) > size and buf:
            parts.append(buf)
            buf = ''
        buf += sent
    if buf:
        parts.append(buf)
    return parts


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    if size <= 0:
        raise ValueError('size 必須 > 0')
    overlap = max(0, min(overlap, size // 2))
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:
        pieces.extend(_split_long(p, size) if len(p) > size else [p])

    chunks: list[str] = []
    buf = ''
    for piece in pieces:
        if buf and len(buf) + len(piece) + 2 > size:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap else ''
            buf = (tail + '\n' + piece) if tail else piece
            if len(buf) > size:                     # 重疊後超長則捨棄重疊
                buf = piece
        else:
            buf = f'{buf}\n\n{piece}' if buf else piece
    if buf:
        chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()]
