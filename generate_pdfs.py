from pathlib import Path
from textwrap import wrap

PAGE_WIDTH = 595
PAGE_HEIGHT = 842
LEFT = 50
TOP = 800
LINE_HEIGHT = 14
FONT_SIZE = 10
MAX_CHARS = 92
LINES_PER_PAGE = 52


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def normalize_lines(text: str) -> list[str]:
    raw_lines = text.splitlines()
    lines: list[str] = []
    for line in raw_lines:
        if not line.strip():
            lines.append("")
            continue
        wrapped = wrap(line, width=MAX_CHARS, replace_whitespace=False, drop_whitespace=False)
        lines.extend(wrapped if wrapped else [""])
    return lines


def paginate(lines: list[str]) -> list[list[str]]:
    pages: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        current.append(line)
        if len(current) >= LINES_PER_PAGE:
            pages.append(current)
            current = []
    if current:
        pages.append(current)
    if not pages:
        pages.append([""])
    return pages


def page_stream(lines: list[str]) -> bytes:
    commands = ["BT", f"/F1 {FONT_SIZE} Tf", f"{LEFT} {TOP} Td", f"{LINE_HEIGHT} TL"]
    for line in lines:
        commands.append(f"({escape_pdf_text(line)}) Tj")
        commands.append("T*")
    commands.append("ET")
    stream = "\n".join(commands).encode("latin-1", errors="replace")
    return stream


def write_text_pdf(src_text: str, out_path: Path) -> None:
    lines = normalize_lines(src_text)
    pages = paginate(lines)

    # Object ids: 1 catalog, 2 pages, 3 font, then pairs of page/content objects.
    objects: dict[int, bytes] = {}
    font_id = 3

    page_ids = []
    content_ids = []
    next_id = 4
    for _ in pages:
        page_ids.append(next_id)
        content_ids.append(next_id + 1)
        next_id += 2

    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"

    kids = " ".join([f"{pid} 0 R" for pid in page_ids])
    objects[2] = f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>".encode("ascii")

    objects[font_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    for idx, lines_in_page in enumerate(pages):
        pid = page_ids[idx]
        cid = content_ids[idx]
        stream = page_stream(lines_in_page)
        objects[cid] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"\nendstream"
        )
        objects[pid] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] ".encode("ascii")
            + f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {cid} 0 R >>".encode("ascii")
        )

    max_id = max(objects)
    offsets = [0] * (max_id + 1)

    payload = bytearray()
    payload.extend(b"%PDF-1.4\n")

    for oid in range(1, max_id + 1):
        if oid not in objects:
            continue
        offsets[oid] = len(payload)
        payload.extend(f"{oid} 0 obj\n".encode("ascii"))
        payload.extend(objects[oid])
        payload.extend(b"\nendobj\n")

    xref_start = len(payload)
    payload.extend(f"xref\n0 {max_id + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for oid in range(1, max_id + 1):
        payload.extend(f"{offsets[oid]:010d} 00000 n \n".encode("ascii"))

    payload.extend(
        (
            f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF\n"
        ).encode("ascii")
    )

    out_path.write_bytes(payload)


def main() -> None:
    mapping = {
        "requirements_source.txt": "requirements.pdf",
        "manual_source.txt": "manual.pdf",
        "replication_source.txt": "replication.pdf",
    }

    for src_name, out_name in mapping.items():
        src = Path(src_name)
        out = Path(out_name)
        write_text_pdf(src.read_text(encoding="utf-8"), out)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
