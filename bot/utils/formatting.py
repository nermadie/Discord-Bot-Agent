import re


def _split_text_chunks(text, chunk_size=1800):
    """Split long text into fixed-size chunks for Discord message limits."""
    content = str(text or "")
    return [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]


def _is_table_like_line(line):
    s = str(line or "").strip()
    if not s:
        return False
    if s.count("|") >= 2:
        return True
    return bool(re.match(r"^[\s\-|:]{6,}$", s))


def _split_table_cells(line):
    s = str(line or "").strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [cell.strip() for cell in s.split("|")]


def _is_separator_row(cells):
    if not cells:
        return False
    normalized = [re.sub(r"\s+", "", str(c)) for c in cells]
    if not any(normalized):
        return False
    return all(bool(re.fullmatch(r"[-:]+", c or "")) for c in normalized if c != "")


def _render_table_block(lines):
    rows = [_split_table_cells(line) for line in (lines or []) if str(line).strip()]
    rows = [row for row in rows if any(str(c).strip() for c in row)]
    if not rows:
        return "\n".join(lines)

    rows = [row for row in rows if not _is_separator_row(row)]
    if len(rows) < 2:
        return "\n".join(lines)

    col_count = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (col_count - len(row)) for row in rows]
    widths = []
    for idx in range(col_count):
        widths.append(max(len(str(row[idx])) for row in normalized_rows))

    def format_row(row):
        return (
            "| "
            + " | ".join(str(row[idx]).ljust(widths[idx]) for idx in range(col_count))
            + " |"
        )

    header = format_row(normalized_rows[0])
    sep = (
        "| " + " | ".join("-" * max(3, widths[idx]) for idx in range(col_count)) + " |"
    )
    body = [format_row(row) for row in normalized_rows[1:]]

    return "```text\n" + "\n".join([header, sep] + body) + "\n```"


def _stylize_line(line):
    raw = str(line or "")
    stripped = raw.strip()
    if not stripped:
        return raw

    heading_match = re.match(r"^#{1,6}\s+(.+)$", stripped)
    if heading_match:
        title = heading_match.group(1).strip()
        return f"**{title}**"

    if (
        re.match(r"^\d+\.\s+.+$", stripped)
        and "|" not in stripped
        and len(stripped) <= 120
    ):
        return f"**{stripped}**"

    if stripped.endswith(":") and len(stripped) <= 100 and "|" not in stripped:
        return f"***{stripped}***"

    if re.search(r"\b(ví dụ|example)\b", stripped, flags=re.IGNORECASE):
        return f"*{stripped}*"

    return raw


def _format_rich_text_for_discord(text):
    """Post-process model output for clearer Discord rendering."""
    content = str(text or "").strip()
    if not content:
        return ""

    segments = re.split(r"(```[\s\S]*?```)", content)
    output_segments = []

    for segment in segments:
        if not segment:
            continue
        if segment.startswith("```") and segment.endswith("```"):
            output_segments.append(segment)
            continue

        lines = segment.splitlines()
        out_lines = []
        idx = 0
        while idx < len(lines):
            if _is_table_like_line(lines[idx]):
                start = idx
                while idx < len(lines) and _is_table_like_line(lines[idx]):
                    idx += 1
                block = lines[start:idx]
                if len(block) >= 2:
                    out_lines.append(_render_table_block(block))
                else:
                    out_lines.extend([_stylize_line(item) for item in block])
                continue

            out_lines.append(_stylize_line(lines[idx]))
            idx += 1

        output_segments.append("\n".join(out_lines).strip())

    return "\n\n".join([item for item in output_segments if item]).strip()


def _build_reason_single_message(prompt, answer_text, model_used):
    """Compose one complete reasoning response block for Discord output."""
    lines = [
        "🧩 **Reasoning Assistant**",
        f"📝 **Bài toán:** **{prompt}**",
        "",
        "**Phân tích:**",
        str(answer_text or "").strip(),
        "",
        f"**Model:** {model_used}",
    ]

    return "\n".join(lines).strip()
