"""Extract searchable text from supported document formats."""
from __future__ import annotations

import csv
import json
from html.parser import HTMLParser
from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".csv", ".xlsx", ".json", ".jsonl",
    ".parquet", ".pdf", ".docx", ".html", ".htm", ".xml",
}


def _append_json_value(lines: list[str], field: str, value: object) -> None:
    if isinstance(value, dict):
        if not value:
            lines.append(f"- {field or 'wartość'}: {{}}")
            return
        for key, nested_value in value.items():
            nested_field = f"{field}.{key}" if field else str(key)
            _append_json_value(lines, nested_field, nested_value)
    elif isinstance(value, list):
        if not value:
            lines.append(f"- {field or 'wartość'}: []")
        elif all(not isinstance(item, (dict, list)) for item in value):
            formatted = ", ".join(json.dumps(item, ensure_ascii=False) for item in value)
            lines.append(f"- {field or 'wartość'}: {formatted}")
        else:
            for index, item in enumerate(value, start=1):
                _append_json_value(lines, f"{field or 'wartość'}[{index}]", item)
    else:
        lines.append(f"- {field or 'wartość'}: {json.dumps(value, ensure_ascii=False)}")


def _extract_json(path: Path) -> str:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Nieprawidłowy JSONL w {path}, wiersz {line_number}, "
                        f"kolumna {error.colno}: {error.msg}"
                    ) from error
    else:
        try:
            with path.open("r", encoding="utf-8-sig") as source:
                data = json.load(source)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Nieprawidłowy JSON w {path}, wiersz {error.lineno}, "
                f"kolumna {error.colno}: {error.msg}"
            ) from error
        records = data if isinstance(data, list) else [data]

    if not records:
        return ""
    lines = [f"# Dane z pliku {path.name}", ""]
    for index, record in enumerate(records, start=1):
        lines.extend((f"## Rekord {index}", ""))
        _append_json_value(lines, "", record)
        lines.append("")
    return "\n".join(lines)


def _extract_csv(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        sample = source.read(8192)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        source.seek(0)
        reader = csv.DictReader(source, dialect=dialect)
        if not reader.fieldnames or not any(name and name.strip() for name in reader.fieldnames):
            raise ValueError(f"Plik CSV nie zawiera nagłówków kolumn: {path}")

        lines = [f"# Dane z pliku {path.name}", ""]
        count = 0
        for count, row in enumerate(reader, start=1):
            lines.extend((f"## Rekord {count}", ""))
            for column, value in row.items():
                if column is None:
                    label = "dodatkowe wartości"
                else:
                    label = column.strip() or "bez nazwy"
                if isinstance(value, list):
                    value = ", ".join(value)
                if value and value.strip():
                    lines.append(f"- {label}: {value.strip()}")
            lines.append("")
    if count == 0:
        raise ValueError(f"Plik CSV nie zawiera żadnych rekordów: {path}")
    return "\n".join(lines)


def _extract_xlsx(path: Path) -> str:
    try:
        from openpyxl import load_workbook
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Do odczytu XLSX zainstaluj zależności poleceniem: pip install -r requirements.txt"
        ) from error

    workbook = load_workbook(path, read_only=True, data_only=True)
    lines = [f"# Dane z pliku {path.name}", ""]
    total_records = 0
    try:
        for worksheet in workbook.worksheets:
            headers: list[str] | None = None
            sheet_records = 0
            for row in worksheet.iter_rows(values_only=True):
                if headers is None:
                    if not any(value is not None and str(value).strip() for value in row):
                        continue
                    headers = [
                        str(value).strip() if value is not None and str(value).strip()
                        else f"Kolumna {index + 1}"
                        for index, value in enumerate(row)
                    ]
                    lines.extend((f"## Arkusz: {worksheet.title}", ""))
                    continue
                if not any(value is not None and str(value).strip() for value in row):
                    continue

                sheet_records += 1
                total_records += 1
                lines.extend((f"### Rekord {sheet_records}", ""))
                for index, value in enumerate(row):
                    if value is None or not str(value).strip():
                        continue
                    label = headers[index] if index < len(headers) else f"Kolumna {index + 1}"
                    lines.append(f"- {label}: {value}")
                lines.append("")
    finally:
        workbook.close()

    if total_records == 0:
        raise ValueError(f"Plik XLSX nie zawiera rekordów pod nagłówkami: {path}")
    return "\n".join(lines)


def _extract_parquet(path: Path) -> str:
    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Do odczytu Parquet zainstaluj zależności poleceniem: pip install -r requirements.txt"
        ) from error

    reader = parquet.ParquetFile(str(path))
    lines = [f"# Dane z pliku {path.name}", ""]
    record_number = 0
    for batch in reader.iter_batches(batch_size=256):
        for record in batch.to_pylist():
            record_number += 1
            lines.extend((f"## Rekord {record_number}", ""))
            for field, value in record.items():
                if value is not None:
                    formatted = json.dumps(value, ensure_ascii=False, default=str)
                    lines.append(f"- {field}: {formatted}")
            lines.append("")

    if record_number == 0:
        raise ValueError(f"Plik Parquet nie zawiera żadnych rekordów: {path}")
    return "\n".join(lines)


class _HTMLTextParser(HTMLParser):
    _BLOCK_TAGS = {
        "address", "article", "aside", "blockquote", "br", "dd", "div", "dl",
        "dt", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr",
        "li", "main", "ol", "p", "section", "table", "td", "th", "tr", "ul",
    }
    _HIDDEN_TAGS = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._HIDDEN_TAGS:
            self._hidden_depth += 1
        elif not self._hidden_depth and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._HIDDEN_TAGS and self._hidden_depth:
            self._hidden_depth -= 1
        elif not self._hidden_depth and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            self.parts.append(data)


def _extract_html(path: Path) -> str:
    parser = _HTMLTextParser()
    parser.feed(path.read_text(encoding="utf-8-sig", errors="replace"))
    parser.close()
    text = "".join(parser.parts)
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def _extract_docx(path: Path) -> str:
    try:
        from docx import Document
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Do odczytu DOCX zainstaluj zależności poleceniem: pip install -r requirements.txt"
        ) from error

    document = Document(str(path))
    parts = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Do odczytu PDF zainstaluj zależności poleceniem: pip install -r requirements.txt"
        ) from error

    reader = PdfReader(str(path))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if not text:
        raise ValueError(
            f"PDF nie zawiera tekstu możliwego do odczytu: {path}. "
            "Skanowane pliki wymagają OCR przed indeksowaniem."
        )
    return text


def extract_document_text(path: Path) -> str:
    extension = path.suffix.lower()
    if extension in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if extension == ".csv":
        return _extract_csv(path)
    if extension == ".xlsx":
        return _extract_xlsx(path)
    if extension == ".parquet":
        return _extract_parquet(path)
    if extension in {".json", ".jsonl"}:
        return _extract_json(path)
    if extension == ".pdf":
        return _extract_pdf(path)
    if extension == ".docx":
        return _extract_docx(path)
    if extension in {".html", ".htm", ".xml"}:
        return _extract_html(path)
    raise ValueError(f"Nieobsługiwane rozszerzenie pliku: {path}")
