"""Portable JSONL export/import for stored vectors and document payloads."""
from __future__ import annotations

import json
import math
from pathlib import Path

import config as C
from vectorstores import StoredPoint, VectorStore

FORMAT_VERSION = 1


def _header(source_db: str) -> dict:
    return {
        "format": "rag-vector-backup",
        "version": FORMAT_VERSION,
        "source_db": source_db,
        "collection": C.COLLECTION,
        "embedding_model": C.EMBED_MODEL,
        "embedding_dimension": C.EMBED_DIM,
    }


def _parse_point(value: object, line_number: int, dimension: int) -> StoredPoint:
    if not isinstance(value, dict):
        raise ValueError(f"Wiersz {line_number}: rekord musi być obiektem JSON")
    point_id = value.get("id")
    vector = value.get("vector")
    text = value.get("text")
    source = value.get("source")
    if isinstance(point_id, bool) or not isinstance(point_id, int) or point_id < 0:
        raise ValueError(f"Wiersz {line_number}: id musi być nieujemną liczbą całkowitą")
    if (
        not isinstance(vector, list)
        or len(vector) != dimension
        or any(
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            for number in vector
        )
    ):
        raise ValueError(
            f"Wiersz {line_number}: wektor musi zawierać {dimension} skończonych liczb"
        )
    if not isinstance(text, str) or not isinstance(source, str):
        raise ValueError(f"Wiersz {line_number}: text i source muszą być tekstem")
    return StoredPoint(point_id, [float(number) for number in vector], text, source)


def _read_header(file, path: Path) -> dict:
    first_line = file.readline()
    try:
        header = json.loads(first_line)
    except json.JSONDecodeError as error:
        raise ValueError(f"Nieprawidłowy nagłówek kopii zapasowej: {path}") from error
    if (
        not isinstance(header, dict)
        or header.get("format") != "rag-vector-backup"
        or header.get("version") != FORMAT_VERSION
    ):
        raise ValueError(f"Nieobsługiwany format kopii zapasowej: {path}")
    if header.get("collection") != C.COLLECTION:
        raise ValueError(
            f"Kopia zawiera kolekcję {header.get('collection')!r}, "
            f"a konfiguracja oczekuje {C.COLLECTION!r}"
        )
    if header.get("embedding_model") != C.EMBED_MODEL:
        raise ValueError(
            f"Kopia używa embeddingów {header.get('embedding_model')!r}, "
            f"a konfiguracja {C.EMBED_MODEL!r}"
        )
    if header.get("embedding_dimension") != C.EMBED_DIM:
        raise ValueError(
            f"Kopia ma wymiar {header.get('embedding_dimension')!r}, "
            f"a konfiguracja oczekuje {C.EMBED_DIM}"
        )
    return header


def _iter_file_points(file, dimension: int):
    for line_number, line in enumerate(file, start=2):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Nieprawidłowy JSON w wierszu {line_number}") from error
        yield _parse_point(value, line_number, dimension)


def validate_backup(path: Path) -> tuple[dict, int]:
    with path.open(encoding="utf-8") as file:
        header = _read_header(file, path)
        count = sum(1 for _ in _iter_file_points(file, C.EMBED_DIM))
    return header, count


def export_backup(store: VectorStore, source_db: str, path: Path) -> int:
    if not store.collection_exists():
        raise ValueError(f"Baza {source_db} nie zawiera kolekcji {C.COLLECTION}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Plik już istnieje; wybierz inną nazwę: {path}")

    temporary_path = path.with_name(path.name + ".tmp")
    if temporary_path.exists():
        raise FileExistsError(f"Istnieje plik tymczasowy kopii: {temporary_path}")

    expected_count = store.count()
    exported = 0
    try:
        with temporary_path.open("x", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(_header(source_db), ensure_ascii=False) + "\n")
            for point in store.iter_points(C.UPSERT_BATCH_SIZE):
                parsed = _parse_point(
                    {
                        "id": point.id,
                        "vector": point.vector,
                        "text": point.text,
                        "source": point.source,
                    },
                    exported + 2,
                    C.EMBED_DIM,
                )
                file.write(json.dumps({
                    "id": parsed.id,
                    "vector": parsed.vector,
                    "text": parsed.text,
                    "source": parsed.source,
                }, ensure_ascii=False, allow_nan=False) + "\n")
                exported += 1
        if exported != expected_count:
            raise RuntimeError(
                f"Liczba eksportowanych punktów ({exported}) różni się od "
                f"liczby w bazie ({expected_count}); nie zmieniaj bazy podczas eksportu."
            )
        temporary_path.rename(path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    return exported


def import_backup(
    store: VectorStore,
    path: Path,
    replace: bool = False,
    batch_size: int = C.UPSERT_BATCH_SIZE,
) -> int:
    _, point_count = validate_backup(path)
    exists = store.collection_exists()
    if exists and not replace:
        raise FileExistsError(
            f"Kolekcja {C.COLLECTION} już istnieje. Użyj --replace, aby ją zastąpić."
        )

    store.recreate(C.EMBED_DIM)
    imported = 0
    ids, vectors, texts, sources = [], [], [], []
    with path.open(encoding="utf-8") as file:
        _read_header(file, path)
        for point in _iter_file_points(file, C.EMBED_DIM):
            ids.append(point.id)
            vectors.append(point.vector)
            texts.append(point.text)
            sources.append(point.source)
            if len(ids) >= batch_size:
                store.upsert(ids, vectors, texts, sources)
                imported += len(ids)
                ids, vectors, texts, sources = [], [], [], []
        if ids:
            store.upsert(ids, vectors, texts, sources)
            imported += len(ids)

    store.finish_ingest()
    actual_count = store.count()
    if imported != point_count or actual_count != point_count:
        raise RuntimeError(
            f"Import zakończony niezgodną liczbą punktów: "
            f"plik={point_count}, zapisano={imported}, baza={actual_count}"
        )
    return imported
