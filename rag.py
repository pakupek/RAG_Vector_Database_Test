"""Potok RAG: dokumenty -> chunking -> embeddingi Qwen3 (Ollama) -> baza wektorowa -> LLM Qwen (Ollama).
Użycie:
  python rag.py ingest --db qdrant --docs ./docs --subdir csv
  python rag.py ask    --db milvus  "Jak działa HNSW?"
  python rag.py ask    --db all     "Jak działa HNSW?"   # to samo pytanie do wszystkich 4 baz
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import ollama
import config as C
from backups import export_backup, import_backup
from document_loaders import SUPPORTED_EXTENSIONS, extract_document_text
from vectorstores import STORES, get_store, Hit

client = ollama.Client(host=C.OLLAMA_HOST)

SYSTEM_PROMPT = (
    "Jesteś asystentem odpowiadającym wyłącznie na podstawie podanego kontekstu. "
    "Jeśli kontekst nie zawiera odpowiedzi, napisz, że nie wiesz. Odpowiadaj po polsku, zwięźle.")


def chunk(text: str, size=C.CHUNK_SIZE, overlap=C.CHUNK_OVERLAP) -> list[str]:
    out, step = [], size - overlap
    for i in range(0, max(len(text), 1), step):
        piece = text[i:i + size].strip()
        if piece:
            out.append(piece)
    return out


def show_progress(label: str, current: int, total: int, width: int = 30) -> None:
    percent = current / total
    filled = int(width * percent)
    bar = "#" * filled + "-" * (width - filled)
    ending = "\n" if current >= total else ""
    sys.stdout.write(f"\r{label}: [{bar}] {current}/{total} ({percent:.0%}){ending}")
    sys.stdout.flush()


def embed(
    texts: list[str],
    batch: int = C.EMBED_BATCH_SIZE,
    progress_label: str | None = None,
) -> list[list[float]]:
    vecs: list[list[float]] = []
    if progress_label and texts:
        show_progress(progress_label, 0, len(texts))
    try:
        for i in range(0, len(texts), batch):
            r = client.embed(model=C.EMBED_MODEL, input=texts[i:i + batch])
            vecs.extend(r["embeddings"])
            if progress_label:
                show_progress(progress_label, len(vecs), len(texts))
    finally:
        if progress_label and len(vecs) < len(texts):
            sys.stdout.write("\n")
            sys.stdout.flush()
    return vecs


def stable_id(source: str, chunk_index: int) -> int:
    key = f"{source}\0{chunk_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") & ((1 << 63) - 1)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_subdir(docs_dir: str, subdir: str) -> tuple[Path, Path, str]:
    docs_root = Path(docs_dir).resolve()
    if not docs_root.is_dir():
        raise NotADirectoryError(f"Katalog dokumentów nie istnieje: {docs_root}")

    relative_path = Path(subdir)
    if relative_path.is_absolute():
        raise ValueError("--subdir musi być ścieżką względną względem --docs")

    selected_root = (docs_root / relative_path).resolve()
    try:
        relative_root = selected_root.relative_to(docs_root)
    except ValueError as error:
        raise ValueError("--subdir musi znajdować się wewnątrz --docs") from error

    if not relative_root.parts or ".rag_state" in relative_root.parts:
        raise ValueError("--subdir musi wskazywać podfolder, a nie cały katalog docs lub .rag_state")
    return docs_root, selected_root, relative_root.as_posix()


def load_documents(
    docs_root: Path,
    selected_root: Path,
) -> dict[str, tuple[str, list[str]]]:
    documents = {}
    paths = sorted(selected_root.rglob("*")) if selected_root.is_dir() else []
    paths = [
        path for path in paths
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and ".rag_state" not in path.relative_to(docs_root).parts
    ]
    if paths:
        show_progress("Wczytywanie dokumentów", 0, len(paths))

    for index, path in enumerate(paths, start=1):
        relative_path = path.relative_to(docs_root)
        source = relative_path.as_posix()
        show_progress(f"Wczytywanie {path.name}", index - 1, len(paths))
        text = extract_document_text(path)
        documents[source] = (
            file_digest(path),
            chunk(text),
        )
        show_progress("Wczytywanie dokumentów", index, len(paths))
    return documents


def source_in_subdir(source: str, subdir: str) -> bool:
    return source == subdir or source.startswith(subdir + "/")


def load_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("documents"), dict):
        raise ValueError(f"Nieprawidłowy plik stanu indeksowania: {path}")
    for source, document in manifest["documents"].items():
        if (
            not isinstance(source, str)
            or not isinstance(document, dict)
            or not isinstance(document.get("sha256"), str)
            or not isinstance(document.get("ids"), list)
            or any(not isinstance(point_id, int) for point_id in document["ids"])
        ):
            raise ValueError(f"Nieprawidłowe dane dokumentu w pliku stanu: {path}")
    return manifest


def make_manifest(documents: dict[str, tuple[str, list[str]]]) -> dict:
    return {
        "version": 1,
        "chunk_size": C.CHUNK_SIZE,
        "chunk_overlap": C.CHUNK_OVERLAP,
        "documents": {
            source: {
                "sha256": digest,
                "ids": [stable_id(source, i) for i in range(len(texts))],
            }
            for source, (digest, texts) in documents.items()
        },
    }


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def ingest(db: str, docs_dir: str, subdir: str, rebuild: bool = False) -> None:
    docs_root, selected_root, selected_subdir = resolve_subdir(docs_dir, subdir)
    documents = load_documents(docs_root, selected_root)
    names = list(STORES) if db == "all" else [db]
    selected_manifest = make_manifest(documents)
    stores = {}
    base_manifests = {}
    previous_selected_documents = {}
    changed_sources_by_db = {}
    removed_sources_by_db = {}
    create_collections = {}
    try:
        for name in names:
            store = get_store(name)
            stores[name] = store
            state_path = docs_root / ".rag_state" / f"{name}.json"
            previous = load_manifest(state_path)
            compatible = (
                previous is not None
                and previous.get("version") == selected_manifest["version"]
                and previous.get("chunk_size") == C.CHUNK_SIZE
                and previous.get("chunk_overlap") == C.CHUNK_OVERLAP
            )
            collection_exists = store.collection_exists()
            point_count = store.count() if collection_exists else 0

            if compatible:
                base_manifest = previous
                needs_create = not collection_exists
                expected_count = sum(
                    len(document.get("ids", []))
                    for document in base_manifest["documents"].values()
                )
                if point_count != expected_count:
                    raise RuntimeError(
                        f"[{name}] liczba punktów ({point_count}) nie zgadza się "
                        f"z manifestem ({expected_count}). Przerywam, aby nie "
                        "przebudować danych spoza wybranego podfolderu."
                    )
                if not collection_exists and expected_count:
                    raise RuntimeError(
                        f"[{name}] manifest istnieje, ale kolekcja nie. "
                        "Najpierw przywróć zgodną bazę wektorową."
                    )
            else:
                if point_count:
                    raise RuntimeError(
                        f"[{name}] baza zawiera {point_count} punktów, ale brakuje "
                        "zgodnego manifestu. Nie można bezpiecznie indeksować "
                        "pojedynczego podfolderu; przywróć manifest albo użyj "
                        "pustej kolekcji."
                    )
                base_manifest = make_manifest({})
                needs_create = True

            old_documents = base_manifest["documents"]
            selected_old = {
                source: document
                for source, document in old_documents.items()
                if source_in_subdir(source, selected_subdir)
            }
            changed_sources = {
                source for source, (digest, _) in documents.items()
                if rebuild or selected_old.get(source, {}).get("sha256") != digest
            }
            removed_sources = set(selected_old) - set(documents)

            base_manifests[name] = base_manifest
            previous_selected_documents[name] = selected_old
            changed_sources_by_db[name] = changed_sources
            removed_sources_by_db[name] = removed_sources
            create_collections[name] = needs_create

        if not documents and not any(previous_selected_documents[name] for name in names):
            extensions = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise SystemExit(
                f"Brak obsługiwanych plików ({extensions}) lub wcześniej "
                f"zaindeksowanych dokumentów w podfolderze {selected_subdir}"
            )

        needed_chunks: dict[tuple[str, int], str] = {}
        for name in names:
            for source in changed_sources_by_db[name]:
                for index, text in enumerate(documents[source][1]):
                    needed_chunks[(source, index)] = text

        chunk_keys = list(needed_chunks)
        t0 = time.perf_counter()
        if chunk_keys:
            embedded = embed(
                [needed_chunks[key] for key in chunk_keys],
                progress_label="Tworzenie embeddingów",
            )
        else:
            embedded = []
            print("Tworzenie embeddingów pominięte: brak nowych lub zmienionych fragmentów.")
        embeddings = dict(zip(chunk_keys, embedded))
        embedding_time = time.perf_counter() - t0

        for name in names:
            store = stores[name]
            state_path = docs_root / ".rag_state" / f"{name}.json"
            t0 = time.perf_counter()
            old_selected = previous_selected_documents[name]
            changed_sources = changed_sources_by_db[name]
            removed_sources = removed_sources_by_db[name]
            if not changed_sources and not removed_sources and not create_collections[name]:
                print(
                    f"[{name}] podfolder {selected_subdir} bez zmian; "
                    f"indeks zawiera {store.count()} fragmentów."
                )
                continue

            if create_collections[name]:
                store.recreate(C.EMBED_DIM)
            else:
                stale_ids = [
                    point_id
                    for source in sorted(changed_sources | removed_sources)
                    for point_id in old_selected.get(source, {}).get("ids", [])
                ]
                deleted = 0
                try:
                    for i in range(0, len(stale_ids), C.UPSERT_BATCH_SIZE):
                        batch_ids = stale_ids[i:i + C.UPSERT_BATCH_SIZE]
                        store.delete(batch_ids)
                        deleted += len(batch_ids)
                        show_progress(f"Usuwanie z {name}", deleted, len(stale_ids))
                finally:
                    if deleted < len(stale_ids):
                        sys.stdout.write("\n")
                        sys.stdout.flush()

            pending_ids, pending_vectors, pending_texts, pending_sources = [], [], [], []
            for source in sorted(changed_sources):
                for index, text in enumerate(documents[source][1]):
                    pending_ids.append(stable_id(source, index))
                    pending_vectors.append(embeddings[(source, index)])
                    pending_texts.append(text)
                    pending_sources.append(source)

            indexed = 0
            try:
                for i in range(0, len(pending_ids), C.UPSERT_BATCH_SIZE):
                    end = i + C.UPSERT_BATCH_SIZE
                    store.upsert(
                        pending_ids[i:end],
                        pending_vectors[i:end],
                        pending_texts[i:end],
                        pending_sources[i:end],
                    )
                    indexed += len(pending_ids[i:end])
                    show_progress(f"Zapis do {name}", indexed, len(pending_ids))
                store.finish_ingest()
            finally:
                if indexed < len(pending_ids):
                    sys.stdout.write("\n")
                    sys.stdout.flush()

            merged_documents = {
                source: document
                for source, document in base_manifests[name]["documents"].items()
                if not source_in_subdir(source, selected_subdir)
            }
            merged_documents.update(selected_manifest["documents"])
            updated_manifest = {
                **selected_manifest,
                "documents": merged_documents,
            }
            count = store.count()
            expected_count = sum(
                len(document.get("ids", []))
                for document in updated_manifest["documents"].values()
            )
            if count != expected_count:
                raise RuntimeError(
                    f"[{name}] po aktualizacji baza zawiera {count} punktów, "
                    f"manifest opisuje {expected_count}."
                )
            save_manifest(state_path, updated_manifest)
            print(
                f"[{name}] podfolder {selected_subdir}: indeks zawiera {count} "
                f"fragmentów; zapisano {len(pending_ids)}, "
                f"usunięto źródeł {len(removed_sources)} "
                f"w {time.perf_counter()-t0:.2f}s "
                f"(embeddingi: {embedding_time:.2f}s)"
            )
    finally:
        for store in stores.values():
            store.close()


def build_prompt(question: str, hits: list[Hit]) -> str:
    ctx = "\n\n".join(f"[{i+1}] ({h.source}) {h.text}" for i, h in enumerate(hits))
    return f"Kontekst:\n{ctx}\n\nPytanie: {question}\nOdpowiedź:"


def generate(question: str, hits: list[Hit]) -> str:
    r = client.chat(
        model=C.LLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": build_prompt(question, hits)}],
        options={
            "temperature": C.LLM_TEMPERATURE,
            "seed": C.LLM_SEED,
            "num_ctx": C.LLM_NUM_CTX,
            "num_predict": C.LLM_NUM_PREDICT,
        },
        think=C.LLM_THINK,
        keep_alive=C.LLM_KEEP_ALIVE,
        stream=True,
    )
    answer = []
    for part in r:
        text = part["message"]["content"]
        if text:
            sys.stdout.write(text)
            sys.stdout.flush()
            answer.append(text)
    result = "".join(answer)
    if not result.strip():
        raise RuntimeError(
            f"Model {C.LLM_MODEL} nie zwrócił treści odpowiedzi. "
            "Sprawdź obsługę parametru think przez używaną wersję Ollama."
        )
    return result


def ask(db: str, question: str) -> None:
    qvec = embed([question])[0]
    for n in (list(STORES) if db == "all" else [db]):
        store = get_store(n)
        try:
            t0 = time.perf_counter(); hits = store.search(qvec); t_ret = (time.perf_counter() - t0) * 1000
            print(f"\n===== {n} | wyszukiwanie {t_ret:.1f} ms =====")
            for h in hits:
                print(f"  id={h.id} score={h.score:.4f} [{h.source}] {h.text[:80]!r}")
            if not hits:
                print("Nie znaleziono pasujących fragmentów; pomijam generowanie odpowiedzi.")
                continue
            print("\nOdpowiedź: ", end="", flush=True)
            t0 = time.perf_counter(); generate(question, hits); t_gen = time.perf_counter() - t0
            print(f"\n\n[generowanie {t_gen:.1f} s]")
        finally:
            store.close()


def backup_export(db: str, output: str) -> None:
    store = get_store(db)
    try:
        count = export_backup(store, db, Path(output))
        print(f"[{db}] wyeksportowano {count} punktów do {output}")
    finally:
        store.close()


def backup_import(db: str, input_path: str, replace: bool = False) -> None:
    store = get_store(db)
    try:
        count = import_backup(store, Path(input_path), replace=replace)
        print(f"[{db}] zaimportowano {count} punktów z {input_path}")
    finally:
        store.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    choices = [*STORES, "all"]
    a = sub.add_parser("ingest"); a.add_argument("--db", choices=choices, required=True); a.add_argument("--docs", required=True)
    a.add_argument(
        "--subdir", required=True,
        help="Podfolder względem --docs; ingest całego katalogu docs jest wyłączony",
    )
    a.add_argument(
        "--rebuild", action="store_true",
        help="Wymuś ponowne embeddingi tylko dla wybranego podfolderu",
    )
    b = sub.add_parser("ask"); b.add_argument("--db", choices=choices, required=True); b.add_argument("question")
    backup_parser = sub.add_parser("backup", help="Eksportuj lub importuj gotowe wektory")
    backup_sub = backup_parser.add_subparsers(dest="backup_action", required=True)
    export_parser = backup_sub.add_parser("export", help="Zapisz punkty bazy do pliku JSONL")
    export_parser.add_argument("--db", choices=list(STORES), required=True)
    export_parser.add_argument("--out", required=True)
    import_parser = backup_sub.add_parser("import", help="Odtwórz punkty z pliku JSONL")
    import_parser.add_argument("--db", choices=list(STORES), required=True)
    import_parser.add_argument("--file", required=True)
    import_parser.add_argument(
        "--replace", action="store_true",
        help="Usuń istniejącą kolekcję przed importem",
    )
    args = ap.parse_args()
    if args.cmd == "ingest":
        ingest(args.db, args.docs, args.subdir, rebuild=args.rebuild)
    elif args.cmd == "ask":
        ask(args.db, args.question)
    elif args.backup_action == "export":
        backup_export(args.db, args.out)
    else:
        backup_import(args.db, args.file, replace=args.replace)
