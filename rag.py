"""Potok RAG: dokumenty -> chunking -> embeddingi Qwen3 (Ollama) -> baza wektorowa -> LLM Qwen (Ollama).
Użycie:
  python rag.py ingest --db qdrant --docs ./docs
  python rag.py ask    --db milvus  "Jak działa HNSW?"
  python rag.py ask    --db all     "Jak działa HNSW?"   # to samo pytanie do wszystkich 4 baz
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import ollama
import config as C
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


def load_documents(docs_root: Path) -> dict[str, tuple[str, list[str]]]:
    documents = {}
    paths = [
        path for path in sorted(docs_root.rglob("*"))
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


def ingest(db: str, docs_dir: str, rebuild: bool = False) -> None:
    docs_root = Path(docs_dir)
    documents = load_documents(docs_root)
    if not any(texts for _, texts in documents.values()):
        extensions = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise SystemExit(
            f"Brak obsługiwanych plików ({extensions}) zawierających tekst w {docs_dir}"
        )

    names = list(STORES) if db == "all" else [db]
    current_manifest = make_manifest(documents)
    previous_manifests = {}
    stores = {}
    full_rebuild = {}
    try:
        for name in names:
            store = get_store(name)
            stores[name] = store
            state_path = docs_root / ".rag_state" / f"{name}.json"
            previous = load_manifest(state_path)
            compatible = (
                previous is not None
                and previous.get("version") == current_manifest["version"]
                and previous.get("chunk_size") == C.CHUNK_SIZE
                and previous.get("chunk_overlap") == C.CHUNK_OVERLAP
            )
            previous_manifests[name] = previous if compatible else None
            expected_count = (
                sum(len(doc.get("ids", [])) for doc in previous["documents"].values())
                if compatible else -1
            )
            full_rebuild[name] = (
                rebuild
                or not compatible
                or not store.collection_exists()
                or store.count() != expected_count
            )

        needed_chunks: dict[tuple[str, int], str] = {}
        for name in names:
            old_documents = (
                previous_manifests[name]["documents"]
                if not full_rebuild[name] else {}
            )
            for source, (digest, texts) in documents.items():
                if old_documents.get(source, {}).get("sha256") != digest:
                    for index, text in enumerate(texts):
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
            old_documents = (
                previous_manifests[name]["documents"]
                if not full_rebuild[name] else {}
            )
            changed_sources = {
                source for source, (digest, _) in documents.items()
                if old_documents.get(source, {}).get("sha256") != digest
            }
            removed_sources = set(old_documents) - set(documents)
            if not full_rebuild[name] and not changed_sources and not removed_sources:
                print(f"[{name}] bez zmian; indeks zawiera {store.count()} fragmentów.")
                continue

            if full_rebuild[name]:
                store.recreate(C.EMBED_DIM)
            else:
                stale_ids = [
                    point_id
                    for source in sorted(changed_sources | removed_sources)
                    for point_id in old_documents[source].get("ids", [])
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

            count = store.count()
            save_manifest(state_path, current_manifest)
            action = "odtworzono" if full_rebuild[name] else "zaktualizowano"
            print(
                f"[{name}] {action} indeks: {count} fragmentów; "
                f"zmieniono {len(pending_ids)}, usunięto źródeł {len(removed_sources)} "
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    choices = [*STORES, "all"]
    a = sub.add_parser("ingest"); a.add_argument("--db", choices=choices, required=True); a.add_argument("--docs", required=True)
    a.add_argument("--rebuild", action="store_true", help="Wymuś pełne odtworzenie indeksu")
    b = sub.add_parser("ask"); b.add_argument("--db", choices=choices, required=True); b.add_argument("question")
    args = ap.parse_args()
    ingest(args.db, args.docs, rebuild=args.rebuild) if args.cmd == "ingest" else ask(args.db, args.question)
