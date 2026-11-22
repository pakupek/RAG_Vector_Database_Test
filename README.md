# RAG: Qdrant / Weaviate / Milvus / Elasticsearch + Qwen (Ollama)

```bash
ollama pull qwen3-embedding:0.6b && ollama pull qwen3.5:9b
docker compose up -d
pip install -r requirements.txt
python rag.py ingest --db all --docs ./docs       # indeksowanie plików: txt/md/csv/xlsx/json/jsonl/parquet/pdf/docx/html/xml
python rag.py ingest --db all --docs ./docs --rebuild  # wymuś pełne odtworzenie indeksów
python rag.py ask --db all "Twoje pytanie"
```

`rag.py ingest` odczytuje bezpośrednio pliki TXT, Markdown, CSV, XLSX, JSON, JSONL, Parquet, PDF, DOCX, HTML/HTM i XML — konwersja do Markdown nie jest wymagana. Dla Parquet, PDF i DOCX zainstaluj zależności przez `pip install -r requirements.txt`. PDF-y skanowane jako obrazy wymagają OCR, którego projekt nie wykonuje.

Indeksowanie używa paczek po 64 teksty dla embeddingów i 512 rekordów przy zapisie. Rozmiar fragmentu zwiększono do 1200 znaków (nakładanie 100 znaków). Przy pierwszym uruchomieniu po aktualizacji lub po zmianie parametrów chunkowania indeks jest tworzony od nowa; kolejne uruchomienia przetwarzają tylko nowe, zmienione i usunięte pliki. Stan dla każdej bazy jest zapisywany w `docs/.rag_state/`. Użyj `--rebuild`, aby wymusić pełne odtworzenie i ponowne policzenie embeddingów. Podczas `ingest` terminal pokazuje postęp generowania embeddingów, usuwania starych rekordów i zapisu do baz.

Odpowiedź LLM jest strumieniowana do terminala. Parametry `LLM_NUM_CTX`, `LLM_NUM_PREDICT`, `LLM_KEEP_ALIVE` i `LLM_THINK` w `config.py` kontrolują rozmiar kontekstu, maksymalną długość odpowiedzi, czas utrzymania modelu w pamięci i tryb rozumowania. Tryb myślenia jest domyślnie wyłączony, aby limit tokenów nie został zużyty na ukryte rozumowanie zamiast odpowiedzi. Gdy wyszukiwanie nie zwróci fragmentów, wywołanie LLM jest pomijane.

Parametry wspólne (slajd 13) są w `config.py` -> `HNSWParams`; mapowanie na nazwy w bazach w `vectorstores.py`.
