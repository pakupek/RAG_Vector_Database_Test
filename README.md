# RAG: Qdrant / Weaviate / Milvus / Elasticsearch + Qwen (Ollama)

```bash
ollama pull qwen3-embedding:0.6b && ollama pull qwen3.5:9b
docker compose up -d
pip install -r requirements.txt
python rag.py ingest --db all --docs ./docs       # indeksowanie plików: txt/md/csv/xlsx/json/jsonl/parquet/pdf/docx/html/xml
python rag.py ingest --db all --docs ./docs --rebuild  # wymuś pełne odtworzenie indeksów
python rag.py ask --db all "Twoje pytanie"
python rag.py backup export --db qdrant --out backups\qdrant.jsonl
python rag.py backup import --db qdrant --file backups\qdrant.jsonl
```

## Przenoszenie gotowego indeksu

Polecenie `backup export` zapisuje wektory, teksty i źródła do przenośnego pliku JSONL. Można go zaimportować do tego samego lub innego backendu bez ponownego liczenia embeddingów:

```powershell
python rag.py backup export --db weaviate --out backups\weaviate.jsonl
# Skopiuj plik na drugi komputer, uruchom tam usługi z docker compose i:
ollama pull qwen3-embedding:0.6b
python rag.py backup import --db milvus --file backups\weaviate.jsonl
```

Wybierz dla eksportu jedną bazę (`qdrant`, `weaviate`, `milvus` lub `elasticsearch`) i wskaż plik wyjściowy. Import domyślnie odmawia nadpisania istniejącej kolekcji. Opcja `--replace` usuwa ją przed odtworzeniem danych:

```powershell
python rag.py backup import --db qdrant --file backups\qdrant.jsonl --replace
```

Eksport wymaga dostępnej kolekcji, a import sprawdza model embeddingów i wymiar wektorów względem `config.py`. Na komputerze docelowym użyj tego samego modelu (`qwen3-embedding:0.6b`), żeby zapytania były osadzane w tej samej przestrzeni. Nie uruchamiaj `ingest --rebuild` po imporcie, jeśli nie chcesz ponownie liczyć embeddingów. Podczas eksportu nie wykonuj równoległego indeksowania. Pliki w `backups/` są ignorowane przez Git.

Jeśli po imporcie chcesz później używać zwykłego `ingest` bez ponownego liczenia embeddingów dla niezmienionych dokumentów, przenieś też katalog dokumentów i plik stanu `docs/.rag_state/<źródłowa_baza>.json`. Przy imporcie do innego backendu skopiuj ten plik pod nazwą docelowej bazy, np. `qdrant.json` jako `milvus.json`.

`rag.py ingest` odczytuje bezpośrednio pliki TXT, Markdown, CSV, XLSX, JSON, JSONL, Parquet, PDF, DOCX, HTML/HTM i XML — konwersja do Markdown nie jest wymagana. Dla Parquet, PDF i DOCX zainstaluj zależności przez `pip install -r requirements.txt`. PDF-y skanowane jako obrazy wymagają OCR, którego projekt nie wykonuje.

Indeksowanie używa paczek po 64 teksty dla embeddingów i 512 rekordów przy zapisie. Rozmiar fragmentu zwiększono do 1200 znaków (nakładanie 100 znaków). Przy pierwszym uruchomieniu po aktualizacji lub po zmianie parametrów chunkowania indeks jest tworzony od nowa; kolejne uruchomienia przetwarzają tylko nowe, zmienione i usunięte pliki. Stan dla każdej bazy jest zapisywany w `docs/.rag_state/`. Użyj `--rebuild`, aby wymusić pełne odtworzenie i ponowne policzenie embeddingów. Podczas `ingest` terminal pokazuje postęp generowania embeddingów, usuwania starych rekordów i zapisu do baz.

Odpowiedź LLM jest strumieniowana do terminala. Parametry `LLM_NUM_CTX`, `LLM_NUM_PREDICT`, `LLM_KEEP_ALIVE` i `LLM_THINK` w `config.py` kontrolują rozmiar kontekstu, maksymalną długość odpowiedzi, czas utrzymania modelu w pamięci i tryb rozumowania. Tryb myślenia jest domyślnie wyłączony, aby limit tokenów nie został zużyty na ukryte rozumowanie zamiast odpowiedzi. Gdy wyszukiwanie nie zwróci fragmentów, wywołanie LLM jest pomijane.

Parametry wspólne (slajd 13) są w `config.py` -> `HNSWParams`; mapowanie na nazwy w bazach w `vectorstores.py`.
