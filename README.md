# RAG z porównaniem baz wektorowych

Projekt stanowi lokalne środowisko do eksperymentów RAG (Retrieval-Augmented Generation). Odpowiedź modelu językowego jest generowana wyłącznie na podstawie fragmentów znalezionych w dokumentach.

Dla porównania dostępne są cztery niezależne backendy wektorowe:

- Qdrant,
- Weaviate,
- Milvus,
- Elasticsearch.

Embeddingi i odpowiedzi są generowane lokalnie przez Ollama. Cały potok można uruchomić na jednym komputerze bez wysyłania danych do zewnętrznej usługi.

## Jak działa projekt

1. Polecenie `ingest` wczytuje jeden wskazany podfolder z katalogu `docs`.
2. Dokumenty są konwertowane do tekstu i dzielone na fragmenty po 1200 znaków z nakładaniem 100 znaków.
3. Ollama generuje dla każdego fragmentu embedding modelu `qwen3-embedding:0.6b` o wymiarze 1024.
4. Fragmenty, ich embeddingi, identyfikatory i względne ścieżki źródeł są zapisane w jednej z baz wektorowych.
5. Polecenie `ask` osadza pytanie tym samym modelem embeddingowym.
6. Baza zwraca do 5 najbardziej podobnych fragmentów według podobieństwa cosinusowego.
7. Ollama generuje odpowiedź modelem `qwen3:4b` na podstawie pobranego kontekstu.

Jeżeli wyszukiwanie nie zwróci fragmentów, model językowy nie jest uruchamiany.

## Składniki

| Komponent | Konfiguracja domyślna |
|---|---|
| Embeddingi | Ollama, `qwen3-embedding:0.6b`, 1024 wymiary |
| Generowanie odpowiedzi | Ollama, `qwen3:4b` |
| Fragmentacja | 1200 znaków, nakładanie 100 znaków |
| Wyszukiwanie | HNSW, cosine, `k = 5`, `ef_search = 128` |
| Nazwa kolekcji/indeksu | `rag_docs` |
| Stan indeksowania | `docs/.rag_state/<nazwa_bazy>.json` |

Konfiguracja wspólna znajduje się w [config.py](config.py), implementacje baz w [vectorstores.py](vectorstores.py), a potok CLI w [rag.py](rag.py).

## Wymagania

- Windows lub inne środowisko obsługujące Python i Docker,
- Docker Desktop z Docker Compose,
- Python 3.10 lub nowszy,
- Ollama uruchomiona pod adresem `http://localhost:11434`.

## Instalacja i uruchomienie

### 1. Pobierz modele Ollama

```powershell
ollama pull qwen3-embedding:0.6b
ollama pull qwen3:4b
```

Dokładne modele i parametry odpowiedzi można zmienić w [config.py](config.py).

### 2. Zainstaluj Python i zależności

W katalogu projektu:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Uruchom bazy wektorowe

```powershell
docker compose up -d
docker compose ps
```

Docker Compose uruchamia:

| Usługa | Port klienta | Port dodatkowy |
|---|---:|---:|
| Qdrant | 6333 | 6334 |
| Weaviate | 8080 | 50051 gRPC |
| Milvus | 19530 | 9091 i 2379 |
| Elasticsearch | 9200 | - |

Dane utrwalają osobne wolumeny Docker: `qdrant_data`, `weaviate_data`, `milvus_data` i `es_data`.

## Przygotowanie dokumentów

Polecenie `ingest` obsługuje następujące rozszerzenia:

- `.txt`, `.md`,
- `.csv`, `.xlsx`,
- `.json`, `.jsonl`,
- `.parquet`,
- `.pdf`, `.docx`,
- `.html`, `.htm`, `.xml`.

Zawartość plików jest konwertowana do tekstu, który następnie trafia do fragmentów. CSV i XLSX są odczytywane wraz z nagłówkami, a JSON i JSONL są przekształcane w rekordy czytelne dla modelu.

PDF musi zawierać tekst możliwy do wyodrębnienia. Projekt nie wykonuje OCR, więc zeskanowane strony PDF bez warstwy tekstowej trzeba przetworzyć poza tym narzędziem.

## Indeksowanie

Indeksowanie musi wskazywać jeden podfolder względem `--docs`. Indeksowanie całego katalogu `docs` jednym poleceniem nie jest obsługiwane.

```powershell
python rag.py ingest --db qdrant --docs .\docs --subdir csv
python rag.py ingest --db milvus --docs .\docs --subdir english
python rag.py ingest --db all --docs .\docs --subdir csv
```

Każdy podfolder można indeksować osobnym poleceniem. Przykładowo:

```powershell
python rag.py ingest --db all --docs .\docs --subdir android
python rag.py ingest --db all --docs .\docs --subdir gaming
```

### Parametry polecenia

- `--db` - `qdrant`, `weaviate`, `milvus`, `elasticsearch` albo `all`,
- `--docs` - katalog główny dokumentów,
- `--subdir` - podfolder z dokumentami względem `--docs`,
- `--rebuild` - ponowne embeddingi i zapis fragmentów tylko dla wskazanego podfolderu.

Domyślnie przy powtórzeniu indeksowania embeddingi są liczone tylko dla nowych lub zmienionych plików. Usunięcie pliku i ponowne uruchomienie dla danego podfolderu usuwa odpowiadające mu punkty z baz.

```powershell
python rag.py ingest --db all --docs .\docs --subdir csv --rebuild
```

Paski postępu wczytywania dokumentów, tworzenia embeddingów, usuwania starych punktów i zapisu do bazy pokazują szacowany pozostały czas aktualnej fazy w minutach lub godzinach. Estymacja jest aktualizowana na podstawie dotychczasowego tempa i może się zmieniać w trakcie pracy.

## Manifest i bezpieczeństwo danych

Dla każdej bazy osobny manifest w `docs/.rag_state/` zapisuje:

- rozmiar i nakładanie fragmentów,
- SHA-256 każdego zaindeksowanego pliku,
- identyfikatory fragmentów,
- ścieżki źródeł względem `docs`.

Operacja aktualizuje tylko manifest i punkty wskazanego podfolderu. Pozostałe podfoldery zostają bez zmian.

Jeżeli baza zawiera punkty, lecz manifest jest brakujący, ma niezgodne ustawienia fragmentów lub liczba punktów nie zgadza się z manifestem, indeksowanie zostaje przerwane. Chroni to dane spoza aktualizowanego podfolderu przed przypadkowym przebudowaniem lub zniszczeniem kolekcji.

Zachowuj te same:

- model embeddingów,
- wymiar wektorów,
- rozmiar i nakładanie fragmentów,
- nazwę kolekcji `rag_docs`.

## Zadawanie pytań

```powershell
python rag.py ask --db qdrant "Jak działa HNSW?"
python rag.py ask --db milvus "Podaj główne wnioski z dokumentacji"
python rag.py ask --db all "W jakiej dziedzinie dotyczy dokument?"
```

Argument pytania można podać jako jeden ciąg. Dla `--db all` pytanie jest wykonane osobno względem każdej z czterech baz.

Terminal wyświetla:

- czas wyszukiwania,
- identyfikator i wynik podobieństwa,
- względną ścieżkę źródła,
- początkowy fragment tekstu,
- strumieniową odpowiedź modelu językowego.

## Eksport i import indeksów

Gotową bazę można przenieść do innego komputera bez ponownego liczenia embeddingów. Kopia zapasowa jest tekstowym plikiem JSONL zawierającym nagłówek oraz wektory, tekst i źródła.

### Eksport

Eksport działa tylko dla jednej konkretnej bazy:

```powershell
python rag.py backup export --db qdrant --out backups\qdrant.jsonl
python rag.py backup export --db weaviate --out backups\weaviate.jsonl
```

Nie uruchamiaj równocześnie indeksowania tej samej bazy podczas eksportu.

### Import

```powershell
python rag.py backup import --db milvus --file backups\qdrant.jsonl
```

Plik importuje się do dowolnego obsługiwanego backendu. Domyślnie import zostanie odrzucony, jeśli kolekcja docelowa już istnieje. Aby zastąpić istniejącą kolekcję:

```powershell
python rag.py backup import --db qdrant --file backups\qdrant.jsonl --replace
```

Import sprawdza format kopii, nazwę kolekcji, model embeddingów i wymiar 1024. Odpowiedni model Ollama musi być dostępny także po stronie docelowej.

Jeżeli chcesz zachować przyrostowe indeksowanie po przeniesieniu danych, skopiuj również odpowiedni manifest `docs/.rag_state/<baza>.json`. Przy zmianie backendu zmień nazwę pliku na nazwę bazy docelowej, na przykład:

```text
qdrant.json -> milvus.json
```

Katalog `backups/` jest przeznaczony na kopie zapasowe i nie jest commitowany do repozytorium.

## Parametry HNSW i wyszukiwania

W [config.py](config.py) istnieje jedna wspólna klasa `HNSWParams`:

| Parametr | Wartość | Znaczenie |
|---|---:|---|
| `m` | 16 | liczba połączeń w indeksie HNSW |
| `ef_construction` | 100 | dokładność budowy indeksu |
| `ef_search` | 128 | dokładność wyszukiwania |
| `k` | 5 | liczba zwracanych fragmentów |
| `metric` | cosine | metryka podobieństwa |

Wektory nie są kwantyzowane, nie są stosowane filtry ani wyszukiwanie hybrydowe. Mapowanie parametrów na API poszczególnych baz znajduje się w [vectorstores.py](vectorstores.py).

## Pozostałe parametry `config.py`

| Parametr | Wartość domyślna |
|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` |
| `LLM_MODEL` | `qwen3:4b` |
| `LLM_TEMPERATURE` | `0.0` |
| `LLM_SEED` | `42` |
| `LLM_THINK` | `False` |
| `LLM_NUM_CTX` | `4096` |
| `LLM_NUM_PREDICT` | `256` |
| `LLM_KEEP_ALIVE` | `5m` |
| `EMBED_BATCH_SIZE` | `64` |
| `UPSERT_BATCH_SIZE` | `512` |

Wyłączenie trybu myślenia ogranicza zużycie limitu odpowiedzi na ukryte rozumowanie. Zwiększenie `LLM_NUM_PREDICT` pozwala generować dłuższe odpowiedzi, ale zwiększa zużycie czasu i pamięci.

## Najczęstsze problemy

### Ollama nie odpowiada

Sprawdź usługę i połączenie:

```powershell
ollama list
curl http://localhost:11434/api/tags
```

Upewnij się, że pobrano oba modele używane przez [config.py](config.py).

### Baza nie działa

```powershell
docker compose ps
docker compose logs qdrant
docker compose logs milvus
docker compose logs elasticsearch
```

### Brak tekstu w PDF

Skanowane PDF-y bez warstwy tekstowej wymagają OCR przed wykonaniem polecenia `ingest`.

### Kolekcja i manifest nie są zgodne

Najpierw przywróć zgodną kopię bazy i odpowiadający jej manifest. Nie usuwaj katalogu `.rag_state` przy istniejących punktach, ponieważ utrata manifestu blokuje bezpieczne indeksowanie pojedynczych podfolderów.
