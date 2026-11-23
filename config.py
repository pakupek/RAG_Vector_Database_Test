"""Wspólna konfiguracja eksperymentu.
Zmienia się TYLKO baza wektorowa; reszta jest stała i jawna."""
from dataclasses import dataclass

# --- Modele (Ollama) -------------------------------------------------------
OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "qwen3-embedding:0.6b"   # 1024 wymiary, kontekst 32K
EMBED_DIM = 1024
LLM_MODEL = "qwen3:4b"               
LLM_TEMPERATURE = 0.0                 
LLM_SEED = 42                          
LLM_THINK = False
LLM_NUM_CTX = 4096
LLM_NUM_PREDICT = 256
LLM_KEEP_ALIVE = "5m"

# --- Parametry wyszukiwania -------------------------------------
@dataclass(frozen=True)
class HNSWParams:
    m: int = 16                 # liczba połączeń M
    ef_construction: int = 100  # budowa indeksu
    ef_search: int = 128        # przeszukiwanie ef
    k: int = 5                  # liczba wyników
    metric: str = "cosine"
    # kwantyzacja, filtry i wyszukiwanie hybrydowe: WYŁĄCZONE we wszystkich bazach

PARAMS = HNSWParams()

# --- Ingest wydajność -------------------------------------------------------
EMBED_BATCH_SIZE = 64
UPSERT_BATCH_SIZE = 512

# --- Chunking (ten sam dla każdej bazy) ------------------------------------
CHUNK_SIZE = 1200     # znaki
CHUNK_OVERLAP = 100

# --- Połączenia ------------------------------------------------------------
QDRANT_URL = "http://localhost:6333"
WEAVIATE_HTTP_PORT, WEAVIATE_GRPC_PORT = 8080, 50051
MILVUS_URI = "http://localhost:19530"
ES_URL = "http://localhost:9200"
COLLECTION = "rag_docs"   # Weaviate wymaga wielkiej litery -> patrz vectorstores.py
