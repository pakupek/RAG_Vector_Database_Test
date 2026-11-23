"""Cztery backendy za jednym interfejsem. Parametry HNSW ustawiane JAWNIE (slajd 13)."""
from __future__ import annotations
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator
import config as C

P = C.PARAMS


@dataclass
class StoredPoint:
    id: int
    vector: list[float]
    text: str
    source: str


@dataclass
class Hit:
    id: int
    text: str
    source: str
    score: float   # podobieństwo cosinusowe (większe = lepsze)


class VectorStore(ABC):
    name: str

    @abstractmethod
    def collection_exists(self) -> bool: ...
    @abstractmethod
    def recreate(self, dim: int) -> None: ...
    @abstractmethod
    def upsert(self, ids: list[int], vectors: list[list[float]], texts: list[str], sources: list[str]) -> None: ...
    @abstractmethod
    def delete(self, ids: list[int]) -> None: ...
    @abstractmethod
    def search(self, vector: list[float]) -> list[Hit]: ...
    @abstractmethod
    def iter_points(self, batch_size: int) -> Iterator[StoredPoint]: ...
    def finish_ingest(self) -> None: pass
    def count(self) -> int: return -1
    def close(self) -> None: pass


# ------------------------------------------------------------------ Qdrant
class QdrantStore(VectorStore):
    name = "qdrant"

    def __init__(self):
        from qdrant_client import QdrantClient
        self.c = QdrantClient(url=C.QDRANT_URL)

    def collection_exists(self): return self.c.collection_exists(C.COLLECTION)

    def recreate(self, dim):
        from qdrant_client import models as m
        if self.c.collection_exists(C.COLLECTION):
            self.c.delete_collection(C.COLLECTION)
        self.c.create_collection(
            C.COLLECTION,
            vectors_config=m.VectorParams(size=dim, distance=m.Distance.COSINE),
            hnsw_config=m.HnswConfigDiff(m=P.m, ef_construct=P.ef_construction),
            quantization_config=None,   # bez kwantyzacji
        )

    def upsert(self, ids, vectors, texts, sources):
        from qdrant_client import models as m
        self.c.upsert(C.COLLECTION, wait=True, points=[
            m.PointStruct(id=i, vector=v, payload={"text": t, "source": s})
            for i, v, t, s in zip(ids, vectors, texts, sources)])

    def delete(self, ids):
        if ids:
            from qdrant_client import models as m
            self.c.delete(C.COLLECTION, points_selector=m.PointIdsList(points=ids), wait=True)

    def search(self, vector):
        from qdrant_client import models as m
        r = self.c.query_points(
            C.COLLECTION, query=vector, limit=P.k,                   # limit
            search_params=m.SearchParams(hnsw_ef=P.ef_search),       # hnsw_ef
            with_payload=True)
        return [Hit(p.id, p.payload["text"], p.payload["source"], p.score) for p in r.points]

    def count(self): return self.c.count(C.COLLECTION).count

    def iter_points(self, batch_size):
        offset = None
        while True:
            points, offset = self.c.scroll(
                collection_name=C.COLLECTION,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for point in points:
                vector = point.vector
                if isinstance(vector, dict):
                    if len(vector) != 1:
                        raise ValueError("Eksport obsługuje tylko pojedynczy wektor na punkt Qdrant")
                    vector = next(iter(vector.values()))
                yield StoredPoint(
                    int(point.id), vector,
                    point.payload["text"], point.payload["source"],
                )
            if offset is None:
                break

    def close(self): self.c.close()


# ---------------------------------------------------------------- Weaviate
class WeaviateStore(VectorStore):
    name = "weaviate"
    COL = C.COLLECTION.replace("_", "").capitalize()   # nazwy klas: wielka litera, bez '_'

    def __init__(self):
        import weaviate
        self.c = weaviate.connect_to_local(
            port=C.WEAVIATE_HTTP_PORT, grpc_port=C.WEAVIATE_GRPC_PORT)

    def collection_exists(self): return self.c.collections.exists(self.COL)

    def recreate(self, dim):
        from weaviate.classes.config import Configure, Property, DataType, VectorDistances
        if self.c.collections.exists(self.COL):
            self.c.collections.delete(self.COL)
        self.c.collections.create(
            self.COL,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
                max_connections=P.m,                # maxConnections
                ef_construction=P.ef_construction,  # efConstruction
                ef=P.ef_search,                     # ef (parametr kolekcji, nie zapytania)
            ),                                      # bez quantizera = bez kwantyzacji
            properties=[Property(name="text", data_type=DataType.TEXT),
                        Property(name="source", data_type=DataType.TEXT),
                        Property(name="doc_id", data_type=DataType.INT)],
        )

    @staticmethod
    def _uid(i): return uuid.uuid5(uuid.NAMESPACE_OID, str(i))

    def upsert(self, ids, vectors, texts, sources):
        from weaviate.classes.data import DataObject
        col = self.c.collections.get(self.COL)
        res = col.data.insert_many([
            DataObject(uuid=self._uid(i), vector=v,
                       properties={"text": t, "source": s, "doc_id": i})
            for i, v, t, s in zip(ids, vectors, texts, sources)])
        if res.has_errors:
            raise RuntimeError(res.errors)

    def delete(self, ids):
        if ids:
            from weaviate.classes.query import Filter
            col = self.c.collections.get(self.COL)
            uuids = [self._uid(i) for i in ids]
            result = col.data.delete_many(where=Filter.by_id().contains_any(uuids))
            if result.failed:
                raise RuntimeError(f"Weaviate nie usunął {result.failed} obiektów")

    def search(self, vector):
        from weaviate.classes.query import MetadataQuery
        col = self.c.collections.get(self.COL)
        r = col.query.near_vector(near_vector=vector, limit=P.k,     # limit
                                  return_metadata=MetadataQuery(distance=True))
        return [Hit(o.properties["doc_id"], o.properties["text"], o.properties["source"],
                    1.0 - o.metadata.distance)                        # cosine distance -> similarity
                for o in r.objects]

    def count(self):
        return self.c.collections.get(self.COL).aggregate.over_all(total_count=True).total_count

    def iter_points(self, batch_size):
        col = self.c.collections.get(self.COL)
        for item in col.iterator(include_vector=True):
            vector = item.vector
            if isinstance(vector, dict):
                if len(vector) != 1:
                    raise ValueError("Eksport obsługuje tylko pojedynczy wektor na obiekt Weaviate")
                vector = next(iter(vector.values()))
            yield StoredPoint(
                int(item.properties["doc_id"]), vector,
                item.properties["text"], item.properties["source"],
            )

    def close(self): self.c.close()


# ------------------------------------------------------------------ Milvus
class MilvusStore(VectorStore):
    name = "milvus"

    def __init__(self):
        from pymilvus import MilvusClient
        self.c = MilvusClient(uri=C.MILVUS_URI)

    def collection_exists(self): return self.c.has_collection(C.COLLECTION)

    def recreate(self, dim):
        from pymilvus import MilvusClient, DataType
        if self.c.has_collection(C.COLLECTION):
            self.c.drop_collection(C.COLLECTION)
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("source", DataType.VARCHAR, max_length=1024)
        idx = self.c.prepare_index_params()
        idx.add_index(field_name="vector", index_type="HNSW", metric_type="COSINE",
                      params={"M": P.m, "efConstruction": P.ef_construction})
        self.c.create_collection(C.COLLECTION, schema=schema, index_params=idx)  # tworzy indeks i ładuje kolekcję

    def upsert(self, ids, vectors, texts, sources):
        self.c.upsert(C.COLLECTION, [
            {"id": i, "vector": v, "text": t, "source": s}
            for i, v, t, s in zip(ids, vectors, texts, sources)])

    def delete(self, ids):
        if ids:
            self.c.delete(C.COLLECTION, ids=ids)

    def finish_ingest(self): self.c.flush(C.COLLECTION)

    def search(self, vector):
        r = self.c.search(
            C.COLLECTION, data=[vector], limit=P.k,                                   # limit
            search_params={"metric_type": "COSINE", "params": {"ef": P.ef_search}},   # ef
            output_fields=["text", "source"])
        return [Hit(h["id"], h["entity"]["text"], h["entity"]["source"], h["distance"]) for h in r[0]]

    def count(self): return self.c.get_collection_stats(C.COLLECTION)["row_count"]

    def iter_points(self, batch_size):
        iterator = self.c.query_iterator(
            collection_name=C.COLLECTION,
            batch_size=batch_size,
            expr="id >= 0",
            output_fields=["id", "vector", "text", "source"],
        )
        try:
            while True:
                rows = iterator.next()
                if not rows:
                    break
                for row in rows:
                    yield StoredPoint(
                        int(row["id"]), row["vector"], row["text"], row["source"],
                    )
        finally:
            iterator.close()

    def close(self): self.c.close()


# ----------------------------------------------------------- Elasticsearch
class ElasticsearchStore(VectorStore):
    name = "elasticsearch"
    INDEX = C.COLLECTION

    def __init__(self):
        from elasticsearch import Elasticsearch
        self.c = Elasticsearch(C.ES_URL)

    def collection_exists(self): return self.c.indices.exists(index=self.INDEX)

    def recreate(self, dim):
        if self.c.indices.exists(index=self.INDEX):
            self.c.indices.delete(index=self.INDEX)
        self.c.indices.create(index=self.INDEX, mappings={"properties": {
            "vector": {
                "type": "dense_vector", "dims": dim, "index": True,
                "similarity": "cosine",
                # Typ jawnie "hnsw" (bez kwantyzacji int8) - slajd 13: "typ indeksu ustaw jawnie"
                "index_options": {"type": "hnsw", "m": P.m, "ef_construction": P.ef_construction},
            },
            "text": {"type": "text"},
            "source": {"type": "keyword"},
        }}, settings={"number_of_shards": 1, "number_of_replicas": 0})

    def upsert(self, ids, vectors, texts, sources):
        from elasticsearch import helpers
        helpers.bulk(self.c, ({"_index": self.INDEX, "_id": str(i),
                               "_source": {"vector": v, "text": t, "source": s}}
                              for i, v, t, s in zip(ids, vectors, texts, sources)), refresh=False)

    def delete(self, ids):
        if ids:
            from elasticsearch import helpers
            helpers.bulk(self.c, ({"_op_type": "delete", "_index": self.INDEX, "_id": str(i)}
                                  for i in ids), ignore_status=(404,), refresh=False)

    def finish_ingest(self): self.c.indices.refresh(index=self.INDEX)

    def search(self, vector):
        r = self.c.search(index=self.INDEX, size=P.k, source=["text", "source"], knn={
            "field": "vector", "query_vector": vector,
            "k": P.k,                           # k
            "num_candidates": P.ef_search})     # num_candidates (odpowiednik ef)
        # ES dla cosine zwraca _score = (1 + cos) / 2 -> przeliczenie na cosinus
        return [Hit(int(h["_id"]), h["_source"]["text"], h["_source"]["source"], 2 * h["_score"] - 1)
                for h in r["hits"]["hits"]]

    def count(self): return self.c.count(index=self.INDEX)["count"]

    def iter_points(self, batch_size):
        from elasticsearch import helpers
        for hit in helpers.scan(
            self.c,
            index=self.INDEX,
            query={"query": {"match_all": {}}},
            _source=["vector", "text", "source"],
            size=batch_size,
        ):
            source = hit["_source"]
            yield StoredPoint(
                int(hit["_id"]), source["vector"], source["text"], source["source"],
            )

    def close(self): self.c.close()


STORES = {s.name: s for s in (QdrantStore, WeaviateStore, MilvusStore, ElasticsearchStore)}

def get_store(name: str) -> VectorStore:
    return STORES[name]()
