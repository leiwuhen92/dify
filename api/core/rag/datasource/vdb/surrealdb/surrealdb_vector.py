from pydantic import BaseModel, model_validator
from typing import Any, Optional
import json
import time
import random
import logging
from surrealdb import Surreal

from configs import dify_config
from core.rag.datasource.vdb.field import Field
from core.rag.datasource.vdb.vector_base import BaseVector
from core.rag.datasource.vdb.vector_factory import AbstractVectorFactory
from core.rag.datasource.vdb.vector_type import VectorType
from core.rag.embedding.embedding_base import Embeddings
from core.rag.models.document import Document
from extensions.ext_redis import redis_client
from models.dataset import Dataset

logger = logging.getLogger(__name__)


class SurrealDBConfig(BaseModel):
    url: str  # ws://127.0.0.1:10943/rpc
    username: str
    password: str
    namespace: str
    database: str
    batch_size: int = 4
    enable_hybrid_search: bool = False  # Flag to enable hybrid search
    hybrid_search_weight: Optional[dict] = {"vector": 0.5, "text": 0.5}  # 添加权重配置

    @model_validator(mode="before")
    @classmethod
    def validate_config(cls, values: dict) -> dict:
        """
        Validate the configuration values.
        Raises ValueError if required fields are missing.
        """
        required_fields = ["url", "username", "password"]
        for field in required_fields:
            if not values.get(field):
                raise ValueError(f"config SURREALDB_{field.upper()} is required")

        return values


class SurrealDBVector(BaseVector):
    """
    SurrealDB vector storage implementation.
    """

    def __init__(self, collection_name: str, config: SurrealDBConfig):
        super().__init__(collection_name)
        self._config = config
        self._client = self._init_client(config)

    def _init_client(self, config: SurrealDBConfig) -> Surreal:
        """
        Initialize and return a SurrealDB client.
        """
        try:
            client = Surreal(config.url)
            # client.timeout = config.timeout
            client.signin({
                "username": config.username,
                "password": config.password
            })
            client.use(namespace=config.namespace, database=config.database)
            return client
        except Exception as e:
            logger.error(f"Failed to initialize SurrealDB client: {e}")
            raise ConnectionError("Failed to connect to SurrealDB")

    def get_type(self) -> str:
        """Get the type of vector storage (SurrealDB)."""
        return VectorType.SURREALDB

    def create(self, texts: list[Document], embeddings: list[list[float]], **kwargs):
        """
        Create a table and add texts with embeddings.
        """
        metadatas = [d.metadata or {} for d in texts]
        self.create_collection(embeddings, metadatas)  # Create table and indexes
        self.add_texts(texts, embeddings)  # Add texts

    def create_collection(
        self,
        embeddings: list,
        metadatas: Optional[list[dict]] = None,
        index_params: Optional[dict] = None,
    ):
        """
        Create a new collection in SurrealDB with the specified schema and index parameters.
        """
        lock_name = f"vector_indexing_lock_{self._collection_name}"
        with redis_client.lock(lock_name, timeout=20):
            cache_key = f"vector_indexing_{self._collection_name}"
            if redis_client.get(cache_key):
                return

            dim = len(embeddings[0])

            # SurrealDB table + vector index(定义向量索引)
            statements = f"""
                DEFINE TABLE {self._collection_name} SCHEMAFULL;
                DEFINE FIELD {Field.CONTENT_KEY.value}
                    ON {self._collection_name}
                    TYPE string;
                DEFINE FIELD {Field.VECTOR.value}
                    ON {self._collection_name}
                    TYPE array<float>
                    ASSERT array::len($value) = {dim};
                DEFINE FIELD {Field.METADATA_KEY.value}
                    ON {self._collection_name}
                    FLEXIBLE TYPE object;
                DEFINE INDEX idx_embedding
                    ON {self._collection_name}
                    FIELDS {Field.VECTOR.value}
                    HNSW
                    DIMENSION {dim}
                    DIST COSINE
                    TYPE F32;
            """
            # Create custom function to support text to sparse vector by BM25
            if self._config.enable_hybrid_search:  # 启用混合搜索时, 创建全文索引
                statements += f"""
                DEFINE ANALYZER my_analyzer TOKENIZERS class, blank FILTERS lowercase, ascii;

                DEFINE INDEX idx_text
                    ON TABLE {self._collection_name}
                    FIELDS {Field.CONTENT_KEY.value}
                    SEARCH ANALYZER my_analyzer BM25;
                """
            logger.info(f"Created collection's statements: {statements}")
            self._client.query(statements)
            logger.info(f"Created collection {self._collection_name} with dim={dim}")

            redis_client.set(cache_key, 1, ex=3600)

    def add_texts(self, documents: list[Document], embeddings: list[list[float]], **kwargs):
        """
        Add texts and their embeddings to the collection.
        """
        logger.info(f"in add_texts, len(documents): {len(documents)}")
        records = []
        for i, doc in enumerate(documents):
            # SurrealDB不支持float后缀f（0.02228437860139816f），需要强制转float（0.02228437860139816）
            vector = [float(x) for x in embeddings[i]]

            records.append({
                Field.CONTENT_KEY.value: doc.page_content,
                Field.VECTOR.value: vector,
                Field.METADATA_KEY.value: doc.metadata,
            })

        total = len(records)
        ids: list[str] = []

        MAX_RETRIES = 5  # 最大重试次数
        batch_size = min(self._config.batch_size, 4)  # 在dify的并发模式下，batch>4 = transaction冲突指数放上升

        lock_key = f"dify:vdb:surreal:{self._collection_name}"
        for i in range(0, total, batch_size):
            batch = records[i:i + batch_size]
            for attempt in range(MAX_RETRIES):
                lock = redis_client.lock(
                    lock_key,
                    timeout=20,  # 必须 < SurrealDB写入时间
                    blocking=False,  # 不阻塞 worker
                )

                try:
                    if not lock.acquire():
                        # 没抢到锁，轻微退避
                        time.sleep(random.uniform(0.002, 0.008))
                        continue

                    time.sleep(random.uniform(0.002, 0.008))  # 抖动错峰

                    sql = f"INSERT INTO {self._collection_name} $records;"
                    result = self._client.query(sql, {"records": batch})
                    # logger.info(f"Batch insert result: {result}")

                    # 判断返回类型，确保是 dict 才取 id
                    if isinstance(result[0], dict):
                        ids.extend([str(r["id"]) for r in result])
                    else:
                        logger.warning(f"Attempt {attempt + 1}: transaction failed, SurrealDB response: {result}")
                        raise RuntimeError("Transaction conflict, retrying...")

                    # 成功就跳出重试循环
                    break

                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(
                            f"Failed to insert batch starting at entity: {i}/{total}, retrying attempt {attempt + 1}...")
                        time.sleep(0.1 * (attempt + 1))  # 小延迟再重试
                        continue
                    else:
                        logger.exception("Failed to insert batch after max retries")
                        raise e
                finally:
                    if lock.locked():
                        try:
                            lock.release()
                        except Exception:
                            pass
        return ids

    def search_by_vector(self, query_vector: list[float], **kwargs: Any) -> list[Document]:
        """
        Search for documents by vector similarity.
        """
        logger.info(f"in search_by_vector".center(44, "*"))
        top_k = kwargs.get("top_k", 4)
        score_threshold = float(kwargs.get("score_threshold") or 0.0)
        document_ids_filter = kwargs.get("document_ids_filter")

        where_clause = ""
        if document_ids_filter:
            ids = ", ".join(f'"{id}"' for id in document_ids_filter)
            where_clause = f'WHERE metadata.document_id IN [{ids}]'

        query = f"""
            SELECT *, vector::similarity::cosine({Field.VECTOR.value}, {query_vector}) AS score
            FROM {self._collection_name}
            {where_clause}
            ORDER BY score DESC
            LIMIT {top_k};
        """
        logger.debug(f"search_by_vector's query: {query}")

        result = self._client.query(query)
        logger.info(f"search_by_vector {len(result)} result")

        docs: list[Document] = []
        for r in result:
            if r["score"] >= score_threshold:
                metadata = r.get(Field.METADATA_KEY.value, {})
                metadata["score"] = r["score"]
                docs.append(
                    Document(
                        page_content=r.get(Field.CONTENT_KEY.value, ""),
                        metadata=metadata,
                    )
                )

        logger.info(f"search_by_vector, get {len(docs)} docs")
        return docs

    def search_by_full_text(self, query: str, **kwargs: Any) -> list[Document]:
        """
        Search for documents by full-text search (if hybrid search is enabled).
        """
        logger.info(f"in search_by_full_text".center(44, "*"))
        if not self._config.enable_hybrid_search:
            logger.warning("Hybrid search is disabled for SurrealDB")
            return []

        top_k = kwargs.get("top_k", 4)
        score_threshold = float(kwargs.get("score_threshold") or 0.0)
        document_ids_filter = kwargs.get("document_ids_filter")

        where_clauses = [f"{Field.CONTENT_KEY.value} @0@ '{query}'"]

        if document_ids_filter:
            where_clauses.append(f"metadata.document_id IN {document_ids_filter}")

        where_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT *, search::score(0) AS score
            FROM {self._collection_name}
            WHERE {where_sql}
            ORDER BY score DESC
            LIMIT {top_k};
        """
        logger.debug(f"search_by_full_text's sql: {sql}")

        result = self._client.query(sql)
        logger.info(f"search_by_full_text {len(result)} result")

        # 解决score为负值导致跟阈值过滤后无数据问题
        min_score = 0
        if result:
            min_score = min(r['score'] for r in result)

        docs: list[Document] = []
        for r in result:
            adjusted_score = r['score'] - min_score
            if adjusted_score >= score_threshold:
                metadata = r.get(Field.METADATA_KEY.value, {})
                metadata["score"] = adjusted_score  # r["score"]

                docs.append(
                    Document(
                        page_content=r.get(Field.CONTENT_KEY.value, ""),
                        metadata=metadata,
                    )
                )

        logger.info(f"search_by_full_text, get {len(docs)} docs")
        return docs

    def delete_by_ids(self, ids: list[str]) -> None:
        """
        Delete documents by their IDs.
        """
        self._client.query(f"DELETE FROM {self._collection_name} WHERE metadata.doc_id IN {ids};")

    def delete(self) -> None:
        """
        Delete the entire collection.
        """
        self._client.query(f"REMOVE TABLE {self._collection_name}")

        # delete collection redis cache
        cache_key = f"vector_indexing_{self._collection_name}"
        redis_client.delete(cache_key)

    def text_exists(self, id: str) -> bool:
        """
        Check if a text with the given ID exists in the collection.
        """
        result = self._client.query(
            f"SELECT id FROM {self._collection_name} WHERE metadata.doc_id = '{id}' LIMIT 1;"
        )
        return len(result) > 0

    def get_ids_by_metadata_field(self, key: str, value: str) -> list[str]:
        """
        Get document IDs by metadata field key and value.
        """
        try:
            query = f'SELECT id FROM {self._collection_name} WHERE metadata.{key} = "{value}";'
            result = self._client.query(query)
            return [item["id"] for item in result]
        except Exception as e:
            logger.error(f"Failed to get IDs by metadata field: {e}")
            return []

    def delete_by_metadata_field(self, key: str, value: str) -> None:
        """Delete documents by metadata field key-value pair."""
        try:
            # Build DELETE query for metadata field
            delete_query = f"""
                DELETE FROM {self._collection_name}
                WHERE {Field.METADATA_KEY.value}.{key} = "{value}"
            """
            result = self._client.query(delete_query)

            # Get count of deleted documents
            deleted_count = len(result)
            logger.info(f"Deleted {deleted_count} documents from {self._collection_name} where {key} = {value}")

        except Exception as e:
            logger.exception(f"Failed to delete documents by metadata field {key}={value}")


class SurrealDBVectorFactory(AbstractVectorFactory):
    """
    Factory class for creating SurrealDBVector instances.
    """

    def init_vector(self, dataset: Dataset, attributes: list, embeddings: Embeddings) -> SurrealDBVector:
        if dataset.index_struct_dict:
            class_prefix = dataset.index_struct_dict["vector_store"]["class_prefix"]
            collection_name = class_prefix
        else:
            dataset_id = dataset.id
            collection_name = Dataset.gen_collection_name_by_id(dataset_id)
            dataset.index_struct = json.dumps(self.gen_index_struct_dict(VectorType.SURREALDB, collection_name))
        logger.info(f"1111111111111111111111111111111111 dify_config.SURREALDB_URL: {dify_config.SURREALDB_URL}, dify_config.SURREALDB_USERNAME: {dify_config.SURREALDB_USERNAME}, dify_config.SURREALDB_PASSWORD: {dify_config.SURREALDB_PASSWORD}")
        return SurrealDBVector(
            collection_name=collection_name,
            config=SurrealDBConfig(
                url=dify_config.SURREALDB_URL,
                username=dify_config.SURREALDB_USERNAME or "",
                password=dify_config.SURREALDB_PASSWORD or "",
                namespace=dify_config.SURREALDB_NAMESPACE or "",
                database=dify_config.SURREALDB_DATABASE or "",
                batch_size=dify_config.SURREALDB_BATCH_SIZE,
                enable_hybrid_search=dify_config.SURREALDB_ENABLE_HYBRID_SEARCH or False,
            ),
        )
