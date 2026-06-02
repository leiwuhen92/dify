import json
from collections import defaultdict
from typing import Any, Optional

from pydantic import BaseModel

from configs import dify_config
from core.rag.datasource.keyword.jieba.jieba_keyword_table_handler import JiebaKeywordTableHandler
from core.rag.datasource.keyword.keyword_base import BaseKeyword
from core.rag.models.document import Document
from extensions.ext_database import db
from extensions.ext_redis import redis_client
from extensions.ext_storage import storage
from models.dataset import Dataset, DatasetKeywordTable, DocumentSegment


class KeywordTableConfig(BaseModel):
    max_keywords_per_chunk: int = 10


class Jieba(BaseKeyword):
    def __init__(self, dataset: Dataset):
        # 初始化数据集和配置
        super().__init__(dataset)
        self._config = KeywordTableConfig()

    def create(self, texts: list[Document], **kwargs) -> BaseKeyword:
        lock_name = "keyword_indexing_lock_{}".format(self.dataset.id)
        # 获取redis分布式锁
        with redis_client.lock(lock_name, timeout=600):
            keyword_table_handler = JiebaKeywordTableHandler()
            # 检查关键词表是否存在，若不存在，创建关键词表dataset_keyword_tables
            keyword_table = self._get_dataset_keyword_table()
            # 遍历文件分块列表： Document对象列表
            for text in texts:
                # 关键词提取
                keywords = keyword_table_handler.extract_keywords(
                    text.page_content, self._config.max_keywords_per_chunk
                )
                if text.metadata is not None:
                    # 更新段落关键词 document_segment表
                    self._update_segment_keywords(self.dataset.id, text.metadata["doc_id"], list(keywords))
                    # 更新关键词表：把关键词保存到字典中
                    keyword_table = self._add_text_to_keyword_table(
                        keyword_table or {}, text.metadata["doc_id"], list(keywords)
                    )

            # 保存关键词表：保存关键词表到数据库或文件系统
            self._save_dataset_keyword_table(keyword_table)

            return self

    def add_texts(self, texts: list[Document], **kwargs):
        lock_name = "keyword_indexing_lock_{}".format(self.dataset.id)
        with redis_client.lock(lock_name, timeout=600):
            keyword_table_handler = JiebaKeywordTableHandler()

            keyword_table = self._get_dataset_keyword_table()
            keywords_list = kwargs.get("keywords_list")
            for i in range(len(texts)):
                text = texts[i]
                if keywords_list:
                    keywords = keywords_list[i]
                    if not keywords:
                        keywords = keyword_table_handler.extract_keywords(
                            text.page_content, self._config.max_keywords_per_chunk
                        )
                else:
                    keywords = keyword_table_handler.extract_keywords(
                        text.page_content, self._config.max_keywords_per_chunk
                    )
                if text.metadata is not None:
                    self._update_segment_keywords(self.dataset.id, text.metadata["doc_id"], list(keywords))
                    keyword_table = self._add_text_to_keyword_table(
                        keyword_table or {}, text.metadata["doc_id"], list(keywords)
                    )

            self._save_dataset_keyword_table(keyword_table)

    def text_exists(self, id: str) -> bool:
        keyword_table = self._get_dataset_keyword_table()
        if keyword_table is None:
            return False
        return id in set.union(*keyword_table.values())

    def delete_by_ids(self, ids: list[str]) -> None:
        lock_name = "keyword_indexing_lock_{}".format(self.dataset.id)
        with redis_client.lock(lock_name, timeout=600):
            keyword_table = self._get_dataset_keyword_table()
            if keyword_table is not None:
                keyword_table = self._delete_ids_from_keyword_table(keyword_table, ids)

            self._save_dataset_keyword_table(keyword_table)

    def search(self, query: str, **kwargs: Any) -> list[Document]:
        # 从dataset_keyword_tables表中获取对应数据集的数据分块记录字典
        keyword_table = self._get_dataset_keyword_table()

        k = kwargs.get("top_k", 4)
        document_ids_filter = kwargs.get("document_ids_filter")
        # 使用Jieba对用户输入的查询字符串进行关键词提取，然后从刚才查询出来的关键词字典中，查询出与查询字符串中国关键词匹配的文本索引id
        sorted_chunk_indices = self._retrieve_ids_by_query(keyword_table or {}, query, k)

        documents = []
        # 根据文本块索引id，从数据库中查询出对应的文本块内容
        for chunk_index in sorted_chunk_indices:
            segment_query = db.session.query(DocumentSegment).filter(
                DocumentSegment.dataset_id == self.dataset.id, DocumentSegment.index_node_id == chunk_index
            )
            if document_ids_filter:
                segment_query = segment_query.filter(DocumentSegment.document_id.in_(document_ids_filter))
            segment = segment_query.first()

            # 以Document对象结构来返回结果
            if segment:
                documents.append(
                    Document(
                        page_content=segment.content,
                        metadata={
                            "doc_id": chunk_index,
                            "doc_hash": segment.index_node_hash,
                            "document_id": segment.document_id,
                            "dataset_id": segment.dataset_id,
                        },
                    )
                )

        return documents

    def delete(self) -> None:
        lock_name = "keyword_indexing_lock_{}".format(self.dataset.id)
        with redis_client.lock(lock_name, timeout=600):
            dataset_keyword_table = self.dataset.dataset_keyword_table
            if dataset_keyword_table:
                db.session.delete(dataset_keyword_table)
                db.session.commit()
                if dataset_keyword_table.data_source_type != "database":
                    file_key = "keyword_files/" + self.dataset.tenant_id + "/" + self.dataset.id + ".txt"
                    storage.delete(file_key)

    def _save_dataset_keyword_table(self, keyword_table):
        """
        将一个关键词表（dataset_keyword_tables）保存到数据库或文件中，具体取决于数据源类型
        :param keyword_table:
        :return:
        """
        # 创建数据字典，保存元数据信息
        keyword_table_dict = {
            "__type__": "keyword_table",
            "__data__": {"index_id": self.dataset.id, "summary": None, "table": keyword_table},
        }
        # 记录数据记得数据来源类型
        dataset_keyword_table = self.dataset.dataset_keyword_table
        keyword_data_source_type = dataset_keyword_table.data_source_type
        if keyword_data_source_type == "database":  # 数据源是数据库，则将字典编码为JSON字符串，并更新数据库中的keyword_table字段
            dataset_keyword_table.keyword_table = json.dumps(keyword_table_dict, cls=SetEncoder)
            db.session.commit()
        else:  # 来源是文件，则构建一个文件键（路径），检查文件是否存在，如存在则删除。将字典编码为JSON并保存到指定的文件路径
            file_key = "keyword_files/" + self.dataset.tenant_id + "/" + self.dataset.id + ".txt"
            if storage.exists(file_key):
                storage.delete(file_key)
            storage.save(file_key, json.dumps(keyword_table_dict, cls=SetEncoder).encode("utf-8"))

    def _get_dataset_keyword_table(self) -> Optional[dict]:
        """
        检查关键词表是否存在，若不存在，创建关键词表dataset_keyword_tables
        :return:
        """
        dataset_keyword_table = self.dataset.dataset_keyword_table
        if dataset_keyword_table:
            keyword_table_dict = dataset_keyword_table.keyword_table_dict
            if keyword_table_dict:
                return dict(keyword_table_dict["__data__"]["table"])
        else:
            keyword_data_source_type = dify_config.KEYWORD_DATA_SOURCE_TYPE
            dataset_keyword_table = DatasetKeywordTable(
                dataset_id=self.dataset.id,
                keyword_table="",
                data_source_type=keyword_data_source_type,
            )
            if keyword_data_source_type == "database":
                dataset_keyword_table.keyword_table = json.dumps(
                    {
                        "__type__": "keyword_table",
                        "__data__": {"index_id": self.dataset.id, "summary": None, "table": {}},
                    },
                    cls=SetEncoder,
                )
            db.session.add(dataset_keyword_table)
            db.session.commit()

        return {}

    def _add_text_to_keyword_table(self, keyword_table: dict, id: str, keywords: list[str]) -> dict:
        """
        更新关键词表：把关键词保存到字典中
        :param keyword_table:
        :param id:
        :param keywords:
        :return:
        """
        for keyword in keywords:
            if keyword not in keyword_table:
                keyword_table[keyword] = set()
            keyword_table[keyword].add(id)
        return keyword_table

    def _delete_ids_from_keyword_table(self, keyword_table: dict, ids: list[str]) -> dict:
        # get set of ids that correspond to node
        node_idxs_to_delete = set(ids)

        # delete node_idxs from keyword to node idxs mapping
        keywords_to_delete = set()
        for keyword, node_idxs in keyword_table.items():
            if node_idxs_to_delete.intersection(node_idxs):
                keyword_table[keyword] = node_idxs.difference(node_idxs_to_delete)
                if not keyword_table[keyword]:
                    keywords_to_delete.add(keyword)

        for keyword in keywords_to_delete:
            del keyword_table[keyword]

        return keyword_table

    def _retrieve_ids_by_query(self, keyword_table: dict, query: str, k: int = 4):
        keyword_table_handler = JiebaKeywordTableHandler()
        keywords = keyword_table_handler.extract_keywords(query)

        # go through text chunks in order of most matching keywords
        chunk_indices_count: dict[str, int] = defaultdict(int)
        keywords_list = [keyword for keyword in keywords if keyword in set(keyword_table.keys())]
        for keyword in keywords_list:
            for node_id in keyword_table[keyword]:
                chunk_indices_count[node_id] += 1

        sorted_chunk_indices = sorted(
            chunk_indices_count.keys(),
            key=lambda x: chunk_indices_count[x],
            reverse=True,
        )

        return sorted_chunk_indices[:k]

    def _update_segment_keywords(self, dataset_id: str, node_id: str, keywords: list[str]):
        """
        更新段落关键词 document_segment表
        :param dataset_id:
        :param node_id:
        :param keywords:
        :return:
        """
        document_segment = (
            db.session.query(DocumentSegment)
            .filter(DocumentSegment.dataset_id == dataset_id, DocumentSegment.index_node_id == node_id)
            .first()
        )
        if document_segment:
            document_segment.keywords = keywords
            db.session.add(document_segment)
            db.session.commit()

    def create_segment_keywords(self, node_id: str, keywords: list[str]):
        keyword_table = self._get_dataset_keyword_table()
        self._update_segment_keywords(self.dataset.id, node_id, keywords)
        keyword_table = self._add_text_to_keyword_table(keyword_table or {}, node_id, keywords)
        self._save_dataset_keyword_table(keyword_table)

    def multi_create_segment_keywords(self, pre_segment_data_list: list):
        keyword_table_handler = JiebaKeywordTableHandler()
        keyword_table = self._get_dataset_keyword_table()
        for pre_segment_data in pre_segment_data_list:
            segment = pre_segment_data["segment"]
            if pre_segment_data["keywords"]:
                segment.keywords = pre_segment_data["keywords"]
                keyword_table = self._add_text_to_keyword_table(
                    keyword_table or {}, segment.index_node_id, pre_segment_data["keywords"]
                )
            else:
                keywords = keyword_table_handler.extract_keywords(segment.content, self._config.max_keywords_per_chunk)
                segment.keywords = list(keywords)
                keyword_table = self._add_text_to_keyword_table(
                    keyword_table or {}, segment.index_node_id, list(keywords)
                )
        self._save_dataset_keyword_table(keyword_table)

    def update_segment_keywords_index(self, node_id: str, keywords: list[str]):
        keyword_table = self._get_dataset_keyword_table()
        keyword_table = self._add_text_to_keyword_table(keyword_table or {}, node_id, keywords)
        self._save_dataset_keyword_table(keyword_table)


class SetEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理set类型"""
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)
