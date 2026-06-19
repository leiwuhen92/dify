import math
from collections import Counter
from typing import Optional

import numpy as np

from core.model_manager import ModelManager
from core.model_runtime.entities.model_entities import ModelType
from core.rag.datasource.keyword.jieba.jieba_keyword_table_handler import JiebaKeywordTableHandler
from core.rag.embedding.cached_embedding import CacheEmbedding
from core.rag.models.document import Document
from core.rag.rerank.entity.weight import VectorSetting, Weights
from core.rag.rerank.rerank_base import BaseRerankRunner


class WeightRerankRunner(BaseRerankRunner):
    def __init__(self, tenant_id: str, weights: Weights) -> None:
        self.tenant_id = tenant_id
        self.weights = weights

    def run(
        self,
        query: str,
        documents: list[Document],
        score_threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        user: Optional[str] = None,
    ) -> list[Document]:
        """
        Run rerank model
        :param query: search query
        :param documents: documents for reranking
        :param score_threshold: score threshold
        :param top_n: top n
        :param user: unique user id if needed

        :return:
        """
        # 1、对document进行去重处理
        unique_documents = []
        doc_ids = set()
        for document in documents:
            if document.metadata is not None and document.metadata["doc_id"] not in doc_ids:
                doc_ids.add(document.metadata["doc_id"])
                unique_documents.append(document)

        documents = unique_documents

        # 2、使用BM25算法对query和每个document进行分词处理，并计算每个文档和query的相似度分数。这样就得到每个document和query的相似度分数
        query_scores = self._calculate_keyword_score(query, documents)
        # 3、对query和document进行向量化处理，并计算每个document和query向量的相似度分数。这样就得到每个document和query的向量的相似度分数
        query_vector_scores = self._calculate_cosine(self.tenant_id, query, documents, self.weights.vector_setting)

        # 4、针对每个document：将基于分词 和 向量的两个分数按照给定的权重综合计算，得到每个document和query相似度的最终评分
        rerank_documents = []
        for document, query_score, query_vector_score in zip(documents, query_scores, query_vector_scores):
            score = (
                self.weights.vector_setting.vector_weight * query_vector_score
                + self.weights.keyword_setting.keyword_weight * query_score
            )    # 计算综合得分
            if score_threshold and score < score_threshold:  # 如果设置了 score_threshold 且文档的综合评分为负数，则跳过
                continue
            if document.metadata is not None:
                document.metadata["score"] = score  # 添加文档的综合评分到元数据中
                rerank_documents.append(document)

        # 5、根据document的得分对其进行排序，并选择得分最高的top_n的个document返回
        rerank_documents.sort(key=lambda x: x.metadata["score"] if x.metadata else 0, reverse=True)
        return rerank_documents[:top_n] if top_n else rerank_documents

    def _calculate_keyword_score(self, query: str, documents: list[Document]) -> list[float]:
        """
        Calculate BM25 scores   文档的关键词提取和相似度分数计算
        :param query: search query
        :param documents: documents for reranking

        :return:
        """
        # 使用Jieba对问题（query）进行分词和关键词提取，同时去掉停用词
        keyword_table_handler = JiebaKeywordTableHandler()
        query_keywords = keyword_table_handler.extract_keywords(query, None)

        # 使用Jieba对document中的文本内容进行分词和关键词提取
        documents_keywords = []
        for document in documents:
            # get the document keywords
            document_keywords = keyword_table_handler.extract_keywords(document.page_content, None)
            if document.metadata is not None:
                document.metadata["keywords"] = document_keywords
                documents_keywords.append(document_keywords)

        # Counter query keywords(TF)  计算query关键词序列的TF
        query_keyword_counts = Counter(query_keywords)

        # total documents
        total_documents = len(documents)

        # calculate all documents' keywords IDF 计算每个文档分块关键词序列的IDF
        all_keywords = set()
        for document_keywords in documents_keywords:
            all_keywords.update(document_keywords)

        keyword_idf = {}
        for keyword in all_keywords:
            # calculate include query keywords' documents
            doc_count_containing_keyword = sum(1 for doc_keywords in documents_keywords if keyword in doc_keywords)
            # IDF
            keyword_idf[keyword] = math.log((1 + total_documents) / (1 + doc_count_containing_keyword)) + 1

        # 计算问题(query)关键词的TF-IDF的值
        query_tfidf = {}

        for keyword, count in query_keyword_counts.items():
            tf = count
            idf = keyword_idf.get(keyword, 0)
            query_tfidf[keyword] = tf * idf

        # calculate all documents' TF-IDF  计算所有文档的IF-IDF的值
        documents_tfidf = []
        for document_keywords in documents_keywords:
            document_keyword_counts = Counter(document_keywords)
            document_tfidf = {}
            for keyword, count in document_keyword_counts.items():
                tf = count
                idf = keyword_idf.get(keyword, 0)
                document_tfidf[keyword] = tf * idf
            documents_tfidf.append(document_tfidf)

        def cosine_similarity(vec1, vec2):
            # 1. 获取共同关键词
            intersection = set(vec1.keys()) & set(vec2.keys())
            # 2. 计算向量点积
            numerator = sum(vec1[x] * vec2[x] for x in intersection)

            # 3. 计算向量模长
            sum1 = sum(vec1[x] ** 2 for x in vec1)
            sum2 = sum(vec2[x] ** 2 for x in vec2)
            denominator = math.sqrt(sum1) * math.sqrt(sum2)

            # 4. 计算余弦相似度
            if not denominator:
                return 0.0
            else:
                return float(numerator) / denominator

        # 计算每个文档内容和问题内容tfidf值的余弦相似度
        similarities = []
        for document_tfidf in documents_tfidf:
            similarity = cosine_similarity(query_tfidf, document_tfidf)
            similarities.append(similarity)

        # for idx, similarity in enumerate(similarities):
        #     print(f"Document {idx + 1} similarity: {similarity}")

        return similarities

    def _calculate_cosine(
        self, tenant_id: str, query: str, documents: list[Document], vector_setting: VectorSetting
    ) -> list[float]:
        """
        Calculate Cosine scores
        :param query: search query
        :param documents: documents for reranking

        :return:
        """
        query_vector_scores = []

        model_manager = ModelManager()

        embedding_model = model_manager.get_model_instance(
            tenant_id=tenant_id,
            provider=vector_setting.embedding_provider_name,
            model_type=ModelType.TEXT_EMBEDDING,
            model=vector_setting.embedding_model_name,
        )
        cache_embedding = CacheEmbedding(embedding_model)
        query_vector = cache_embedding.embed_query(query)
        for document in documents:
            # calculate cosine similarity
            if document.metadata and "score" in document.metadata:
                query_vector_scores.append(document.metadata["score"])
            else:
                # transform to NumPy
                vec1 = np.array(query_vector)
                vec2 = np.array(document.vector)

                # calculate dot product
                dot_product = np.dot(vec1, vec2)

                # calculate norm
                norm_vec1 = np.linalg.norm(vec1)
                norm_vec2 = np.linalg.norm(vec2)

                # calculate cosine similarity
                cosine_sim = dot_product / (norm_vec1 * norm_vec2)
                query_vector_scores.append(cosine_sim)

        return query_vector_scores
