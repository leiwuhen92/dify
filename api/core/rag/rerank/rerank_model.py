from typing import Optional

from core.model_manager import ModelInstance
from core.rag.models.document import Document
from core.rag.rerank.rerank_base import BaseRerankRunner


class RerankModelRunner(BaseRerankRunner):
    def __init__(self, rerank_model_instance: ModelInstance) -> None:
        self.rerank_model_instance = rerank_model_instance

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
        # 1、对document进行去重处理，若是dify文档，记录doc_id
        docs = []
        doc_ids = set()
        unique_documents = []
        for document in documents:
            if (
                document.provider == "dify"
                and document.metadata is not None
                and document.metadata["doc_id"] not in doc_ids
            ):
                doc_ids.add(document.metadata["doc_id"])
                docs.append(document.page_content)
                unique_documents.append(document)
            elif document.provider == "external":
                if document not in unique_documents:
                    docs.append(document.page_content)
                    unique_documents.append(document)

        documents = unique_documents

        # 2、调用rerank模型来对document进行重排处理，并输出重排序后的document列表
        rerank_result = self.rerank_model_instance.invoke_rerank(
            query=query, docs=docs, score_threshold=score_threshold, top_n=top_n, user=user
        )

        rerank_documents = []

        # 3、格式化重新排序后的document，生成Document对象，并把排序后的分数添加到document.metadata["score"]字段中
        for result in rerank_result.docs:
            # format document
            rerank_document = Document(
                page_content=result.text,
                metadata=documents[result.index].metadata,
                provider=documents[result.index].provider,
            )
            if rerank_document.metadata is not None:
                rerank_document.metadata["score"] = result.score
                rerank_documents.append(rerank_document)

        return rerank_documents
