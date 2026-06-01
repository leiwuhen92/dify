import datetime
import logging
import time

import click
from celery import shared_task  # type: ignore

from configs import dify_config
from core.indexing_runner import DocumentIsPausedError, IndexingRunner
from extensions.ext_database import db
from models.dataset import Dataset, Document
from services.feature_service import FeatureService


@shared_task(queue="dataset")
def document_indexing_task(dataset_id: str, document_ids: list):
    """
    Async process document
    :param dataset_id:
    :param document_ids:

    Usage: document_indexing_task.delay(dataset_id, document_ids)
    """
    documents = []
    start_at = time.perf_counter()

    # 1、查询并验证数据集是否存在，如果数据集不存在，记录日志并返回
    dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        logging.info(click.style("Dataset is not found: {}".format(dataset_id), fg="yellow"))
        return

    # 2、check document limit  检查租户的功能限制：验证批量上传限制和向量空间限制，若超出限制，将所有文档编辑为错误状态
    features = FeatureService.get_features(dataset.tenant_id)
    try:
        if features.billing.enabled:
            vector_space = features.vector_space
            count = len(document_ids)
            batch_upload_limit = int(dify_config.BATCH_UPLOAD_LIMIT)
            if count > batch_upload_limit:                      # 检查批量上传限制
                raise ValueError(f"You have reached the batch upload limit of {batch_upload_limit}.")
            if 0 < vector_space.limit <= vector_space.size:     # 检查向量空间限制
                raise ValueError(
                    "Your total number of documents plus the number of uploads have over the limit of "
                    "your subscription."
                )
    except Exception as e:
        # 处理错误：更新所有相关文档的状态为error
        for document_id in document_ids:
            document = (
                db.session.query(Document).filter(Document.id == document_id, Document.dataset_id == dataset_id).first()
            )
            if document:
                document.indexing_status = "error"
                document.error = str(e)
                document.stopped_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
                db.session.add(document)
        db.session.commit()
        return

    # 根据任务参数中的document_ids从数据库中拿到所有的documents
    for document_id in document_ids:
        logging.info(click.style("Start process document: {}".format(document_id), fg="green"))

        document = (
            db.session.query(Document).filter(Document.id == document_id, Document.dataset_id == dataset_id).first()
        )

        if document:
            document.indexing_status = "parsing"  # 初始化状态为parsing
            document.processing_started_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
            documents.append(document)
            db.session.add(document)
    db.session.commit()

    try:
        # 调用 IndexingRunner 执行索引任务
        indexing_runner = IndexingRunner()
        indexing_runner.run(documents)    # IndexingRunner.run()包含了RAG索引的实现细节
        end_at = time.perf_counter()
        logging.info(click.style("Processed dataset: {} latency: {}".format(dataset_id, end_at - start_at), fg="green"))
    except DocumentIsPausedError as ex:
        logging.info(click.style(str(ex), fg="yellow"))
    except Exception:
        pass
