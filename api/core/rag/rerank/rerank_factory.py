from core.rag.rerank.rerank_base import BaseRerankRunner
from core.rag.rerank.rerank_model import RerankModelRunner
from core.rag.rerank.rerank_type import RerankMode
from core.rag.rerank.weight_rerank import WeightRerankRunner


class RerankRunnerFactory:
    @staticmethod
    def create_rerank_runner(runner_type: str, *args, **kwargs) -> BaseRerankRunner:
        match runner_type:
            case RerankMode.RERANKING_MODEL.value:
                return RerankModelRunner(*args, **kwargs)   # 使用reranker模型
            case RerankMode.WEIGHTED_SCORE.value:
                return WeightRerankRunner(*args, **kwargs)  # 使用权重计算
            case _:
                raise ValueError(f"Unknown runner type: {runner_type}")
