import re
from typing import Optional, cast


class JiebaKeywordTableHandler:
    def __init__(self):
        import jieba.analyse  # type: ignore

        from core.rag.datasource.keyword.jieba.stopwords import STOPWORDS

        jieba.analyse.default_tfidf.stop_words = STOPWORDS  # type: ignore

    def extract_keywords(self, text: str, max_keywords_per_chunk: Optional[int] = 10) -> set[str]:
        """Extract keywords with JIEBA tfidf.
        关键词提取主要完成：从文本中提取关键词，支持停用词过滤和子词提取
        """
        import jieba.analyse  # type: ignore

        # 1、使用TFIDF算法提取关键词
        keywords = jieba.analyse.extract_tags(
            sentence=text,
            topK=max_keywords_per_chunk,
        )
        # jieba.analyse.extract_tags returns list[Any] when withFlag is False by default.
        keywords = cast(list[str], keywords)

        # 2、扩展子词并过滤停用词
        return set(self._expand_tokens_with_subtokens(set(keywords)))

    def _expand_tokens_with_subtokens(self, tokens: set[str]) -> set[str]:
        """Get subtokens from a list of tokens., filtering for stopwords.
        获取tokens的子词，并过滤停用词
        """
        from core.rag.datasource.keyword.jieba.stopwords import STOPWORDS

        results = set()
        for token in tokens:
            # 1、添加原始token
            results.add(token)
            # 2、使用正则提取子词
            sub_tokens = re.findall(r"\w+", token)
            # 3、如果存在多个子词，过滤停用词并添加到结果集
            if len(sub_tokens) > 1:
                results.update({w for w in sub_tokens if w not in list(STOPWORDS)})

        return results
