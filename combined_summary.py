from summarizeNews import summarize_article_text
from cluster import summarize_article_debug

def combined_summarize(article_text, stopwords_path="stopwords.txt", title=None):

    cloud_summary = summarize_article_debug(article_text)
    result = summarize_article_text(article_text, stopwords_path=stopwords_path, title=title)
    keywords = result.get("keywords", [])

    if any("요약 불가" in s for s in cloud_summary):
        tr_result = summarize_article_text(article_text, stopwords_path=stopwords_path, title=title)
        tr_summary = tr_result.get("summary", [])
        combined_summary = [f"{i+1}. {line}" for i, line in enumerate(tr_summary)]
        return combined_summary, tr_result.get("keywords", [])
    else:
        return cloud_summary, keywords

# 사용 예시:
