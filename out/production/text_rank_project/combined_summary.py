from summarizeNews import summarize_article_text
from dasd import summarize_article_debug

def combined_summarize(article_text, stopwords_path="stopwords.txt", title=None):
    """
    1) 클라우드(클러스터링) 방식으로 3줄 요약 시도
    2) '요약 불가' 메시지 포함 시 텍스트랭크(pagerank) 방식으로 대체
    """
    # 1. 클라우드 요약 시도
    cloud_summary = summarize_article_debug(article_text)
    result = summarize_article_text(article_text, stopwords_path=stopwords_path, title=title)
    keywords = result.get("keywords", [])

# 클라우드 요약 결과가 리스트라고 가정, 예: ["1. 요약문", "2. 요약문", "3. 요약문"] 또는 ["요약 불가 (문단 부족)"] 등

    # 2. 실패 판단: "요약 불가" 포함 여부 체크
    if any("요약 불가" in s for s in cloud_summary):
        print("[INFO] 클라우드 요약 실패 - 텍스트랭크 방식으로 대체")
        tr_result = summarize_article_text(article_text, stopwords_path=stopwords_path, title=title)
        # summarize_article_text 함수 반환값 구조: {"summary": [...], "keywords": [...]}
        tr_summary = tr_result.get("summary", [])
        # 리스트형태로 반환하기 위해 인덱스 앞 숫자 붙여 리턴
        combined_summary = [f"{i+1}. {line}" for i, line in enumerate(tr_summary)]
        return combined_summary, tr_result.get("keywords", [])
    else:
        # 클라우드 요약 성공
        # 키워드는 기존 텍스트랭크 함수에서 따로 뽑는 게 좋으니 빈 리스트 반환
        return cloud_summary, keywords

# 사용 예시:
