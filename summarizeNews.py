# summarizeNews.py

import re
import kss
from konlpy.tag import Komoran
from summarizer import KeywordSummarizer, KeysentenceSummarizer

def load_stopwords(path):
    with open(path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def remove_redundant_sentences(keysents, threshold=0.9):
    selected = []
    for idx, score, sent in keysents:
        sent_nospace = sent.replace(" ", "")
        if any(jaccard_similarity(sent_nospace, other.replace(" ", "")) > threshold for _, _, other in selected):
            continue
        selected.append((idx, score, sent))
    return selected[:3]

def jaccard_similarity(a, b):
    set_a = set(a)
    set_b = set(b)
    return len(set_a & set_b) / len(set_a | set_b)

def merge_if_incomplete(sents):
    merged = []
    i = 0
    while i < len(sents):
        if i + 1 < len(sents) and not sents[i].endswith(('.', '?', '!', '.”', '."')):
            merged.append(sents[i] + ' ' + sents[i + 1])
            i += 2
        else:
            merged.append(sents[i])
            i += 1
    return merged

def summarize_article_text(article_text: str, stopwords_path="stopwords.txt", title=None):
    # 1. 기본 전처리
    article = re.sub(r'(▲|&nbsp;)+', ' ', article_text).strip()
    article = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b', '', article)
    article = re.sub(r'\[[^\[\]]{2,30}?\]', '', article)
    article = re.sub(r'[\w가-힣\s]+기자\s*=\s*', '', article)
    article = re.sub(r'사진=[^\s,\[\]\n"]+', '', article)

    if article.startswith('['): article = article[1:]
    if article.endswith(']'): article = article[:-1]

    # 2. 문장 분리
    sents = kss.split_sentences(article)
    sents = merge_if_incomplete(sents)

    # 3. 형태소 분석 및 요약
    komoran = Komoran()
    def komoran_tokenize(sent):
        words = komoran.pos(sent, join=True)
        return [w for w in words if '/NN' in w or '/VA' in w or '/VV' in w]

    keyword_summarizer = KeywordSummarizer(tokenize=komoran_tokenize, window=-1, verbose=False)
    keysent_summarizer = KeysentenceSummarizer(tokenize=komoran_tokenize, min_sim=0.3)

    keywords = keyword_summarizer.summarize(sents, topk=30, title=title)
    keysents = keysent_summarizer.summarize(sents, topk=5)
    keysents = remove_redundant_sentences(keysents)

    # 4. 불용어 필터링
    stopwords = load_stopwords(stopwords_path)
    filtered_keywords = [
        kw.split('/')[0]
        for kw, _ in keywords
        if len(kw.split('/')[0]) > 1 and kw.split('/')[0] not in stopwords
    ]

    return {
        "keywords": filtered_keywords,
        "summary": [s for _, _, s in keysents[:3]]
    }
