# summarizeNews.py

import re
import kss
from konlpy.tag import Komoran
from summarizer import KeywordSummarizer, KeysentenceSummarizer
#from konlpy.tag import Okt

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
def protect_quotes(text):
    quote_map = {}
    def repl(match):
        key = f"__QUOTE{len(quote_map)}__"
        quote_map[key] = match.group(0)
        return key
    protected = re.sub(r'"[^"]+?"', repl, text)
    return protected, quote_map

def restore_quotes(texts, quote_map):
    return [re.sub(r'__QUOTE\d+__', lambda m: quote_map[m.group(0)], t) for t in texts]

def split_by_da_dot_after_kss(sentences):
    result = []
    for s in sentences:
        # 1. 따옴표 안 문장 보호
        protected, quote_map = protect_quotes(s)

        # 2. "다." 기준 분리
        splits = re.findall(r'.*?다\.', protected)

        # 3. 따옴표 복원
        splits = restore_quotes(splits, quote_map)

        result.extend([ss.strip() for ss in splits if ss.strip()])
    return result
def summarize_article_text(article_text: str, stopwords_path="stopwords.txt",title=None):
    # 1. 기본 전처리
    email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    pattern = re.compile(r'[\[\(][가-힣]{2,}=[가-힣A-Za-z0-9]+[\]\)]')
    article = re.sub(r'[\w가-힣\s]+기자\s*=\s*', '', article_text)
    article = re.sub(r'사진=[^\s,\[\]\n"]+', '', article)
    article = article.replace("\n"," ")
    article = article.replace('.',". ")


    if article.startswith('['): article = article[1:]
    if article.endswith(']'): article = article[:-1]

    article = re.sub(pattern, '', article)
    article = re.sub(email_pattern, '', article)
    article = re.sub(r'\b(?:co\.kr|kr|com|net|org)\b', '', article)


    sents = kss.split_sentences(article)
    sents = merge_if_incomplete(sents)
    sents = split_by_da_dot_after_kss(sents)

    if len(sents) >1 :
        komoran = Komoran()
        def komoran_tokenize(sent):
            words = komoran.pos(sent, join=True)
            return [w for w in words if '/NN' in w or '/VA' in w or '/VV' in w]

        keyword_summarizer = KeywordSummarizer(tokenize=komoran_tokenize, window=-1, verbose=False)
        keysent_summarizer = KeysentenceSummarizer(tokenize=komoran_tokenize, min_sim=0.3)

        keywords = keyword_summarizer.summarize(sents, topk=30, title=title)
        #print(keywords)
        keysents = keysent_summarizer.summarize(sents, topk=5)
        keysents = remove_redundant_sentences(keysents)


        stopwords = load_stopwords(stopwords_path)
        filtered_keywords = [
            kw.split('/')[0]
            for kw, _ in keywords
            if len(kw.split('/')[0]) > 1 and kw.split('/')[0] not in stopwords and not kw.endswith('/VV') and not kw.endswith('/VA') and not kw.endswith('/NNB')
        ]

        return {
            "keywords": filtered_keywords,
            "summary": [s for _, _, s in keysents[:3]]
        }
    else:
        return {
            "keywords" : [],
            "summary" : sents[:3]
        }

