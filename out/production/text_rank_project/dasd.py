from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import kss
import torch

def clean_article_text(article_text: str) -> str:
    email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    pattern = re.compile(r'[\[\(][가-힣]{2,}=[가-힣A-Za-z0-9]+[\]\)]')
    article = re.sub(r'[\w가-힣\s]+기자\s*=\s*', '', article_text)
    article = re.sub(r'사진=[^\s,\[\]\n"]+', '', article)
    article = article.replace("\n", " ")
    article = article.replace('.', ". ")
    if article.startswith('['):
        article = article[1:]
    if article.endswith(']'):
        article = article[:-1]
    article = re.sub(pattern, '', article)
    article = re.sub(email_pattern, '', article)
    article = re.sub(r'\b(?:co\.kr|kr|com|net|org)\b', '', article)
    return article

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


def split_sentences_to_paragraphs(sentences, n=3):
    paragraphs = []
    for i in range(0, len(sentences), n):
        chunk = sentences[i:i+n]
        if len(chunk) == n:  # 3문장 미만은 포함하지 않음
            paragraph = ' '.join(chunk)
            paragraphs.append(paragraph)
    return paragraphs

def embed_paragraphs(paragraphs, model):
    return model.encode(paragraphs)

def cluster_paragraphs(paragraphs, embeddings, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    clustered = {i: [] for i in range(n_clusters)}
    for i, label in enumerate(labels):
        clustered[label].append(paragraphs[i])
    return clustered

def get_representative_paragraphs(cluster, embedder, top_n=1):
    if len(cluster) <= top_n:
        return " ".join(cluster)
    embeddings = embedder.encode(cluster)
    centroid = np.mean(embeddings, axis=0)
    scores = cosine_similarity([centroid], embeddings)[0]
    top_idxs = np.argsort(scores)[-top_n:]
    top_idxs = sorted(top_idxs)
    return " ".join([cluster[i] for i in top_idxs])

def summarize(text, model, tokenizer):
    input_ids = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=1024)
    summary_ids = model.generate(
        input_ids,
        max_length=128,
        min_length=15,
        num_beams=4,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_article_debug(article_text):
    article = clean_article_text(article_text)
    sents = kss.split_sentences(article)
    sents = merge_if_incomplete(sents)
    paragraphs = split_sentences_to_paragraphs(sents, n=3)
    print("\n🧪 [1단계] 문단 수:", len(paragraphs))
    for i, p in enumerate(paragraphs):
        print(f"  문단 {i+1}: {p[:1000]}...")

    if len(paragraphs) < 3:
        return ["요약 불가 (문단 부족)"] * 3

    embedder = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    embeddings = embed_paragraphs(paragraphs, embedder)
    clustered = cluster_paragraphs(paragraphs, embeddings, n_clusters=3)

    print("\n🧪 [2단계] 클러스터 분포:")
    for i in range(3):
        print(f"  클러스터 {i}: {len(clustered[i])} 문단")

    tokenizer = PreTrainedTokenizerFast.from_pretrained("digit82/kobart-summarization")
    model = BartForConditionalGeneration.from_pretrained("digit82/kobart-summarization")

    summary_result = []

    for i in range(3):
        cluster_paras = clustered.get(i, [])
        cluster_text = get_representative_paragraphs(cluster_paras, embedder)
        print(f"\n🧪 [3단계] 클러스터 {i} 입력 길이: {len(cluster_text)}")

        if not cluster_text.strip():
            summary_result.append(f"{i+1}. (요약 불가)")
            print("  ⛔️ 입력 없음")
        else:
            try:
                summary = summarize(cluster_text, model, tokenizer)
                summary_result.append(f"{i+1}. {summary}")
                print(f"  ✅ 요약 결과: {summary}")
            except Exception as e:
                summary_result.append(f"{i+1}. (요약 실패: {str(e)[:50]}...)")
                print(f"  ❌ 에러: {str(e)}")

    return summary_result

