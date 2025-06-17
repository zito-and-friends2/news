# summarizer.py
from textrank_utils import *
import numpy as np
import scipy as sp

class KeywordSummarizer:
    def __init__(self, sents=None, tokenize=None, min_count=2,
                 window=-1, min_cooccurrence=2, vocab_to_idx=None,
                 df=0.85, max_iter=30, verbose=False):
        self.tokenize = tokenize
        self.min_count = min_count
        self.window = window
        self.min_cooccurrence = min_cooccurrence
        self.vocab_to_idx = vocab_to_idx
        self.df = df
        self.max_iter = max_iter
        self.verbose = verbose
        if sents is not None:
            self.train_textrank(sents)

    def train_textrank(self, sents, title=None):
        full_vocab_sents = sents[:]
        if title:
            full_vocab_sents.append(title)

        g, self.idx_to_vocab = word_graph(
            full_vocab_sents, self.tokenize, self.min_count,
            self.window, self.min_cooccurrence,
            self.vocab_to_idx, self.verbose
        )

        bias = None
        if title is not None:
            bias = np.ones(len(self.idx_to_vocab))
            title_tokens = self.tokenize(title)
            for i, word in enumerate(self.idx_to_vocab):
                if word in title_tokens:
                    bias[i] *= 5.0  # 제목 단어에 가중치 부여
            bias = bias / bias.sum()

        self.R = pagerank(g, self.df, self.max_iter, bias).reshape(-1)
        if self.verbose:
            print('trained TextRank. n words = {}'.format(self.R.shape[0]))

    def keywords(self, topk=30):
        if not hasattr(self, 'R'):
            return []
        idxs = self.R.argsort()[-topk:]
        keywords = [(self.idx_to_vocab[idx], self.R[idx]) for idx in reversed(idxs)]
        return keywords

    def summarize(self, sents, topk=30, title=None):
        self.train_textrank(sents, title=title)
        return self.keywords(topk)

class KeysentenceSummarizer:
    def __init__(self, sents=None, tokenize=None, min_count=2,
                 min_sim=0.3, similarity=None, vocab_to_idx=None,
                 df=0.85, max_iter=30, verbose=False):
        self.tokenize = tokenize
        self.min_count = min_count
        self.min_sim = min_sim
        self.similarity = similarity
        self.vocab_to_idx = vocab_to_idx
        self.df = df
        self.max_iter = max_iter
        self.verbose = verbose
        if sents is not None and len(sents):
            self.train_textrank(sents)

    def train_textrank(self, sents, bias=None):
        g = sent_graph(sents, self.tokenize, self.min_count,
                       self.min_sim, self.similarity, self.vocab_to_idx, self.verbose)
        self.R = pagerank(g, self.df, self.max_iter, bias).reshape(-1)
        if self.verbose:
            print('trained TextRank. n sentences = {}'.format(self.R.shape[0]))

    def summarize(self, sents, topk=30, bias=None):

        n_sents = len(sents)

        if isinstance(bias, np.ndarray):
            if bias.shape != (n_sents,):
                raise ValueError('The shape of bias must be (n_sents,) but {}'.format(bias.shape))
        elif bias is not None:
            raise ValueError('The type of bias must be None or numpy.ndarray but the type is {}'.format(type(bias)))
        self.train_textrank(sents, bias)
        idxs = self.R.argsort()[-topk:]
        keysents = [(idx, self.R[idx], sents[idx]) for idx in reversed(idxs)]
        return keysents
