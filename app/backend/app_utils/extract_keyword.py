from typing import List

import gzip
import os
import pickle
import random

import numpy as np
from konlpy.tag import Mecab
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class TFIDF:
    def __init__(self, train_path="../data/train.pickle", valid_path="../data/validation.pickle"):
        self.mecab = Mecab()
        self._set_seed()
        self.train_data, self.valid_data = self._load_data(train_path, valid_path)

    def add_compound_noun(self, phrase: str) -> List:
        """
        Mecab으로 형태소 분석 후,
        명사(일반 명사(NNG), 고유 명사(NNP))가 연속으로 나올 때 복합 명사로 합치고,
        복합 명사에 사용되지 않은 기존 명사도 포함하여 반환
        """
        result = []
        pos_result = self.mecab.pos(phrase)
        idx = 0
        comp_word = ""

        while idx < len(pos_result):
            word, pos = pos_result[idx]
            if pos in ["NNG", "NNP"]:
                if comp_word != "":
                    comp_word = f"{comp_word} {word}"  # 띄어쓰기로 구분하여 명사 합성
                else:
                    comp_word += word
            else:
                if comp_word != "":
                    result.append(comp_word)
                comp_word = ""  # 일반 명사, 고유 명사가 아닌 토큰이 나오면 다시 초기화
            idx += 1
        return result

    def extract_compound_noun(self, phrase: str) -> List:
        """
        Mecab으로 형태소 분석 후,
        명사(일반 명사(NNG), 고유 명사(NNP))가 연속으로 나올 때 복합 명사로 합쳐서
        복합 명사만 반환
        """
        result = []
        pos_result = self.mecab.pos(phrase)
        idx = 0
        comp_word = ""

        while idx < len(pos_result):
            word, pos = pos_result[idx]
            if pos in ["NNG", "NNP"]:
                if comp_word != "":
                    comp_word = f"{comp_word} {word}"  # 띄어쓰기로 구분하여 명사 합성
                else:
                    comp_word += word
            else:
                if len(comp_word.split(" ")) > 1:  # 복합 명사일 경우만 결과 리스트에 추가
                    result.append(comp_word)
                comp_word = ""  # 일반 명사, 고유 명사가 아닌 토큰이 나오면 다시 초기화
            idx += 1
        return result

    def compute_tfidf_matrix(self, data, method="comp_only"):
        summaries = [ann["summary1"] for ann in data["annotation"]]

        if method == "comp_only":
            vectorizer = CountVectorizer(tokenizer=lambda x: self.extract_compound_noun(x))  # 복합명사만
        elif method == "noun_only":
            vectorizer = CountVectorizer(tokenizer=lambda x: self.mecab.nouns(x))  # 명사만
        elif method == "noun_and_comp":
            vectorizer = CountVectorizer(tokenizer=lambda x: self.add_compound_noun(x))  # 복합명사 + 명사

        tf_matrix = vectorizer.fit_transform(summaries)

        transformer = TfidfTransformer()
        tfidf_matrix = transformer.fit_transform(tf_matrix)

        return summaries, vectorizer, tfidf_matrix

    def idx2vocab(self, tfidf_matrix, vectorizer):
        argsorted_index = np.argsort(tfidf_matrix.toarray(), axis=1)
        vocabulary = [vocab for vocab, _ in sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1])]
        return argsorted_index, vocabulary

    def get_topk_keyword(self, summaries, index, idx2vocab):
        topk = 3
        queries = []

        for i in range(len(summaries)):
            lst = [idx2vocab[index[i][::-1][j]] for j in range(topk)]
            queries.append(lst)

        return queries

    def dump_keywords(self, data, query_keywords, filename):
        data["query"] = query_keywords
        saved_name = f"{filename}.pickle"
        with open(saved_name, "wb") as fw:
            pickle.dump(data, fw, protocol=pickle.HIGHEST_PROTOCOL)

    def extract_keyword(self, data, filename):
        summaries, vectorizer, tfidf_matrix = self.compute_tfidf_matrix(data)
        argsorted_index, vocabulary = self.idx2vocab(tfidf_matrix, vectorizer)
        queries = self.get_topk_keyword(summaries, argsorted_index, vocabulary)
        self.dump_keywords(data, queries, filename)

    def run_tfidf_extract_keyword(self):
        self.extract_keyword(self.train_data, "train_query")
        self.extract_keyword(self.valid_data, "valid_query")

    def _set_seed(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def _load_data(self, train_path, valid_path):
        with gzip.open(train_path, "rb") as f:
            train_data = pickle.load(f)

        with gzip.open(valid_path, "rb") as f:
            valid_data = pickle.load(f)

        return train_data, valid_data
