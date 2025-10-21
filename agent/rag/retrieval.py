# agent/rag/retrieval.py
import os
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class SimpleTFIDFRetriever:
    def __init__(self, docs_dir="docs", chunk_size=300, overlap=50):
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = []  # list of dict {id, text, source}
        self._vectorizer = None
        self._matrix = None
        self._build_index()

    def _read_files(self):
        files = []
        for fn in sorted(os.listdir(self.docs_dir)):
            path = os.path.join(self.docs_dir, fn)
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    files.append((fn, f.read()))
        return files

    def _chunk_text(self, text, fn):
        words = text.split()
        i = 0
        n = len(words)
        chunks = []
        cid = 0
        while i < n:
            chunk_words = words[i:i+self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({"id": f"{fn}::chunk{cid}", "text": chunk_text, "source": fn})
            cid += 1
            i += self.chunk_size - self.overlap
        return chunks

    def _build_index(self):
        files = self._read_files()
        chunks = []
        for fn, txt in files:
            chunks.extend(self._chunk_text(txt, fn))
        self.chunks = chunks
        texts = [c["text"] for c in chunks] or ["empty"]
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        self._matrix = self._vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k=3) -> List[Dict]:
        if self._matrix is None:
            self._build_index()
        qv = self._vectorizer.transform([query])
        scores = (self._matrix @ qv.T).toarray().squeeze()
        if np.all(scores == 0):
            return []
        idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in idx:
            if scores[i] <= 0:
                continue
            c = self.chunks[i]
            results.append({"id": c["id"], "source": c["source"], "text": c["text"], "score": float(scores[i])})
        return results
