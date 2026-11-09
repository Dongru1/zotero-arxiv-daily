"""Local embedding cache encodes only missing texts."""

from types import SimpleNamespace

import numpy as np

from zotero_arxiv_daily.reranker.local import LocalReranker


def test_local_reranker_reuses_cached_vectors(tmp_path, monkeypatch, config):
    calls = []

    class FakeEncoder:
        def encode(self, texts, **kwargs):
            calls.append(list(texts))
            return np.ones((len(texts), 3))

        def similarity(self, left, right):
            return SimpleNamespace(numpy=lambda: np.zeros((len(left), len(right))))

    monkeypatch.setattr("zotero_arxiv_daily.reranker.local.project_root", lambda: tmp_path)
    reranker = LocalReranker(config)
    reranker._encoder = FakeEncoder()
    reranker.get_similarity_score(["alpha"], ["beta"])
    reranker.get_similarity_score(["alpha"], ["gamma"])
    encoded = [text for batch in calls for text in batch]
    assert encoded.count("alpha") == 1
    assert "beta" in encoded and "gamma" in encoded
