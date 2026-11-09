from .base import BaseReranker, register_reranker
import hashlib
import logging
import warnings
from pathlib import Path
import numpy as np

from ..openreview_state import project_root

_MAX_CHARS = 2000
_CACHE_DIR = Path("data") / "embeddings"


def _clip(text: str) -> str:
    return (text or "")[:_MAX_CHARS]


def _key(text: str) -> str:
    return hashlib.sha256(_clip(text).encode()).hexdigest()


@register_reranker("local")
class LocalReranker(BaseReranker):
    def __init__(self, config):
        super().__init__(config)
        self._encoder = None

    def _cache_dir(self) -> Path:
        path = project_root() / _CACHE_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _encoder_model(self):
        if self._encoder is not None:
            return self._encoder
        from sentence_transformers import SentenceTransformer
        if not self.config.executor.debug:
            from transformers.utils import logging as transformers_logging
            from huggingface_hub.utils import logging as hf_logging

            transformers_logging.set_verbosity_error()
            hf_logging.set_verbosity_error()
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=FutureWarning)
        self._encoder = SentenceTransformer(self.config.reranker.local.model, trust_remote_code=True)
        return self._encoder

    def _encode(self, texts: list[str]) -> np.ndarray:
        encode_kwargs = dict(self.config.reranker.local.encode_kwargs or {})
        encode_kwargs.setdefault("batch_size", 64)
        keys = [_key(text) for text in texts]
        vectors: list[np.ndarray | None] = [None] * len(texts)
        missing: list[int] = []
        cache_dir = self._cache_dir()
        for index, key in enumerate(keys):
            path = cache_dir / f"{key}.npy"
            if path.is_file():
                vectors[index] = np.load(path)
            else:
                missing.append(index)
        if missing:
            encoded = self._encoder_model().encode(
                [_clip(texts[index]) for index in missing],
                **encode_kwargs,
                show_progress_bar=True,
            )
            for row, index in zip(encoded, missing):
                array = np.asarray(row)
                np.save(cache_dir / f"{keys[index]}.npy", array)
                vectors[index] = array
        return np.stack(vectors)

    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        features = self._encode(s1 + s2)
        s1_feature = features[:len(s1)]
        s2_feature = features[len(s1):]
        encoder = self._encoder_model()
        sim = encoder.similarity(s1_feature, s2_feature)
        return sim.numpy()
