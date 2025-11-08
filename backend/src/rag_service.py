import json
import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

try:
    import faiss
except Exception:
    faiss = None


class RagService:
    def __init__(self, config: Dict):
        self.config = config
        self.project_root = Path(__file__).resolve().parent.parent

        def _abs(p: str | os.PathLike) -> Path:
            p_str = str(p)
            # Expand ~ to home directory
            if p_str.startswith("~"):
                p_str = os.path.expanduser(p_str)
            pth = Path(p_str)
            return pth if pth.is_absolute() else (self.project_root / pth)

        self.model_path = _abs(config.get("embedding_model_path", "src/final_model"))
        self.faiss_index_path = _abs(config.get("faiss_index_path", "data/index.faiss"))
        self.docstore_path = _abs(config.get("docstore_path", "data/docstore.json"))
        self.vectors_path = _abs(config.get("vectors_path", "data/vectors.npy"))
        self.vectorizer_path = _abs(config.get("vectorizer_path", "data/vectorizer.pkl"))
        self.tfidf_matrix_path = _abs(config.get("tfidf_matrix_path", "data/tfidf.npz"))
        self.use_faiss = bool(config.get("use_faiss", False))
        self.backend = config.get("backend", "sbert")
        self.top_k_default = int(config.get("top_k", 5))

        self.embedder = None
        self.tfidf_vectorizer: TfidfVectorizer | None = None
        if self.backend in ("sbert", "gemma"):
            from sentence_transformers import SentenceTransformer
            logger.info(f"[RAG] Đang load embedding model từ {self.model_path} (backend: {self.backend})")
            try:
                # Thử load với device="cpu" trước
                try:
                    self.embedder = SentenceTransformer(str(self.model_path), device="cpu")
                    logger.info(f"[RAG] ✓ Embedding model đã load thành công")
                except (TypeError, FileNotFoundError, OSError) as e:
                    # Nếu fail, thử load không có device parameter
                    try:
                        self.embedder = SentenceTransformer(str(self.model_path))
                        logger.info(f"[RAG] ✓ Embedding model đã load thành công (không chỉ định device)")
                    except Exception as e2:
                        # Nếu vẫn fail, log lỗi và để embedder = None
                        logger.warning(f"[RAG] ✗ Không thể load embedding model từ {self.model_path}: {e2}")
                        logger.warning("[RAG] Embedding sẽ không hoạt động. Hãy kiểm tra đường dẫn model hoặc chuyển sang backend 'tfidf'")
                        self.embedder = None
            except Exception as e:
                logger.error(f"[RAG] ✗ Lỗi khi khởi tạo SentenceTransformer: {e}")
                self.embedder = None
        elif self.backend == "tfidf":
            logger.info("[RAG] Sử dụng backend TF-IDF")
            self.tfidf_vectorizer = TfidfVectorizer(max_features=50000)

        self.index = None
        self.nn_index: NearestNeighbors | None = None
        self.vectors_db: np.ndarray | None = None
        self.tfidf_matrix = None
        self.docstore: List[Dict] = []

    def _ensure_paths(self):
        Path(self.faiss_index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.docstore_path).parent.mkdir(parents=True, exist_ok=True)

    def build_index(self, chunks: List[Dict]):
        """Build và lưu index theo backend."""
        self._ensure_paths()
        texts = [c["text"] for c in chunks]
        with open(self.docstore_path, "w", encoding="utf-8") as f:
            json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)

        if self.backend in ("sbert", "gemma"):
            vectors = self.encode(texts)
            if self.use_faiss:
                if faiss is None:
                    raise RuntimeError("FAISS chưa cài đặt nhưng use_faiss=True. Hãy tắt use_faiss hoặc cài FAISS.")
                dim = vectors.shape[1]
                faiss_index = faiss.IndexFlatIP(dim)
                faiss_index.add(vectors.astype(np.float32))
                faiss.write_index(faiss_index, str(self.faiss_index_path))
            else:
                np.save(self.vectors_path, vectors.astype(np.float32))
        elif self.backend == "tfidf":
            self.tfidf_vectorizer.fit(texts)
            X = self.tfidf_vectorizer.transform(texts)
            with open(self.vectorizer_path, "wb") as f:
                pickle.dump(self.tfidf_vectorizer, f)
            try:
                from scipy import sparse
                sparse.save_npz(self.tfidf_matrix_path, X)
            except Exception:
                np.save(str(self.tfidf_matrix_path).replace('.npz', '.npy'), X.toarray())

    def load_index(self):
        logger.info(f"[RAG] Đang load index (backend: {self.backend}, use_faiss: {self.use_faiss})")
        if not Path(self.docstore_path).exists():
            raise FileNotFoundError("Chưa có docstore để load.")
        with open(self.docstore_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.docstore = data.get("chunks", [])
        logger.info(f"[RAG] Đã load {len(self.docstore)} chunks từ docstore")

        if self.backend in ("sbert", "gemma"):
            if self.use_faiss:
                if faiss is None:
                    raise RuntimeError("FAISS chưa cài đặt - không thể load index.")
                if not Path(self.faiss_index_path).exists():
                    raise FileNotFoundError("Chưa có FAISS index để load.")
                self.index = faiss.read_index(str(self.faiss_index_path))
                logger.info(f"[RAG] ✓ Đã load FAISS index với {self.index.ntotal} vectors")
                self.nn_index = None
                self.vectors_db = None
            else:
                if not Path(self.vectors_path).exists():
                    raise FileNotFoundError("Chưa có vectors để load.")
                self.vectors_db = np.load(self.vectors_path)
                self.nn_index = NearestNeighbors(metric="cosine")
                self.nn_index.fit(self.vectors_db)
                logger.info(f"[RAG] ✓ Đã load {len(self.vectors_db)} vectors và build kNN index")
                self.index = None
        elif self.backend == "tfidf":
            if not Path(self.vectorizer_path).exists():
                raise FileNotFoundError("Chưa có vectorizer để load.")
            with open(self.vectorizer_path, "rb") as f:
                self.tfidf_vectorizer = pickle.load(f)
            try:
                from scipy import sparse
                self.tfidf_matrix = sparse.load_npz(self.tfidf_matrix_path)
            except Exception:
                alt = str(self.tfidf_matrix_path).replace('.npz', '.npy')
                if Path(alt).exists():
                    self.tfidf_matrix = np.load(alt)
                else:
                    raise FileNotFoundError("Chưa có ma trận TF-IDF để load.")
            logger.info(f"[RAG] ✓ Đã load TF-IDF vectorizer và matrix")

    def is_ready(self) -> bool:
        backend_ready = False
        if self.backend in ("sbert", "gemma"):
            backend_ready = (self.index is not None) or (self.nn_index is not None)
        elif self.backend == "tfidf":
            backend_ready = self.tfidf_vectorizer is not None and self.tfidf_matrix is not None
        return backend_ready and len(self.docstore) > 0

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.backend not in ("sbert", "gemma"):
            raise RuntimeError("encode chỉ dùng cho backend sbert hoặc gemma.")
        if self.embedder is None:
            raise RuntimeError(
                f"Embedding model chưa được load. "
                f"Kiểm tra đường dẫn model: {self.model_path} "
                f"hoặc chuyển sang backend 'tfidf' trong settings.json"
            )
        logger.debug(f"[EMBEDDING] Đang encode {len(texts)} texts")
        emb = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        if isinstance(emb, list):
            emb = np.array(emb)
        logger.debug(f"[EMBEDDING] ✓ Đã encode thành công, shape: {emb.shape}")
        return emb.astype(np.float32)

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        k = int(top_k or self.top_k_default)
        results = []
        if self.backend in ("sbert", "gemma"):
            logger.debug(f"[SEARCH] Đang encode query và search với {self.backend} backend")
            q_emb = self.encode([query])
            if self.use_faiss:
                if self.index is None:
                    raise RuntimeError("FAISS index chưa sẵn sàng.")
                distances, indices = self.index.search(q_emb, k)
                logger.debug(f"[SEARCH] FAISS search hoàn thành, tìm thấy {len(indices[0])} kết quả")
                for score, idx in zip(distances[0], indices[0]):
                    if idx < 0 or idx >= len(self.docstore):
                        continue
                    item = self.docstore[idx]
                    results.append({
                        "text": item.get("text"),
                        "source": item.get("source"),
                        "score": float(score)
                    })
            else:
                if self.nn_index is None or self.vectors_db is None:
                    raise RuntimeError("kNN index chưa sẵn sàng.")
                distances, indices = self.nn_index.kneighbors(q_emb, n_neighbors=k, return_distance=True)
                logger.debug(f"[SEARCH] kNN search hoàn thành, tìm thấy {len(indices[0])} kết quả")
                for score, idx in zip(distances[0], indices[0]):
                    if idx < 0 or idx >= len(self.docstore):
                        continue
                    item = self.docstore[idx]
                    results.append({
                        "text": item.get("text"),
                        "source": item.get("source"),
                        "score": float(1.0 - score)
                    })
        elif self.backend == "tfidf":
            logger.debug(f"[SEARCH] Đang search với TF-IDF backend")
            if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
                raise RuntimeError("TF-IDF index chưa sẵn sàng.")
            q_vec = self.tfidf_vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
            top_idx = np.argsort(-sims)[:k]
            logger.debug(f"[SEARCH] TF-IDF search hoàn thành, tìm thấy {len(top_idx)} kết quả")
            for idx in top_idx:
                item = self.docstore[idx]
                results.append({
                    "text": item.get("text"),
                    "source": item.get("source"),
                    "score": float(sims[idx])
                })
        return results
