# ============================================================
# Module: Embedding Engine (embedding_engine.py)
# 模块：向量化引擎
#
# Generates embeddings via Gemini API (OpenAI-compatible),
# stores them in SQLite, and provides cosine similarity search.
# 通过 Gemini API（OpenAI 兼容）生成 embedding，
# 存储在 SQLite 中，提供余弦相似度搜索。
#
# Depended on by: server.py, bucket_manager.py
# 被谁依赖：server.py, bucket_manager.py
# ============================================================

import os
import json
import math
import hashlib
import sqlite3
import logging
import asyncio
from pathlib import Path

from openai import AsyncOpenAI

logger = logging.getLogger("ombre_brain.embedding")


class EmbeddingEngine:
    """
    Embedding generation + SQLite vector storage + cosine search.
    向量生成 + SQLite 向量存储 + 余弦搜索。
    """

    def __init__(self, config: dict):
        dehy_cfg = config.get("dehydration", {})
        embed_cfg = config.get("embedding", {})

        self.api_key = embed_cfg.get("api_key") or dehy_cfg.get("api_key", "")
        self.base_url = (
            embed_cfg.get("base_url")
            or dehy_cfg.get("base_url")
            or "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.model = embed_cfg.get("model", "gemini-embedding-001")
        self.enabled = bool(self.api_key) and embed_cfg.get("enabled", True)
        self.max_chars = self._int_between(embed_cfg.get("max_chars", 6000), 6000, 500, 32000)
        self.query_instruction = str(
            embed_cfg.get("query_instruction")
            or "Given a memory search query, retrieve relevant long-term memory passages."
        ).strip()
        self.document_instruction = str(embed_cfg.get("document_instruction") or "").strip()
        self.scene_chunk_chars = self._int_between(
            embed_cfg.get("scene_chunk_chars", 900), 900, 400, 2400
        )
        self.scene_chunk_overlap = self._int_between(
            embed_cfg.get("scene_chunk_overlap", 160), 160, 0, 600
        )
        self.scene_chunk_min_chars = self._int_between(
            embed_cfg.get("scene_chunk_min_chars", 120), 120, 40, 600
        )

        # --- SQLite path: buckets_dir/embeddings.db ---
        db_path = os.path.join(config["buckets_dir"], "embeddings.db")
        self.db_path = db_path

        # --- Initialize client ---
        if self.enabled:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=30.0,
            )
        else:
            self.client = None

        # --- Initialize SQLite ---
        self._init_db()

    def _init_db(self):
        """Create embeddings table if not exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                bucket_id TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                model TEXT,
                dimension INTEGER,
                updated_at TEXT NOT NULL
            )
        """)
        self._ensure_column(conn, "embeddings", "model", "TEXT")
        self._ensure_column(conn, "embeddings", "dimension", "INTEGER")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scene_embedding_chunks (
                scene_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                model TEXT,
                dimension INTEGER,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(scene_id, ordinal)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scene_embedding_chunks_scene "
            "ON scene_embedding_chunks(scene_id)"
        )
        conn.commit()
        conn.close()

    async def generate_and_store(self, bucket_id: str, content: str) -> bool:
        """
        Generate embedding for content and store in SQLite.
        为内容生成 embedding 并存入 SQLite。
        Returns True on success, False on failure.
        """
        if not self.enabled or not content or not content.strip():
            return False

        try:
            embedding = await self._generate_embedding(content, kind="document")
            if not embedding:
                return False
            self._store_embedding(bucket_id, embedding)
            return True
        except Exception as e:
            logger.warning(f"Embedding generation failed for {bucket_id}: {e}")
            return False

    async def generate_and_store_scene(self, scene_id: str, content: str) -> bool:
        """Embed one whole Scene plus deterministic raw spans with no child identity."""
        ok = await self.generate_and_store(scene_id, content)
        if not ok:
            self._replace_scene_chunks(scene_id, [])
            return False

        rows: list[dict] = []
        for chunk in self.scene_text_chunks(content):
            embedding = await self._generate_embedding(chunk["text"], kind="document")
            if not embedding:
                self._replace_scene_chunks(scene_id, [])
                logger.warning("Scene chunk embedding failed; cleared stale chunks for %s", scene_id)
                return True
            rows.append({**chunk, "embedding": embedding})
        self._replace_scene_chunks(scene_id, rows)
        return True

    def scene_text_chunks(self, content: str) -> list[dict]:
        """Return overlapping verbatim spans only when a Scene is genuinely long."""
        text = str(content or "")
        max_chars = self.scene_chunk_chars
        if len(text.strip()) <= max_chars:
            return []

        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        chunks: list[dict] = []
        start = 0
        ordinal = 0
        while start < len(text):
            hard_end = min(len(text), start + max_chars)
            end = hard_end
            if hard_end < len(text):
                floor = min(hard_end, start + self.scene_chunk_min_chars)
                candidates = [
                    text.rfind(marker, floor, hard_end)
                    for marker in ("\n\n", "。", "！", "？", "\n")
                ]
                boundary = max(candidates)
                if boundary >= floor:
                    end = boundary + (2 if text.startswith("\n\n", boundary) else 1)

            raw = text[start:end]
            left_trim = len(raw) - len(raw.lstrip())
            right_trim = len(raw) - len(raw.rstrip())
            chunk_start = start + left_trim
            chunk_end = end - right_trim
            chunk_text = text[chunk_start:chunk_end]
            if len(chunk_text) >= self.scene_chunk_min_chars:
                chunks.append(
                    {
                        "ordinal": ordinal,
                        "start_offset": chunk_start,
                        "end_offset": chunk_end,
                        "text": chunk_text,
                        "content_hash": content_hash,
                    }
                )
                ordinal += 1
            if end >= len(text):
                break
            start = max(start + 1, end - self.scene_chunk_overlap)
        return chunks

    def _replace_scene_chunks(self, scene_id: str, rows: list[dict]) -> None:
        from utils import now_iso

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM scene_embedding_chunks WHERE scene_id = ?", (scene_id,))
            for row in rows:
                embedding = row["embedding"]
                conn.execute(
                    """
                    INSERT INTO scene_embedding_chunks
                    (scene_id, ordinal, content_hash, start_offset, end_offset, text,
                     embedding, model, dimension, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scene_id,
                        int(row["ordinal"]),
                        str(row["content_hash"]),
                        int(row["start_offset"]),
                        int(row["end_offset"]),
                        str(row["text"]),
                        json.dumps(embedding),
                        self.model,
                        len(embedding),
                        now_iso(),
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    async def _generate_embedding(self, text: str, *, kind: str = "document") -> list[float]:
        """Call API to generate embedding vector."""
        # Truncate to avoid token limits
        prepared = self._prepare_embedding_input(text, kind=kind)
        truncated = prepared[: self.max_chars]
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=truncated,
            )
            if response.data and len(response.data) > 0:
                return response.data[0].embedding
            return []
        except Exception as e:
            logger.warning(f"Embedding API call failed: {e}")
            return []

    async def embed_query(self, text: str) -> list[float]:
        """Generate one query-space vector without searching the bucket index."""
        if not self.enabled or not str(text or "").strip():
            return []
        return await self._generate_embedding(text, kind="query")

    async def embed_document(self, text: str) -> list[float]:
        """Generate one document-space vector without writing the body index."""
        if not self.enabled or not str(text or "").strip():
            return []
        return await self._generate_embedding(text, kind="document")

    def _store_embedding(self, bucket_id: str, embedding: list[float]):
        """Store embedding in SQLite."""
        from utils import now_iso
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT OR REPLACE INTO embeddings (bucket_id, embedding, model, dimension, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (bucket_id, json.dumps(embedding), self.model, len(embedding), now_iso()),
        )
        conn.commit()
        conn.close()

    def delete_embedding(self, bucket_id: str):
        """Remove embedding when bucket is deleted."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM embeddings WHERE bucket_id = ?", (bucket_id,))
        conn.execute("DELETE FROM scene_embedding_chunks WHERE scene_id = ?", (bucket_id,))
        conn.commit()
        conn.close()

    async def get_embedding(self, bucket_id: str) -> list[float] | None:
        """Retrieve stored embedding for a bucket. Returns None if not found."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT embedding, model, dimension FROM embeddings WHERE bucket_id = ?", (bucket_id,)
        ).fetchone()
        conn.close()
        if row:
            try:
                embedding = json.loads(row[0])
                if not self._row_matches_current_model(row[1], row[2], embedding):
                    return None
                return embedding
            except json.JSONDecodeError:
                return None
        return None

    async def get_embeddings(self, bucket_ids: list[str]) -> dict[str, list[float]]:
        """Retrieve stored embeddings for several buckets with one SQLite read."""
        unique_ids = list(
            dict.fromkeys(
                str(item or "").strip()
                for item in bucket_ids
                if str(item or "").strip()
            )
        )
        if not unique_ids:
            return {}
        placeholders = ",".join("?" for _ in unique_ids)
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                f"SELECT bucket_id, embedding, model, dimension FROM embeddings WHERE bucket_id IN ({placeholders})",
                unique_ids,
            ).fetchall()
        finally:
            conn.close()

        output: dict[str, list[float]] = {}
        for bucket_id, payload, model, dimension in rows:
            try:
                embedding = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                continue
            if self._row_matches_current_model(model, dimension, embedding):
                output[str(bucket_id)] = embedding
        return output

    async def search_similar(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Search for buckets similar to query text.
        Returns list of (bucket_id, similarity_score) sorted by score desc.
        搜索与查询文本相似的桶。返回 (bucket_id, 相似度分数) 列表。
        """
        if not self.enabled:
            return []

        try:
            query_embedding = await self.embed_query(query)
            if not query_embedding:
                return []
        except Exception as e:
            logger.warning(f"Query embedding failed: {e}")
            return []

        return await self.search_similar_by_embedding(query_embedding, top_k=top_k)

    async def search_similar_by_embedding(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search the bucket index using an already-generated query vector."""
        if not self.enabled or not query_embedding:
            return []

        # Load all embeddings from SQLite
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT bucket_id, embedding, model, dimension FROM embeddings").fetchall()
        chunk_rows = conn.execute(
            "SELECT scene_id, embedding, model, dimension FROM scene_embedding_chunks"
        ).fetchall()
        conn.close()

        if not rows and not chunk_rows:
            return []

        # Aggregate all derived vectors back to their owning Scene. Chunk ids
        # never leave this function and never become memory references.
        best_scores: dict[str, float] = {}
        for bucket_id, emb_json, model, dimension in rows:
            try:
                stored_embedding = json.loads(emb_json)
                if not self._row_matches_current_model(model, dimension, stored_embedding):
                    continue
                sim = self._cosine_similarity(query_embedding, stored_embedding)
                key = str(bucket_id)
                best_scores[key] = max(best_scores.get(key, -1.0), sim)
            except (json.JSONDecodeError, Exception):
                continue

        for scene_id, emb_json, model, dimension in chunk_rows:
            try:
                stored_embedding = json.loads(emb_json)
                if not self._row_matches_current_model(model, dimension, stored_embedding):
                    continue
                sim = self._cosine_similarity(query_embedding, stored_embedding)
                key = str(scene_id)
                best_scores[key] = max(best_scores.get(key, -1.0), sim)
            except (json.JSONDecodeError, Exception):
                continue

        results = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def search_scene_evidence_by_embedding(
        self,
        query_embedding: list[float],
        *,
        scene_ids: set[str],
        top_k: int = 10,
    ) -> list[dict]:
        """Return Scene scores with body-only index provenance.

        Canonical Scene rows are written from ``bucket_text_for_embedding``
        (the verbatim Scene body) and ``scene_embedding_chunks`` (verbatim body
        spans). The caller supplies vetted canonical Scene ids, so legacy
        bucket vectors, tags, aliases, graph edges, and Word Map data cannot
        enter this result.
        """
        allowed = {str(scene_id) for scene_id in scene_ids if str(scene_id)}
        if not self.enabled or not query_embedding or not allowed:
            return []

        placeholders = ",".join("?" for _ in allowed)
        params = tuple(sorted(allowed))
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            f"SELECT bucket_id, embedding, model, dimension FROM embeddings "
            f"WHERE bucket_id IN ({placeholders})",
            params,
        ).fetchall()
        chunk_rows = conn.execute(
            f"SELECT scene_id, ordinal, start_offset, end_offset, embedding, model, dimension "
            f"FROM scene_embedding_chunks WHERE scene_id IN ({placeholders})",
            params,
        ).fetchall()
        conn.close()

        best: dict[str, dict] = {}

        def consider(scene_id: str, embedding_json: str, model, dimension, evidence: dict) -> None:
            try:
                stored = json.loads(embedding_json)
                if not self._row_matches_current_model(model, dimension, stored):
                    return
                score = self._cosine_similarity(query_embedding, stored)
            except Exception:
                return
            current = best.get(str(scene_id))
            if current is None or score > float(current["score"]):
                best[str(scene_id)] = {
                    "scene_id": str(scene_id),
                    "score": score,
                    **evidence,
                }

        for scene_id, embedding_json, model, dimension in rows:
            consider(
                str(scene_id),
                embedding_json,
                model,
                dimension,
                {"field": "scene_body", "chunk_ordinal": None},
            )
        for scene_id, ordinal, start_offset, end_offset, embedding_json, model, dimension in chunk_rows:
            consider(
                str(scene_id),
                embedding_json,
                model,
                dimension,
                {
                    "field": "scene_body_chunk",
                    "chunk_ordinal": int(ordinal),
                    "start_offset": int(start_offset),
                    "end_offset": int(end_offset),
                },
            )

        return sorted(
            best.values(),
            key=lambda item: (-float(item["score"]), str(item["scene_id"])),
        )[: max(1, int(top_k))]

    def _prepare_embedding_input(self, text: str, *, kind: str) -> str:
        raw = str(text or "")
        if kind == "query" and self.query_instruction:
            return f"Instruct: {self.query_instruction}\nQuery: {raw}"
        if kind == "document" and self.document_instruction:
            return f"Instruct: {self.document_instruction}\nDocument: {raw}"
        return raw

    def _row_matches_current_model(self, model: str | None, dimension: int | None, embedding: list[float]) -> bool:
        if not embedding:
            return False
        if model != self.model:
            return False
        try:
            stored_dimension = int(dimension)
        except (TypeError, ValueError):
            return False
        return stored_dimension == len(embedding)

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        if any(row[1] == column for row in rows):
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    @staticmethod
    def _int_between(value, default: int, min_value: int, max_value: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = default
        return max(min_value, min(max_value, number))

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
