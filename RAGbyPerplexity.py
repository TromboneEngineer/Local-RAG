"""
Complete Local RAG System with Attention Basin Fix

v13 (by Perplexity auto-version naming,
 but based on local RAGbyPerplexityV12.py plus sensitivity analysis and auto optimizer scripts)

CRITICAL UPDATE:
- Position-aware retrieval prevents attention basin problem
- Preserves all v11/v12 features and compatibility
- GTX 1050 (2GB) fully supported

Key Features:
- Intelligent GPU memory management
- Smart fallback system
- Complete settings tracking
- Basin detection & visualization
- Ollama crash recovery
"""

import os
import sys

# ============================================================================
# CRITICAL: Environment Setup BEFORE Any Imports
# ============================================================================
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_JIT"] = "0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TORCH_USE_CUDA_DSA"] = "0"
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True
torch.backends.cudnn.enabled = False

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    GPU_COMPUTE = torch.cuda.get_device_capability(0)
    GPU_COMPUTE_STR = f"{GPU_COMPUTE[0]}.{GPU_COMPUTE[1]}"
    if GPU_COMPUTE[0] < 7:
        torch.set_float32_matmul_precision('high')
else:
    GPU_NAME = "No GPU detected"
    GPU_MEMORY = 0
    GPU_COMPUTE = (0, 0)
    GPU_COMPUTE_STR = "N/A"

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import tempfile
import sqlite3
import json
import requests
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document
import uuid
import pandas as pd
import numpy as np
import time

st.set_page_config(
    page_title="Complete Biomedical RAG v13",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Complete Biomedical RAG System v13 - Basin Fixed")

if CUDA_AVAILABLE:
    st.caption(f"GPU: {GPU_NAME} ({GPU_MEMORY:.1f}GB) | CUDA {GPU_COMPUTE_STR}")
    if GPU_COMPUTE[0] < 7:
        st.info(f"ℹ️ CUDA {GPU_COMPUTE_STR} - Compatibility mode (Triton disabled)")
else:
    st.caption("CPU-only mode")

# ============================================================================
# NEW: POSITION-AWARE RETRIEVAL - THE CRITICAL FIX
# ============================================================================

class PositionAwareRetrieval:
    """
    Anti-basin mechanism - prevents over-concentration at document end
    This is the key innovation that solves the attention basin problem
    """

    @staticmethod
    def diversify_by_position(retrieved_chunks: List[Tuple[str, float, dict, float]],
                              total_chunks: int,
                              diversity_weight: float = 0.3) -> List[Tuple[str, float, dict, float]]:
        """
        Re-score chunks to encourage positional diversity

        Args:
            retrieved_chunks: List of (text, distance, metadata, quality_score)
            total_chunks: Total number of chunks in document
            diversity_weight: 0.0-1.0 (0=pure semantic, 1=pure diversity)

        Returns:
            Re-ranked chunks with position diversity applied
        """
        if not retrieved_chunks or diversity_weight == 0:
            return retrieved_chunks

        reweighted = []
        for text, dist, meta, qual in retrieved_chunks:
            chunk_id = meta.get('chunk_id', 0)
            position = chunk_id / max(total_chunks, 1)

            # Semantic score (inverse of distance)
            sem_score = 1.0 / (1.0 + dist)

            # Position score (penalize document end)
            if position > 0.8:      # Last 20%
                pos_score = 0.3
            elif position > 0.6:    # 60-80%
                pos_score = 0.7
            elif position > 0.4:    # Middle
                pos_score = 1.0
            elif position > 0.2:    # 20-40%
                pos_score = 0.9
            else:                   # First 20%
                pos_score = 0.8

            # Combine semantic + position scores
            combined = (sem_score * (1 - diversity_weight)) + (pos_score * diversity_weight)

            reweighted.append((text, combined, meta, qual))

        # Re-sort by combined score
        reweighted.sort(key=lambda x: x[1], reverse=True)
        return reweighted

# ============================================================================
# NEW: ATTENTION BASIN DETECTOR
# ============================================================================

class AttentionBasinDetector:
    @staticmethod
    def analyze_chunk_distribution(retrieved_chunks: List[Tuple], total_chunks: int) -> dict:
        if not retrieved_chunks:
            return {'detected': False, 'end_ratio': 0, 'severity': 'none', 'positions': [], 'chunk_ids': []}

        # Handle both 4-tuple and 5-tuple formats
        chunk_ids = []
        for item in retrieved_chunks:
            if len(item) == 5:
                _, _, meta, _, _ = item  # 5-tuple: (chunk, score, meta, quality, rerank)
            elif len(item) == 4:
                _, _, meta, _ = item     # 4-tuple: (chunk, score, meta, quality)
            else:
                continue  # Skip malformed items
            chunk_ids.append(meta.get('chunk_id', 0))

        positions = [chunk_id / max(total_chunks, 1) for chunk_id in chunk_ids]
        end_chunks = sum(1 for pos in positions if pos > 0.8)
        end_ratio = end_chunks / len(positions) if positions else 0
        detected = end_ratio > 0.6
        severity = 'high' if end_ratio > 0.75 else 'medium' if end_ratio > 0.6 else 'low'

        return {
            'detected': detected,
            'end_ratio': end_ratio,
            'severity': severity,
            'positions': positions,
            'chunk_ids': chunk_ids
        }

# ============================================================================
# NEW: QUALITY METRICS
# ============================================================================

class QualityMetrics:
    """Compute response quality metrics"""

    @staticmethod
    def compute_metrics(response: str, query: str, retrieved_chunks: List[Tuple]) -> dict:
        metrics = {}

        # Response length
        metrics['response_words'] = len(response.split())

        # Query coverage
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        metrics['query_coverage'] = len(query_words & response_words) / max(len(query_words), 1)

        # Substantive content
        substantive_patterns = [
            r'(?i)(found|discovered|showed|demonstrated|revealed)',
            r'(?i)(conclusion|result|finding)',
            r'(?i)(study|research|experiment|analysis)'
        ]
        substantive_count = sum(len(re.findall(p, response)) for p in substantive_patterns)
        metrics['substantive_density'] = substantive_count / max(len(response.split()), 1)

        # Chunk diversity - handle both 4-tuple and 5-tuple
        chunk_ids = []
        for item in retrieved_chunks:
            if len(item) == 5:
                _, _, meta, _, _ = item  # 5-tuple
            elif len(item) == 4:
                _, _, meta, _ = item     # 4-tuple
            else:
                continue
            chunk_ids.append(meta.get('chunk_id', 0))

        if chunk_ids:
            metrics['chunk_spread'] = max(chunk_ids) - min(chunk_ids)
        else:
            metrics['chunk_spread'] = 0

        return metrics

# ============================================================================
# NEW: OLLAMA CRASH RECOVERY
# ============================================================================

class OllamaManager:
    """Manage Ollama with crash recovery"""

    @staticmethod
    def force_unload_all():
        """Force unload all Ollama models"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                for model in response.json().get('models', []):
                    try:
                        requests.post("http://localhost:11434/api/generate",
                                    json={"model": model['model'], "prompt": "", "keep_alive": 0},
                                    timeout=2)
                    except:
                        pass
        except:
            pass
        time.sleep(0.5)

    @staticmethod
    def invoke_with_recovery(llm, prompt: str, max_retries: int = 3) -> Tuple[str, bool]:
        """
        Invoke LLM with crash recovery
        Returns: (response, success)
        """
        for attempt in range(max_retries):
            try:
                response = llm.invoke(prompt)
                if "model runner" not in response.lower():
                    return response, True
                raise RuntimeError("Ollama crashed")
            except Exception as e:
                error_msg = str(e).lower()
                if "model runner" in error_msg or "resource" in error_msg or "memory" in error_msg:
                    if attempt < max_retries - 1:
                        if CUDA_AVAILABLE:
                            torch.cuda.empty_cache()
                        OllamaManager.force_unload_all()
                        time.sleep(2)
                    else:
                        return f"Error after {max_retries} retries: {e}", False
                else:
                    return f"Error: {e}", False
        return "Failed after retries", False

# ============================================================================
# EMBEDDING MODEL CONFIGURATIONS (from v11/v12)
# ============================================================================

EMBEDDING_MODELS = {
    "granite-embedding-r2": {
        "model_id": "ibm-granite/granite-embedding-english-r2",
        "dimensions": 768,
        "params": "149M",
        "vram_mb": 600,
        "speed": "Moderate",
        "accuracy": "90.1%",
        "context_window": 8192,
        "description": "🏆 Granite 4 co-designed, 8K context (BEST FOR GRANITE 4)",
        "instruction_prefix": None,
        "requires_trust": False,
        "download_size_mb": 600,
        "domain": "🔬 Scientific",
        "recommended_chunk_size": 2000,
        "recommended_overlap": 400
    },
    "granite-embedding-small-r2": {
        "model_id": "ibm-granite/granite-embedding-small-english-r2",
        "dimensions": 384,
        "params": "47M",
        "vram_mb": 190,
        "speed": "Fast",
        "accuracy": "87.3%",
        "context_window": 8192,
        "description": "⭐ Basin-resistant, 8K context (RECOMMENDED FOR BASIN AVOIDANCE)",
        "instruction_prefix": None,
        "requires_trust": False,
        "download_size_mb": 190,
        "domain": "🔬 Scientific",
        "recommended_chunk_size": 2000,
        "recommended_overlap": 400
    },
    "pubmedbert-ms-marco": {
        "model_id": "pritamdeka/S-PubMedBert-MS-MARCO",
        "dimensions": 768,
        "params": "110M",
        "vram_mb": 540,
        "speed": "Moderate",
        "accuracy": "92.4%",
        "context_window": 512,
        "description": "🏆 PubMed-trained (BEST FOR BIOMEDICAL)",
        "instruction_prefix": None,
        "requires_trust": False,
        "download_size_mb": 540,
        "domain": "🧬 Biomedical",
        "recommended_chunk_size": 1600,
        "recommended_overlap": 320
    },
    "pubmedbert-embeddings": {
        "model_id": "NeuML/pubmedbert-base-embeddings",
        "dimensions": 768,
        "params": "110M",
        "vram_mb": 540,
        "speed": "Moderate",
        "accuracy": "91.8%",
        "context_window": 512,
        "description": "PubMed sentence embeddings",
        "instruction_prefix": None,
        "requires_trust": False,
        "download_size_mb": 540,
        "domain": "🧬 Biomedical",
        "recommended_chunk_size": 1600,
        "recommended_overlap": 320
    },
    "bge-small-en-v1.5": {
        "model_id": "BAAI/bge-small-en-v1.5",
        "dimensions": 384,
        "params": "33.4M",
        "vram_mb": 130,
        "speed": "Fast",
        "accuracy": "84.7%",
        "context_window": 512,
        "description": "General purpose balanced",
        "instruction_prefix": "Represent this sentence for searching relevant passages: ",
        "requires_trust": False,
        "download_size_mb": 130,
        "domain": "📚 General",
        "recommended_chunk_size": 1400,
        "recommended_overlap": 280
    },
    "nomic-embed-text-v1.5": {
        "model_id": "nomic-ai/nomic-embed-text-v1.5",
        "dimensions": 768,
        "params": "137M",
        "vram_mb": 550,
        "speed": "Moderate",
        "accuracy": "86.2%",
        "context_window": 8192,
        "description": "High accuracy, 8K context",
        "instruction_prefix": "search_document: ",
        "requires_trust": True,
        "download_size_mb": 550,
        "domain": "📚 General",
        "recommended_chunk_size": 2000,
        "recommended_overlap": 400
    },
    "all-MiniLM-L6-v2": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": 384,
        "params": "22.7M",
        "vram_mb": 90,
        "speed": "Very Fast",
        "accuracy": "78.1%",
        "context_window": 256,
        "description": "Fastest, smallest (fallback)",
        "instruction_prefix": None,
        "requires_trust": False,
        "download_size_mb": 90,
        "domain": "📚 General",
        "recommended_chunk_size": 800,
        "recommended_overlap": 160
    }
}

CHUNK_CONFIGS = {
    "granite-optimized": {
        "chunk_size": 2000,
        "chunk_overlap": 400,
        "description": "🏆 Granite Optimized: Best for Granite 4 + Granite embeddings (RECOMMENDED)"
    },
    "biomedical-precise": {
        "chunk_size": 1600,
        "chunk_overlap": 320,
        "description": "🧬 Biomedical Precise: Best for PubMedBERT models"
    },
    "balanced": {
        "chunk_size": 1400,
        "chunk_overlap": 280,
        "description": "📚 Balanced: Good for general embeddings"
    },
    "contextual": {
        "chunk_size": 2400,
        "chunk_overlap": 480,
        "description": "📖 Contextual: Maximum context"
    },
    "compact": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "description": "⚡ Compact: Fast retrieval"
    }
}

# ============================================================================
# DOCUMENT CLEANER (from v11/v12 - unchanged)
# ============================================================================

class DocumentCleaner:
    """Filter boilerplate content"""

    BOILERPLATE_PATTERNS = [
        r'(?i)^(data availability|ethics statement|competing interests|author contributions|acknowledgements?|supplementary information|funding|copyright|license)',
        r'(?i)(all authors? (?:have )?read and agreed|correspondence to|equal contribution|supplementary material)',
        r'(?i)(^[©]|springer nature|creative commons|cc by|all rights reserved)',
        r'(?i)(^references?$|^bibliography$|^cited by$)',
        r'(?i)(peer[ -]?review|manuscript received|accepted for publication)',
        r'(?i)(^doi:|^pmid:|^pmc[0-9]+)',
    ]

    SUBSTANTIVE_INDICATORS = [
        r'(?i)(we (?:found|show|demonstrate|report|observe)|results? (?:show|indicate|suggest))',
        r'(?i)(methods?:|materials? and methods?|experimental|statistical analysis)',
        r'(?i)(conclusion|discussion|introduction|background)',
        r'(?i)((?:figure|table|supplementary) [0-9]+)',
        r'(?i)(p\s*[<>=]\s*0\.[0-9]+)',
    ]

    MIN_SUBSTANTIVE_WORDS = 20
    MAX_REPETITIVE_RATIO = 0.3

    @classmethod
    def is_boilerplate(cls, text: str) -> Tuple[bool, str]:
        for pattern in cls.BOILERPLATE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True, "Boilerplate"

        words = text.split()
        if len(words) < cls.MIN_SUBSTANTIVE_WORDS:
            return True, "Too short"

        unique_words = len(set(words))
        repetitive_ratio = 1 - (unique_words / len(words))
        if repetitive_ratio > cls.MAX_REPETITIVE_RATIO:
            return True, "Repetitive"

        return False, "OK"

    @classmethod
    def score_chunk_quality(cls, text: str) -> float:
        score = 0.5

        for pattern in cls.SUBSTANTIVE_INDICATORS:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.1

        for pattern in cls.BOILERPLATE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                score -= 0.2

        biomedical_terms = [
            r'(?i)(protein|gene|cell|patient|treatment|therapy|disease|clinical|study)',
            r'(?i)(receptor|pathway|signaling|expression|mutation|variant)',
            r'(?i)(significant|correlation|regression|model|analysis|p-value)'
        ]

        for pattern in biomedical_terms:
            if re.search(pattern, text):
                score += 0.05

        return max(0.0, min(1.0, score))

# ============================================================================
# OLLAMA MODEL MANAGEMENT (from v11/v12 - unchanged)
# ============================================================================

def get_ollama_models():
    """Fetch available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            return [model['model'] for model in models_data.get('models', [])]
        else:
            return ["granite4:tiny-h"]
    except Exception:
        return ["granite4:tiny-h"]

# ============================================================================
# DATABASE - COMPLETE VERSION
# ============================================================================

class ConversationDB:
    """SQLite database with complete settings tracking"""

    def __init__(self, db_path="./conversation_history.db",
                 auto_export_dir="./conversation_exports"):
        self.db_path = db_path
        self.auto_export_dir = Path(auto_export_dir)
        self.auto_export_dir.mkdir(exist_ok=True)
        self.init_database()
        self.migrate_database()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                full_path TEXT,
                upload_timestamp TEXT NOT NULL,
                collection_name TEXT NOT NULL,
                num_chunks INTEGER,
                doc_size_bytes INTEGER,
                chunk_size_chars INTEGER,
                chunk_overlap_chars INTEGER,
                chunk_config_name TEXT,
                embedding_model_name TEXT,
                embedding_model_id TEXT,
                embedding_dimensions INTEGER,
                embedding_params TEXT,
                embedding_context_window INTEGER,
                gpu_acceleration_enabled INTEGER,
                gpu_device_name TEXT,
                gpu_batch_size INTEGER,
                gpu_vram_gb REAL,
                filtering_enabled INTEGER,
                chunks_raw INTEGER,
                chunks_filtered_out INTEGER,
                min_quality_threshold REAL,
                processing_llm_name TEXT,
                processing_llm_context INTEGER,
                session_id TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                query_text TEXT NOT NULL,
                response_text TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                num_sources_requested INTEGER,
                num_sources_returned INTEGER,
                top_k_initial INTEGER,
                query_llm_name TEXT,
                query_llm_context_window INTEGER,
                query_llm_temperature REAL,
                query_llm_top_p REAL,
                query_llm_top_k INTEGER,
                reranking_enabled INTEGER,
                reranker_model_name TEXT,
                response_time_seconds REAL,
                embedding_time_seconds REAL,
                reranking_time_seconds REAL,
                session_id TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_rank INTEGER,
                similarity_score REAL,
                quality_score REAL,
                rerank_score REAL,
                chunk_length_chars INTEGER,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
            )
        """)
        conn.commit()
        conn.close()

    def migrate_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check and add missing columns to documents table
        cursor.execute("PRAGMA table_info(documents)")
        doc_columns = [col[1] for col in cursor.fetchall()]

        new_doc_columns = {
            'full_path': 'TEXT', 'chunk_size_chars': 'INTEGER',
            'chunk_overlap_chars': 'INTEGER', 'chunk_config_name': 'TEXT',
            'embedding_model_name': 'TEXT', 'embedding_model_id': 'TEXT',
            'embedding_dimensions': 'INTEGER', 'embedding_params': 'TEXT',
            'embedding_context_window': 'INTEGER', 'gpu_acceleration_enabled': 'INTEGER',
            'gpu_device_name': 'TEXT', 'gpu_batch_size': 'INTEGER',
            'gpu_vram_gb': 'REAL', 'filtering_enabled': 'INTEGER',
            'chunks_raw': 'INTEGER', 'chunks_filtered_out': 'INTEGER',
            'min_quality_threshold': 'REAL', 'processing_llm_name': 'TEXT',
            'processing_llm_context': 'INTEGER', 'session_id': 'TEXT'
        }

        for col_name, col_type in new_doc_columns.items():
            if col_name not in doc_columns:
                try:
                    cursor.execute(f"ALTER TABLE documents ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError:
                    pass

        # ⭐ NEW: Add basin-related columns to conversations table
        cursor.execute("PRAGMA table_info(conversations)")
        conv_columns = [col[1] for col in cursor.fetchall()]

        new_conv_columns = {
            'diversity_weight': 'REAL',
            'basin_detected': 'INTEGER',
            'basin_end_ratio': 'REAL',
            'basin_severity': 'TEXT',
            'quality_query_coverage': 'REAL',
            'quality_response_words': 'INTEGER',
            'quality_substantive_density': 'REAL',
            'quality_chunk_spread': 'INTEGER'
        }

        for col_name, col_type in new_conv_columns.items():
            if col_name not in conv_columns:
                try:
                    cursor.execute(f"ALTER TABLE conversations ADD COLUMN {col_name} {col_type}")
                except sqlite3.OperationalError:
                    pass

        conn.commit()
        conn.close()

    def save_document(self, doc_id, filename, full_path, collection_name,
                     num_chunks, doc_size, session_id, settings_dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO documents VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            doc_id, filename, full_path, datetime.now().isoformat(),
            collection_name, num_chunks, doc_size,
            settings_dict['chunk_size_chars'], settings_dict['chunk_overlap_chars'],
            settings_dict['chunk_config_name'], settings_dict['embedding_model_name'],
            settings_dict['embedding_model_id'], settings_dict['embedding_dimensions'],
            settings_dict['embedding_params'], settings_dict['embedding_context_window'],
            settings_dict['gpu_acceleration_enabled'], settings_dict['gpu_device_name'],
            settings_dict['gpu_batch_size'], settings_dict['gpu_vram_gb'],
            settings_dict['filtering_enabled'], settings_dict['chunks_raw'],
            settings_dict['chunks_filtered_out'], settings_dict['min_quality_threshold'],
            settings_dict.get('processing_llm_name'), settings_dict.get('processing_llm_context'),
            session_id
        ))
        conn.commit()
        conn.close()

    def save_conversation(self, conversation_id, doc_id, query, response,
                         sources, session_id, settings_dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # ⭐ UPDATED: Include basin and quality metrics
        cursor.execute("""
            INSERT INTO conversations VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            conversation_id, doc_id, query, response, datetime.now().isoformat(),
            settings_dict['num_sources_requested'], settings_dict['num_sources_returned'],
            settings_dict['top_k_initial'], settings_dict['query_llm_name'],
            settings_dict['query_llm_context_window'], settings_dict['query_llm_temperature'],
            settings_dict['query_llm_top_p'], settings_dict['query_llm_top_k'],
            settings_dict['reranking_enabled'], settings_dict.get('reranker_model_name'),
            settings_dict['response_time_seconds'], settings_dict.get('embedding_time_seconds', 0),
            settings_dict.get('reranking_time_seconds', 0), session_id,
            # ⭐ NEW: Basin and quality metrics
            settings_dict.get('diversity_weight', 0.0),
            settings_dict.get('basin_detected', 0),
            settings_dict.get('basin_end_ratio', 0.0),
            settings_dict.get('basin_severity', 'none'),
            settings_dict.get('quality_query_coverage', 0.0),
            settings_dict.get('quality_response_words', 0),
            settings_dict.get('quality_substantive_density', 0.0),
            settings_dict.get('quality_chunk_spread', 0)
        ))

        for rank, (chunk, similarity_score, quality_score, rerank_score) in enumerate(sources):
            cursor.execute("""
                INSERT INTO sources (
                    conversation_id, chunk_text, chunk_rank,
                    similarity_score, quality_score, rerank_score, chunk_length_chars
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (conversation_id, chunk, rank + 1, similarity_score, quality_score,
                  rerank_score, len(chunk)))

        conn.commit()
        conn.close()
        self._auto_export_all()

    def _auto_export_all(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = self.auto_export_dir / f"conversations_auto_{timestamp}.csv"
        json_path = self.auto_export_dir / f"conversations_auto_{timestamp}.json"
        try:
            self.export_to_csv_comprehensive(str(csv_path))
            self.export_to_json_comprehensive(str(json_path))
        except Exception:
            pass

    def export_to_csv_comprehensive(self, output_path, session_id=None):
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT
                d.filename, d.full_path, d.upload_timestamp, d.doc_size_bytes, d.num_chunks,
                d.chunk_size_chars, d.chunk_overlap_chars, d.chunk_config_name,
                d.embedding_model_name, d.embedding_model_id, d.embedding_dimensions,
                d.embedding_params, d.embedding_context_window, d.gpu_acceleration_enabled,
                d.gpu_device_name, d.gpu_batch_size, d.gpu_vram_gb, d.filtering_enabled,
                d.chunks_raw, d.chunks_filtered_out, d.min_quality_threshold,
                d.processing_llm_name, d.processing_llm_context,
                c.timestamp, c.query_text, c.response_text, c.num_sources_requested,
                c.num_sources_returned, c.top_k_initial, c.query_llm_name,
                c.query_llm_context_window, c.query_llm_temperature, c.query_llm_top_p,
                c.query_llm_top_k, c.reranking_enabled, c.reranker_model_name,
                c.response_time_seconds, c.embedding_time_seconds, c.reranking_time_seconds,
                c.diversity_weight, c.basin_detected, c.basin_end_ratio, c.basin_severity,
                c.quality_query_coverage, c.quality_response_words,
                c.quality_substantive_density, c.quality_chunk_spread,
                c.conversation_id, c.session_id
            FROM conversations c
            JOIN documents d ON c.doc_id = d.doc_id
        """
        if session_id:
            query += f" WHERE c.session_id = '{session_id}'"
        query += " ORDER BY c.timestamp"

        df = pd.read_sql_query(query, conn)

        for idx, row in df.iterrows():
            conv_id = row['conversation_id']
            cursor = conn.cursor()
            cursor.execute("""
                SELECT chunk_text, chunk_rank, similarity_score,
                       quality_score, rerank_score, chunk_length_chars
                FROM sources WHERE conversation_id = ?
                ORDER BY chunk_rank
            """, (conv_id,))
            sources = cursor.fetchall()
            for i, (text, rank, sim, qual, rerank, length) in enumerate(sources, 1):
                df.at[idx, f'source_{i}_text'] = text
                df.at[idx, f'source_{i}_similarity'] = sim
                df.at[idx, f'source_{i}_quality'] = qual
                df.at[idx, f'source_{i}_rerank_score'] = rerank
                df.at[idx, f'source_{i}_length_chars'] = length

        df = df.drop('conversation_id', axis=1)
        conn.close()
        df.to_csv(output_path, index=False)
        return output_path

    def export_to_json_comprehensive(self, output_path, session_id=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        query = """
            SELECT
                d.filename, d.full_path, d.upload_timestamp, d.doc_size_bytes, d.num_chunks,
                d.chunk_size_chars, d.chunk_overlap_chars, d.chunk_config_name,
                d.embedding_model_name, d.embedding_model_id, d.embedding_dimensions,
                d.embedding_params, d.embedding_context_window, d.gpu_acceleration_enabled,
                d.gpu_device_name, d.gpu_batch_size, d.gpu_vram_gb, d.filtering_enabled,
                d.chunks_raw, d.chunks_filtered_out, d.min_quality_threshold,
                d.processing_llm_name, d.processing_llm_context,
                c.conversation_id, c.query_text, c.response_text, c.timestamp,
                c.num_sources_requested, c.num_sources_returned, c.top_k_initial,
                c.query_llm_name, c.query_llm_context_window, c.query_llm_temperature,
                c.query_llm_top_p, c.query_llm_top_k, c.reranking_enabled,
                c.reranker_model_name, c.response_time_seconds,
                c.embedding_time_seconds, c.reranking_time_seconds, c.session_id
            FROM conversations c
            JOIN documents d ON c.doc_id = d.doc_id
        """
        if session_id:
            query += f" WHERE c.session_id = '{session_id}'"
        query += " ORDER BY c.timestamp"

        cursor.execute(query)
        conversations = []

        for row in cursor.fetchall():
            conv_id = row[23]
            cursor.execute("""
                SELECT chunk_text, chunk_rank, similarity_score,
                       quality_score, rerank_score, chunk_length_chars
                FROM sources WHERE conversation_id = ?
                ORDER BY chunk_rank
            """, (conv_id,))
            sources = [{
                "text": s[0], "rank": s[1], "similarity": s[2],
                "quality": s[3], "rerank_score": s[4], "length_chars": s[5]
            } for s in cursor.fetchall()]

            conversations.append({
                "document": {"filename": row[0], "full_path": row[1], "uploaded": row[2], "size_bytes": row[3], "num_chunks": row[4]},
                "chunking_settings": {"chunk_size_chars": row[5], "chunk_overlap_chars": row[6], "config_name": row[7]},
                "embedding_settings": {"model_name": row[8], "model_id": row[9], "dimensions": row[10], "parameters": row[11], "context_window": row[12]},
                "gpu_settings": {"acceleration_enabled": bool(row[13]), "device_name": row[14], "batch_size": row[15], "vram_gb": row[16]},
                "filtering_settings": {"enabled": bool(row[17]), "chunks_raw": row[18], "chunks_filtered_out": row[19], "min_quality_threshold": row[20]},
                "processing_llm": {"name": row[21], "context_window": row[22]},
                "conversation_id": conv_id,
                "query": row[24], "response": row[25], "timestamp": row[26],
                "retrieval_settings": {"sources_requested": row[27], "sources_returned": row[28], "top_k_initial": row[29]},
                "query_llm_settings": {"model_name": row[30], "context_window": row[31], "temperature": row[32], "top_p": row[33], "top_k": row[34]},
                "enhancement_settings": {"reranking_enabled": bool(row[35]), "reranker_model": row[36]},
                "performance_metrics": {"response_time_seconds": row[37], "embedding_time_seconds": row[38], "reranking_time_seconds": row[39]},
                "session_id": row[40],
                "sources": sources
            })

        conn.close()
        with open(output_path, 'w') as f:
            json.dump(conversations, f, indent=2)
        return output_path

    def get_session_stats(self, session_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM conversations WHERE session_id = ?", (session_id,))
            conv_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM conversations WHERE session_id = ?", (session_id,))
            doc_count = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            conv_count = 0
            doc_count = 0
        conn.close()
        return {'conversations': conv_count, 'documents': doc_count}

# Instantiate database
db = ConversationDB()

# ============================================================================
# SESSION STATE (from v11/v12 - unchanged)
# ============================================================================

if 'session_id' not in st.session_state:
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:12]}"

for key in ['messages', 'doc_processed', 'doc_id', 'collection_name', 'filename',
            'full_path', 'selected_embedding', 'selected_chunk_config', 'use_gpu',
            'gpu_batch_size', 'enable_filtering', 'enable_reranking']:
    if key not in st.session_state:
        if key == 'selected_embedding':
            st.session_state[key] = "granite-embedding-small-r2"  # Changed default to basin-resistant
        elif key == 'selected_chunk_config':
            st.session_state[key] = "granite-optimized"
        elif key == 'use_gpu':
            st.session_state[key] = CUDA_AVAILABLE
        elif key == 'gpu_batch_size':
            st.session_state[key] = 8
        elif key == 'enable_filtering':
            st.session_state[key] = True
        elif key == 'enable_reranking':
            st.session_state[key] = True
        elif key == 'messages':
            st.session_state[key] = []
        else:
            st.session_state[key] = False if 'doc' in key else None

# NEW: Add diversity weight to session state
if 'diversity_weight' not in st.session_state:
    st.session_state.diversity_weight = 0.3  # Default 30% position weighting

if 'processing_llm' not in st.session_state:
    st.session_state.processing_llm = None

if 'query_llm' not in st.session_state:
    st.session_state.query_llm = "granite4:tiny-h"

if 'llm_temperature' not in st.session_state:
    st.session_state.llm_temperature = 0.7

if 'llm_top_p' not in st.session_state:
    st.session_state.llm_top_p = 0.9

if 'llm_top_k' not in st.session_state:
    st.session_state.llm_top_k = 40

# ============================================================================
# RAG COMPONENTS (from v11/v12 with modifications)
# ============================================================================

@st.cache_resource
def load_embedding_model(model_key, use_gpu=False):
    """Load embedding model with intelligent fallback and memory management"""
    model_config = EMBEDDING_MODELS[model_key]
    model_id = model_config["model_id"]
    device = 'cuda' if (use_gpu and CUDA_AVAILABLE) else 'cpu'

    # Check GPU memory before attempting GPU load
    if use_gpu and CUDA_AVAILABLE:
        try:
            free_memory = torch.cuda.mem_get_info(0)[0] / (1024**3)
            required_memory = model_config['vram_mb'] / 1024
            if free_memory < required_memory:
                st.warning(f"⚠️ Insufficient GPU memory: {free_memory:.2f}GB free, {required_memory:.2f}GB required")
                st.info(f"🔄 Loading {model_key} on CPU instead")
                device = 'cpu'
        except Exception as e:
            st.warning(f"Could not check GPU memory: {e}")

    if use_gpu and not CUDA_AVAILABLE:
        st.warning("⚠️ GPU requested but not available")
        device = 'cpu'

    try:
        kwargs = {'device': device}
        if model_config.get("requires_trust", False):
            st.info(f"🔐 {model_key} requires trust_remote_code")
            kwargs['trust_remote_code'] = True

        model = SentenceTransformer(model_id, **kwargs)
        device_str = 'GPU' if device == 'cuda' else 'CPU'
        st.success(f"✅ Loaded {model_key} on {device_str}")
        return model, model_config

    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "out of memory" in str(e):
            st.error(f"❌ GPU OOM: {str(e)[:100]}...")
            st.info(f"🔄 Retrying {model_key} on CPU")
            try:
                if CUDA_AVAILABLE:
                    torch.cuda.empty_cache()
                kwargs = {'device': 'cpu'}
                if model_config.get("requires_trust", False):
                    kwargs['trust_remote_code'] = True
                model = SentenceTransformer(model_id, **kwargs)
                st.success(f"✅ Loaded {model_key} on CPU")
                return model, model_config
            except Exception as cpu_error:
                st.error(f"❌ Failed to load {model_key} on CPU: {str(cpu_error)[:100]}")
                st.warning("🔄 Final fallback: all-MiniLM-L6-v2 on CPU")
                fallback = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
                return fallback, EMBEDDING_MODELS['all-MiniLM-L6-v2']
        else:
            st.error(f"❌ Failed: {str(e)[:150]}")
            st.warning(f"🔄 Trying {model_key} on CPU")
            try:
                kwargs = {'device': 'cpu'}
                if model_config.get("requires_trust", False):
                    kwargs['trust_remote_code'] = True
                model = SentenceTransformer(model_id, **kwargs)
                st.success(f"✅ Loaded {model_key} on CPU")
                return model, model_config
            except Exception as fb_error:
                st.error(f"❌ Fallback failed: {fb_error}")
                st.warning("🔄 Using all-MiniLM-L6-v2 on CPU")
                fallback = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
                return fallback, EMBEDDING_MODELS['all-MiniLM-L6-v2']

@st.cache_resource
def load_reranker():
    """Load reranking model"""
    try:
        return CrossEncoder('BAAI/bge-reranker-base', max_length=512, device='cpu')
    except Exception as e:
        st.warning(f"Reranker unavailable: {e}")
        return None

def load_llm(model_name, context_window=25000, temperature=0.7, top_p=0.9, top_k=40):
    """Load Ollama LLM with configurable parameters"""
    return OllamaLLM(
        model=model_name,
        num_ctx=context_window,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )

@st.cache_resource
def init_chromadb():
    """Initialize ChromaDB"""
    return chromadb.PersistentClient(
        path="./chromadb_storage",
        settings=Settings(anonymized_telemetry=False)
    )

def save_uploaded_file(uploaded_file, save_dir="./uploaded_documents"):
    """Save uploaded file"""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"{timestamp}_{uploaded_file.name}"
    full_path = save_path / unique_filename
    with open(full_path, 'wb') as f:
        f.write(uploaded_file.getvalue())
    return str(full_path.absolute())

def process_document(uploaded_file):
    """Process uploaded PDF with selected settings"""

    # Save file
    full_path = save_uploaded_file(uploaded_file)
    st.session_state.full_path = full_path
    st.session_state.filename = uploaded_file.name

    # Load PDF
    try:
        loader = PyMuPDFLoader(full_path)
        pages = loader.load()
        full_text = "\n\n".join([page.page_content for page in pages])
        st.success(f"✅ Loaded {len(pages)} pages")
    except Exception as e:
        st.error(f"❌ Failed to load PDF: {e}")
        return

    # Get chunking config
    chunk_cfg = CHUNK_CONFIGS[st.session_state.selected_chunk_config]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_cfg['chunk_size'],
        chunk_overlap=chunk_cfg['chunk_overlap'],
        length_function=len
    )

    # Split into chunks
    raw_chunks = splitter.split_text(full_text)
    st.info(f"📝 Created {len(raw_chunks)} chunks")

    # Filter chunks if enabled
    if st.session_state.enable_filtering:
        filtered_chunks = []
        filtered_qualities = []
        filtered_out = 0

        progress_bar = st.progress(0)
        for i, chunk in enumerate(raw_chunks):
            is_boilerplate, reason = DocumentCleaner.is_boilerplate(chunk)
            if not is_boilerplate:
                quality = DocumentCleaner.score_chunk_quality(chunk)
                if quality >= 0.3:
                    filtered_chunks.append(chunk)
                    filtered_qualities.append(quality)
                else:
                    filtered_out += 1
            else:
                filtered_out += 1
            progress_bar.progress((i + 1) / len(raw_chunks))

        progress_bar.empty()
        st.success(f"✅ Kept {len(filtered_chunks)} chunks (filtered {filtered_out})")
        chunks = filtered_chunks
        qualities = filtered_qualities
    else:
        chunks = raw_chunks
        qualities = [0.5] * len(chunks)
        filtered_out = 0

    if len(chunks) == 0:
        st.error("❌ No chunks after filtering!")
        return

    # Create embeddings
    st.info(f"🧮 Creating embeddings with {st.session_state.selected_embedding}...")

    embedding_model, model_config = load_embedding_model(
        st.session_state.selected_embedding,
        st.session_state.use_gpu
    )

    # Initialize ChromaDB collection
    client = init_chromadb()
    collection_name = f"doc_{uuid.uuid4().hex[:8]}"

    try:
        collection = client.create_collection(collection_name)
    except:
        client.delete_collection(collection_name)
        collection = client.create_collection(collection_name)

    # Batch encode
    batch_size = st.session_state.gpu_batch_size if st.session_state.use_gpu else 32
    progress_bar = st.progress(0)

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_qualities = qualities[i:i + batch_size]

        try:
            with torch.no_grad():
                embeddings = embedding_model.encode(
                    batch_chunks,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )

            # Add to collection
            collection.add(
                embeddings=embeddings.tolist(),
                documents=batch_chunks,
                ids=[f"chunk_{i+j}" for j in range(len(batch_chunks))],
                metadatas=[
                    {'chunk_id': i+j, 'quality_score': batch_qualities[j]}
                    for j in range(len(batch_chunks))
                ]
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                st.warning("⚠️ GPU OOM, reducing batch size...")
                torch.cuda.empty_cache()
                st.session_state.gpu_batch_size = max(2, batch_size // 2)
                st.rerun() # Changed from st.experimental_rerun()
            else:
                raise e

        progress_bar.progress((i + len(batch_chunks)) / len(chunks))

    progress_bar.empty()
    st.success(f"✅ Embedded {len(chunks)} chunks")

    # Save to database
    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    settings_dict = {
        'chunk_size_chars': chunk_cfg['chunk_size'],
        'chunk_overlap_chars': chunk_cfg['chunk_overlap'],
        'chunk_config_name': st.session_state.selected_chunk_config,
        'embedding_model_name': st.session_state.selected_embedding,
        'embedding_model_id': model_config['model_id'],
        'embedding_dimensions': model_config['dimensions'],
        'embedding_params': model_config['params'],
        'embedding_context_window': model_config['context_window'],
        'gpu_acceleration_enabled': int(st.session_state.use_gpu),
        'gpu_device_name': GPU_NAME if st.session_state.use_gpu else 'CPU',
        'gpu_batch_size': st.session_state.gpu_batch_size,
        'gpu_vram_gb': GPU_MEMORY if st.session_state.use_gpu else 0,
        'filtering_enabled': int(st.session_state.enable_filtering),
        'chunks_raw': len(raw_chunks),
        'chunks_filtered_out': filtered_out,
        'min_quality_threshold': 0.3,
        'processing_llm_name': None,
        'processing_llm_context': None
    }

    db.save_document(
        doc_id, uploaded_file.name, full_path, collection_name,
        len(chunks), uploaded_file.size, st.session_state.session_id,
        settings_dict
    )

    # Update session state
    st.session_state.doc_processed = True
    st.session_state.doc_id = doc_id
    st.session_state.collection_name = collection_name
    st.session_state.messages = []

    st.success("✅ Document processed! Ask questions below.")
    st.rerun() # Changed from st.experimental_rerun()

# MODIFIED: query_documents with position-aware retrieval
def query_documents(question, collection_name, embedding_key, use_gpu,
                   enable_reranking, diversity_weight, top_k=5):
    """
    Query with position-aware retrieval and optional reranking

    NEW: diversity_weight parameter enables position-aware retrieval
    """
    # Get more candidates if using diversity or reranking
    initial_k = top_k * 4 if diversity_weight > 0 else (top_k * 3 if enable_reranking else top_k)

    embedding_model, model_config = load_embedding_model(embedding_key, use_gpu)

    prefix = model_config.get("instruction_prefix")
    question_with_prefix = (prefix + question) if prefix else question

    embed_start = datetime.now()
    with torch.no_grad():
        question_embedding = embedding_model.encode(
            [question_with_prefix],
            convert_to_numpy=True,
            normalize_embeddings=False
        )
    embed_time = (datetime.now() - embed_start).total_seconds()

    client = init_chromadb()
    collection = client.get_collection(collection_name)
    total_chunks = collection.count()

    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=min(initial_k, total_chunks),
        include=['documents', 'distances', 'metadatas']
    )

    candidates = results['documents'][0]
    distances = results['distances'][0]
    metadatas = results['metadatas'][0]

    # Convert to (text, distance, metadata, quality_score) format
    retrieved_chunks = [
        (doc, dist, meta, meta.get('quality_score', 0.5))
        for doc, dist, meta in zip(candidates, distances, metadatas)
    ]

    # ⭐ APPLY POSITION-AWARE DIVERSIFICATION ⭐
    if diversity_weight > 0:
        retrieved_chunks = PositionAwareRetrieval.diversify_by_position(
            retrieved_chunks,
            total_chunks,
            diversity_weight
        )

    rerank_time = 0
    if enable_reranking:
        reranker = load_reranker()
        if reranker:
            rerank_start = datetime.now()
            # Take top candidates after diversification for reranking
            top_candidates = retrieved_chunks[:top_k * 2]
            pairs = [[question, chunk[0]] for chunk in top_candidates]
            rerank_scores = reranker.predict(pairs)
            rerank_time = (datetime.now() - rerank_start).total_seconds()

            ranked = sorted(
                zip(top_candidates, rerank_scores),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]

            final_results = []
            for (doc, dist_or_score, meta, qual), rerank_score in ranked:
                final_results.append((doc, dist_or_score, meta, qual, rerank_score))
            return final_results, embed_time, rerank_time, total_chunks

    # No reranking - just return top_k
    return [(doc, dist_or_score, meta, qual, None)
            for doc, dist_or_score, meta, qual in retrieved_chunks[:top_k]], embed_time, rerank_time, total_chunks

# MODIFIED: generate_answer with crash recovery
def generate_answer(question, context_chunks, model_name, temperature=0.7, top_p=0.9, top_k=40):
    """Generate answer with Ollama crash recovery"""
    context = "\n\n".join([chunk for chunk, _, _, _, _ in context_chunks])

    prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

    # Force unload before starting
    OllamaManager.force_unload_all()

    llm = load_llm(model_name, temperature=temperature, top_p=top_p, top_k=top_k)
    start_time = datetime.now()

    # Use crash recovery
    response, success = OllamaManager.invoke_with_recovery(llm, prompt)
    response_time = (datetime.now() - start_time).total_seconds()

    if not success:
        st.error(f"❌ LLM generation failed: {response}")
        return None, response_time

    return response, response_time

# ============================================================================
# SIDEBAR WITH GPU MEMORY DISPLAY (from v11/v12 - PRESERVED with additions)
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    if CUDA_AVAILABLE:
        st.subheader("🚀 GPU Acceleration")

        # Show GPU memory status
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            free_gb = free_mem / (1024**3)
            total_gb = total_mem / (1024**3)
            used_gb = total_gb - free_gb

            st.success(f"**{GPU_NAME}**")
            st.caption(f"CUDA {GPU_COMPUTE_STR} | {total_gb:.1f}GB total")
            st.progress(used_gb / total_gb)
            st.caption(f"Used: {used_gb:.2f}GB | Free: {free_gb:.2f}GB")

            if free_gb < 0.5:
                st.warning(f"⚠️ Low GPU memory!")
                st.caption("Consider: Lower batch size or CPU mode")
        except Exception:
            st.success(f"**{GPU_NAME}**")
            st.caption(f"CUDA {GPU_COMPUTE_STR} | {GPU_MEMORY:.1f}GB")

        use_gpu = st.checkbox(
            "Enable GPU for Embeddings",
            value=st.session_state.use_gpu,
            help="Auto-manages memory and falls back gracefully"
        )
        st.session_state.use_gpu = use_gpu

        if use_gpu:
            gpu_batch_size = st.slider(
                "GPU Batch Size",
                min_value=2,
                max_value=32,
                value=st.session_state.gpu_batch_size,
                step=2,
                help="Auto-reduces if OOM. Start with 4-8 for 2GB GPU"
            )
            st.session_state.gpu_batch_size = gpu_batch_size

            selected_emb = st.session_state.selected_embedding
            if selected_emb in EMBEDDING_MODELS:
                model_vram = EMBEDDING_MODELS[selected_emb]["vram_mb"]
                batch_vram = (gpu_batch_size * 512 * 4) / 1024
                total_vram = model_vram + batch_vram
                st.caption(f"**Est. VRAM:** ~{total_vram:.0f}MB")
                if total_vram > (GPU_MEMORY * 1024 * 0.9):
                    st.warning("⚠️ May exceed VRAM!")

        st.divider()
    else:
        st.warning("⚠️ CPU-only mode")
        use_gpu = False
        gpu_batch_size = 8
        st.divider()

    # ⭐ NEW: Anti-Basin Controls ⭐
    st.subheader("🎯 Anti-Basin Controls")
    diversity_weight = st.slider(
        "Position Diversity Weight",
        min_value=0.0,
        max_value=0.5,
        value=st.session_state.diversity_weight,
        step=0.05,
        help="⭐ CRITICAL: 0.3 = 30% position weighting (prevents basin). 0.0 = standard retrieval (may cause basin)"
    )
    st.session_state.diversity_weight = diversity_weight

    if diversity_weight > 0:
        st.success(f"✅ Position-aware: {diversity_weight:.0%} diversity")
        st.caption("Prevents over-concentration at document end")
    else:
        st.error("⚠️ Standard retrieval - basin likely!")
        st.caption("Enable diversity weight to avoid basin")

    st.divider()

    # ⭐ MISSING: Document Upload Section ⭐
    st.subheader("📄 Document")
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=['pdf'],
        help="Upload a PDF to process with RAG"
    )

    if uploaded_file:
        st.info(f"📄 {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

        # Embedding model selection
        st.subheader("🧮 Embedding Model")
        selected_embedding = st.selectbox(
            "Select Embedding Model",
            options=list(EMBEDDING_MODELS.keys()),
            index=list(EMBEDDING_MODELS.keys()).index(st.session_state.selected_embedding),
            format_func=lambda x: f"{x}: {EMBEDDING_MODELS[x]['description']}"
        )
        st.session_state.selected_embedding = selected_embedding

        # Show model info
        model_info = EMBEDDING_MODELS[selected_embedding]
        st.caption(f"📊 {model_info['dimensions']}D | {model_info['vram_mb']}MB | {model_info['domain']}")

        st.divider()

        # Chunking strategy
        st.subheader("✂️ Chunking Strategy")
        selected_chunk_config = st.selectbox(
            "Chunk Configuration",
            options=list(CHUNK_CONFIGS.keys()),
            index=list(CHUNK_CONFIGS.keys()).index(st.session_state.selected_chunk_config),
            format_func=lambda x: CHUNK_CONFIGS[x]['description']
        )
        st.session_state.selected_chunk_config = selected_chunk_config

        chunk_cfg = CHUNK_CONFIGS[selected_chunk_config]
        st.caption(f"Size: {chunk_cfg['chunk_size']} chars | Overlap: {chunk_cfg['chunk_overlap']} chars")

        st.divider()

        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            st.session_state.enable_filtering = st.checkbox(
                "Enable Content Filtering",
                value=st.session_state.enable_filtering,
                help="Remove boilerplate sections from scientific papers"
            )

            st.session_state.enable_reranking = st.checkbox(
                "Enable Reranking",
                value=st.session_state.enable_reranking,
                help="Use CrossEncoder for better retrieval quality"
            )

            if 'top_k' not in st.session_state:
                st.session_state.top_k = 5

            st.session_state.top_k = st.slider(
                "Top-K Chunks",
                min_value=3,
                max_value=15,
                value=st.session_state.get('top_k', 5),
                help="Number of chunks to retrieve"
            )

        st.divider()

        # LLM settings
        st.subheader("🤖 LLM Settings")

        available_models = get_ollama_models()
        if st.session_state.query_llm not in available_models and available_models:
            st.session_state.query_llm = available_models[0]

        st.session_state.query_llm = st.selectbox(
            "Query LLM",
            options=available_models,
            index=available_models.index(st.session_state.query_llm) if st.session_state.query_llm in available_models else 0
        )

        with st.expander("🎛️ LLM Parameters"):
            st.session_state.llm_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.llm_temperature,
                step=0.1
            )

            st.session_state.llm_top_p = st.slider(
                "Top-P",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.llm_top_p,
                step=0.05
            )

            st.session_state.llm_top_k = st.slider(
                "Top-K",
                min_value=1,
                max_value=100,
                value=st.session_state.llm_top_k,
                step=1
            )

        st.divider()

        # Process button
        if st.button("📊 Process Document", type="primary", use_container_width=True):
            with st.spinner("Processing document..."):
                process_document(uploaded_file)

# ============================================================================
# MAIN CHAT (from v11/v12 with basin analysis additions)
# ============================================================================

if st.session_state.doc_processed:
    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # ⭐ NEW: Display basin analysis if available
            if "basin_analysis" in message:
                basin = message["basin_analysis"]
                if basin['detected']:
                    st.error(f"⚠️ ATTENTION BASIN ({basin['severity']} severity)")
                    st.caption(f"{basin['end_ratio']:.0%} of chunks from document end")
                else:
                    st.success("✅ No attention basin detected")

            if "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source_data in enumerate(message["sources"]):
                        # ⭐ FIX: Handle variable tuple lengths more robustly
                        if len(source_data) == 5:
                            chunk, sim, meta, qual, rerank = source_data
                        elif len(source_data) == 4:
                            # Could be (chunk, sim, meta, qual) OR (chunk, sim, qual, rerank)
                            # Check if third element is dict (metadata) or float (quality)
                            if isinstance(source_data[2], dict):
                                chunk, sim, meta, qual = source_data
                                rerank = None
                            else:
                                chunk, sim, qual, rerank = source_data
                                meta = {}
                        elif len(source_data) == 3:
                            chunk, sim, qual = source_data
                            meta = {}
                            rerank = None
                        else:
                            continue  # Skip malformed

                        # Handle cases where qual or sim might be dict/meta
                        if isinstance(qual, dict):
                            qual = qual.get('quality_score', 0.5)
                        if isinstance(sim, dict):
                            sim = 0.5

                        # Build score text safely
                        score_text = f"score: {float(sim):.3f}, qual: {float(qual):.2f}"
                        if rerank is not None:
                            score_text += f", rerank: {float(rerank):.3f}"

                        st.text_area(
                            f"Source {i+1} ({score_text})",
                            chunk, height=100,
                            key=f"src_{message.get('msg_id', i)}_{i}"
                        )

    if prompt := st.chat_input("Ask about your document..."):
        msg_id = uuid.uuid4().hex[:8]
        st.session_state.messages.append({"role": "user", "content": prompt, "msg_id": msg_id})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"Processing with {st.session_state.query_llm}..."):
                chunks_with_scores, embed_time, rerank_time, total_chunks = query_documents(
                    prompt, st.session_state.collection_name,
                    st.session_state.selected_embedding, st.session_state.use_gpu,
                    st.session_state.enable_reranking,
                    st.session_state.diversity_weight,  # ⭐ NEW PARAMETER
                    st.session_state.get('top_k', 5)
                )

                # ⭐ NEW: Analyze for attention basin
                basin_analysis = AttentionBasinDetector.analyze_chunk_distribution(
                    chunks_with_scores, total_chunks
                )

                response, response_time = generate_answer(
                    prompt, chunks_with_scores, st.session_state.query_llm,
                    temperature=st.session_state.llm_temperature,
                    top_p=st.session_state.llm_top_p,
                    top_k=st.session_state.llm_top_k
                )

                if response is None:
                    st.error("Failed to generate response")
                    st.stop()

                st.markdown(response)

                # ⭐ NEW: Display basin warning
                if basin_analysis['detected']:
                    st.error(f"⚠️ ATTENTION BASIN DETECTED ({basin_analysis['severity']} severity)")
                    st.caption(f"{basin_analysis['end_ratio']:.0%} of chunks from document end")
                    if st.session_state.diversity_weight == 0:
                        st.warning("💡 Tip: Enable Position Diversity in sidebar to prevent basin")
                else:
                    st.success("✅ No attention basin detected")

                # ⭐ NEW: Compute quality metrics
                quality_metrics = QualityMetrics.compute_metrics(response, prompt, chunks_with_scores)

                timing = f"⏱️ Total: {response_time:.2f}s | Embed: {embed_time:.3f}s"
                if rerank_time > 0:
                    timing += f" | Rerank: {rerank_time:.3f}s"
                st.caption(f"{timing} | Model: {st.session_state.query_llm}")
                st.caption(f"Quality: {quality_metrics['query_coverage']:.0%} coverage, {quality_metrics['response_words']} words")

                with st.expander("📚 Sources & Analysis"):
                    # Basin visualization
                    st.subheader("📍 Chunk Position Distribution")
                    positions = basin_analysis['positions']
                    chunk_ids = basin_analysis['chunk_ids']

                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 2))
                    ax.scatter(positions, [0]*len(positions), s=100, alpha=0.6)
                    ax.axvspan(0.8, 1.0, alpha=0.2, color='red', label='Document end (basin zone)')
                    ax.set_xlim(0, 1)
                    ax.set_ylim(-0.5, 0.5)
                    ax.set_xlabel('Document Position (0=start, 1=end)')
                    ax.set_yticks([])
                    ax.legend()
                    st.pyplot(fig)

                    st.caption(f"Retrieved chunks: {chunk_ids}")

                    st.subheader("📄 Source Chunks")
                    for i, chunk_data in enumerate(chunks_with_scores):
                        if len(chunk_data) == 5:
                            chunk, score, meta, qual, rerank = chunk_data
                        else:
                            chunk, score, meta, qual = chunk_data[:4]
                            rerank = None

                        chunk_id = meta.get('chunk_id', 0)
                        position = chunk_id / total_chunks

                        score_text = f"score: {score:.3f}, qual: {qual:.2f}, pos: {position:.1%}"
                        if rerank is not None:
                            score_text += f", rerank: {rerank:.3f}"
                        st.text_area(
                            f"Source {i+1} - Chunk {chunk_id} ({score_text})",
                            chunk, height=100,
                            key=f"src_new_{msg_id}_{i}"
                        )

                # ============================================================
                # SAVE TO DATABASE WITH COMPLETE AUDITABILITY
                # ============================================================

                # Prepare conversation settings with ALL parameters
                conversation_settings = {
                    # Retrieval settings
                    'num_sources_requested': st.session_state.get('top_k', 5),
                    'num_sources_returned': len(chunks_with_scores),
                    'top_k_initial': st.session_state.get('top_k', 5) * 4 if st.session_state.diversity_weight > 0 else (st.session_state.get('top_k', 5) * 3 if st.session_state.enable_reranking else st.session_state.get('top_k', 5)),

                    # LLM settings
                    'query_llm_name': st.session_state.query_llm,
                    'query_llm_context_window': 25000,
                    'query_llm_temperature': st.session_state.llm_temperature,
                    'query_llm_top_p': st.session_state.llm_top_p,
                    'query_llm_top_k': st.session_state.llm_top_k,

                    # Enhancement settings
                    'reranking_enabled': int(st.session_state.enable_reranking),
                    'reranker_model_name': 'BAAI/bge-reranker-base' if st.session_state.enable_reranking else None,

                    # Performance metrics
                    'response_time_seconds': response_time,
                    'embedding_time_seconds': embed_time,
                    'reranking_time_seconds': rerank_time,

                    # ⭐ NEW: Basin prevention metrics
                    'diversity_weight': st.session_state.diversity_weight,
                    'basin_detected': int(basin_analysis['detected']),
                    'basin_end_ratio': basin_analysis['end_ratio'],
                    'basin_severity': basin_analysis['severity'],

                    # ⭐ NEW: Quality metrics
                    'quality_query_coverage': quality_metrics['query_coverage'],
                    'quality_response_words': quality_metrics['response_words'],
                    'quality_substantive_density': quality_metrics['substantive_density'],
                    'quality_chunk_spread': quality_metrics['chunk_spread']
                }

                # Generate conversation ID
                conversation_id = f"conv_{uuid.uuid4().hex[:12]}"

                # Convert chunks_with_scores to save format (handle both 4 and 5 tuple)
                sources_to_save = []
                for chunk_data in chunks_with_scores:
                    if len(chunk_data) == 5:
                        chunk, score, meta, qual, rerank = chunk_data
                    elif len(chunk_data) == 4:
                        chunk, score, meta, qual = chunk_data
                        rerank = None
                    else:
                        continue  # Skip malformed
                    sources_to_save.append((chunk, score, qual, rerank))

                # Save to database (auto-exports to CSV/JSON)
                db.save_conversation(
                    conversation_id=conversation_id,
                    doc_id=st.session_state.doc_id,
                    query=prompt,
                    response=response,
                    sources=sources_to_save,
                    session_id=st.session_state.session_id,
                    settings_dict=conversation_settings
                )

                st.success("✅ Saved & auto-exported to CSV/JSON!")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": chunks_with_scores,
                    "basin_analysis": basin_analysis,
                    "quality_metrics": quality_metrics,
                    "msg_id": msg_id
                })
else:
    st.markdown("""
    ## 🔬 Complete Biomedical RAG System v13 - Basin Fixed

    ### 🎯 NEW: Attention Basin Prevention
    - **Position-aware retrieval** prevents over-concentration at document endings
    - **Real-time basin detection** with visual feedback
    - **Quality metrics** show response coverage and substantive density

    ### Intelligent GPU Management
    - **Smart Fallback**: Tries selected model on CPU before smallest model
    - **Auto Batch Adjustment**: Reduces batch size if GPU OOM
    - **Memory Monitoring**: Real-time GPU memory display
    - **GTX 1050 Compatible**: Works with 2GB VRAM

    ### Features
    - **Dual LLM Support**: Separate models for processing and queries
    - **Optimized Chunks**: Granite (2000 chars) | PubMedBERT (1600 chars)
    - **Complete Auditability**: All settings tracked and auto-exported
    - **Biomedical Filtering**: Removes boilerplate from scientific papers
    - **Crash Recovery**: Ollama crash detection and recovery

    ### Recommended for Granite 4
    - Embedding: `granite-embedding-small-r2` (⭐ basin-resistant)
    - Chunking: `granite-optimized`
    - Query LLM: `granite4:tiny-h`
    - GPU Batch: 4-8 (for 2GB GPU)
    - **Position Diversity: 0.3** (30% weighting)

    ### Recommended for Biomedical Papers
    - Embedding: `pubmedbert-ms-marco`
    - Chunking: `biomedical-precise`
    - Filtering: Enabled
    - Reranking: Enabled
    - **Position Diversity: 0.3**

    Upload a document to get started!
    """)

    with st.expander("📊 System Stats"):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Session", st.session_state.session_id[:12] + "...")
        session_stats = db.get_session_stats(st.session_state.session_id)
        col2.metric("Queries", session_stats['conversations'])
        col3.metric("Embedding", st.session_state.selected_embedding.split('-')[0])
        col4.metric("Query LLM", st.session_state.query_llm.split(':')[0])
