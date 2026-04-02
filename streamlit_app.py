"""
Research assistant UI (Step 1: upload → ask → answer + sources).

Run from project root:
  cd /path/to/Learning_basics
  streamlit run streamlit_app.py

Why imports live here: Streamlit runs this file as the main script, so we ensure the
project root is on sys.path so `app.*` matches how FastAPI loads the same package.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent of this file — same folder layout FastAPI expects for `app`
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load secrets before importing Generator (reads GOOGLE_API_KEY).
# Try app/.env first (where this repo often keeps keys), then repo-root .env.
load_dotenv(_ROOT / "app" / ".env")
load_dotenv(_ROOT / ".env")

import streamlit as st

# Streamlit Community Cloud: set GOOGLE_API_KEY in app Settings → Secrets (not in git).
# Generator uses os.getenv; map dashboard secrets into the environment.
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

from app.db.vector_store import VectorStore
from app.services.document_loader import load_and_chunk_pdf
from app.services.generator import Generator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _get_clients():
    """One embedding model + one generator per session — avoids reloading on every widget interaction."""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()
    if "generator" not in st.session_state:
        st.session_state.generator = Generator()
    return st.session_state.vector_store, st.session_state.generator


def main() -> None:
    st.set_page_config(page_title="Research assistant", layout="wide")
    st.title("Research assistant")
    st.caption("Upload PDFs, ask questions, inspect retrieved sources.")

    vs, gen = _get_clients()

    uploaded = st.file_uploader(
        "PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Each file is chunked and indexed; duplicate filenames get a numeric prefix so chunks do not overwrite each other.",
    )

    if uploaded:
        for idx, file in enumerate(uploaded):
            # Unique "paper" id so two files with the same name do not share Chroma ids.
            paper_key = f"{idx}_{file.name}"
            tmp = Path(_ROOT) / f"_st_upload_{paper_key.replace('/', '_')}"
            tmp.write_bytes(file.getvalue())
            try:
                chunks = load_and_chunk_pdf(str(tmp))
                if not chunks:
                    st.warning(f"No text extracted from **{file.name}** (empty or unreadable).")
                else:
                    vs.add_documents(chunks, paper=paper_key)
                    st.success(f"Indexed **{file.name}**: {len(chunks)} chunk(s).")
            finally:
                tmp.unlink(missing_ok=True)

    try:
        n_docs = vs.collection.count()
    except Exception:
        n_docs = 0

    query = st.text_input("Question", placeholder="Ask something about your uploaded documents…")

    if st.button("Answer", type="primary"):
        if not query.strip():
            st.info("Enter a question.")
        elif n_docs == 0:
            st.warning("No documents in the index yet. Upload at least one PDF first.")
        else:
            t0 = time.perf_counter()
            sources = vs.hybrid_search(query.strip(), n_results=st.session_state.get("top_k", 5))
            latency_retrieval = time.perf_counter() - t0

            if not sources:
                st.warning("No relevant information found in uploaded documents (retrieval returned no chunks).")
                logger.info("query=%r retrieval_ms=%.1f chunks=0", query, latency_retrieval * 1000)
            else:
                context = [s["text"] for s in sources]
                t1 = time.perf_counter()
                try:
                    answer = gen.generate(query.strip(), context)
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    logger.exception("generator failed")
                    return
                latency_gen = time.perf_counter() - t1

                logger.info(
                    "query=%r retrieval_ms=%.1f gen_ms=%.1f n_sources=%d",
                    query,
                    latency_retrieval * 1000,
                    latency_gen * 1000,
                    len(sources),
                )

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Sources (retrieved chunks)")
                for i, s in enumerate(sources, start=1):
                    with st.expander(f"Source {i} — {s.get('paper', 'unknown')} (chunk {s.get('chunk_id', '?')})"):
                        st.caption(f"score: {s.get('score', 0):.4f}")
                        st.write(s.get("text", ""))
                st.caption(
                    f"Retrieval ~{latency_retrieval * 1000:.0f} ms · generation ~{latency_gen * 1000:.0f} ms"
                )


if __name__ == "__main__":
    main()
