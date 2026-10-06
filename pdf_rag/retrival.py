"""
Question -> Qwen3 query embedding -> search ChromaDB -> BGE rerank -> top matching chunks

Run embedding.py first (it creates chroma_db/).

Run:  python search.py                          # ask questions in a loop
      python search.py "When was ChatGPT released?"
"""

import json
import math
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

DB_PATH = "chroma_db"
COLLECTION_NAME = "chatgpt"
CHUNKS_PATH = Path("chatgpt_chunks.jsonl")
TOP_K = 3                     # final chunks sent to the LLM
RETRIEVAL_TOP_K = 20          # retrieve more candidates first, then rerank
RERANKER_MODEL = "BAAI/bge-reranker-base"


@lru_cache(maxsize=1)
def load_reranker():
    print(f"Loading reranker {RERANKER_MODEL}...")
    return CrossEncoder(RERANKER_MODEL, max_length=512)


def load_index():
    """Open the existing collection and load the SAME model used in embedding.py."""
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)   # get, not create

    model_name = collection.metadata["model"]             # saved by embedding.py
    print(f"Collection '{COLLECTION_NAME}': {collection.count()} chunks")
    print(f"Loading model {model_name}...")
    model = SentenceTransformer(model_name)
    return collection, model


def search(question: str, collection, model, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
    # Questions use the "query" prompt; chunks were embedded without a prompt
    q_vector = model.encode(question, prompt_name="query", normalize_embeddings=True)

    result = collection.query(
        query_embeddings=[q_vector.tolist()],
        n_results=top_k,
    )

    # Chroma returns lists of lists (one inner list per question); we sent one question
    hits = []
    for doc, meta, dist in zip(result["documents"][0],
                               result["metadatas"][0],
                               result["distances"][0]):
        hits.append({
            "chunk_id": meta["chunk_id"],
            "page": meta["page"],
            "score": 1 - dist,          # cosine distance -> similarity (higher = better)
            "text": doc,
        })
    return hits


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def load_chunk_texts() -> list[dict]:
    if not CHUNKS_PATH.exists():
        return []

    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            chunks.append({
                "chunk_id": int(item["id"]),
                "page": int(item["page"]),
                "text": item["text"],
            })
    return chunks


def lexical_search(question: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
    """Simple BM25-style lexical score over stored chunks."""
    chunks = load_chunk_texts()
    if not chunks:
        return []

    q_terms = [t for t in _tokenize(question) if len(t) > 1]
    if not q_terms:
        return []

    doc_freq = Counter()
    for chunk in chunks:
        doc_freq.update(set(_tokenize(chunk["text"])))

    total_docs = len(chunks)
    scored = []
    for chunk in chunks:
        terms = _tokenize(chunk["text"])
        term_counts = Counter(terms)
        score = 0.0
        for term in q_terms:
            if term not in term_counts:
                continue
            df = doc_freq.get(term, 1)
            idf = math.log((1 + total_docs) / (1 + df)) + 1.0
            score += term_counts[term] * idf

        if score > 0:
            scored.append({
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "score": float(score),
                "text": chunk["text"],
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def hybrid_search(question: str, collection, model, top_k: int = TOP_K, vector_weight: float = 0.7, lexical_weight: float = 0.3) -> list[dict]:
    """Combine semantic retrieval with keyword-based relevance."""
    vector_hits = search(question, collection, model, top_k=max(top_k * 4, 10))
    lexical_hits = lexical_search(question, top_k=max(top_k * 4, 10))

    if not vector_hits:
        return lexical_hits[:top_k]
    if not lexical_hits:
        return vector_hits[:top_k]

    lexical_map = {hit["chunk_id"]: hit for hit in lexical_hits}
    max_lex = max(hit["score"] for hit in lexical_hits)

    combined = {}
    for hit in vector_hits:
        chunk_id = hit["chunk_id"]
        combined[chunk_id] = {
            "chunk_id": chunk_id,
            "page": hit["page"],
            "text": hit["text"],
            "score": vector_weight * float(hit["score"]),
        }

    for hit in lexical_hits:
        chunk_id = hit["chunk_id"]
        lex_score = float(hit["score"]) / max_lex if max_lex > 0 else 0.0
        if chunk_id in combined:
            combined[chunk_id]["score"] += lexical_weight * lex_score
        else:
            combined[chunk_id] = {
                "chunk_id": chunk_id,
                "page": hit["page"],
                "text": hit["text"],
                "score": lexical_weight * lex_score,
            }

    result = list(combined.values())
    result.sort(key=lambda x: x["score"], reverse=True)
    return result[:top_k]


def rerank(question: str, hits: list[dict], top_k: int = TOP_K) -> list[dict]:
    """Reorder retrieved chunks with a stronger relevance model."""
    if not hits:
        return []

    pairs = [[question, hit["text"]] for hit in hits]
    scores = load_reranker().predict(pairs)

    reranked = []
    for hit, score in zip(hits, scores):
        item = dict(hit)
        item["rerank_score"] = float(score)
        item["score"] = float(score)  # keep a single score name for downstream code
        reranked.append(item)

    return sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


def show(question: str, hits: list[dict]):
    print(f"\nQuestion: {question}")
    for rank, h in enumerate(hits, start=1):
        score = h.get("rerank_score", h["score"])
        print(f"\n#{rank}  chunk {h['chunk_id']}  score {score:.3f}")
        print(h["text"][:400])
    print("\n" + "-" * 60)


def main():
    collection, model = load_index()

    # One question from the command line
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        hits = hybrid_search(question, collection, model, top_k=TOP_K)
        show(question, rerank(question, hits, top_k=TOP_K))
        return

    # Otherwise keep asking until the user types exit
    print("\nAsk a question (type 'exit' to quit)")
    while True:
        question = input("\n> ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if question:
            hits = hybrid_search(question, collection, model, top_k=TOP_K)
            show(question, rerank(question, hits, top_k=TOP_K))


if __name__ == "__main__":
    main()