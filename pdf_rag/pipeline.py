"""
Full RAG pipeline in one command:

  1. PDF -> Markdown        (parse_pdf.py)
  2. Markdown -> chunks     (chunking_markdown.py)
  3. chunks -> ChromaDB     (embedding.py)
  4. Question -> answer     (retrival.py + generation.py)

Steps 1-3 are skipped if their output already exists (they are slow).
If a step runs again, every step after it runs again too.

Run:  python pipeline.py                         # build (if needed) + ask questions
      python pipeline.py "When was ChatGPT released?"
      python pipeline.py --force                 # rebuild everything from the PDF
"""

import argparse
from pathlib import Path

from . import chunking_markdown, embedding, generation, parse_pdf
from .retrival import load_index

PDF_PATH = Path("data/chatgpt.pdf")

# (step name, output it creates, function that runs it)
STEPS = [
    ("1. PDF -> Markdown", Path("chatgpt.md"), parse_pdf.main),
    ("2. Markdown -> chunks", Path("chatgpt_chunks.jsonl"), chunking_markdown.main),
    ("3. chunks -> ChromaDB", Path("chroma_db"), embedding.main),
]


def build(force: bool):
    rerun = force
    for name, output, run in STEPS:
        if rerun or not output.exists():
            print(f"\n===== {name} =====")
            run()
            rerun = True          # this step changed, so later steps must run too
        else:
            print(f"Skipping {name} ({output} already exists)")


def ingest_pdf() -> None:
    """Rebuild the document index from the PDF saved at PDF_PATH."""
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    build(force=True)


def run_query(question: str, force: bool = False) -> str:
    """Build the pipeline if needed, then answer a single question."""
    if not PDF_PATH.exists():
        raise SystemExit(f"PDF not found: {PDF_PATH}")

    build(force)

    collection, model = load_index()
    print(f"LLM: {generation.LLM_MODEL} (local Transformers)")
    return generation.ask(question, collection, model)


def main():
    ap = argparse.ArgumentParser(description="Run the full RAG pipeline")
    ap.add_argument("question", nargs="*", help="ask one question and exit")
    ap.add_argument("--force", action="store_true", help="rebuild all steps")
    args = ap.parse_args()

    if not PDF_PATH.exists():
        raise SystemExit(f"PDF not found: {PDF_PATH}")

    if args.question:
        run_query(" ".join(args.question), force=args.force)
        return

    build(args.force)

    print("\n===== 4. Question -> answer =====")
    collection, model = load_index()
    print(f"LLM: {generation.LLM_MODEL} (local Transformers)")

    print("\nAsk a question (type 'exit' to quit)")
    while True:
        question = input("\n> ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if question:
            generation.ask(question, collection, model)


if __name__ == "__main__":
    main()