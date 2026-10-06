"""
Question -> retrieval (top candidates) -> BGE rerank -> local Transformers model -> answer

The first run downloads the model from Hugging Face.

Run:  python generation.py
      python generation.py "When was ChatGPT released?"
"""

import sys
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from retrival import hybrid_search, load_index, rerank

LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TOP_K = 3
RETRIEVAL_TOP_K = 20

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a document.
Rules:
- Answer ONLY using the context given below.
- If the answer is not in the context, say: "I couldn't find this in the document."
- Do not use outside knowledge and do not make things up.
- Cite the page number for each factual claim, like [page 2]."""


@lru_cache(maxsize=1)
def load_generator():
    print(f"Loading local language model {LLM_MODEL} (first run downloads it)...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def build_prompt(question: str, hits: list[dict]) -> list[dict[str, str]]:
    """Put the retrieved chunks and question into a chat conversation."""
    context = "\n\n".join(
        f"[chunk {hit['chunk_id']}, page {hit['page']}]\n{hit['text']}"
        for hit in hits
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def generate(question: str, hits: list[dict]) -> str:
    """Generate and print a document-grounded answer locally."""
    tokenizer, model, device = load_generator()
    prompt = tokenizer.apply_chat_template(
        build_prompt(question, hits),
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer_tokens = output[0, inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    print(answer)
    return answer


def ask(question: str, collection, model):
    hits = hybrid_search(question, collection, model, top_k=TOP_K)
    hits = rerank(question, hits, top_k=TOP_K)

    print("\nSources:")
    for hit in hits:
        print(f"  chunk {hit['chunk_id']}, page {hit['page']}  (score {hit['score']:.3f})")

    print("\nAnswer:")
    answer = generate(question, hits)
    print("-" * 60)
    return answer


def main():
    collection, model = load_index()
    print(f"LLM: {LLM_MODEL} (local Transformers)")

    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]), collection, model)
        return

    print("\nAsk a question (type 'exit' to quit)")
    while True:
        question = input("\n> ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if question:
            ask(question, collection, model)


if __name__ == "__main__":
    main()