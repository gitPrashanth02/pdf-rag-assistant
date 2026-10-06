import json
from pathlib import Path

from mineru.parser import parse


def main():
    result = parse("data/chatgpt.pdf", tier="flash")
    markdown_text = result.markdown()
    json_result = result.to_json()

    Path("chatgpt.md").write_text(markdown_text, encoding="utf-8")
    json_text = (
        json_result
        if isinstance(json_result, str)
        else json.dumps(json_result, ensure_ascii=False, indent=2)
    )
    Path("chatgpt.json").write_text(json_text, encoding="utf-8")

    print(f"Characters: {len(markdown_text)}")
    print(markdown_text[:500])


if __name__ == "__main__":
    main()