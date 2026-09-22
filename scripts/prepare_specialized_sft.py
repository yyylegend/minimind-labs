"""Convert sampled coding, math, and general data into MiniMind SFT JSONL."""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Callable, Iterable


def _message(role: str, content: str) -> dict[str, str]:
    """Emit every field expected by the project's explicit SFT schema."""
    return {
        "role": role,
        "content": content,
        "reasoning_content": "",
        "tools": "",
        "tool_calls": "",
    }


def _as_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _normalize_conversations(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list):
        return None

    conversations = []
    for raw_message in value:
        if not isinstance(raw_message, dict) or not raw_message.get("role"):
            return None
        conversations.append(
            {
                "role": str(raw_message["role"]),
                "content": _as_string(raw_message.get("content")),
                "reasoning_content": _as_string(raw_message.get("reasoning_content")),
                "tools": _as_string(raw_message.get("tools")),
                "tool_calls": _as_string(raw_message.get("tool_calls")),
            }
        )

    has_user = any(message["role"] == "user" and message["content"].strip() for message in conversations)
    has_assistant = any(
        message["role"] == "assistant"
        and (message["content"].strip() or message["tool_calls"].strip())
        for message in conversations
    )
    return conversations if has_user and has_assistant else None


def _strip_code_fence(solution: str) -> str:
    solution = solution.strip()
    solution = re.sub(r"^`{1,3}(?:python)?\s*", "", solution, count=1, flags=re.IGNORECASE)
    solution = re.sub(r"\s*`{1,3}$", "", solution, count=1)
    return solution.strip()


def _code_conversations(row: dict[str, Any]) -> list[dict[str, str]] | None:
    try:
        if float(row.get("test_reward") or 0) < 1.0:
            return None
    except (TypeError, ValueError):
        return None

    verification = row.get("verification_info") or {}
    if not isinstance(verification, dict) or not verification.get("test_cases"):
        return None

    problem = _as_string(row.get("problem")).strip()
    solution = _strip_code_fence(_as_string(row.get("gold_standard_solution")))
    if not problem or not solution:
        return None
    return [_message("user", problem), _message("assistant", solution)]


def _math_conversations(row: dict[str, Any]) -> list[dict[str, str]] | None:
    question = _as_string(row.get("question")).strip()
    answer = _as_string(row.get("answer")).strip()
    if not question or not answer:
        return None
    return [_message("user", question), _message("assistant", answer)]


def _iter_parquet_rows(directory: Path) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("读取 Parquet 需要 pyarrow；请安装项目训练依赖。") from exc

    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"目录中没有 Parquet 文件：{directory}")
    for path in files:
        reader = parquet.ParquetFile(path)
        for batch in reader.iter_batches(batch_size=256):
            yield from batch.to_pylist()


def _iter_jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到通用 SFT 数据：{path}")
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def _reservoir_sample(
    rows: Iterable[dict[str, Any]],
    convert: Callable[[dict[str, Any]], list[dict[str, str]] | None],
    count: int,
    tokenizer,
    max_seq_len: int,
    seed: int,
    source_name: str,
    batch_size: int = 256,
) -> tuple[list[list[dict[str, str]]], int, int]:
    """Sample eligible examples uniformly without holding the source dataset in RAM."""
    rng = random.Random(seed)
    reservoir: list[list[dict[str, str]]] = []
    scanned = 0
    eligible = 0
    next_progress = 10000
    pending: list[list[dict[str, str]]] = []

    def consume_batch(batch: list[list[dict[str, str]]]) -> None:
        nonlocal eligible
        if not batch:
            return
        rendered = [
            tokenizer.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=False,
            )
            for conversations in batch
        ]
        tokenized = tokenizer(rendered, add_special_tokens=False, truncation=False)
        for conversations, token_ids in zip(batch, tokenized["input_ids"]):
            if len(token_ids) > max_seq_len:
                continue
            eligible += 1
            if len(reservoir) < count:
                reservoir.append(conversations)
                continue
            replace_at = rng.randrange(eligible)
            if replace_at < count:
                reservoir[replace_at] = conversations

    for row in rows:
        scanned += 1
        conversations = convert(row)
        if conversations is None:
            continue
        pending.append(conversations)
        if len(pending) == batch_size:
            consume_batch(pending)
            pending = []
        if scanned >= next_progress:
            consume_batch(pending)
            pending = []
            print(f"{source_name}: scanned={scanned}, eligible={eligible}", flush=True)
            next_progress += 10000
    consume_batch(pending)
    return reservoir, scanned, eligible


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code_dir", type=Path, default=Path("data/raw/code/data"))
    parser.add_argument("--math_dir", type=Path, default=Path("data/raw/math/data"))
    parser.add_argument("--general_data", type=Path, default=Path("data/sft_t2t_mini.jsonl"))
    parser.add_argument("--tokenizer_path", type=str, default="../minimind/model")
    parser.add_argument("--output", type=Path, default=Path("data/sft_code_math_mix.jsonl"))
    parser.add_argument("--code_samples", type=int, default=5000)
    parser.add_argument("--math_samples", type=int, default=10000)
    parser.add_argument("--general_samples", type=int, default=5000)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for name in ("code_samples", "math_samples", "general_samples", "max_seq_len"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} 必须大于 0")
    if args.output.exists():
        parser.error(f"输出文件已存在，为避免覆盖请先改名或移走：{args.output}")

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("加载 tokenizer 需要 transformers；请安装项目训练依赖。") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    sources = (
        (
            "code",
            _iter_parquet_rows(args.code_dir),
            _code_conversations,
            args.code_samples,
            args.seed,
        ),
        (
            "math",
            _iter_parquet_rows(args.math_dir),
            _math_conversations,
            args.math_samples,
            args.seed + 1,
        ),
        (
            "general",
            _iter_jsonl_rows(args.general_data),
            lambda row: _normalize_conversations(row.get("conversations")),
            args.general_samples,
            args.seed + 2,
        ),
    )

    selected: list[tuple[str, list[dict[str, str]]]] = []
    for name, rows, convert, count, seed in sources:
        samples, scanned, eligible = _reservoir_sample(
            rows,
            convert,
            count,
            tokenizer,
            args.max_seq_len,
            seed,
            name,
        )
        print(
            f"{name}: scanned={scanned}, under_{args.max_seq_len}_tokens={eligible}, "
            f"selected={len(samples)}/{count}"
        )
        selected.extend((name, conversations) for conversations in samples)

    random.Random(args.seed).shuffle(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        for _, conversations in selected:
            output.write(json.dumps({"conversations": conversations}, ensure_ascii=False) + "\n")

    print(f"已生成 {len(selected)} 条样本：{args.output}")


if __name__ == "__main__":
    main()
