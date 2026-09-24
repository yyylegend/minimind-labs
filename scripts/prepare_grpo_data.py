"""建立排除 held-out 问题的 GRPO 训练副本，并记录 SHA-256。"""

import argparse
import hashlib
import json
import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Any


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_user_prompt(messages: Any, *, require_assistant: bool) -> str:
    if not isinstance(messages, list) or not messages:
        raise ValueError("每条记录必须包含非空 conversations 列表")
    if require_assistant and messages[-1].get("role") != "assistant":
        raise ValueError("RLAIF 训练记录的最后一条消息必须是 assistant")

    context = messages[:-1] if messages[-1].get("role") == "assistant" else messages
    user_messages = [
        str(message.get("content", ""))
        for message in context
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if not user_messages or not user_messages[-1].strip():
        raise ValueError("记录中找不到非空的 user prompt")
    return user_messages[-1]


def _prompt_key(prompt: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", prompt).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _heldout_prompt(row: dict[str, Any]) -> str:
    if isinstance(row.get("prompt"), str):
        return row["prompt"]
    if "conversations" in row:
        return _latest_user_prompt(row["conversations"], require_assistant=False)
    raise ValueError("held-out JSONL 每行必须包含 prompt 字符串或 conversations 列表")


def prepare_grpo_data(
    source_path: str,
    heldout_path: str,
    output_path: str,
    manifest_path: str,
    source_url: str,
    source_revision: str,
    source_license: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(source_path).resolve()
    heldout = Path(heldout_path).resolve()
    output = Path(output_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    if not source.is_file() or not heldout.is_file():
        raise FileNotFoundError("source_path 和 heldout_path 都必须是现有文件")
    if not source_url.strip() or not source_revision.strip() or not source_license.strip():
        raise ValueError("必须记录 RLAIF source URL、revision 和 license")
    if output in {source, heldout} or manifest_file in {source, heldout, output}:
        raise ValueError("输出和 manifest 路径不能覆盖输入文件或互相覆盖")
    if not overwrite and (output.exists() or manifest_file.exists()):
        raise FileExistsError("输出或 manifest 已存在；确认后可使用 --overwrite 替换")

    heldout_keys = set()
    with heldout.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                heldout_keys.add(_prompt_key(_heldout_prompt(row)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"held-out JSONL 第 {line_number} 行无效：{exc}") from exc
    if not heldout_keys:
        raise ValueError("held-out JSONL 中没有有效 prompt")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output.parent,
            prefix=f"{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as prepared:
            temporary_output = Path(prepared.name)
            rows_scanned = 0
            rows_removed = 0
            with source.open("r", encoding="utf-8", newline="") as raw:
                for line_number, line in enumerate(raw, start=1):
                    if not line.strip():
                        continue
                    rows_scanned += 1
                    try:
                        row = json.loads(line)
                        prompt = _latest_user_prompt(
                            row.get("conversations"), require_assistant=True
                        )
                    except (json.JSONDecodeError, ValueError) as exc:
                        raise ValueError(f"RLAIF JSONL 第 {line_number} 行无效：{exc}") from exc
                    if _prompt_key(prompt) in heldout_keys:
                        rows_removed += 1
                        continue
                    prepared.write(line if line.endswith("\n") else line + "\n")

        if rows_scanned == rows_removed:
            raise ValueError("去重后没有可训练记录；拒绝生成空训练集")

        os.replace(temporary_output, output)
        temporary_output = None
        result = {
            "decontamination": "exact normalized match of the final user message",
            "source_url": source_url,
            "source_revision": source_revision,
            "source_license": source_license,
            "source_path": str(source),
            "source_sha256": _file_sha256(source),
            "heldout_path": str(heldout),
            "heldout_sha256": _file_sha256(heldout),
            "training_path": str(output),
            "training_sha256": _file_sha256(output),
            "rows_scanned": rows_scanned,
            "rows_removed": rows_removed,
            "rows_kept": rows_scanned - rows_removed,
            "heldout_prompt_count": len(heldout_keys),
        }
        temporary_manifest = manifest_file.with_name(manifest_file.name + ".tmp")
        try:
            temporary_manifest.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(temporary_manifest, manifest_file)
        finally:
            temporary_manifest.unlink(missing_ok=True)
        return result
    finally:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="去除 GRPO 训练数据中的 held-out prompt 并记录哈希")
    parser.add_argument("--source", required=True, help="只读的官方 RLAIF JSONL")
    parser.add_argument("--heldout", required=True, help="JSONL：每行含 prompt 或 conversations")
    parser.add_argument("--output", required=True, help="去重后的训练 JSONL 副本")
    parser.add_argument("--manifest", required=True, help="数据来源、过滤数量及 SHA-256 记录")
    parser.add_argument("--source_url", required=True, help="官方数据集来源 URL")
    parser.add_argument("--source_revision", required=True, help="数据集 revision 或 commit")
    parser.add_argument("--source_license", required=True, help="数据集许可证名称")
    parser.add_argument("--overwrite", action="store_true", help="允许替换已存在的 output/manifest")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = prepare_grpo_data(
        args.source,
        args.heldout,
        args.output,
        args.manifest,
        args.source_url,
        args.source_revision,
        args.source_license,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
