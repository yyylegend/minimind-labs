"""Compare baseline and specialized MiniMind checkpoints on held-out tasks."""

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import resource
except ImportError:  # pragma: no cover - the code runner is Linux-only
    resource = None


GENERAL_PROMPTS = [
    "你有什么特长？",
    "为什么天空是蓝色的？",
    "请用 Python 写一个计算斐波那契数列的函数。",
    "解释一下光合作用的基本过程。",
    "如果明天下雨，我应该如何出门？",
    "比较一下猫和狗作为宠物的优缺点。",
    "解释什么是机器学习。",
    "推荐一些中国美食，并说明它们各自的特点。",
]
NUMBER_PATTERN = re.compile(
    r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?:\s*/\s*[+-]?\d+(?:\.\d+)?)?%?"
)
ANSWER_MARKER = re.compile(
    r"(?:final\s+answer|answer\s*(?:is|:)|####|答案\s*(?:是|为|：|:)|最终答案)",
    re.IGNORECASE,
)
CODE_FENCE = re.compile(r"```(?:python|py)?\s*([\s\S]*?)```", re.IGNORECASE)


def _prompt_hash(prompt: str) -> bytes:
    return hashlib.sha256(prompt.strip().encode("utf-8")).digest()


def _load_training_prompts(path: Path) -> set[bytes]:
    """Store fixed-size prompt fingerprints instead of the full 1.6 GB text set."""
    prompts: set[bytes] = set()
    rows = 0
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            for message in row.get("conversations", []):
                if message.get("role") == "user":
                    prompts.add(_prompt_hash(str(message.get("content", ""))))
            if rows % 100000 == 0:
                print(f"decontamination: {path.name} scanned={rows}", flush=True)
    return prompts


def _render_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        tools=None,
    )


def _iter_parquet_rows(directory: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("读取评测 Parquet 需要 pyarrow。") from exc

    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"目录中没有 Parquet 文件：{directory}")
    row_index = 0
    for path in files:
        reader = parquet.ParquetFile(path)
        for batch in reader.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                row_index += 1
                yield row_index, row


def _select_tasks(
    directory: Path,
    build_task: Callable[[int, dict[str, Any]], dict[str, Any] | None],
    tokenizer,
    excluded_prompts: set[bytes],
    count: int,
    max_prompt_tokens: int,
    seed: int,
    label: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    scanned = 0
    eligible = 0
    next_progress = 10000

    def consume_batch(batch: list[dict[str, Any]]) -> None:
        nonlocal eligible
        if not batch:
            return
        rendered = [_render_prompt(tokenizer, task["eval_prompt"]) for task in batch]
        tokenized = tokenizer(rendered, add_special_tokens=False, truncation=False)
        for task, token_ids in zip(batch, tokenized["input_ids"]):
            if len(token_ids) > max_prompt_tokens:
                continue
            eligible += 1
            if len(reservoir) < count:
                reservoir.append(task)
                continue
            replacement = rng.randrange(eligible)
            if replacement < count:
                reservoir[replacement] = task

    for row_index, row in _iter_parquet_rows(directory):
        scanned += 1
        task = build_task(row_index, row)
        if task is not None and _prompt_hash(task["prompt"]) not in excluded_prompts:
            pending.append(task)
        if len(pending) >= 128:
            consume_batch(pending)
            pending = []
        if scanned >= next_progress:
            consume_batch(pending)
            pending = []
            print(f"{label}: scanned={scanned}, eligible={eligible}", flush=True)
            next_progress += 10000
    consume_batch(pending)
    print(f"{label}: held_out={eligible}, selected={len(reservoir)}/{count}")
    return reservoir


def _numeric_value(text: str) -> Fraction | None:
    matches = list(NUMBER_PATTERN.finditer(text))
    if not matches:
        return None
    markers = list(ANSWER_MARKER.finditer(text))
    marked_match = NUMBER_PATTERN.search(text[markers[-1].end() :]) if markers else None
    token = (marked_match or matches[-1]).group().replace(",", "").replace(" ", "")
    is_percent = token.endswith("%")
    if is_percent:
        token = token[:-1]
    try:
        value = Fraction(token)
        return value / 100 if is_percent else value
    except (ValueError, ZeroDivisionError):
        return None


def _build_math_task(row_index: int, row: dict[str, Any]) -> dict[str, Any] | None:
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    target = _numeric_value(answer)
    if not question or target is None:
        return None
    prompt = question + "\n\nGive the final numeric answer clearly on the last line."
    return {
        "id": f"math-{row_index}",
        "task": "math",
        "prompt": question,
        "eval_prompt": prompt,
        "gold_answer": answer,
        "gold_value": str(target),
    }


def _build_code_task(row_index: int, row: dict[str, Any]) -> dict[str, Any] | None:
    try:
        if float(row.get("test_reward") or 0) < 1.0:
            return None
    except (TypeError, ValueError):
        return None
    problem = str(row.get("problem") or "").strip()
    verification = row.get("verification_info") or {}
    cases = verification.get("test_cases", []) if isinstance(verification, dict) else []
    if not problem or not cases:
        return None
    if any(case.get("type") != "stdin_stdout" for case in cases):
        return None
    prompt = problem + "\n\nReturn only a complete Python 3 program."
    return {
        "id": str(row.get("problem_id") or f"code-{row_index}"),
        "task": "code",
        "prompt": problem,
        "eval_prompt": prompt,
        "test_cases": cases,
    }


def _load_model(checkpoint: Path, tokenizer, args, device, dtype):
    import torch

    from minimind_lab import MiniMindConfig, MiniMindForCausalLM

    config = MiniMindConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=args.max_seq_len,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or 0,
    )
    model = MiniMindForCausalLM(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(state, strict=True)
    return model.to(device=device, dtype=dtype).eval()


def _generate(model, tokenizer, task: dict[str, Any], device, max_new_tokens: int) -> str:
    import torch

    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": task["eval_prompt"]}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        tools=None,
    ).to(device)
    generated = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_p=1.0,
        do_sample=False,
    )
    return tokenizer.decode(generated[0, prompt_ids.shape[-1] :], skip_special_tokens=True).strip()


def _extract_code(answer: str) -> str:
    blocks = CODE_FENCE.findall(answer)
    return blocks[0].strip() if blocks else answer.strip()


def _set_child_limits() -> None:
    if resource is None:
        return
    resource.setrlimit(resource.RLIMIT_CPU, (3, 4))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _outputs_match(actual: str, expected: str) -> bool:
    actual_tokens, expected_tokens = actual.split(), expected.split()
    if len(actual_tokens) != len(expected_tokens):
        return False
    for actual_token, expected_token in zip(actual_tokens, expected_tokens):
        try:
            left, right = float(actual_token), float(expected_token)
        except ValueError:
            if actual_token.casefold() != expected_token.casefold():
                return False
        else:
            if not math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6):
                return False
    return True


def _bwrap_command(bwrap: str, code: str, temp_dir: str) -> list[str]:
    return [
        bwrap,
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-net",
        "--unshare-uts",
        "--die-with-parent",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/bin", "/bin",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64",
        "--dir", "/etc",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "--dir", "/workspace",
        "--bind", temp_dir, "/workspace",
        "--chdir", "/workspace",
        "--clearenv",
        "--setenv", "HOME", "/tmp",
        "--setenv", "PATH", "/usr/bin:/bin",
        "--setenv", "PYTHONIOENCODING", "utf-8",
        "--",
        sys.executable,
        "-I",
        "-S",
        "-c",
        code,
    ]


def _check_bwrap(bwrap: str) -> None:
    command = [
        bwrap,
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-net",
        "--unshare-uts",
        "--die-with-parent",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/bin", "/bin",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64",
        "--dir", "/etc",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "--clearenv",
        "--setenv", "HOME", "/tmp",
        "--setenv", "PATH", "/usr/bin:/bin",
        "--setenv", "PYTHONIOENCODING", "utf-8",
        "--",
        sys.executable,
        "-I",
        "-S",
        "-c",
        "pass",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("bubblewrap 无法启动；代码评测未执行。") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"bubblewrap 隔离环境不可用；代码评测未执行：{detail}")


def _run_code_case(code: str, test_case: dict[str, Any], timeout: float, bwrap: str) -> bool:
    if os.name != "posix" or resource is None:
        raise RuntimeError("代码测试需要 Linux/POSIX 资源限制，请在服务器 Linux 环境运行。")

    with tempfile.TemporaryDirectory(prefix="minimind-code-eval-") as temp_dir:
        os.chmod(temp_dir, 0o777)
        with tempfile.TemporaryFile(mode="w+b") as stdout_file:
            try:
                process = subprocess.Popen(
                    _bwrap_command(bwrap, code, temp_dir),
                    stdin=subprocess.PIPE,
                    stdout=stdout_file,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    preexec_fn=_set_child_limits,
                )
                try:
                    process.communicate(
                        input=str(test_case.get("input", "")).encode("utf-8"),
                        timeout=timeout,
                    )
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    process.wait()
                    return False
                if process.returncode != 0:
                    return False
                stdout_file.seek(0)
                actual = stdout_file.read(1024 * 1024 + 1).decode("utf-8", errors="replace")
                if len(actual) > 1024 * 1024:
                    return False
                expected = str(test_case.get("output", ""))
                return _outputs_match(actual, expected)
            except (OSError, ValueError, subprocess.SubprocessError):
                return False


def _score_code(answer: str, cases: list[dict[str, Any]], timeout: float, bwrap: str) -> dict[str, Any]:
    code = _extract_code(answer)
    if not code:
        return {"passed": 0, "checked": 0, "total": len(cases), "all_passed": False}
    passed = 0
    checked = 0
    for case in cases:
        checked += 1
        if _run_code_case(code, case, timeout, bwrap):
            passed += 1
        else:
            break
    return {
        "passed": passed,
        "checked": checked,
        "total": len(cases),
        "all_passed": passed == len(cases),
    }


def _run_model(label: str, checkpoint: Path, tasks: list[dict[str, Any]], tokenizer, args, device, dtype):
    import torch

    print(f"加载 {label}: {checkpoint}", flush=True)
    model = _load_model(checkpoint, tokenizer, args, device, dtype)
    outputs = []
    with torch.inference_mode():
        for index, task in enumerate(tasks, start=1):
            outputs.append(_generate(model, tokenizer, task, device, args.max_new_tokens))
            if index % 10 == 0 or index == len(tasks):
                print(f"{label}: generated={index}/{len(tasks)}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_checkpoint", type=Path, required=True)
    parser.add_argument("--candidate_checkpoint", type=Path, required=True)
    parser.add_argument(
        "--train_data",
        type=Path,
        nargs="+",
        required=True,
        help="从评测题中排除 user prompt 的训练 JSONL，可传多个文件",
    )
    parser.add_argument("--code_dir", type=Path, default=Path("data/raw/code/data"))
    parser.add_argument("--math_dir", type=Path, default=Path("data/raw/math/data"))
    parser.add_argument("--tokenizer_path", default="../minimind/model")
    parser.add_argument("--output", type=Path, default=Path("out/evaluations/sft_comparison.json"))
    parser.add_argument("--code_samples", type=int, default=50)
    parser.add_argument("--math_samples", type=int, default=50)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--code_timeout", type=float, default=2.0)
    parser.add_argument("--skip_code_execution", action="store_true")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    for checkpoint in (args.base_checkpoint, args.candidate_checkpoint):
        if not checkpoint.is_file():
            parser.error(f"找不到文件：{checkpoint}")
    for train_data in args.train_data:
        if not train_data.is_file():
            parser.error(f"找不到训练数据：{train_data}")
    if args.code_samples <= 0 or args.math_samples <= 0 or args.max_new_tokens <= 0:
        parser.error("样本数和 max_new_tokens 必须大于 0")
    if args.code_timeout <= 0:
        parser.error("code_timeout 必须大于 0")
    if args.max_new_tokens >= args.max_seq_len:
        parser.error("max_new_tokens 必须小于 max_seq_len")
    if args.output.exists():
        parser.error(f"评测结果已存在，为避免覆盖请指定新的 --output：{args.output}")
    bwrap = None
    if not args.skip_code_execution:
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            parser.error("代码通过率需要 bubblewrap；请先安装 bubblewrap，或指定 --skip_code_execution。")
        try:
            _check_bwrap(bwrap)
        except RuntimeError as exc:
            parser.error(str(exc))

    import torch
    from transformers import AutoTokenizer
    from trainer.trainer_utils import resolve_device

    device = resolve_device(args.device)
    if args.dtype == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        parser.error("当前 GPU 不支持 BF16，请改用 --dtype float16")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    if device.type != "cuda":
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    training_prompts: set[str] = set()
    for train_data in args.train_data:
        training_prompts.update(_load_training_prompts(train_data))
    max_prompt_tokens = args.max_seq_len - args.max_new_tokens

    general_tasks = [
        {
            "id": f"general-{index + 1}",
            "task": "general",
            "prompt": prompt,
            "eval_prompt": prompt,
        }
        for index, prompt in enumerate(GENERAL_PROMPTS)
    ]
    math_tasks = _select_tasks(
        args.math_dir,
        _build_math_task,
        tokenizer,
        training_prompts,
        args.math_samples,
        max_prompt_tokens,
        args.seed,
        "math",
    )
    code_tasks = _select_tasks(
        args.code_dir,
        _build_code_task,
        tokenizer,
        training_prompts,
        args.code_samples,
        max_prompt_tokens,
        args.seed + 1,
        "code",
    )
    tasks = general_tasks + math_tasks + code_tasks
    if not math_tasks or not code_tasks:
        parser.error("没有得到可评测的 math/code 样本，请检查路径和训练集排除结果。")

    baseline_answers = _run_model(
        "baseline", args.base_checkpoint, tasks, tokenizer, args, device, dtype
    )
    candidate_answers = _run_model(
        "candidate", args.candidate_checkpoint, tasks, tokenizer, args, device, dtype
    )

    examples = []
    for task, baseline, candidate in zip(tasks, baseline_answers, candidate_answers):
        item = {
            "id": task["id"],
            "task": task["task"],
            "prompt": task["prompt"],
            "baseline_answer": baseline,
            "candidate_answer": candidate,
        }
        if task["task"] == "math":
            target = Fraction(task["gold_value"])
            item["gold_answer"] = task["gold_answer"]
            item["baseline_correct"] = _numeric_value(baseline) == target
            item["candidate_correct"] = _numeric_value(candidate) == target
        elif task["task"] == "code":
            item["test_cases"] = len(task["test_cases"])
            if bwrap is not None:
                item["baseline_code"] = _score_code(
                    baseline, task["test_cases"], args.code_timeout, bwrap
                )
                item["candidate_code"] = _score_code(
                    candidate, task["test_cases"], args.code_timeout, bwrap
                )
        examples.append(item)

    math_examples = [item for item in examples if item["task"] == "math"]
    code_examples = [item for item in examples if item["task"] == "code"]
    summary = {
        "general_prompts": len(general_tasks),
        "math_exact_match": {
            "baseline": sum(item["baseline_correct"] for item in math_examples) / len(math_examples),
            "candidate": sum(item["candidate_correct"] for item in math_examples) / len(math_examples),
            "samples": len(math_examples),
        },
    }
    if bwrap is not None:
        baseline_code_pass = sum(item["baseline_code"]["all_passed"] for item in code_examples)
        candidate_code_pass = sum(item["candidate_code"]["all_passed"] for item in code_examples)
        summary["code_pass_at_1"] = {
            "baseline": baseline_code_pass / len(code_examples),
            "candidate": candidate_code_pass / len(code_examples),
            "samples": len(code_examples),
        }
    else:
        summary["code_pass_at_1"] = {"status": "skipped", "samples": len(code_examples)}
    report = {
        "baseline_checkpoint": str(args.base_checkpoint),
        "candidate_checkpoint": str(args.candidate_checkpoint),
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "summary": summary,
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"完整并排结果：{args.output}")


if __name__ == "__main__":
    main()
