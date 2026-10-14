#!/usr/bin/env python3
"""Collect replayable TVM MetaSchedule trials for an LLM-derived workload.

The native MetaSchedule JSON database is preserved verbatim.  In addition, this
script writes one self-contained JSON object per measured candidate to
``trials.jsonl``.  Each object contains the initial TIR, the complete schedule
trace, the scheduled TIR, before/after AST summaries, raw latency samples, and
build/run/replay verification status.

This example uses TVM's ``tvm.tirx`` and ``tvm.s_tir.meta_schedule`` APIs
and requires a compatible TVM build.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

SCHEMA_VERSION = "verified-shortcuts.tvm-trial.v2"


@dataclasses.dataclass(frozen=True)
class ModelPreset:
    """Static dimensions for a decoder SwiGLU subgraph."""

    name: str
    hidden_size: int
    intermediate_size: int
    default_dtype: str
    description: str
    source_url: str


MODEL_PRESETS: dict[str, ModelPreset] = {
    "llama3.1-8b": ModelPreset(
        name="meta-llama/Llama-3.1-8B",
        hidden_size=4096,
        intermediate_size=14336,
        # This TVM revision has FP16 CUDA MMA rules but no BF16 CUDA tensor
        # intrinsic wired into the default MetaSchedule tensor-core space.
        default_dtype="float16",
        description="Llama 3.1 8B decoder SwiGLU MLP",
        source_url=("https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/config.json"),
    ),
    "paper-llama2-7b": ModelPreset(
        name="example/Llama-2-7B-shape",
        hidden_size=4096,
        intermediate_size=11008,
        default_dtype="bfloat16",
        description="Llama 2 7B-shaped decoder SwiGLU MLP",
        source_url="https://huggingface.co/meta-llama/Llama-2-7b-hf",
    ),
    "tinyllama-1.1b": ModelPreset(
        name="TinyLlama/TinyLlama-1.1B",
        hidden_size=2048,
        intermediate_size=5632,
        default_dtype="float16",
        description="TVM tutorial-compatible TinyLlama decoder SwiGLU MLP",
        source_url="https://tvm.apache.org/docs/how_to/tutorials/optimize_llm.html",
    ),
    "smoke": ModelPreset(
        name="synthetic/Llama-SwiGLU-smoke",
        hidden_size=128,
        intermediate_size=352,
        default_dtype="float16",
        description="Small shape for end-to-end installation and callback tests",
        source_url="local-test-preset",
    ),
}


def utc_now() -> str:
    """Return an RFC 3339 timestamp in UTC."""

    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    """Serialize with stable separators for IDs and line-oriented output."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def to_jsonable(value: Any) -> Any:
    """Convert TVM containers/immediates and Python objects to JSON-safe values."""

    if value is None or isinstance(value, bool | int | float | str):
        return value
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(to_jsonable(key)): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [to_jsonable(item) for item in value]

    # IntImm/FloatImm, DataType, ShapeTuple values, and other TVM scalar wrappers.
    scalar_value = getattr(value, "value", None)
    if scalar_value is not None and isinstance(scalar_value, bool | int | float | str):
        return scalar_value
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)


def latency_statistics(run_secs: Sequence[Any] | None) -> dict[str, Any] | None:
    """Summarize MetaSchedule samples, retaining raw seconds elsewhere."""

    if not run_secs:
        return None
    values = [float(getattr(value, "value", value)) for value in run_secs]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        return None
    millis = [value * 1e3 for value in values]
    return {
        "unit": "ms",
        "count": len(millis),
        "mean": statistics.fmean(millis),
        "median": statistics.median(millis),
        "min": min(millis),
        "max": max(millis),
        "population_stddev": statistics.pstdev(millis) if len(millis) > 1 else 0.0,
    }


def camel_to_snake(name: str) -> str:
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value).lower()


def semantic_category(kind: str, python_line: str) -> str | None:
    """Map a low-level schedule instruction to a semantic edit."""

    if kind == "Split":
        return "loop_tiling"
    if kind == "Reorder":
        return "reorder_loops"
    if kind == "Bind":
        if "blockIdx" in python_line:
            return "bind_cuda_blocks"
        if "threadIdx" in python_line or "vthread" in python_line:
            return "bind_cuda_threads"
        return "bind_loop_axis"
    if kind in {"CacheRead", "CacheWrite", "SetScope"}:
        return "memory_hierarchy"
    if kind in {"ComputeAt", "ReverseComputeAt"}:
        return "compute_placement"
    if kind in {"ComputeInline", "ReverseComputeInline"}:
        return "inline_compute"
    if kind in {"DecomposeReduction", "RFactor"}:
        return "decompose_reduction"
    if kind == "Tensorize":
        return "tensorize"
    if kind == "Vectorize":
        return "vectorize"
    if kind == "Parallel":
        return "parallelize"
    if kind == "Unroll":
        return "unroll"
    if kind in {
        "TransformLayout",
        "TransformBlockLayout",
        "SetAxisSeparator",
        "StorageAlign",
        "ReIndex",
        "Reindex",
    }:
        return "memory_layout"
    if kind in {"Annotate", "Unannotate"}:
        return "schedule_annotation"
    if kind in {"Blockize", "AddUnitLoop", "Merge", "Fuse"}:
        return camel_to_snake(kind)
    if kind == "EnterPostproc":
        return None
    return None


TRANSFORM_LABELS: dict[str, str] = {
    "loop_tiling": "Tile loop(s)",
    "reorder_loops": "Reorder loop nesting",
    "bind_cuda_blocks": "Map loop(s) to CUDA blocks",
    "bind_cuda_threads": "Map loop(s) to CUDA threads",
    "bind_loop_axis": "Bind a loop axis",
    "memory_hierarchy": "Change the cache or storage hierarchy",
    "compute_placement": "Move computation within the loop nest",
    "inline_compute": "Inline computation",
    "decompose_reduction": "Decompose or factor a reduction",
    "tensorize": "Replace computation with a tensor intrinsic",
    "vectorize": "Vectorize a loop",
    "parallelize": "Parallelize a loop",
    "unroll": "Unroll a loop",
    "memory_layout": "Transform a buffer or block layout",
    "schedule_annotation": "Update a schedule annotation",
    "blockize": "Blockize a loop nest",
    "add_unit_loop": "Add a unit loop",
    "merge": "Merge loops",
    "fuse": "Fuse loops",
}


def group_semantic_edits(
    python_lines: Sequence[str],
    kind_names: Sequence[str],
    pure_flags: Sequence[bool],
) -> list[dict[str, Any]]:
    """Group samplers with the mutating edit that consumes their decisions.

    In particular, ``SamplePerfectTile`` followed by ``Split`` is represented as
    one ``loop_tiling`` edit.
    """

    count = min(len(python_lines), len(kind_names), len(pure_flags))
    pending_samples: list[int] = []
    edits: list[dict[str, Any]] = []
    for index in range(count):
        kind = kind_names[index]
        line = str(python_lines[index])
        is_sample = kind.startswith("Sample")
        if is_sample:
            pending_samples.append(index)
            continue
        if pure_flags[index]:
            continue

        category = semantic_category(kind, line)
        if category is None:
            pending_samples.clear()
            continue
        indices = [*pending_samples, index]
        pending_samples.clear()
        edits.append(
            {
                "edit_index": len(edits),
                "kind": category,
                "trace_instruction_indices": indices,
                "instruction_kinds": [kind_names[item] for item in indices],
                "python": [str(python_lines[item]) for item in indices],
            }
        )
    return edits


def build_transform_chain(semantic_edits: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Create a readable, ordered projection of the exact schedule trace.

    The replayable source of truth remains ``trace_json``. Each chain step
    points back to its exact trace instructions and retains their TVM schedule
    Python, while summarizing the operation in plain language.
    """

    chain: list[dict[str, Any]] = []
    for step_number, edit in enumerate(semantic_edits, start=1):
        kind = str(edit["kind"])
        instruction_kinds = [str(item) for item in edit["instruction_kinds"]]
        readable_instructions = [
            camel_to_snake(item).replace("_", " ") for item in instruction_kinds
        ]
        operation = ", then ".join(readable_instructions)
        label = TRANSFORM_LABELS.get(kind, kind.replace("_", " ").capitalize())
        summary = f"{label}: {operation}." if operation else f"{label}."
        chain.append(
            {
                "step": step_number,
                "kind": kind,
                "summary": summary,
                "trace_instruction_indices": [
                    int(item) for item in edit["trace_instruction_indices"]
                ],
                "schedule_python": [str(item) for item in edit["python"]],
            }
        )
    return chain


def atomic_write_text(path: pathlib.Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(contents, encoding="utf-8")
    os.replace(temporary, path)


def run_git(args: Sequence[str], cwd: pathlib.Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.rstrip("\n")


def tvm_source_metadata(tvm_root: pathlib.Path, work_dir: pathlib.Path) -> dict[str, Any]:
    head = run_git(["rev-parse", "HEAD"], tvm_root)
    describe = run_git(["describe", "--tags", "--always", "--dirty"], tvm_root)
    status = run_git(["status", "--porcelain=v1"], tvm_root)
    diff = run_git(["diff", "--no-ext-diff"], tvm_root)
    diff_path = None
    if diff:
        diff_path = work_dir / "tvm_worktree.diff"
        atomic_write_text(diff_path, diff + "\n")
    return {
        "root": str(tvm_root.resolve()),
        "git_commit": head,
        "git_describe": describe,
        "git_dirty": bool(status),
        "git_status": status.splitlines() if status else [],
        "git_diff_sha256": sha256_text(diff) if diff else None,
        "git_diff_file": str(diff_path.resolve()) if diff_path else None,
    }


def script_text(obj: Any) -> str:
    """Print TVMScript while tolerating API differences in printer keywords."""

    try:
        return str(obj.script(show_meta=True))
    except TypeError:
        return str(obj.script())


def structural_hash_hex(ms: Any, mod: Any) -> str:
    try:
        return str(ms.utils.shash2hex(mod))
    except Exception:  # pylint: disable=broad-exception-caught
        import tvm_ffi  # pylint: disable=import-outside-toplevel

        return f"{int(tvm_ffi.structural_hash(mod)) & ((1 << 64) - 1):016x}"


def shape_value(dim: Any) -> int | str:
    value = getattr(dim, "value", None)
    if isinstance(value, int):
        return value
    try:
        return int(dim)
    except (TypeError, ValueError):
        return str(dim)


def ast_features(tvm: Any, mod: Any, ms: Any) -> dict[str, Any]:
    """Create an interpretable, JSON-safe TensorIR AST feature summary."""

    from tvm import tirx  # pylint: disable=import-outside-toplevel

    node_histogram: Counter[str] = Counter()
    loop_kinds: Counter[str] = Counter()
    thread_bindings: Counter[str] = Counter()
    block_names: list[str] = []
    blocks: list[dict[str, Any]] = []
    buffers: dict[str, dict[str, Any]] = {}
    primfunc_count = 0

    for global_var, func in mod.functions_items():
        if not isinstance(func, tirx.PrimFunc):
            continue
        primfunc_count += 1
        for buffer in func.buffer_map.values():
            name = str(buffer.name)
            try:
                scope = str(buffer.scope())
            except (AttributeError, TypeError):
                scope = ""
            buffers[name] = {
                "name": name,
                "dtype": str(buffer.dtype),
                "shape": [shape_value(dim) for dim in buffer.shape],
                "scope": scope,
                "is_function_parameter": True,
            }

        def visit(node: Any) -> None:
            node_histogram[type(node).__name__] += 1
            if isinstance(node, tirx.For):
                kind_value = int(node.kind)
                kind_map = {
                    int(tirx.ForKind.SERIAL): "serial",
                    int(tirx.ForKind.PARALLEL): "parallel",
                    int(tirx.ForKind.VECTORIZED): "vectorized",
                    int(tirx.ForKind.UNROLLED): "unrolled",
                    int(tirx.ForKind.THREAD_BINDING): "thread_binding",
                }
                loop_kinds[kind_map.get(kind_value, str(kind_value))] += 1
                if node.thread_binding is not None:
                    thread_bindings[str(node.thread_binding.thread_tag)] += 1
            elif isinstance(node, tirx.SBlock):
                block_name = str(node.name_hint)
                block_names.append(block_name)
                blocks.append(
                    {
                        "name": block_name,
                        "iter_vars": len(node.iter_vars),
                        "reads": len(node.reads),
                        "writes": len(node.writes),
                        "alloc_buffers": len(node.alloc_buffers),
                        "annotations": to_jsonable(node.annotations),
                    }
                )
                for buffer in node.alloc_buffers:
                    name = str(buffer.name)
                    try:
                        scope = str(buffer.scope())
                    except (AttributeError, TypeError):
                        scope = ""
                    buffers[name] = {
                        "name": name,
                        "dtype": str(buffer.dtype),
                        "shape": [shape_value(dim) for dim in buffer.shape],
                        "scope": scope,
                        "is_function_parameter": False,
                    }

        tirx.stmt_functor.post_order_visit(func.body, visit)

    return {
        "structural_hash": structural_hash_hex(ms, mod),
        "primfunc_count": primfunc_count,
        "node_count": sum(node_histogram.values()),
        "node_type_histogram": dict(sorted(node_histogram.items())),
        "loop_kinds": dict(sorted(loop_kinds.items())),
        "thread_bindings": dict(sorted(thread_bindings.items())),
        "block_count": len(block_names),
        "block_names": block_names,
        "blocks": blocks,
        "buffers": sorted(buffers.values(), key=lambda item: item["name"]),
    }


def build_llama_mlp_ir(
    tvm: Any,
    relax: Any,
    nn: Any,
    preset: ModelPreset,
    batch_size: int,
    seq_len: int,
    dtype: str,
    fused_gate_up: bool,
) -> tuple[Any, list[tuple[str, Any]]]:
    """Author a static Llama SwiGLU subgraph with TVM's Relax frontend."""

    op = nn.op

    if fused_gate_up:

        class LlamaMLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate_up_proj = nn.Linear(
                    preset.hidden_size,
                    2 * preset.intermediate_size,
                    bias=False,
                    dtype=dtype,
                    out_dtype="float32",
                )
                self.down_proj = nn.Linear(
                    preset.intermediate_size,
                    preset.hidden_size,
                    bias=False,
                    dtype=dtype,
                    out_dtype="float32",
                )

            def forward(self, x):
                gate_up = self.gate_up_proj(x).astype(dtype)
                gate, up = op.split(gate_up, 2, axis=-1)
                return self.down_proj(op.silu(gate) * up).astype(dtype)

    else:

        class LlamaMLP(nn.Module):  # type: ignore[no-redef]
            def __init__(self) -> None:
                super().__init__()
                self.gate_proj = nn.Linear(
                    preset.hidden_size,
                    preset.intermediate_size,
                    bias=False,
                    dtype=dtype,
                    out_dtype="float32",
                )
                self.up_proj = nn.Linear(
                    preset.hidden_size,
                    preset.intermediate_size,
                    bias=False,
                    dtype=dtype,
                    out_dtype="float32",
                )
                self.down_proj = nn.Linear(
                    preset.intermediate_size,
                    preset.hidden_size,
                    bias=False,
                    dtype=dtype,
                    out_dtype="float32",
                )

            def forward(self, x):
                gate = self.gate_proj(x).astype(dtype)
                up = self.up_proj(x).astype(dtype)
                return self.down_proj(op.silu(gate) * up).astype(dtype)

    model = LlamaMLP()
    model.to(dtype=dtype)
    spec = {
        "forward": {
            "x": nn.spec.Tensor((batch_size, seq_len, preset.hidden_size), dtype),
        }
    }
    mod, named_params = model.export_tvm(spec)
    return mod, named_params


def lower_relax_to_tir(tvm: Any, relax: Any, mod: Any, target: Any) -> Any:
    """Use the same pre-lowering stages as TVM's official LLM tutorial."""

    with target:
        return tvm.transform.Sequential(
            [
                relax.transform.FuseTransposeMatmul(),
                relax.transform.DecomposeOpsForInference(),
                relax.transform.CanonicalizeBindings(),
                relax.get_pipeline("zero"),
            ]
        )(mod)


class TrialState:
    """Chronological counters and best-so-far state, including resumed trials."""

    def __init__(self) -> None:
        self.global_count = 0
        self.task_counts: defaultdict[str, int] = defaultdict(int)
        self.status_counts: Counter[str] = Counter()
        self.verification_counts: Counter[str] = Counter()
        self.first_measured_ms: dict[str, float] = {}
        self.best_measured_ms: dict[str, float] = {}
        self.first_verified_ms: dict[str, float] = {}
        self.best_verified_ms: dict[str, float] = {}

    def observe_existing(self, record: dict[str, Any]) -> None:
        self.global_count += 1
        workload_id = str(record.get("workload", {}).get("workload_id", "unknown"))
        self.task_counts[workload_id] += 1
        measurement = record.get("measurement", {})
        verification = record.get("verification", {})
        status = str(measurement.get("status", "unknown"))
        self.status_counts[status] += 1
        search_valid = bool(verification.get("search_valid"))
        verified = bool(verification.get("verified_for_training"))
        self.verification_counts["search_valid" if search_valid else "search_invalid"] += 1
        self.verification_counts["verified" if verified else "unverified"] += 1
        stats = measurement.get("latency")
        if status == "ok" and stats and stats.get("median") is not None:
            latency_ms = float(stats["median"])
            self.first_measured_ms.setdefault(workload_id, latency_ms)
            previous = self.best_measured_ms.get(workload_id)
            if previous is None or latency_ms < previous:
                self.best_measured_ms[workload_id] = latency_ms
            if verified:
                self.first_verified_ms.setdefault(workload_id, latency_ms)
                verified_previous = self.best_verified_ms.get(workload_id)
                if verified_previous is None or latency_ms < verified_previous:
                    self.best_verified_ms[workload_id] = latency_ms


class TrialWriter:
    """Append-only, crash-resilient sidecar for measured candidates."""

    def __init__(self, path: pathlib.Path, run_id: str, resume: bool) -> None:
        self.path = path
        self.run_id = run_id
        self.state = TrialState()
        self.start_monotonic = time.monotonic()
        self._workload_cache: dict[str, dict[str, Any]] = {}
        if resume and path.exists():
            with path.open("r", encoding="utf-8") as existing:
                for line_number, line in enumerate(existing, start=1):
                    if not line.strip():
                        continue
                    try:
                        self.state.observe_existing(json.loads(line))
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Malformed existing JSONL at {path}:{line_number}: {error}"
                        ) from error
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("a", encoding="utf-8", buffering=1)

    def close(self) -> None:
        self._stream.flush()
        self._stream.close()

    def workload_record(
        self,
        tvm: Any,
        ms: Any,
        task_name: str,
        task_weight: float,
        initial_mod: Any,
        target: Any,
        work_dir: pathlib.Path,
        store_tvm_json: bool,
    ) -> dict[str, Any]:
        ir_text = script_text(initial_mod)
        target_text = str(target)
        structural_hash = structural_hash_hex(ms, initial_mod)
        workload_key = structural_hash + target_text + ir_text
        workload_id = f"w_{sha256_text(workload_key)[:24]}"
        if workload_id in self._workload_cache:
            return self._workload_cache[workload_id]

        record = {
            "workload_id": workload_id,
            "task_name": task_name,
            "task_weight": task_weight,
            "target": target_text,
            "structural_hash": structural_hash,
            "initial_ir": ir_text,
            "initial_ast_features": ast_features(tvm, initial_mod, ms),
        }
        if store_tvm_json:
            record["initial_ir_tvm_json"] = tvm.ir.save_json(initial_mod)
        workload_file = work_dir / "workloads" / f"{workload_id}.json"
        persisted = dict(record)
        atomic_write_text(workload_file, json.dumps(persisted, indent=2, ensure_ascii=False) + "\n")
        record["workload_file"] = str(workload_file.resolve())
        self._workload_cache[workload_id] = record
        return record

    def append(self, record: dict[str, Any]) -> None:
        self._stream.write(canonical_json(record) + "\n")
        self._stream.flush()
        self.state.observe_existing(record)


def trace_json_sha256(trace_json: Any) -> str:
    """Hash TVM's dedicated, round-trippable schedule-trace JSON."""

    return sha256_text(canonical_json(to_jsonable(trace_json)))


def normalize_trace_json_for_comparison(value: Any) -> Any:
    """Normalize equivalent IndexMap encodings emitted across the C++/Python FFI boundary."""

    safe = to_jsonable(value)
    if isinstance(safe, str):
        stripped = safe.strip()
        if stripped.startswith("T.index_map(") or (
            "root_index" in stripped and "tirx.IndexMap" in stripped
        ):
            return "<tvm-index-map>"
        return safe
    if isinstance(safe, list):
        return [normalize_trace_json_for_comparison(item) for item in safe]
    if isinstance(safe, dict):
        return {
            str(key): normalize_trace_json_for_comparison(item)
            for key, item in sorted(safe.items())
        }
    return safe


def portable_trace_sha256(trace_json: Any) -> str:
    """Hash a trace while tolerating equivalent IndexMap string encodings."""

    return sha256_text(canonical_json(normalize_trace_json_for_comparison(trace_json)))


def trace_decisions_by_instruction(trace_json: Any) -> dict[int, Any]:
    """Return the stochastic decision at each serialized instruction index."""

    safe = to_jsonable(trace_json)
    if not isinstance(safe, list) or len(safe) != 2 or not isinstance(safe[1], list):
        return {}
    decisions: dict[int, Any] = {}
    for item in safe[1]:
        if isinstance(item, list) and len(item) == 2:
            decisions[int(item[0])] = item[1]
    return decisions


def trace_instructions(trace_json: Any) -> list[Any]:
    safe = to_jsonable(trace_json)
    if isinstance(safe, list) and len(safe) == 2 and isinstance(safe[0], list):
        return safe[0]
    return []


def instruction_kind_at(instructions: Sequence[Any], index: int) -> str | None:
    if 0 <= index < len(instructions):
        instruction = instructions[index]
        if isinstance(instruction, list) and instruction:
            return str(instruction[0])
    return None


def provenance_to_jsonable(
    value: Any,
    trace_json_converter: Callable[[Any], Any],
    field_name: str | None = None,
) -> Any:
    """Convert provenance while preserving TVM's dedicated trace JSON encoding."""

    if field_name is not None and (
        field_name.endswith("trace_json") or field_name == "final_trace_json_without_postproc"
    ):
        return to_jsonable(trace_json_converter(value))
    if isinstance(value, Mapping):
        return {
            str(to_jsonable(key)): provenance_to_jsonable(
                item,
                trace_json_converter,
                str(to_jsonable(key)),
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [provenance_to_jsonable(item, trace_json_converter) for item in value]
    return to_jsonable(value)


def enrich_search_provenance(
    provenance: Any,
    final_trace_without_postproc: Any,
    trace_json_converter: Callable[[Any], Any] = to_jsonable,
) -> dict[str, Any]:
    """Add readable diffs and consistency checks to native evolutionary provenance."""

    if provenance is None:
        return {
            "available": False,
            "strategy": None,
            "reason": "The search strategy did not attach native candidate provenance",
            "mutations": [],
            "mutation_count": 0,
            "lineage_consistent": None,
        }

    result = provenance_to_jsonable(provenance, trace_json_converter)
    if not isinstance(result, dict):
        return {
            "available": False,
            "strategy": None,
            "reason": f"Unsupported provenance payload type: {type(result).__name__}",
            "raw_value": result,
            "mutations": [],
            "mutation_count": 0,
            "lineage_consistent": False,
        }

    root_trace = result.get("root_trace_json")
    root_sha256 = trace_json_sha256(root_trace)
    previous_sha256 = root_sha256
    previous_trace = root_trace
    chain_links_match = True
    enriched_mutations: list[dict[str, Any]] = []

    for position, raw_mutation in enumerate(result.get("mutations", [])):
        mutation = dict(raw_mutation)
        parent_trace = mutation.get("parent_trace_json")
        mutated_trace = mutation.get("mutated_trace_json")
        parent_sha256 = trace_json_sha256(parent_trace)
        mutated_sha256 = trace_json_sha256(mutated_trace)
        parent_instructions = trace_instructions(parent_trace)
        mutated_instructions = trace_instructions(mutated_trace)
        parent_decisions = trace_decisions_by_instruction(parent_trace)
        mutated_decisions = trace_decisions_by_instruction(mutated_trace)

        changed_decisions = []
        for instruction_index in sorted(parent_decisions.keys() | mutated_decisions.keys()):
            before = parent_decisions.get(instruction_index)
            after = mutated_decisions.get(instruction_index)
            if before != after:
                changed_decisions.append(
                    {
                        "instruction_index": instruction_index,
                        "instruction_kind": (
                            instruction_kind_at(mutated_instructions, instruction_index)
                            or instruction_kind_at(parent_instructions, instruction_index)
                        ),
                        "before": before,
                        "after": after,
                    }
                )

        changed_instructions = []
        for instruction_index in range(max(len(parent_instructions), len(mutated_instructions))):
            before = (
                parent_instructions[instruction_index]
                if instruction_index < len(parent_instructions)
                else None
            )
            after = (
                mutated_instructions[instruction_index]
                if instruction_index < len(mutated_instructions)
                else None
            )
            if before != after:
                changed_instructions.append(
                    {
                        "instruction_index": instruction_index,
                        "kind_before": instruction_kind_at(parent_instructions, instruction_index),
                        "kind_after": instruction_kind_at(mutated_instructions, instruction_index),
                        "before": before,
                        "after": after,
                    }
                )

        mutator_type = str(mutation.get("mutator_type"))
        mutator_name = mutator_type.rsplit(".", maxsplit=1)[-1]
        changed_kinds = sorted(
            {
                str(item["instruction_kind"])
                for item in changed_decisions
                if item["instruction_kind"] is not None
            }
        )
        if changed_decisions:
            change_description = f"changed {len(changed_decisions)} stochastic decision(s)" + (
                f" ({', '.join(changed_kinds)})" if changed_kinds else ""
            )
        elif changed_instructions:
            change_description = f"changed {len(changed_instructions)} trace instruction(s)"
        else:
            change_description = "returned an equivalent pre-postprocessing trace"

        mutation.update(
            {
                "mutation_index": position,
                "mutator_name": mutator_name,
                "summary": f"{mutator_name} {change_description}.",
                "parent_trace_sha256": parent_sha256,
                "mutated_trace_sha256": mutated_sha256,
                "parent_trace_portable_sha256": portable_trace_sha256(parent_trace),
                "mutated_trace_portable_sha256": portable_trace_sha256(mutated_trace),
                "changed_decisions": changed_decisions,
                "changed_decision_count": len(changed_decisions),
                "changed_instructions": changed_instructions,
                "changed_instruction_count": len(changed_instructions),
                "parent_matches_previous_step": parent_sha256 == previous_sha256,
            }
        )
        chain_links_match = chain_links_match and parent_sha256 == previous_sha256
        previous_sha256 = mutated_sha256
        previous_trace = mutated_trace
        enriched_mutations.append(mutation)

    final_trace_sha256 = trace_json_sha256(final_trace_without_postproc)
    final_trace_portable_sha256 = portable_trace_sha256(final_trace_without_postproc)
    native_final_trace = result.get("final_trace_json_without_postproc")
    native_final_sha256 = trace_json_sha256(native_final_trace)
    native_final_portable_sha256 = portable_trace_sha256(native_final_trace)
    terminal_portable_sha256 = portable_trace_sha256(previous_trace)
    terminal_exact_matches_candidate = previous_sha256 == final_trace_sha256
    native_final_exact_matches_candidate = native_final_sha256 == final_trace_sha256
    terminal_matches_candidate = terminal_portable_sha256 == final_trace_portable_sha256
    native_final_matches_candidate = native_final_portable_sha256 == final_trace_portable_sha256
    result.update(
        {
            "available": bool(result.get("available", True)),
            "root_trace_sha256": root_sha256,
            "root_trace_portable_sha256": portable_trace_sha256(root_trace),
            "mutations": enriched_mutations,
            "mutation_count": len(enriched_mutations),
            "lineage_terminal_trace_sha256": previous_sha256,
            "lineage_terminal_trace_portable_sha256": terminal_portable_sha256,
            "candidate_trace_without_postproc_sha256": final_trace_sha256,
            "candidate_trace_without_postproc_portable_sha256": (final_trace_portable_sha256),
            "native_final_trace_without_postproc_sha256": native_final_sha256,
            "native_final_trace_without_postproc_portable_sha256": (native_final_portable_sha256),
            "chain_links_match": chain_links_match,
            "lineage_terminal_exact_json_matches_candidate": (terminal_exact_matches_candidate),
            "native_final_exact_json_matches_candidate": (native_final_exact_matches_candidate),
            "lineage_terminal_matches_candidate": terminal_matches_candidate,
            "native_final_matches_candidate": native_final_matches_candidate,
            "lineage_consistent": bool(
                chain_links_match and terminal_matches_candidate and native_final_matches_candidate
            ),
        }
    )
    return result


def trace_metadata(trace: Any) -> dict[str, Any]:
    raw_json = trace.as_json(remove_postproc=False)
    raw_without_postproc = trace.as_json(remove_postproc=True)
    python_lines = [str(line) for line in trace.as_python(remove_postproc=False)]
    kinds = [str(inst.kind.name) for inst in trace.insts]
    pure = [bool(inst.kind.is_pure) for inst in trace.insts]
    semantic_edits = group_semantic_edits(python_lines, kinds, pure)
    transform_chain = build_transform_chain(semantic_edits)
    json_safe_trace = to_jsonable(raw_json)
    json_safe_without_postproc = to_jsonable(raw_without_postproc)
    return {
        "trace_json": json_safe_trace,
        "trace_json_without_postproc": json_safe_without_postproc,
        "trace_python": python_lines,
        "instruction_kinds": kinds,
        "instruction_is_pure": pure,
        "raw_trace_length": len(kinds),
        "semantic_edits": semantic_edits,
        "semantic_edit_count": len(semantic_edits),
        "transform_chain": transform_chain,
        "transform_chain_step_count": len(transform_chain),
        "trace_sha256": trace_json_sha256(json_safe_trace),
        "trace_without_postproc_sha256": trace_json_sha256(json_safe_without_postproc),
        "_raw_trace_json": raw_json,
    }


def replay_trace_payload(tvm: Any, initial_mod: Any, trace_json: Any) -> dict[str, Any]:
    """Replay one pre-postprocessing trace from the initial TensorIR."""

    from tvm.s_tir import Schedule  # pylint: disable=import-outside-toplevel
    from tvm.s_tir.schedule import Trace  # pylint: disable=import-outside-toplevel

    result: dict[str, Any] = {"passed": False, "error": None}
    try:
        replayed = Schedule(initial_mod, debug_mask="all", seed=1)
        Trace.apply_json_to_schedule(trace_json, replayed)
        result["passed"] = True
    except Exception as error:  # pylint: disable=broad-exception-caught
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def verify_search_lineage_traces(
    tvm: Any,
    initial_mod: Any,
    lineage: dict[str, Any],
) -> dict[str, Any]:
    """Replay every unique native root/parent/child trace in one lineage."""

    if lineage.get("available") is not True:
        lineage["trace_replay"] = {
            "attempted": False,
            "all_passed": None,
            "reason": "No native lineage was attached",
            "traces": [],
        }
        return lineage

    payloads: list[tuple[str, Any]] = [("root", lineage.get("root_trace_json"))]
    for mutation in lineage.get("mutations", []):
        index = int(mutation["mutation_index"])
        payloads.extend(
            [
                (f"mutation_{index}_parent", mutation.get("parent_trace_json")),
                (f"mutation_{index}_child", mutation.get("mutated_trace_json")),
            ]
        )
    payloads.append(
        (
            "native_final_without_postproc",
            lineage.get("final_trace_json_without_postproc"),
        )
    )

    seen: dict[str, dict[str, Any]] = {}
    replay_records = []
    for role, payload in payloads:
        digest = trace_json_sha256(payload)
        if digest not in seen:
            seen[digest] = replay_trace_payload(tvm, initial_mod, payload)
        replay_records.append(
            {
                "role": role,
                "trace_sha256": digest,
                **seen[digest],
            }
        )

    all_passed = all(item["passed"] for item in replay_records)
    lineage["trace_replay"] = {
        "attempted": True,
        "all_passed": all_passed,
        "unique_trace_count": len(seen),
        "traces": replay_records,
    }
    lineage["lineage_consistent"] = bool(lineage.get("lineage_consistent") is True and all_passed)
    return lineage


def replay_verification(
    tvm: Any, initial_mod: Any, scheduled_mod: Any, raw_trace: Any
) -> dict[str, Any]:
    """Replay serialized instructions and compare the resulting TensorIR."""

    import tvm_ffi  # pylint: disable=import-outside-toplevel

    from tvm.s_tir import Schedule  # pylint: disable=import-outside-toplevel
    from tvm.s_tir.schedule import Trace  # pylint: disable=import-outside-toplevel

    result: dict[str, Any] = {"attempted": True, "passed": False, "error": None}
    try:
        replayed = Schedule(initial_mod, debug_mask="all", seed=1)
        Trace.apply_json_to_schedule(raw_trace, replayed)
        result["passed"] = bool(tvm_ffi.structural_equal(replayed.mod, scheduled_mod))
        if not result["passed"]:
            result["error"] = "Trace replayed, but resulting IR was not structurally equal"
    except Exception as error:  # pylint: disable=broad-exception-caught
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def well_formed_verification(tvm: Any, mod: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"attempted": True, "passed": False, "error": None}
    try:
        generic = bool(tvm.tirx.analysis.verify_well_formed(mod, assert_mode=False))
        tirx = bool(tvm.tirx.analysis.verify_tirx_well_formed(mod, assert_mode=False))
        result.update({"passed": generic and tirx, "generic": generic, "tirx": tirx})
    except Exception as error:  # pylint: disable=broad-exception-caught
        result["error"] = f"{type(error).__name__}: {error}"
    return result


class NumericalValidator:
    """Optional deterministic comparison against the original TIR on LLVM.

    The reference is cached once per workload.  Validation deliberately occurs
    before RemoveBuildArtifact, so the candidate that MetaSchedule measured is
    exactly the artifact checked here.
    """

    def __init__(
        self,
        tvm: Any,
        ms: Any,
        mode: str,
        seed: int,
        rtol: float,
        atol: float,
    ) -> None:
        self.tvm = tvm
        self.ms = ms
        self.mode = mode
        self.seed = seed
        self.rtol = rtol
        self.atol = atol
        self.cache: dict[str, tuple[list[Any], list[Any]]] = {}

    def _numpy_dtype(self, dtype: str) -> Any:
        import numpy as np  # pylint: disable=import-outside-toplevel

        if dtype == "bfloat16":
            import ml_dtypes  # noqa: F401

        return np.dtype(dtype)

    def _reference(self, workload_id: str, initial_mod: Any) -> tuple[list[Any], list[Any]]:
        if workload_id in self.cache:
            return self.cache[workload_id]
        import numpy as np  # pylint: disable=import-outside-toplevel

        args_info = self.ms.arg_info.ArgInfo.from_entry_func(initial_mod)
        seed_material = int(sha256_text(f"{self.seed}:{workload_id}")[:16], 16)
        rng = np.random.default_rng(seed_material)
        initial_values: list[Any] = []
        for info in args_info:
            dtype = str(info.dtype)
            shape = tuple(int(dim) for dim in info.shape)
            if dtype.startswith("float") or dtype == "bfloat16":
                value = rng.uniform(-0.125, 0.125, size=shape).astype(self._numpy_dtype(dtype))
            elif dtype.startswith("uint"):
                value = rng.integers(0, 4, size=shape, dtype=self._numpy_dtype(dtype))
            else:
                value = rng.integers(-2, 3, size=shape, dtype=self._numpy_dtype(dtype))
            initial_values.append(value)

        reference_lib = self.tvm.compile(initial_mod, target="llvm")
        reference_args = [self.tvm.runtime.tensor(value) for value in initial_values]
        reference_lib(*reference_args)
        expected = [arg.numpy() for arg in reference_args]
        self.cache[workload_id] = (initial_values, expected)
        return initial_values, expected

    def validate(
        self,
        workload_id: str,
        initial_mod: Any,
        artifact_path: str | None,
    ) -> dict[str, Any]:
        if self.mode == "none":
            return {
                "attempted": False,
                "passed": None,
                "method": None,
                "reason": "Disabled; use --numerical-validation=llvm to check numerical correctness",
            }
        result: dict[str, Any] = {
            "attempted": True,
            "passed": False,
            "method": "original_tir_on_llvm_vs_measured_artifact_on_cuda",
            "rtol": self.rtol,
            "atol": self.atol,
            "error": None,
        }
        try:
            if not artifact_path:
                raise ValueError("Candidate has no build artifact")
            import numpy as np  # pylint: disable=import-outside-toplevel

            initial_values, expected = self._reference(workload_id, initial_mod)
            device = self.tvm.cuda(0)
            runtime_module = self.tvm.runtime.load_module(artifact_path)
            actual_args = [
                self.tvm.runtime.tensor(value, device=device) for value in initial_values
            ]
            runtime_module(*actual_args)
            device.sync()
            actual = [arg.numpy() for arg in actual_args]
            max_abs_error: list[float] = []
            passed = True
            for expected_value, actual_value in zip(expected, actual):
                difference = np.abs(
                    expected_value.astype("float32") - actual_value.astype("float32")
                )
                max_abs_error.append(float(np.max(difference)) if difference.size else 0.0)
                if not np.allclose(
                    expected_value,
                    actual_value,
                    rtol=self.rtol,
                    atol=self.atol,
                    equal_nan=False,
                ):
                    passed = False
            result["passed"] = passed
            result["max_abs_error_per_argument"] = max_abs_error
        except Exception as error:  # pylint: disable=broad-exception-caught
            result["error"] = f"{type(error).__name__}: {error}"
        return result


def make_trial_callback(
    tvm: Any,
    ms: Any,
    writer: TrialWriter,
    work_dir: pathlib.Path,
    run_id: str,
    model_metadata: dict[str, Any],
    verify_trace: bool,
    store_tvm_json: bool,
    numerical_validator: NumericalValidator,
) -> Any:
    """Create a TVM-derived callback without importing TVM at module import time."""

    from tvm.ir.utils import derived_object  # pylint: disable=import-outside-toplevel

    @derived_object
    class TrialRecorder(ms.measure_callback.PyMeasureCallback):
        def apply(
            self,
            task_scheduler: Any,
            task_id: int,
            measure_candidates: Sequence[Any],
            builder_results: Sequence[Any],
            runner_results: Sequence[Any],
        ) -> None:
            task_record = task_scheduler.tasks_[task_id]
            context = task_record.ctx
            initial_mod = context.mod
            task_name = str(context.task_name)
            task_weight = float(task_record.task_weight)
            workload = writer.workload_record(
                tvm=tvm,
                ms=ms,
                task_name=task_name,
                task_weight=task_weight,
                initial_mod=initial_mod,
                target=context.target,
                work_dir=work_dir,
                store_tvm_json=store_tvm_json,
            )
            workload_id = workload["workload_id"]

            for candidate, build_result, run_result in zip(
                measure_candidates, builder_results, runner_results
            ):
                global_index = writer.state.global_count
                task_index = writer.state.task_counts[workload_id]
                build_error = str(build_result.error_msg) if build_result.error_msg else None
                run_error = str(run_result.error_msg) if run_result.error_msg else None
                run_secs = []
                if run_result.run_secs is not None:
                    run_secs = [
                        float(getattr(value, "value", value)) for value in run_result.run_secs
                    ]
                latency = latency_statistics(run_secs)
                if build_error:
                    status = "build_error"
                elif run_error:
                    status = "run_error"
                elif latency is None:
                    status = "invalid_latency"
                else:
                    status = "ok"

                try:
                    from tvm.s_tir.schedule.trace import (  # pylint: disable=import-outside-toplevel
                        _json_from_tvm as trace_json_from_tvm,
                    )

                    trace = trace_metadata(candidate.sch.trace)
                    raw_trace = trace.pop("_raw_trace_json")
                    trace["search_lineage"] = verify_search_lineage_traces(
                        tvm,
                        initial_mod,
                        enrich_search_provenance(
                            getattr(candidate, "provenance", None),
                            trace["trace_json_without_postproc"],
                            trace_json_converter=trace_json_from_tvm,
                        ),
                    )
                    scheduled_mod = candidate.sch.mod
                    scheduled_ir = script_text(scheduled_mod)
                    scheduled_features = ast_features(tvm, scheduled_mod, ms)
                    replay = (
                        replay_verification(tvm, initial_mod, scheduled_mod, raw_trace)
                        if verify_trace
                        else {
                            "attempted": False,
                            "passed": None,
                            "error": "Disabled with --no-verify-trace-replay",
                        }
                    )
                    well_formed = well_formed_verification(tvm, scheduled_mod)
                except Exception as error:  # pylint: disable=broad-exception-caught
                    trace = {
                        "trace_json": None,
                        "trace_json_without_postproc": None,
                        "trace_python": [],
                        "instruction_kinds": [],
                        "instruction_is_pure": [],
                        "raw_trace_length": None,
                        "semantic_edits": [],
                        "semantic_edit_count": None,
                        "transform_chain": [],
                        "transform_chain_step_count": None,
                        "trace_sha256": None,
                        "trace_without_postproc_sha256": None,
                        "search_lineage": {
                            "available": False,
                            "strategy": None,
                            "reason": "Collector failed before provenance could be serialized",
                            "mutations": [],
                            "mutation_count": 0,
                            "lineage_consistent": False,
                        },
                        "collector_error": f"{type(error).__name__}: {error}",
                    }
                    scheduled_mod = candidate.sch.mod
                    scheduled_ir = None
                    scheduled_features = None
                    replay = {"attempted": False, "passed": False, "error": str(error)}
                    well_formed = {"attempted": False, "passed": False, "error": str(error)}

                numerical = (
                    numerical_validator.validate(
                        workload_id,
                        initial_mod,
                        str(build_result.artifact_path) if build_result.artifact_path else None,
                    )
                    if status == "ok"
                    else {
                        "attempted": False,
                        "passed": None,
                        "method": numerical_validator.mode,
                        "reason": f"Skipped because measurement status is {status}",
                    }
                )

                search_valid = bool(
                    status == "ok"
                    and replay.get("passed") is True
                    and well_formed.get("passed") is True
                )
                verified_for_training = bool(
                    search_valid and numerical.get("attempted") and numerical.get("passed") is True
                )

                first_measured = writer.state.first_measured_ms.get(workload_id)
                best_measured_before = writer.state.best_measured_ms.get(workload_id)
                first_verified = writer.state.first_verified_ms.get(workload_id)
                best_verified_before = writer.state.best_verified_ms.get(workload_id)
                median_ms = float(latency["median"]) if latency else None

                measured_first_effective = first_measured
                measured_speedup = None
                measured_reduction = None
                measured_best_speedup = None
                is_new_measured_best = False
                if status == "ok" and median_ms is not None:
                    measured_first_effective = (
                        first_measured if first_measured is not None else median_ms
                    )
                    measured_speedup = measured_first_effective / median_ms
                    measured_reduction = (
                        (measured_first_effective - median_ms) / measured_first_effective * 100.0
                    )
                    is_new_measured_best = (
                        best_measured_before is None or median_ms < best_measured_before
                    )
                    measured_best_speedup = (
                        best_measured_before / median_ms
                        if best_measured_before is not None
                        else None
                    )

                verified_first_effective = first_verified
                verified_speedup = None
                verified_reduction = None
                verified_best_speedup = None
                is_new_verified_best = False
                if verified_for_training and median_ms is not None:
                    verified_first_effective = (
                        first_verified if first_verified is not None else median_ms
                    )
                    verified_speedup = verified_first_effective / median_ms
                    verified_reduction = (
                        (verified_first_effective - median_ms) / verified_first_effective * 100.0
                    )
                    is_new_verified_best = (
                        best_verified_before is None or median_ms < best_verified_before
                    )
                    verified_best_speedup = (
                        best_verified_before / median_ms
                        if best_verified_before is not None
                        else None
                    )

                improvement = {
                    "baseline_definition": "first fully verified candidate for this workload",
                    "first_valid_latency_ms": verified_first_effective,
                    "best_latency_before_ms": best_verified_before,
                    "speedup_vs_first_valid": verified_speedup,
                    "latency_reduction_vs_first_valid_percent": verified_reduction,
                    "is_new_best": is_new_verified_best,
                    "speedup_vs_best_before": verified_best_speedup,
                    "measured": {
                        "baseline_definition": "first successfully measured candidate",
                        "first_latency_ms": measured_first_effective,
                        "best_latency_before_ms": best_measured_before,
                        "speedup_vs_first": measured_speedup,
                        "latency_reduction_vs_first_percent": measured_reduction,
                        "is_new_best": is_new_measured_best,
                        "speedup_vs_best_before": measured_best_speedup,
                    },
                    "verified": {
                        "baseline_definition": "first fully verified candidate",
                        "first_latency_ms": verified_first_effective,
                        "best_latency_before_ms": best_verified_before,
                        "speedup_vs_first": verified_speedup,
                        "latency_reduction_vs_first_percent": verified_reduction,
                        "is_new_best": is_new_verified_best,
                        "speedup_vs_best_before": verified_best_speedup,
                    },
                }
                record_id_material = (
                    f"{run_id}:{global_index}:{workload_id}:{trace.get('trace_sha256')}"
                )
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "record_id": f"tr_{sha256_text(record_id_material)[:24]}",
                    "run_id": run_id,
                    "global_trial_index": global_index,
                    "task_trial_index": task_index,
                    "task_scheduler_id": int(task_id),
                    "timestamp_utc": utc_now(),
                    "search_elapsed_seconds": time.monotonic() - writer.start_monotonic,
                    "model": model_metadata,
                    "workload": workload,
                    "schedule": {
                        **trace,
                        "scheduled_ir": scheduled_ir,
                        "scheduled_ast_features": scheduled_features,
                        "scheduled_ir_tvm_json": (
                            tvm.ir.save_json(scheduled_mod) if store_tvm_json else None
                        ),
                    },
                    "measurement": {
                        "status": status,
                        "run_secs": run_secs,
                        "latency": latency,
                        "build_error": build_error,
                        "runner_error": run_error,
                        "improvement": improvement,
                    },
                    "verification": {
                        "trace_replay": replay,
                        "well_formed": well_formed,
                        "build_passed": build_error is None,
                        "run_passed": run_error is None and latency is not None,
                        "numerical_correctness": numerical,
                        "search_valid": search_valid,
                        "verified_for_training": verified_for_training,
                    },
                }
                writer.append(record)

    return TrialRecorder()


def summarize_trials(path: pathlib.Path) -> dict[str, Any]:
    state = TrialState()
    workload_names: dict[str, str] = {}
    run_ids: set[str] = set()
    provenance_available = 0
    provenance_consistent = 0
    mutation_type_counts: Counter[str] = Counter()
    selection_source_counts: Counter[str] = Counter()
    total_recorded_mutations = 0
    if path.exists():
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                state.observe_existing(record)
                workload = record.get("workload", {})
                workload_names[str(workload.get("workload_id"))] = str(workload.get("task_name"))
                run_ids.add(str(record.get("run_id")))
                lineage = record.get("schedule", {}).get("search_lineage", {})
                if lineage.get("available") is True:
                    provenance_available += 1
                if lineage.get("lineage_consistent") is True:
                    provenance_consistent += 1
                mutations = lineage.get("mutations", [])
                total_recorded_mutations += len(mutations)
                for mutation in mutations:
                    mutation_type_counts[str(mutation.get("mutator_name"))] += 1
                selection = lineage.get("selection", {})
                if selection.get("source") is not None:
                    selection_source_counts[str(selection.get("source"))] += 1
    workloads = []
    for workload_id in sorted(state.task_counts):
        measured_first = state.first_measured_ms.get(workload_id)
        measured_best = state.best_measured_ms.get(workload_id)
        verified_first = state.first_verified_ms.get(workload_id)
        verified_best = state.best_verified_ms.get(workload_id)
        workloads.append(
            {
                "workload_id": workload_id,
                "task_name": workload_names.get(workload_id),
                "trial_count": state.task_counts[workload_id],
                "measured": {
                    "first_latency_ms": measured_first,
                    "best_latency_ms": measured_best,
                    "best_speedup_vs_first": (
                        measured_first / measured_best
                        if measured_first is not None and measured_best is not None
                        else None
                    ),
                },
                "verified": {
                    "first_latency_ms": verified_first,
                    "best_latency_ms": verified_best,
                    "best_speedup_vs_first": (
                        verified_first / verified_best
                        if verified_first is not None and verified_best is not None
                        else None
                    ),
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "trial_file": str(path.resolve()),
        "total_trials": state.global_count,
        "status_counts": dict(sorted(state.status_counts.items())),
        "verification_counts": dict(sorted(state.verification_counts.items())),
        "provenance": {
            "available_records": provenance_available,
            "consistent_records": provenance_consistent,
            "total_recorded_mutations": total_recorded_mutations,
            "mutation_type_counts": dict(sorted(mutation_type_counts.items())),
            "selection_source_counts": dict(sorted(selection_source_counts.items())),
        },
        "run_ids": sorted(run_ids),
        "workloads": workloads,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lower a Llama MLP to TIR, run MetaSchedule, and record every measured trace.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--work-dir", type=pathlib.Path, default=pathlib.Path("tuning_logs/llama31_mlp")
    )
    parser.add_argument("--model-preset", choices=sorted(MODEL_PRESETS), default="llama3.1-8b")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument(
        "--separate-gate-up",
        action="store_true",
        help="Use separate gate/up projections instead of TVM's tutorial-style fused projection.",
    )
    parser.add_argument("--max-trials-global", type=int, default=2000)
    parser.add_argument("--max-trials-per-task", type=int, default=512)
    parser.add_argument("--num-trials-per-iter", type=int, default=64)
    parser.add_argument("--num-tuning-cores", type=int, default=8)
    parser.add_argument("--builder-workers", type=int, default=8)
    parser.add_argument("--builder-timeout-sec", type=float, default=120.0)
    parser.add_argument("--runner-timeout-sec", type=float, default=120.0)
    parser.add_argument("--number", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--min-repeat-ms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cost-model", choices=["xgb", "random"], default="xgb")
    parser.add_argument(
        "--strategy",
        choices=["evolutionary", "replay-trace", "replay-func"],
        default="evolutionary",
    )
    parser.add_argument(
        "--task-scheduler", choices=["round-robin", "gradient"], default="round-robin"
    )
    parser.add_argument(
        "--task-regex", default=None, help="Tune only extracted task names matching this regex."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Warm-start and append to an existing work directory."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only lower the model and list extracted tasks."
    )
    parser.add_argument(
        "--self-test", action="store_true", help="Run TVM-independent utility tests and exit."
    )
    parser.add_argument(
        "--verify-trace-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replay each serialized trace and require structural equality with the candidate IR.",
    )
    parser.add_argument(
        "--store-tvm-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store lossless TVM object JSON for independently replayable trial records.",
    )
    parser.add_argument(
        "--numerical-validation",
        choices=["none", "llvm"],
        default="llvm",
        help=(
            "Compare every successful measured artifact against the original "
            "unscheduled TIR on LLVM."
        ),
    )
    parser.add_argument("--correctness-rtol", type=float, default=1e-2)
    parser.add_argument("--correctness-atol", type=float, default=1e-2)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    args = parser.parse_args(argv)
    for name in (
        "batch_size",
        "seq_len",
        "max_trials_global",
        "max_trials_per_task",
        "num_trials_per_iter",
        "num_tuning_cores",
        "builder_workers",
        "number",
        "repeat",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    return args


def run_self_test() -> None:
    stats = latency_statistics([0.001, 0.002, 0.003])
    assert stats is not None
    assert stats["mean"] == 2.0
    assert stats["median"] == 2.0
    edits = group_semantic_edits(
        [
            "v1, v2 = sch.sample_perfect_tile(loop=l0, n=2, decision=[1, 16])",
            "l1, l2 = sch.split(loop=l0, factors=[v1, v2])",
            'sch.bind(loop=l1, thread_axis="blockIdx.x")',
            'sch.bind(loop=l2, thread_axis="threadIdx.x")',
        ],
        ["SamplePerfectTile", "Split", "Bind", "Bind"],
        [True, False, False, False],
    )
    assert [edit["kind"] for edit in edits] == [
        "loop_tiling",
        "bind_cuda_blocks",
        "bind_cuda_threads",
    ]
    assert edits[0]["trace_instruction_indices"] == [0, 1]
    chain = build_transform_chain(edits)
    assert chain[0] == {
        "step": 1,
        "kind": "loop_tiling",
        "summary": "Tile loop(s): sample perfect tile, then split.",
        "trace_instruction_indices": [0, 1],
        "schedule_python": edits[0]["python"],
    }
    assert chain[1]["summary"] == "Map loop(s) to CUDA blocks: bind."
    assert semantic_category("ReIndex", "sch.reindex(...)") == "memory_layout"

    sample_parent = [
        [["SampleCategorical", [], [[32, 64], [0.5, 0.5]], ["v0"]]],
        [[0, 0]],
    ]
    sample_child = [
        [["SampleCategorical", [], [[32, 64], [0.5, 0.5]], ["v0"]]],
        [[0, 1]],
    ]
    lineage = enrich_search_provenance(
        {
            "available": True,
            "strategy": "evolutionary",
            "origin": {"type": "random_design_space"},
            "root_trace_json": sample_parent,
            "mutations": [
                {
                    "mutator_type": "s_tir.meta_schedule.MutateThreadBinding",
                    "parent_trace_json": sample_parent,
                    "mutated_trace_json": sample_child,
                }
            ],
            "selection": {"source": "cost_model_best"},
            "final_trace_json_without_postproc": sample_child,
        },
        sample_child,
    )
    assert lineage["lineage_consistent"] is True
    assert lineage["mutations"][0]["changed_decisions"][0]["before"] == 0
    assert lineage["mutations"][0]["changed_decisions"][0]["after"] == 1
    print("Self-test passed")


def prepare_work_dir(work_dir: pathlib.Path, resume: bool, dry_run: bool) -> None:
    if work_dir.exists() and any(work_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"Work directory is not empty: {work_dir}. Use --resume or choose a new directory."
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return


def main(argv: Sequence[str] | None = None) -> int:  # pylint: disable=too-many-locals
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0

    # Delay imports so --self-test can run before the TVM Python module is installed.
    try:
        import tvm
        from tvm import relax
        from tvm.ir.utils import derived_object  # noqa: F401
        from tvm.relax.frontend import nn
        from tvm.s_tir import meta_schedule as ms
    except ImportError as error:
        raise RuntimeError(
            "This example requires TVM with the tirx and s_tir.meta_schedule APIs."
        ) from error

    cuda_compile_mode = os.environ.get("TVM_CUDA_COMPILE_MODE", "nvrtc").lower()
    if cuda_compile_mode == "nvrtc":
        try:
            from cuda.bindings import nvrtc as _nvrtc  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "TVM defaults to NVRTC, but cuda-bindings is unavailable. "
                "Install it with 'python -m pip install cuda-bindings' or set "
                "TVM_CUDA_COMPILE_MODE=nvcc."
            ) from error
    elif cuda_compile_mode == "nvcc":
        if shutil.which("nvcc") is None:
            raise RuntimeError("TVM_CUDA_COMPILE_MODE=nvcc, but nvcc is not on PATH")
    else:
        raise ValueError(f"Invalid TVM_CUDA_COMPILE_MODE={cuda_compile_mode!r}; use nvrtc or nvcc")

    import logging

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("tvm.s_tir.meta_schedule").setLevel(getattr(logging, args.log_level))

    work_dir = args.work_dir.resolve()
    prepare_work_dir(work_dir, resume=args.resume, dry_run=args.dry_run)
    preset = MODEL_PRESETS[args.model_preset]
    dtype = args.dtype or preset.default_dtype
    run_id = (
        f"run_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    )
    tvm_root = pathlib.Path(__file__).resolve().parents[2]

    device = tvm.cuda(0)
    if not device.exist:
        raise RuntimeError(
            "CUDA device 0 is not available. Set CUDA_VISIBLE_DEVICES to the desired GPU."
        )
    target = tvm.target.Target.from_device(device)

    model_metadata = {
        "preset": args.model_preset,
        "name": preset.name,
        "description": preset.description,
        "source_url": preset.source_url,
        "component": "decoder_swiglu_mlp",
        "batch_size": args.batch_size,
        "sequence_length": args.seq_len,
        "hidden_size": preset.hidden_size,
        "intermediate_size": preset.intermediate_size,
        "dtype": dtype,
        "accumulation_dtype": "float32",
        "projection_output_cast_dtype": dtype,
        "fused_gate_up": not args.separate_gate_up,
        "expression": "down_proj(silu(gate_proj(x)) * up_proj(x))",
    }
    search_metadata = {
        "max_trials_global": args.max_trials_global,
        "max_trials_per_task": args.max_trials_per_task,
        "num_trials_per_iter": args.num_trials_per_iter,
        "num_tuning_cores": args.num_tuning_cores,
        "builder_workers": args.builder_workers,
        "builder_timeout_sec": args.builder_timeout_sec,
        "runner_timeout_sec": args.runner_timeout_sec,
        "evaluator": {
            "number": args.number,
            "repeat": args.repeat,
            "min_repeat_ms": args.min_repeat_ms,
            "latency_label": "median",
            "raw_unit": "seconds",
        },
        "seed": args.seed,
        "cost_model": args.cost_model,
        "strategy": args.strategy,
        "task_scheduler": args.task_scheduler,
        "task_regex": args.task_regex,
        "module_equality": "structural",
        "numerical_validation": args.numerical_validation,
        "capture_native_mutator_provenance": args.strategy == "evolutionary",
    }

    relax_mod, named_params = build_llama_mlp_ir(
        tvm=tvm,
        relax=relax,
        nn=nn,
        preset=preset,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dtype=dtype,
        fused_gate_up=not args.separate_gate_up,
    )
    lowered_mod = lower_relax_to_tir(tvm, relax, relax_mod, target)
    atomic_write_text(work_dir / "relax_model.py", script_text(relax_mod) + "\n")
    atomic_write_text(work_dir / "lowered_model.py", script_text(lowered_mod) + "\n")

    extracted = ms.relax_integration.extract_tasks(
        lowered_mod,
        target,
        params={},
        module_equality="structural",
    )
    if args.task_regex:
        pattern = re.compile(args.task_regex)
        extracted = [task for task in extracted if pattern.search(str(task.task_name))]
    if not extracted:
        raise RuntimeError("No tunable TIR tasks were extracted from the lowered Relax module")

    extracted_metadata = [
        {
            "task_index": index,
            "task_name": str(task.task_name),
            "weight": float(task.weight),
            "dispatched_variants": len(task.dispatched),
            "structural_hash": structural_hash_hex(ms, task.dispatched[0]),
        }
        for index, task in enumerate(extracted)
    ]
    for item in extracted_metadata:
        print(
            f"Task {item['task_index']}: {item['task_name']} "
            f"weight={item['weight']} hash={item['structural_hash']}"
        )

    source_metadata = tvm_source_metadata(tvm_root, work_dir)
    collector_path = pathlib.Path(__file__).resolve()
    collector_source = collector_path.read_text(encoding="utf-8")
    collector_snapshot = work_dir / "collector_snapshot.py"
    atomic_write_text(collector_snapshot, collector_source)
    device_metadata = {
        "name": str(device.device_name),
        "compute_version": str(device.compute_version),
        "multi_processor_count": int(device.multi_processor_count),
        "max_threads_per_block": int(device.max_threads_per_block),
    }
    experiment_identity = {
        "model": model_metadata,
        "target": str(target),
        "device": {
            "name": device_metadata["name"],
            "compute_version": device_metadata["compute_version"],
        },
        "evaluator": search_metadata["evaluator"],
        "seed": args.seed,
        "cost_model": args.cost_model,
        "strategy": args.strategy,
        "task_scheduler": args.task_scheduler,
        "task_regex": args.task_regex,
        "module_equality": "structural",
        "verify_trace_replay": args.verify_trace_replay,
        "store_tvm_json": args.store_tvm_json,
        "numerical_validation": args.numerical_validation,
        "correctness_rtol": args.correctness_rtol,
        "correctness_atol": args.correctness_atol,
        "tvm_git_commit": source_metadata.get("git_commit"),
        "tvm_git_diff_sha256": source_metadata.get("git_diff_sha256"),
        "collector_sha256": sha256_text(collector_source),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "dry_run" if args.dry_run else "running",
        "started_at_utc": utc_now(),
        "experiment_identity": experiment_identity,
        "experiment_identity_sha256": sha256_text(canonical_json(experiment_identity)),
        "model": model_metadata,
        "search": search_metadata,
        "target": str(target),
        "device": device_metadata,
        "software": {
            "tvm_version": str(tvm.__version__),
            "python": sys.version,
            "platform": platform.platform(),
            "tvm_source": source_metadata,
            "collector": {
                "path": str(collector_path),
                "snapshot": str(collector_snapshot),
                "sha256": sha256_text(collector_source),
            },
        },
        "relax_parameter_count": len(named_params),
        "extracted_tasks": extracted_metadata,
        "outputs": {
            "native_database_workloads": str(work_dir / "database_workload.json"),
            "native_database_records": str(work_dir / "database_tuning_record.json"),
            "trials_jsonl": str(work_dir / "trials.jsonl"),
            "summary": str(work_dir / "summary.json"),
        },
    }
    manifest_path = work_dir / "manifest.json"
    if args.resume and manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_identity = previous_manifest.get("experiment_identity")
        if previous_identity != experiment_identity:
            raise ValueError(
                "--resume immutable model/target/measurement configuration differs from "
                "the existing manifest; choose a new work directory"
            )
        manifest["resumed_from_run_id"] = previous_manifest.get("run_id")
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    if args.dry_run:
        print(f"Dry run complete; lowered IR and manifest written to {work_dir}")
        return 0

    tasks, task_weights = ms.relax_integration.extracted_tasks_to_tune_contexts(
        extracted_tasks=extracted,
        work_dir=str(work_dir),
        space="post-order-apply",
        strategy=args.strategy,
        num_threads=args.num_tuning_cores,
        seed=args.seed,
    )
    if not tasks:
        raise RuntimeError("All extracted tasks were already scheduled or otherwise skipped")

    writer = TrialWriter(work_dir / "trials.jsonl", run_id=run_id, resume=args.resume)
    numerical_validator = NumericalValidator(
        tvm=tvm,
        ms=ms,
        mode=args.numerical_validation,
        seed=args.seed,
        rtol=args.correctness_rtol,
        atol=args.correctness_atol,
    )
    trial_callback = make_trial_callback(
        tvm=tvm,
        ms=ms,
        writer=writer,
        work_dir=work_dir,
        run_id=run_id,
        model_metadata=model_metadata,
        verify_trace=args.verify_trace_replay,
        store_tvm_json=args.store_tvm_json,
        numerical_validator=numerical_validator,
    )
    builder = ms.builder.LocalBuilder(
        max_workers=args.builder_workers,
        timeout_sec=args.builder_timeout_sec,
    )
    runner = ms.runner.LocalRunner(
        timeout_sec=args.runner_timeout_sec,
        evaluator_config=ms.runner.EvaluatorConfig(
            number=args.number,
            repeat=args.repeat,
            min_repeat_ms=args.min_repeat_ms,
            enable_cpu_cache_flush=False,
        ),
        alloc_repeat=1,
    )
    callbacks = [
        ms.measure_callback.AddToDatabase(),
        trial_callback,
        ms.measure_callback.RemoveBuildArtifact(),
        ms.measure_callback.UpdateCostModel(),
    ]

    try:
        database = ms.tune_tasks(
            tasks=tasks,
            task_weights=task_weights,
            work_dir=str(work_dir),
            max_trials_global=args.max_trials_global,
            max_trials_per_task=args.max_trials_per_task,
            num_trials_per_iter=args.num_trials_per_iter,
            builder=builder,
            runner=runner,
            database="json",
            cost_model=args.cost_model,
            measure_callbacks=callbacks,
            task_scheduler=args.task_scheduler,
            module_equality="structural",
        )
        summary = summarize_trials(work_dir / "trials.jsonl")
        summary["native_database_records"] = len(database)
        atomic_write_text(work_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
        if summary["status_counts"].get("ok", 0) == 0:
            raise RuntimeError(
                "Search completed without a single usable latency measurement; "
                "inspect build/runner errors in trials.jsonl"
            )
        if (
            args.numerical_validation != "none"
            and summary["verification_counts"].get("verified", 0) == 0
        ):
            raise RuntimeError(
                "Numerical validation was requested, but no trial is verified for training"
            )
        if args.strategy == "evolutionary":
            provenance_summary = summary["provenance"]
            if provenance_summary["available_records"] != summary["total_trials"]:
                raise RuntimeError(
                    "Native evolutionary provenance was missing from one or more measured trials"
                )
            if provenance_summary["consistent_records"] != summary["total_trials"]:
                raise RuntimeError(
                    "One or more recorded mutator lineages failed internal consistency checks"
                )
        manifest["status"] = "complete"
        manifest["finished_at_utc"] = utc_now()
        manifest["summary"] = summary
        atomic_write_text(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    except Exception as error:
        manifest["status"] = "failed"
        manifest["finished_at_utc"] = utc_now()
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["traceback"] = traceback.format_exc()
        atomic_write_text(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        raise
    finally:
        writer.close()

    print(f"Collection complete: {summary['total_trials']} measured candidates")
    print(f"Dataset: {work_dir / 'trials.jsonl'}")
    print(f"Summary: {work_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
