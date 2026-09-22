"""Verify and summarize the published long-context systems receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath


MODEL = "google/gemma-4-12B-it"
MODEL_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
LENGTHS = (8192, 16384, 32768, 65536, 131072, 262144)
RECEIPTS = {
    length: (
        f"context-{length}-offload-v1/probe-receipt.json"
        if length == 262144
        else f"context-{length}-v1/probe-receipt.json"
    )
    for length in LENGTHS
}
OFFLOAD_COMPARISON = "offload-check-8192-v1/offload-comparison.json"
KERNEL_FILES = tuple(f"kernel-evidence/kernel-v{version}.json" for version in (1, 3, 4, 5, 6, 7))
REQUIRED_FILES = frozenset((*RECEIPTS.values(), OFFLOAD_COMPARISON, *KERNEL_FILES))


class VerificationError(ValueError):
    """The artifact is incomplete or inconsistent."""


def _read_json(path: Path, *, finite: bool = True) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise VerificationError(f"{path} must contain a JSON object")
    if finite:
        _reject_nonfinite(value, str(path))
    return value


def _reject_nonfinite(value, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise VerificationError(f"non-finite number at {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{location}[{index}]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_inventory(artifact_dir: Path) -> set[str]:
    if not artifact_dir.is_dir():
        raise VerificationError(f"artifact directory does not exist: {artifact_dir}")
    if any(path.is_symlink() for path in artifact_dir.rglob("*")):
        raise VerificationError("artifact inventory must not contain symlinks")
    actual = {
        path.relative_to(artifact_dir).as_posix()
        for path in artifact_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if not REQUIRED_FILES <= actual:
        raise VerificationError(f"artifact inventory is missing: {sorted(REQUIRED_FILES - actual)}")

    checksum_path = artifact_dir / "SHA256SUMS"
    try:
        lines = checksum_path.read_text(encoding="ascii").splitlines()
    except OSError as error:
        raise VerificationError("artifact is missing SHA256SUMS") from error
    hashes: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise VerificationError("SHA256SUMS contains an invalid line")
        digest, name = parts
        path = PurePosixPath(name)
        if (
            any(character not in "0123456789abcdef" for character in digest)
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != name
            or name in hashes
            or name == "SHA256SUMS"
        ):
            raise VerificationError("SHA256SUMS contains an unsafe or duplicate entry")
        hashes[name] = digest
    if set(hashes) != actual:
        raise VerificationError("SHA256SUMS must cover the closed artifact inventory")
    for name, expected in hashes.items():
        if _sha256(artifact_dir / name) != expected:
            raise VerificationError(f"SHA256 mismatch: {name}")
    return actual


def _positive_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise VerificationError(f"{label} must be a positive finite number")
    return float(value)


def _verify_gradient_step(step: dict, label: str, expected_index: int) -> None:
    if not isinstance(step, dict) or step.get("index") != expected_index:
        raise VerificationError(f"{label} has the wrong step index")
    if step.get("gradients_finite") is not True:
        raise VerificationError(f"{label} does not report finite gradients")
    for field in (
        "gradient_tensor_count",
        "gradient_element_count",
        "nonzero_gradient_tensor_count",
        "nonzero_gradient_element_count",
        "total_fwd_bwd_optim_seconds",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
    ):
        _positive_number(step.get(field), f"{label}.{field}")


def _verify_receipt(receipt: dict, tokens: int) -> dict:
    label = RECEIPTS[tokens]
    if receipt.get("schema") != "openjev-long-context-memory-gradient-probe-v1":
        raise VerificationError(f"{label} has the wrong schema")
    if receipt.get("status") != "completed":
        raise VerificationError(f"{label} is not completed")
    requested = receipt.get("requested", {})
    model = receipt.get("model", {})
    if (
        requested.get("model") != MODEL
        or requested.get("revision") != MODEL_REVISION
        or requested.get("quantization") != "nf4"
        or requested.get("adapter") is not None
        or requested.get("adapter_revision") is not None
        or model.get("source") != MODEL
        or model.get("revision") != MODEL_REVISION
        or model.get("quantization") != "nf4"
    ):
        raise VerificationError(f"{label} does not use the pinned base model")
    if requested.get("tokens") != tokens or receipt.get("workload", {}).get("actual_tokens") != tokens:
        raise VerificationError(f"{label} does not contain exactly {tokens} tokens")
    if requested.get("warmup_steps") != 1 or requested.get("measured_steps") != 2:
        raise VerificationError(f"{label} must request one warmup and two measured steps")

    offload = requested.get("activation_offload", False)
    if offload is not (tokens == 262144):
        raise VerificationError(f"{label} has the wrong activation-offload setting")
    warmup = receipt.get("warmup")
    _verify_gradient_step(warmup, f"{label}.warmup", -1)
    steps = receipt.get("steps")
    if not isinstance(steps, list) or len(steps) != 2:
        raise VerificationError(f"{label} must contain two measured steps")
    for index, step in enumerate(steps):
        _verify_gradient_step(step, f"{label}.steps[{index}]", index)

    if offload:
        for index, step in enumerate((warmup, *steps)):
            stats = step.get("activation_offload", {})
            if stats.get("mechanism") != "native-save-on-cpu-checkpoint-inputs":
                raise VerificationError(f"{label} step {index - 1} lacks the pinned offload mechanism")
            _positive_number(stats.get("offloaded_bytes"), f"{label}.activation_offload.offloaded_bytes")

    peak_allocated = max(step["cuda_peak_allocated_bytes"] for step in steps)
    peak_reserved = max(step["cuda_peak_reserved_bytes"] for step in steps)
    declared_peak = receipt.get("measured_peak", {})
    if (
        declared_peak.get("cuda_peak_allocated_bytes") != peak_allocated
        or declared_peak.get("cuda_peak_reserved_bytes") != peak_reserved
    ):
        raise VerificationError(f"{label} measured peak does not match its steps")
    totals = receipt.get("measured_totals", {})
    for field in ("forward_seconds", "backward_seconds", "optimizer_seconds", "total_fwd_bwd_optim_seconds"):
        recomputed = sum(_positive_number(step.get(field), f"{label}.{field}") for step in steps)
        declared = _positive_number(totals.get(field), f"{label}.measured_totals.{field}")
        if not math.isclose(declared, recomputed, rel_tol=0, abs_tol=1e-9):
            raise VerificationError(f"{label} measured total for {field} does not match its steps")
    mean_seconds = sum(step["total_fwd_bwd_optim_seconds"] for step in steps) / 2
    return {
        "tokens": tokens,
        "activation_offload": offload,
        "seed": requested.get("seed"),
        "mean_measured_step_seconds": mean_seconds,
        "peak_cuda_allocated_gib": peak_allocated / (1024**3),
        "peak_cuda_reserved_gib": peak_reserved / (1024**3),
    }


def _verify_offload_comparison(comparison: dict) -> dict:
    if (
        comparison.get("schema") != "openjev-checkpoint-offload-comparison-v1"
        or comparison.get("status") != "completed"
        or comparison.get("tokens") != 8192
        or comparison.get("model") != MODEL
        or comparison.get("revision") != MODEL_REVISION
        or comparison.get("numerically_equivalent") is not True
    ):
        raise VerificationError("offload comparison is incomplete or not pinned")
    checks = comparison.get("comparison", {})
    gradients = checks.get("gradients")
    if not isinstance(gradients, dict) or not gradients:
        raise VerificationError("offload comparison has no gradient checks")
    all_checks = [checks.get("logits"), checks.get("loss"), *gradients.values()]
    if any(not isinstance(check, dict) or check.get("passed") is not True for check in all_checks):
        raise VerificationError("offload comparison contains a failed numerical check")
    max_error = max(_positive_or_zero(check.get("max_abs_diff"), "max_abs_diff") for check in gradients.values())
    mechanism = comparison.get("offload", {}).get("offload", {}).get("mechanism")
    if mechanism != "native-save-on-cpu-checkpoint-inputs":
        raise VerificationError("offload comparison has the wrong mechanism")
    return {
        "tokens": 8192,
        "mechanism": mechanism,
        "numerically_equivalent": True,
        "gradient_tensors_compared": len(gradients),
        "max_gradient_abs_error": max_error,
    }


def _positive_or_zero(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise VerificationError(f"{label} must be a finite nonnegative number")
    return float(value)


def _verify_kernels(artifact_dir: Path) -> dict:
    evidence = []
    for name in KERNEL_FILES:
        receipt = _read_json(artifact_dir / name, finite=False)
        status = receipt.get("status")
        if status not in {"pass", "fail"}:
            raise VerificationError(f"{name} has an invalid status")
        if status == "pass":
            _reject_nonfinite(receipt, name)
        entry = {"artifact": name, "status": status}
        if receipt.get("error"):
            entry["error"] = receipt["error"]
        evidence.append(entry)
    if evidence[-1] != {"artifact": "kernel-evidence/kernel-v7.json", "status": "pass"}:
        raise VerificationError("final kernel evidence v7 did not pass")
    if not any(entry["status"] == "fail" for entry in evidence[:-1]):
        raise VerificationError("kernel evidence does not retain the failed iterations")
    final = _read_json(artifact_dir / KERNEL_FILES[-1])
    if (
        final.get("schema_version") != 1
        or final.get("probe") != "gemma4-12b-long-context-flex-attention"
        or final.get("mask_equivalence", {}).get("status") != "pass"
    ):
        raise VerificationError("final kernel evidence lacks the pinned successful probe")
    cuda = final.get("metadata", {}).get("cuda", {})
    if cuda.get("device_count") != 1 or cuda.get("device_name") != "NVIDIA B200":
        raise VerificationError("final kernel evidence must report one NVIDIA B200")
    expected_shapes = {
        "local_causal_gqa": (2048, 16, 8, 256, 1024),
        "global_causal_gqa": (2048, 16, 1, 512, None),
    }
    cases = final.get("cases")
    if not isinstance(cases, list) or {case.get("name") for case in cases if isinstance(case, dict)} != set(expected_shapes):
        raise VerificationError("final kernel evidence has the wrong bounded cases")
    for case in cases:
        shape = case.get("shape", {})
        observed = tuple(shape.get(field) for field in ("sequence_length", "query_heads", "kv_heads", "head_dim", "window_size"))
        checks = case.get("checks", {})
        if (
            case.get("status") != "pass"
            or observed != expected_shapes[case["name"]]
            or set(checks) != {"output", "q_gradient", "k_gradient", "v_gradient"}
            or any(check.get("all_finite") is not True or check.get("within_tolerance") is not True for check in checks.values())
        ):
            raise VerificationError(f"final kernel case {case.get('name')} did not pass its bounded checks")
    return {"final_version": 7, "final_status": "pass", "evidence": evidence}


def verify_artifact(artifact_dir: Path) -> dict:
    """Verify *artifact_dir* and return its summary derived from raw receipts."""
    artifact_dir = Path(artifact_dir)
    inventory = _verify_inventory(artifact_dir)
    receipts = {tokens: _read_json(artifact_dir / name) for tokens, name in RECEIPTS.items()}
    probes = [_verify_receipt(receipts[tokens], tokens) for tokens in LENGTHS]

    first_hardware = receipts[LENGTHS[0]].get("hardware", {})
    hardware_fields = (
        "name",
        "compute_capability",
        "cuda_runtime",
        "nvidia_driver",
        "total_memory_bytes",
        "visible_device_count",
    )
    if any(
        any(receipts[tokens].get("hardware", {}).get(field) != first_hardware.get(field) for field in hardware_fields)
        for tokens in LENGTHS[1:]
    ):
        raise VerificationError("probe hardware metadata is inconsistent")
    if first_hardware.get("visible_device_count") != 1:
        raise VerificationError("receipts must report exactly one visible GPU")
    if (
        first_hardware.get("name") != "NVIDIA B200"
        or first_hardware.get("compute_capability") != [10, 0]
        or _positive_number(first_hardware.get("total_memory_bytes"), "hardware.total_memory_bytes") <= 0
    ):
        raise VerificationError("receipts must report the pinned NVIDIA B200 hardware")

    summary = {
        "schema": "openjev-long-context-systems-summary-v1",
        "status": "verified",
        "model": {"source": MODEL, "revision": MODEL_REVISION, "quantization": "nf4"},
        "hardware": {
            "name": first_hardware.get("name"),
            "compute_capability": first_hardware.get("compute_capability"),
            "cuda_runtime": first_hardware.get("cuda_runtime"),
            "nvidia_driver": first_hardware.get("nvidia_driver"),
            "total_memory_gib": first_hardware.get("total_memory_bytes") / (1024**3),
            "visible_device_count": 1,
        },
        "probes": probes,
        "offload_validation": _verify_offload_comparison(_read_json(artifact_dir / OFFLOAD_COMPARISON)),
        "kernel_validation": _verify_kernels(artifact_dir),
        "caveats": [
            "The first five probes do not record a seed; the 262144-token offload probe records seed 42.",
            "These are synthetic systems measurements, not semantic-quality results or a trained-model release.",
            "The maximum observed run is native-context 262144 tokens on one NVIDIA B200; this evidence makes no 1M-token claim.",
        ],
    }
    if "summary.json" in inventory:
        existing = _read_json(artifact_dir / "summary.json")
        if existing != summary:
            raise VerificationError("summary.json does not match the raw receipts")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = verify_artifact(args.artifact_dir)
    except VerificationError as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
