import hashlib
import json

import pytest

from benchmarks.verify_long_context import (
    KERNEL_FILES,
    LENGTHS,
    MODEL,
    MODEL_REVISION,
    OFFLOAD_COMPARISON,
    RECEIPTS,
    VerificationError,
    verify_artifact,
)


def _write_json(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _checksums(root):
    names = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (root / "SHA256SUMS").write_text(
        "".join(f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n" for name in names),
        encoding="ascii",
    )


def _step(index, seconds, peak, *, offload=False):
    value = {
        "index": index,
        "gradients_finite": True,
        "gradient_tensor_count": 2,
        "gradient_element_count": 4,
        "nonzero_gradient_tensor_count": 1,
        "nonzero_gradient_element_count": 2,
        "total_fwd_bwd_optim_seconds": seconds,
        "forward_seconds": seconds / 4,
        "backward_seconds": seconds / 2,
        "optimizer_seconds": seconds / 4,
        "cuda_peak_allocated_bytes": peak,
        "cuda_peak_reserved_bytes": peak * 2,
    }
    if offload:
        value["activation_offload"] = {
            "mechanism": "native-save-on-cpu-checkpoint-inputs",
            "offloaded_bytes": 1024,
        }
    return value


def _fixture(tmp_path):
    hardware = {
        "name": "NVIDIA B200",
        "compute_capability": [10, 0],
        "cuda_runtime": "12.8",
        "nvidia_driver": "580.1",
        "total_memory_bytes": 8 * 1024**3,
        "visible_device_count": 1,
    }
    for tokens, name in RECEIPTS.items():
        offload = tokens == 262144
        steps = [_step(0, 2.0, 1024**3, offload=offload), _step(1, 4.0, 2 * 1024**3, offload=offload)]
        requested = {
            "model": MODEL,
            "revision": MODEL_REVISION,
            "quantization": "nf4",
            "tokens": tokens,
            "warmup_steps": 1,
            "measured_steps": 2,
        }
        if offload:
            requested.update({"activation_offload": True, "seed": 42})
        receipt = {
            "schema": "openjev-long-context-memory-gradient-probe-v1",
            "status": "completed",
            "requested": requested,
            "workload": {"actual_tokens": tokens},
            "model": {"source": MODEL, "revision": MODEL_REVISION, "quantization": "nf4"},
            "hardware": hardware,
            "warmup": _step(-1, 5.0, 1024**3, offload=offload),
            "steps": steps,
            "measured_peak": {
                "cuda_peak_allocated_bytes": 2 * 1024**3,
                "cuda_peak_reserved_bytes": 4 * 1024**3,
            },
            "measured_totals": {
                "forward_seconds": 1.5,
                "backward_seconds": 3.0,
                "optimizer_seconds": 1.5,
                "total_fwd_bwd_optim_seconds": 6.0,
            },
        }
        _write_json(tmp_path, name, receipt)

    _write_json(
        tmp_path,
        OFFLOAD_COMPARISON,
        {
            "schema": "openjev-checkpoint-offload-comparison-v1",
            "status": "completed",
            "tokens": 8192,
            "model": MODEL,
            "revision": MODEL_REVISION,
            "numerically_equivalent": True,
            "offload": {"offload": {"mechanism": "native-save-on-cpu-checkpoint-inputs"}},
            "comparison": {
                "logits": {"passed": True, "max_abs_diff": 0.0},
                "loss": {"passed": True, "max_abs_diff": 0.0},
                "gradients": {
                    "a": {"passed": True, "max_abs_diff": 0.0},
                    "b": {"passed": True, "max_abs_diff": 0.000001},
                },
            },
        },
    )
    for name in KERNEL_FILES:
        version = int(name.removesuffix(".json").rsplit("v", 1)[1])
        kernel = {"schema_version": 1, "status": "pass" if version >= 6 else "fail"}
        if version == 7:
            kernel.update(
                {
                    "probe": "gemma4-12b-long-context-flex-attention",
                    "metadata": {"cuda": {"device_count": 1, "device_name": "NVIDIA B200"}},
                    "mask_equivalence": {"status": "pass"},
                    "cases": [
                        _kernel_case("local_causal_gqa", 16, 8, 256, 1024),
                        _kernel_case("global_causal_gqa", 16, 1, 512, None),
                    ],
                }
            )
        _write_json(tmp_path, name, kernel)
    _checksums(tmp_path)
    return tmp_path


def _kernel_case(name, query_heads, kv_heads, head_dim, window_size):
    check = {"all_finite": True, "within_tolerance": True}
    return {
        "name": name,
        "status": "pass",
        "shape": {
            "sequence_length": 2048,
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "window_size": window_size,
        },
        "checks": {field: dict(check) for field in ("output", "q_gradient", "k_gradient", "v_gradient")},
    }


def test_verifier_recomputes_summary_from_minimal_fixture(tmp_path):
    artifact = _fixture(tmp_path)
    summary = verify_artifact(artifact)
    assert summary["status"] == "verified"
    assert [probe["tokens"] for probe in summary["probes"]] == list(LENGTHS)
    assert summary["probes"][0]["mean_measured_step_seconds"] == 3.0
    assert summary["probes"][0]["peak_cuda_allocated_gib"] == 2.0
    assert summary["probes"][-1]["activation_offload"] is True
    assert summary["offload_validation"]["max_gradient_abs_error"] == 0.000001
    assert summary["kernel_validation"]["final_status"] == "pass"

    _write_json(artifact, "summary.json", summary)
    _checksums(artifact)
    assert verify_artifact(artifact) == summary


def test_verifier_rejects_bad_actual_length(tmp_path):
    artifact = _fixture(tmp_path)
    path = artifact / RECEIPTS[32768]
    receipt = json.loads(path.read_text())
    receipt["workload"]["actual_tokens"] = 32767
    path.write_text(json.dumps(receipt))
    _checksums(artifact)
    with pytest.raises(VerificationError, match="exactly 32768"):
        verify_artifact(artifact)


def test_verifier_rejects_nonfinite_measurement(tmp_path):
    artifact = _fixture(tmp_path)
    path = artifact / RECEIPTS[8192]
    receipt = json.loads(path.read_text())
    receipt["steps"][0]["total_fwd_bwd_optim_seconds"] = float("nan")
    path.write_text(json.dumps(receipt))
    _checksums(artifact)
    with pytest.raises(VerificationError, match="non-finite"):
        verify_artifact(artifact)


def test_verifier_rejects_unmatched_hash(tmp_path):
    artifact = _fixture(tmp_path)
    path = artifact / RECEIPTS[16384]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(VerificationError, match="SHA256 mismatch"):
        verify_artifact(artifact)
