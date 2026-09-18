"""Routed, multimodal EXL3 calibration for one complete DeepSeek expert.

The gate/up Hessian uses actual routed inputs. The down Hessian is collected
after the packed quantized gate/up, trained clamps and pre-down route scaling.
Held-out rows are diagnostic only and never enter either Hessian.
"""
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
from .exl3_moe import make_linear_exl3


class RoutedInputs:
    """Load one layer once on CPU, avoiding a corpus scan for every expert."""

    def __init__(self, capture, corpus, layer):
        self.capture = Path(capture)
        identity_bytes = (self.capture / "capture-config.json").read_bytes()
        self.capture_sha256 = hashlib.sha256(identity_bytes).hexdigest()
        identity = json.loads(identity_bytes)
        corpus_bytes = (Path(corpus) / "records.jsonl").read_bytes()
        if hashlib.sha256(corpus_bytes).hexdigest() != identity["corpus_sha256"]:
            raise ValueError("Corpus does not match captured activations")
        records = {record["id"]: record for record in map(json.loads, corpus_bytes.splitlines())}
        self.splits = {}
        for split in ("calibration", "heldout"):
            rows = []
            for record_id in identity["records"]:
                if records[record_id]["split"] == split:
                    path = self.capture / "expert-inputs" / f"{layer:02d}" / (record_id + ".safetensors")
                    data = load_file(path)
                    if not torch.isfinite(data["inputs"]).all() or not torch.isfinite(data["route_weights"]).all():
                        raise ValueError(f"Nonfinite captured data: {path}")
                    rows.append(data)
            if not rows:
                raise ValueError(f"No {split} records in capture")
            self.splits[split] = {key: torch.cat([row[key] for row in rows]) for key in rows[0]}

    def select(self, expert, split, maximum, seed):
        data = self.splits[split]
        row, slot = torch.where(data["route_ids"] == expert)
        available = len(row)
        if maximum > 0 and len(row) > maximum:
            generator = torch.Generator().manual_seed(seed + 1009 * expert + (split == "heldout"))
            chosen = torch.randperm(len(row), generator=generator)[:maximum]
            row, slot = row[chosen], slot[chosen]
        selected = {"inputs": data["inputs"][row].cuda().half().contiguous(),
                    "weights": data["route_weights"][row, slot].cuda().float().unsqueeze(1),
                    "image_mask": data["image_mask"][row].cuda()}
        if not torch.isfinite(selected["inputs"]).all():
            raise ValueError("FP16 EXL3 input overflow")
        metadata = {"available": available, "selected": len(row),
                    "selected_image_tokens": selected["image_mask"].sum().item(),
                    "selected_indices_sha256": hashlib.sha256(torch.stack((row, slot)).numpy().tobytes()).hexdigest()}
        return selected, metadata


def swiglu(gate, up, weights, limit):
    gate, up = gate.float(), up.float()
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return (F.silu(gate) * up * weights).half().contiguous()


def hessian(x, key):
    value = torch.zeros((x.shape[1], x.shape[1]), device=x.device, dtype=torch.float32)
    for batch in x.split(1024):
        batch = batch.float()
        value.addmm_(batch.T, batch)
    if not len(x) or not torch.isfinite(value).all():
        raise ValueError(f"Empty or nonfinite Hessian: {key}")
    return {"H": value, "count": len(x), "finalized": False, "device": str(x.device), "first_key": key}


def quant_linear(weight, hd, key, bits, seed):
    qa = {"K": bits, "seed": seed, "devices": ["cuda:0"], "mul1": True, "apply_out_scales": True}
    _, proxy, packed = quantize_exl3(weight.T.contiguous().clone(), hd, qa, False)
    if qa["q_fallback"] or not 0 <= proxy < 1:
        raise ValueError(f"Uncalibrated fallback or invalid proxy for {key}: {qa}, {proxy}")
    layer = make_linear_exl3(config=None, in_features=weight.shape[1], out_features=weight.shape[0],
                      out_dtype=torch.float16, key=key, **packed)
    return layer, packed, {"proxy_nmse": float(proxy), "uncalibrated_fallback": False,
                           "stored_bytes": sum(t.numel() * t.element_size() for t in packed.values())}


def forward_chunks(layer, inputs):
    return torch.cat([layer.forward(batch.contiguous(), {}) for batch in inputs.split(1024)])


def error_metrics(actual, expected, image_mask):
    result = {}
    for name, mask in (("all", torch.ones_like(image_mask)), ("image", image_mask), ("text", ~image_mask)):
        count = mask.sum().item()
        if not count:
            result[name] = {"tokens": 0, "nmse": None}
            continue
        a, e = actual[mask].float(), expected[mask].float()
        if not torch.isfinite(a).all() or not torch.isfinite(e).all():
            raise ValueError("Nonfinite expert output")
        squared_error, squared_signal = (a - e).square().sum().item(), e.square().sum().item()
        result[name] = {"tokens": count, "squared_error": squared_error, "squared_signal": squared_signal,
                        "nmse": squared_error / max(squared_signal, 1e-30)}
    return result


@torch.inference_mode()
def quantize_expert(runtime, inputs, layer_id, expert_id, bits, seed, max_calibration, max_heldout, minimum):
    key = f"layers.{layer_id}.ffn.experts.{expert_id}"
    calibration, cal_meta = inputs.select(expert_id, "calibration", max_calibration, seed)
    heldout, held_meta = inputs.select(expert_id, "heldout", max_heldout, seed)
    if cal_meta["selected"] < minimum:
        raise ValueError(f"Insufficient routed calibration for {key}: {cal_meta}; require {minimum}")
    dense = {name: runtime.weights.dequantize(key + "." + name) for name in ("w1", "w3", "w2")}
    # Exactly the source expert (FP4 weights with FP8 activation quantization),
    # used only for diagnostics. It contributes no weights to the output quant.
    with torch.device("cuda"), runtime.ref.set_dtype(torch.bfloat16):
        source_expert = runtime.ref.Expert(runtime.args.dim, runtime.args.moe_inter_dim,
                                          dtype=torch.float4_e2m1fn_x2, swiglu_limit=runtime.args.swiglu_limit)
    runtime.load_parameters(source_expert, key)
    hd = hessian(calibration["inputs"], key + ".w1,w3")
    native, packed, stats = {}, {}, {}
    for name in ("w1", "w3"):
        native[name], tensors, stats[name] = quant_linear(dense[name], hd, key + "." + name, bits, seed)
        packed.update({key + "." + name + "." + suffix: tensor for suffix, tensor in tensors.items()})
    del hd
    gate, up = (forward_chunks(native[name], calibration["inputs"]) for name in ("w1", "w3"))
    down_inputs = swiglu(gate, up, calibration["weights"], runtime.args.swiglu_limit)
    hd = hessian(down_inputs, key + ".w2")
    native["w2"], tensors, stats["w2"] = quant_linear(dense["w2"], hd, key + ".w2", bits, seed)
    packed.update({key + ".w2." + suffix: tensor for suffix, tensor in tensors.items()})
    del hd, down_inputs, gate, up
    metrics = {}
    for split, data in (("calibration", calibration), ("heldout", heldout)):
        if not len(data["inputs"]):
            metrics[split] = {"tokens": 0}
            continue
        # Calibration diagnostics can guide bit allocation. Held-out diagnostics
        # must not be used to choose bits or hyperparameters.
        x, weights = data["inputs"], data["weights"]
        gate, up = (forward_chunks(native[name], x) for name in ("w1", "w3"))
        actual = forward_chunks(native["w2"], swiglu(gate, up, weights, runtime.args.swiglu_limit))
        reference = torch.cat([source_expert(batch.bfloat16(), weight)
                               for batch, weight in zip(x.split(1024), weights.split(1024))])
        gate, up = (x.float() @ dense[name].T for name in ("w1", "w3"))
        dequantized = swiglu(gate, up, weights, runtime.args.swiglu_limit).float() @ dense["w2"].T
        metrics[split] = {"vs_source_fp4": error_metrics(actual, reference, data["image_mask"]),
                          "vs_dequantized_weights": error_metrics(actual, dequantized, data["image_mask"])}
    return {name: tensor.cpu().contiguous() for name, tensor in packed.items()}, {
        "expert": key, "bits": bits, "codebook": "mul1", "calibration": cal_meta, "heldout": held_meta,
        "matrices": stats, "metrics": metrics,
        "stored_bytes": sum(t.numel() * t.element_size() for t in packed.values()),
        "method": "actual_routed_inputs; down_H_after_packed_quantized_gate_up_and_route_weight; fp16_EXL3",
        "scope": "One expert on source-model inputs. Not end-to-end quantized-model accuracy."}


def save_result(tensors, report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    # Safetensors serializes metadata through a HashMap: multiple entries may
    # change header order across processes despite identical tensor bytes.
    # Keep one standard entry; quantization metadata lives in the paired report.
    save_file(tensors, partial, metadata={"format": "pt"})
    partial.replace(path)
    report["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    report_path = path.with_suffix(".json")
    partial_report = report_path.with_suffix(".partial.json")
    partial_report.write_text(json.dumps(report, indent=2) + "\n")
    partial_report.replace(report_path)
