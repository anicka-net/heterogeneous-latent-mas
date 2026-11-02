#!/usr/bin/env python3
"""Prepare and run one intermittent Vision Wormhole codec-training burst.

Only codec weights are trained; model weights remain frozen by the upstream trainer.
Training uses the upstream per-model self-distillation objective, not a learned
reward from model-to-model communication. Never promote candidates automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_MODELS = "Qwen/Qwen3-VL-2B-Thinking,google/gemma-3-4b-it"
DEFAULT_REPLAY = "data/vision_codec_anchor_text/mixed_cose_ocr_prm800k_small.jsonl"
DEFAULT_ARCH = {
    "codec_dim": 512, "codec_tokens": 1024, "codec_img_tokens": 256,
    "codec_heads": 8, "codec_layers": 6, "codec_dropout": 0.10,
    "codec_gate_init_bias": -4.0,
}
ARCH_FLAGS = {k: "--vision_" + k for k in DEFAULT_ARCH}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for buf in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def load_texts(path: Path) -> list[str]:
    """Read strict JSONL: string or object with text/prompt/message string."""
    output: list[str] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {error}") from error
            if isinstance(obj, str):
                value = obj
            elif isinstance(obj, dict):
                value = next((obj[k] for k in ("text", "prompt", "message")
                              if isinstance(obj.get(k), str)), None)
            else:
                value = None
            if not isinstance(value, str):
                raise ValueError(f"{path}:{line_no}: expected a text/prompt/message string")
            value = value.strip()
            if value:
                output.append(value)
    return list(dict.fromkeys(output))


def split_and_mix(
    replay: list[str], fresh: list[str], *, seed: int, train_new: int,
    replay_count: int, holdout_fraction: float,
    reserved_holdout: list[str] | None = None,
) -> tuple[list[str], list[str], dict]:
    """Deterministic holdout split; excludes every fresh and reserved item from replay.

    This prevents leakage within a round. Keep prior holdouts in reserved_holdout
    to prevent leakage across rounds; also maintain an independent benchmark set.
    """
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")
    if train_new < 1 or replay_count < 0:
        raise ValueError("train_new must be >=1 and replay_count >=0")
    rng = random.Random(seed)
    fresh = list(dict.fromkeys(fresh))
    reserved = set(reserved_holdout or [])
    training_pool: list[str] = []
    current_holdout: list[str] = []
    for value in fresh:
        if value in reserved:
            continue
        digest = hashlib.sha256(f"{seed}\\0{value}".encode()).digest()
        fraction = int.from_bytes(digest[:8], "big") / 2**64
        (current_holdout if fraction < holdout_fraction else training_pool).append(value)
    if not training_pool or not current_holdout:
        raise ValueError("Need enough distinct fresh examples for train and holdout")
    rng.shuffle(training_pool)
    new_sample = training_pool[:train_new]
    exclude = set(fresh) | reserved
    eligible_replay = [x for x in dict.fromkeys(replay) if x not in exclude]
    old_sample = rng.sample(eligible_replay, min(replay_count, len(eligible_replay)))
    training = list(dict.fromkeys(new_sample + old_sample))
    rng.shuffle(training)
    return training, current_holdout, {
        "seed": seed, "fresh_total": len(fresh),
        "new_train_selected": len(new_sample),
        "replay_selected": len(old_sample), "train_total": len(training),
        "holdout_total": len(current_holdout), "reserved_holdout_total": len(reserved),
        "holdout_fraction": holdout_fraction,
    }


def write_jsonl(path: Path, texts: list[str]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for value in texts:
            stream.write(json.dumps({"text": value}, ensure_ascii=False) + "\n")


def parent_config(path: Path | None, models: str) -> tuple[dict, int]:
    if path is None:
        return DEFAULT_ARCH.copy(), 0
    import torch  # Imported only when the user requests a GPU-training continuation.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    names = models.split(",")
    if (not isinstance(ckpt, dict) or ckpt.get("version") != 2
            or ckpt.get("is_partial", True)):
        raise ValueError("The parent must be a complete version-2 codec checkpoint")
    if ckpt.get("models") != names:
        raise ValueError("Parent models/order do not match --models")
    config = ckpt.get("config", {})
    if set(config) != set(DEFAULT_ARCH):
        raise ValueError(f"Unexpected parent codec architecture fields: {sorted(config)}")
    reference = ckpt.get("ref_model_name")
    if reference not in names:
        raise ValueError("Parent reference model is absent from models")
    return dict(config), names.index(reference)


def make_command(
    *, trainer: Path, candidate: Path, parent: Path | None,
    anchors: Path, models: str, arch: dict, ref_idx: int,
    steps: int, lr: float, seed: int, latent_steps: int,
) -> list[str]:
    cmd = [
        sys.executable, str(trainer),
        "--agent_model_names", models,
        "--vision_codec_path", str(candidate),
        "--vision_codec_anchor_texts_path", str(anchors),
        "--vision_codec_train_steps", str(steps),
        "--vision_codec_train_lr", str(lr),
        "--vision_codec_train_batch_size", "2",
        "--vision_codec_save_per_model", "1",
        "--vision_codec_align_max_anchors", "100",
        "--vision_codec_ref_idx", str(ref_idx),
        "--latent_steps", str(latent_steps), "--seed", str(seed),
    ]
    for key, flag in ARCH_FLAGS.items():
        cmd.extend([flag, str(arch[key])])
    if parent:
        cmd += ["--vision_codec_resume_path", str(parent)]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-file", type=Path, required=True)
    parser.add_argument("--replay-file", type=Path, default=Path(DEFAULT_REPLAY))
    parser.add_argument("--reserved-holdout", type=Path, help="JSONL of prior holdouts to exclude")
    parser.add_argument("--burst-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--parent", type=Path, help="Full previous checkpoint (omit for bootstrap)")
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--steps", type=int, help="Default: 400 bootstrap; 25 on continuation")
    parser.add_argument("--lr", type=float, help="Default: 2e-4 bootstrap; 2e-5 on continuation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-new", type=int, default=128)
    parser.add_argument("--replay", type=int, default=128)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--latent-steps", type=int, default=128)
    parser.add_argument("--trainer", type=Path, default=Path("train_vision_latent_mas_codec_new.py"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    steps = args.steps if args.steps is not None else (25 if args.parent else 400)
    lr = args.lr if args.lr is not None else (2e-5 if args.parent else 2e-4)
    if steps < 1 or lr <= 0 or args.latent_steps < 1:
        parser.error("--steps, --lr, and --latent-steps must be positive")
    if len(args.models.split(",")) != 2:
        parser.error("Prototype expects exactly two comma-separated models")
    if args.parent and args.parent.resolve() == args.candidate.resolve():
        parser.error("Cannot overwrite the parent checkpoint")
    if args.candidate.exists():
        parser.error(f"Candidate checkpoint already exists: {args.candidate}")
    if args.burst_dir.exists() and any(args.burst_dir.iterdir()):
        parser.error(f"Burst directory must be new/empty: {args.burst_dir}")
    if not args.trainer.is_file():
        parser.error(f"Trainer does not exist: {args.trainer}")
    if args.parent and not args.parent.is_file():
        parser.error(f"Parent checkpoint does not exist: {args.parent}")
    fresh = load_texts(args.new_file)
    replay = load_texts(args.replay_file)
    reserved = load_texts(args.reserved_holdout) if args.reserved_holdout else []
    training, holdout, sampling = split_and_mix(
        replay, fresh, seed=args.seed, train_new=args.train_new,
        replay_count=args.replay, holdout_fraction=args.holdout_fraction,
        reserved_holdout=reserved,
    )
    if len(training) < 4:
        parser.error("At least four train anchors are required")
    config, ref_idx = parent_config(args.parent, args.models)
    train_path = args.burst_dir / "train.jsonl"
    holdout_path = args.burst_dir / "holdout.jsonl"
    cmd = make_command(
        trainer=args.trainer, candidate=args.candidate, parent=args.parent,
        anchors=train_path, models=args.models, arch=config, ref_idx=ref_idx,
        steps=steps, lr=lr, seed=args.seed, latent_steps=args.latent_steps,
    )
    manifest = {
        "parent": str(args.parent) if args.parent else None,
        "parent_sha256": sha256(args.parent) if args.parent else None,
        "candidate": str(args.candidate), "train": str(train_path),
        "holdout": str(holdout_path), "fresh_source_sha256": sha256(args.new_file),
        "replay_source_sha256": sha256(args.replay_file),
        "reserved_holdout_sha256": sha256(args.reserved_holdout) if args.reserved_holdout else None,
        "sampling": sampling, "command": cmd, "status": "prepared",
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        print("Command:", shlex.join(cmd))
        return
    args.burst_dir.mkdir(parents=True, exist_ok=True)
    args.candidate.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_path, training)
    write_jsonl(holdout_path, holdout)
    manifest_path = args.burst_dir / "manifest.json"
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        subprocess.run(cmd, check=True)
    except BaseException:
        manifest["status"] = "failed_or_interrupted"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        raise
    manifest["status"] = "trained_unvalidated_candidate"
    manifest["candidate_sha256"] = sha256(args.candidate)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Candidate requires held-out communication evaluation:", args.candidate)


if __name__ == "__main__":
    main()
