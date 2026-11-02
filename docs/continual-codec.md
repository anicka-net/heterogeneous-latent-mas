# Continual codec training: Gemma 3 4B ↔ Qwen3-VL 2B

**Status: experimental, CPU-only orchestration tests; no model weights trained yet.**

This adds intermittent **warm-start** training to the [Vision Wormhole](../README.md)
codec training code. The base Gemma/Qwen weights remain frozen. Codec encoder and
decoder weights carry across bursts, while the AdamW optimizer restarts each
time. A candidate checkpoint is produced for evaluation, not automatically
promoted.

## What actually learns

This retains upstream's **per-model teacher/student distillation** objective.
The user-provided text prompts determine which latent trajectories are
distilled. Providing conversation examples does **not** itself train a
negotiated inter-model language, and it does **not** optimize a communication
success reward. This is infrastructure for a later end-to-end experiment.

## Requirements

Use the upstream CUDA/PyTorch/Transformers environment described in the root
README, with weights for both VLMs accessible. The default pair is
`Qwen/Qwen3-VL-2B-Thinking,google/gemma-3-4b-it`.

A standalone training process sequentially trains each model codec and then
refits cross-model alignment. Memory and runtime have not been measured on
our hardware. Start with small samples and a generous memory margin.

## Input data

Make a fresh examples file with **one text prompt per JSONL line**:

```jsonl
{"text": "A hidden clue can only reach the receiver via the latent channel."}
{"text": "Explain a counterexample to a generalization."}
```

The sample must contain enough distinct prompts to populate both sides of the
train/holdout split. Do not place sensitive material in Git or in shared logs.

The replay file may be the included
`data/vision_codec_anchor_text/mixed_cose_ocr_prm800k_small.jsonl`.
An optional `--reserved-holdout` file of previously withheld JSONL examples
prevents those items from accidentally appearing in replay or new training
examples on subsequent rounds.

**Keep an entirely independent, never-trained-on communication benchmark.**
The per-round holdout is a convenience for experiment iteration, not a
substitute for a permanent evaluation corpus.

## Bootstrap (no parent yet)

```bash
python scripts/continual_codec.py \
  --new-file data/my_fresh_round_000.jsonl \
  --burst-dir experiments/bootstrap \
  --candidate checkpoints/gemma_qwen_bootstrap.pt \
  --dry-run

# Once satisfied, rerun without --dry-run on a CUDA machine.
```

Defaults for bootstrap: 400 training steps per model, learning rate 2e-4.
The runner passes the 512-dimensional, 1024-token codec architecture to the
original trainer, and writes output only under the paths you choose.

## An intermittent burst

```bash
python scripts/continual_codec.py \
  --new-file data/my_fresh_round_001.jsonl \
  --replay-file data/vision_codec_anchor_text/mixed_cose_ocr_prm800k_small.jsonl \
  --reserved-holdout data/all_prior_holdouts.jsonl \
  --burst-dir experiments/round_001 \
  --parent checkpoints/gemma_qwen_bootstrap.pt \
  --candidate checkpoints/round_001_candidate.pt \
  --steps 25 --lr 2e-5 \
  --dry-run

# Remove --dry-run to actually execute.
```

If you haven't yet built a prior holdout registry, omit the
`--reserved-holdout` flag for the first round, then curate that file from the
round's generated `holdout.jsonl` before further training.

Each burst writes:

- `train.jsonl` and `holdout.jsonl` with no within-round overlap.
- `manifest.json` including data and parent hashes, command, and status.
- A **new, unvalidated** checkpoint under `--candidate`.

The runner refuses overwriting a candidate, a nonempty burst directory, and
using the parent checkpoint as output. A resume is rejected if the parent is
partial, the model order/reference differs, or the codec architecture differs.
It intentionally **does not** restore optimizer state or RNG trajectories.
The continued checkpoint records `continual_history` while remaining version
2 for the existing inference reader.

## Promotion must be evaluated

Do **not** accept lower codec loss as evidence of improved collaboration.

Before promoting a candidate, compare parent and candidate with the same
frozen held-out tasks and seeds; evaluate sender-only private-clue transfer
through the latent port against zero-channel, shuffled-channel and ordinary
text-message baselines. Treat negative or ambiguous results as failures to
improve, not an excuse to tune on the holdout. The current patch **does not
implement this evaluation or promotion gate**.

## Local tests

```bash
python -m unittest discover -s tests -p 'test_continual_codec.py' -v
```

These run without CUDA or downloaded weights and exercise deterministic
sampling, leakage exclusions, dry runs, mock burst completion and source
syntax. GPU numerical stability and actual transfer accuracy are untested.

## Experimental next step

Build a sender/receiver private-information task and instrument actual latent
messages. Use a frozen held-out evaluation suite with message shuffling,
zeroed messages and text baselines. Only after verifying genuine information
transfer should a task-level learning signal be introduced. Log all codec
versions, model revisions and data hashes to make drift detectable.
