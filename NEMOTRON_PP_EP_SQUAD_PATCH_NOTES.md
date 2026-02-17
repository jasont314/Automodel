# Nemotron PP/EP + SQuAD Patch Notes

This document summarizes the code-level changes prepared for a PR to make Nemotron-Nano-v3 PP/EP training and fixed-length SQuAD SFT stable, debuggable, and reproducible.

## Scope

The patch set keeps core behavior unchanged for existing non-PP/non-EP paths while addressing:
- PP schedule robustness and diagnostics,
- EP mesh/dispatch safety,
- fixed-length SQuAD supervision correctness,
- PR-level cleanup and configuration handoff.

## Major Functional Changes

### 1) Nemotron PP compatibility and PP runtime safeguards

Files:
- `nemo_automodel/components/distributed/pipelining/functional.py`
- `nemo_automodel/recipes/llm/train_ft.py`

Key changes:
- Added explicit invalid-style handling in `stage_ids_this_rank(...)` (`ValueError` on unknown style).
- Isolated `NEMOAUTOMODEL_PP_SKIP_OUTPUT_MERGE` behavior behind a guarded helper (`_enable_skip_output_merge_if_supported`) with compatibility checks before patching private schedule internals.
- Kept skip-output-merge behavior available for train/benchmark runs where schedule outputs are not consumed.

Why:
- Prevent implicit `None` returns and harder-to-debug failures.
- Make PP skip-merge behavior safer across PyTorch internals drift.

### 2) Nemotron EP safety and guardrails

Files:
- `nemo_automodel/components/moe/parallelizer.py`
- `nemo_automodel/recipes/llm/train_ft.py`

Key changes:
- Added null guard for `ep_shard_axis_names` when `moe_mesh` is not available.
- In LLM setup, `ep_axis_name` / `ep_shard_axis_names` are only passed when corresponding mesh dims exist.

Why:
- Avoid null dereference and confusing startup crashes in mixed EP/non-EP code paths.

### 3) AutoPipeline device typing cleanup

Files:
- `nemo_automodel/components/distributed/pipelining/autopipeline.py`
- `nemo_automodel/recipes/llm/train_ft.py`
- `nemo_automodel/recipes/biencoder/train_biencoder.py`

Key changes:
- AutoPipeline now normalizes `device` input (`torch.device | int | str`) to `torch.device` at construction.
- Call sites now pass `torch.device("cuda", torch.cuda.current_device())` instead of raw `int`.

Why:
- Remove type mismatch and prevent API ambiguity/drift.

### 4) FSDP2 diagnostics clarity

File:
- `nemo_automodel/components/distributed/fsdp2.py`

Key change:
- Corrected divisibility error message to match logic using `tp_size * cp_size * pp_size`.

Why:
- Better debugging clarity in distributed setup failures.

### 5) Logging observability default

File:
- `nemo_automodel/components/loggers/log_utils.py`

Key changes:
- `setup_logging(..., filter_warning=False)` by default.
- Added env override: `NEMOAUTOMODEL_FILTER_WARNINGS=1` to re-enable global warning filtering.

Why:
- Avoid hiding warnings by default during PP/EP debugging and PR validation.

## Fixed-Length SQuAD Supervision (NaN-loss root cause)

Files:
- `nemo_automodel/components/datasets/llm/formatting_utils.py`
- `nemo_automodel/components/datasets/llm/squad.py`

Key changes:
- Made prompt-completion mask generation truncation-aware.
- For fixed-length SQuAD (`seq_length`, `padding=max_length`, `truncation=true`), forced truncation settings that preserve supervised answer tokens.
- Disabled chat-template path for this fixed-length SQuAD mode to avoid all-masked labels.

Observed effect:
- `num_label_tokens` moved from `0` to large nonzero values on optimized SFT runs.
- `loss` and `grad_norm` became finite/nonzero.

## Observed SFT Throughput (from training JSONL)

From:
- `checkpoints/baseline_training.jsonl`
- `checkpoints/optimized_training.jsonl`

Measured `tps`:
- Baseline mean `tps`: ~326.10
- Optimized mean `tps`: ~12109.64
- Mean throughput uplift: ~37.1x

For reference, last logged step:
- Baseline last-step `tps`: ~284.51
- Optimized last-step `tps`: ~12104.76
- Last-step throughput uplift: ~42.5x

## New Example Config

Added:
- `examples/llm_finetune/nemotron/nemotron_nano_v3_pp_ep_squad.yaml`

This is the optimized PP+EP SQuAD SFT recipe used for reproducible runs with:
- `pp_size=4`, `ep_size=2`,
- manual PP module mapping,
- fixed-length SQuAD.

## Recommended Runtime Env (PP+EP)

```bash
TORCH_NCCL_USE_COMM_NONBLOCKING=1
PYTORCH_ALLOC_CONF=expandable_segments:True
NEMOTRONH_EP_USE_DEEPEP_DISPATCH=1
NEMOTRONH_EP_REQUIRE_DEEPEP=1
NEMOTRONH_EP_PHYSICAL_PARTITION=1
NEMOTRONH_EP_SYNC_INACTIVE_EXPERTS=1
NEMOTRONH_EP_EXPERT_RESHARD_AFTER_FORWARD=0
NEMOAUTOMODEL_PP_SKIP_OUTPUT_MERGE=1
```

Meaning of each variable:
- `TORCH_NCCL_USE_COMM_NONBLOCKING=1`: enables NCCL non-blocking error handling to reduce hard hangs.
- `PYTORCH_ALLOC_CONF=expandable_segments:True`: reduces allocator fragmentation under large transient GPU allocations.
- `NEMOTRONH_EP_USE_DEEPEP_DISPATCH=1`: uses DeepEP token dispatch path for EP.
- `NEMOTRONH_EP_REQUIRE_DEEPEP=1`: fails fast if DeepEP is unavailable (prevents silent fallback).
- `NEMOTRONH_EP_PHYSICAL_PARTITION=1`: uses physical expert partition ownership across EP ranks.
- `NEMOTRONH_EP_SYNC_INACTIVE_EXPERTS=1`: keeps EP/FSDP collectives synchronized even for inactive experts.
- `NEMOTRONH_EP_EXPERT_RESHARD_AFTER_FORWARD=0`: avoids immediate post-forward expert reshard to reduce short-run overhead.
- `NEMOAUTOMODEL_PP_SKIP_OUTPUT_MERGE=1`: skips last-stage output merge/concat when schedule outputs are unused, lowering PP memory pressure.

## Minimal Validation Checklist for PR

1. Fixed-length SQuAD train smoke (few steps):
- `num_label_tokens > 0`
- finite `loss`, finite `grad_norm`

2. PP+EP startup:
- no mesh null dereference in EP shard setup
- PP schedule builds with/without skip-merge patch

3. Lint/syntax:
- no duplicate `nn` imports in MoE parallelizer
- all edited files compile (`py_compile`)
