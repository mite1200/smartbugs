# Reentrancy Scenarios Dataset preparation

This directory records benchmark-preparation facts for the Reentrancy Scenarios
Dataset (RSD). It does not define or execute an LLM experiment.

## Pinned source

- Repository: `https://github.com/ca-foscari-reentrancy-research-group/reentrancy-detection-llms.git`
- Commit: `d7c9496121931cff7c55b2047d18fb048fefe2cb`
- Source directory: `benchmarks/RSD/src`
- Evaluation unit: one Solidity source file, treated by the source repository's
  result tables as one labeled scenario. A file may contain several contract or
  interface declarations, which remain together as one source input.
- Ground truth: the terminal filename component `_reeN` means `Reentrant` and
  `_safeN` means `Safe`.

The pinned directory contains 143 Solidity files: 71 Reentrant and 72 Safe.
There are no other labels. All 143 files use `pragma solidity ^0.8.20` and none
contains an import directive. Forty files contain more than one contract or
interface declaration; those declarations are part of the same source scenario,
not separate samples.

The repository README and result artifacts contain stale counts of 150 or 154.
The pinned history shows that 11 `16_Human*` source files were removed, while
the checked-in analyzer-result CSVs and bytecode directories were not updated.
Two active files were also renamed from `00_BasicCall_*` to `00_Basic_*` after
the result CSVs were produced. The manifest therefore follows the pinned source
tree rather than documentation or stale result rows.

## Generate the manifest

From the SmartBugs repository root:

```console
python experiments/reentrancy/scripts/prepare_rsd_benchmark.py \
  /tmp/reentrancy-detection-llms-rsd-prep \
  --timestamp 2026-10-14T00:00:00+00:00
```

This writes `rsd.csv`, `rsd.json`, and `rsd-metadata.json` in this directory.
The metadata includes class, Solidity-version, declaration-count, duplicate,
and exact Aggregated Benchmark overlap statistics.

## Duplicates and Aggregated overlap

The 143 samples have 142 unique byte contents. One same-label duplicate group
contains `04_CrossModFold_safe1.sol` and `04_CrossMod_safe3.sol`. No duplicate
group crosses labels. No RSD source bytes occur in the pinned 432-file
Aggregated Benchmark.

## Protocol compatibility and caveats

RSD can use the same one-source-prompt binary classification protocol as the
Aggregated Benchmark when the unit is explicitly described as a source-file
scenario rather than an individual declared contract. Auxiliary interfaces and
contracts are embedded in the same `.sol` file; there are no imports or separate
attacker source files to resolve.

The frozen preprocessor succeeds unchanged and preserves line and character
positions while replacing comments with spaces before adding line numbers.
RSD uses comments for explanatory text such as control-flow notes and, in a few
read-only-reentrancy files, commented-out attacker scaffolding. Removing these
comments does not alter executable Solidity semantics and avoids exposing
author-written hints, but it means the prompt contains only the active source.

The repository includes compiled bytecode and per-file analyzer results for the
older 154-file state. Those artifacts are useful historical baselines but do not
correspond path-for-path to the pinned 143-source tree and must not be treated as
an exact current benchmark run without reconciliation.

## Exact offline Qwen token counts

With the exact pinned tokenizer available locally:

```console
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python experiments/reentrancy/scripts/analyze_rsd_qwen_context.py \
  /tmp/reentrancy-detection-llms-rsd-prep \
  --tokenizer-path /tmp/qwen38-tokenizer
```

This renders all four frozen prompt strategies through the existing reentrancy
profile and preprocessing pipeline. It does not invoke a model or provider API.