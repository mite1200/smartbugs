# Reentrancy experiments

The canonical Aggregated Benchmark manifests are generated from the Ca' Foscari
`reentrancy-detection-llms` repository at commit
`d7c9496121931cff7c55b2047d18fb048fefe2cb`.

Acquire the benchmark separately:

```console
python experiments/reentrancy/scripts/acquire_benchmark.py /path/to/reentrancy-detection-llms
```

Regenerate and validate the manifests:

```console
python experiments/reentrancy/scripts/generate_aggregated_manifest.py \
  /path/to/reentrancy-detection-llms
python experiments/reentrancy/scripts/validate_aggregated_manifest.py \
  /path/to/reentrancy-detection-llms
```

`contract_id` and `contract_path` are the same POSIX-normalized path relative
to the benchmark repository root. Ground truth comes only from the immediate
`reentrant` or `safe` source directory. The optional `source` value is derived
only from terminal `_cgt`, `_hg`, or `_rs` filename suffixes; all other files
are marked `unspecified`.

## Experiment execution

The frozen analyzer tag is `thesis-llm-analyzer-v1`. The deterministic plan is:

```console
python experiments/reentrancy/scripts/generate_run_plan.py
```

It contains 27 model/strategy/repeat conditions and 11,664 expected Aggregated
Benchmark observations.

Plan one condition without provider calls:

```console
python experiments/reentrancy/scripts/run_experiment.py \
  /path/to/reentrancy-detection-llms \
  --model-condition gemini-3.8-flash \
  --strategy zero_shot \
  --repeat 1
```

Plan a three-contract infrastructure subset without changing the manifest:

```console
python experiments/reentrancy/scripts/run_experiment.py \
  /path/to/reentrancy-detection-llms \
  --model-condition gemini-3.8-flash \
  --strategy zero_shot \
  --repeat 1 \
  --limit-contracts 3
```

Plan or execute exactly one canonical-manifest contract by its full, exact
`contract_id`:

```console
python experiments/reentrancy/scripts/run_experiment.py \
  /path/to/reentrancy-detection-llms \
  --model-condition gemini-3.8-flash \
  --strategy zero_shot \
  --repeat 1 \
  --contract-id benchmarks/aggregated-benchmark/src/reentrant/example.sol
```

`--contract-id` and `--limit-contracts` are mutually exclusive. An unknown ID
fails before analyzer or provider execution.

Add `--execute` only when provider credentials are loaded and paid execution is
intended. `run_matrix.py` plans all 27 conditions by default and executes them
sequentially only when passed `--execute`.

### Resume semantics

`llm-provenance.json` is the authoritative paid-observation marker because the
frozen analyzer writes it before returning to SmartBugs. A matching provenance
record is terminal and skipped for valid classifications, invalid model output,
and technical errors. A nonempty expected result directory without matching
provenance is blocked as a conflict and is never automatically overwritten or
retried. Only a completely absent expected result directory is scheduled.

Results retain the native SmartBugs tree under:

```text
experiments/reentrancy/runs/results/<runid>/<benchmark-relative-contract-path>/
```

Collect the raw per-inference table and coverage report with:

```console
python experiments/reentrancy/scripts/collect_predictions.py
```

This writes `evaluation/predictions.csv` and `evaluation/coverage.json`. It does
not perform majority voting or calculate evaluation metrics.

## Qwen Cerebras context analysis

The exact frozen prompts can be measured with the official pinned
`Qwen/Qwen3.8-27B` tokenizer and chat template without invoking the model:

```console
python -m pip install transformers jinja2
python experiments/reentrancy/scripts/analyze_qwen_context.py \
  /path/to/reentrancy-detection-llms
```

The script renders the same preprocessed source and profile prompt used by the
analyzer, then serializes one user message with `add_generation_prompt=true`
and `reasoning_effort=medium`. It writes per-prompt counts and a summary under
`analysis/qwen3.8-27b-cerebras-context/`.