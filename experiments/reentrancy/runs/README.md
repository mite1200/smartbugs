main-v2/
  Primary Aggregated Benchmark experiment:
  3 models × 3 prompting strategies × 3 repeats × 432 samples.

combined-v1/
  Pre-planned combined prompting extension:
  3 models × 1 combined strategy × 3 repeats × 432 samples.

rsd-v1/
  Reentrancy Scenarios Dataset experiment:
  3 models × 4 prompting strategies × 3 repeats × 143 scenarios.

preflight-*/
  Small technical validation runs performed before the final experiments.
  They were used to verify provider connectivity, output handling,
  concurrency, context limits, pacing, and execution/resume behavior.
  They are not part of the reported thesis evaluation.

locks/
  Runtime lock files used by the experiment orchestration layer.
  Not experimental observations.

Final thesis metrics are generated from main-v2, combined-v1, and rsd-v1; preflight runs are excluded from the evaluation.