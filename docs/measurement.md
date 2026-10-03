# Input-pipeline memory measurement

The frozen measurement was completed on 3 October 2026. All 30 worker receipts
and summary calculations were verified. See [measured results](measurement-results.md)
for the observations, machine context and limits. The measurement design is
preserved below; the commands use the public repository layout.

## Property and comparison

Measure the **peak process resident memory**, reported in MiB, while consuming one
complete shuffled input-dataset pass. This tests the practical memory cost of the
former decoded-image shuffle and the revised index-before-decode implementation.
The comparison calls the real `dataset()` functions from frozen `data.py` files:

- Baseline: upstream commit `5f2e2ba9796b00255a3d93808f7ecdefdb6c5ecc`.
- Indexed: the measured revision, identified by SHA-256 in
  the plan. Future edits do not silently change the frozen experiment.

The earlier implementation supports only two classes, so **both use the same
binary-label input mode**. This is a pipeline-memory comparison, not a comparison
of classifier quality or three-class training. It compares whole implementations;
it cannot attribute every difference solely to the shuffle representation.

Windows records `GetProcessMemoryInfo().PeakWorkingSetSize`, in bytes, divided by
`2**20` for MiB. The high-water mark includes the fresh interpreter, TensorFlow
startup, manifest/source/image checks, pipeline construction, full iteration and
identical streaming correctness checks. It is read before receipt serialization.
It excludes parent fixture generation, CNN training, predictions and cameras.
It is **not** isolated shuffle-buffer memory, a sampled instantaneous value, GPU
memory, or a latency measurement. Startup peaks are not subtracted.

The helper also defines the Linux `ru_maxrss` conversion; Windows and Linux
methods must never be pooled. The frozen local plan requires its recorded
platform, Python version and full installed-package inventory.

## Frozen workload

| Setting | Value |
|---|---|
| Synthetic corpus | Fixed 16x16 PNGs, constant RGB pixels encoding each unique integer ID |
| Cases | 256, 1,024 and 4,096 images |
| Input tensors | 128x128x3 float32; binary float32 labels equal to ID modulo 2 |
| Batch size | 32; one full shuffled pass |
| Paired seeds | 17, 29, 43, 71, 101 |
| Runs | 30 sequential fresh processes; five per implementation per case |
| Process order | Seed-1729 shuffled pairs; baseline-first/indexed-first as balanced as possible |
| Runtime | CPU-only; two intra-op threads, one inter-op thread; deterministic TensorFlow operations |
| Failure policy | Stop, retain all available receipts; no partial comparative summary or automatic retry |

Each pair is adjacent and shares its case and seed. With five pairs per case,
first position is split 3/2 or 2/3, and 8/7 overall; exact balance is impossible.
Filesystem caches are not flushed. The worker verifies shape, dtype, every pixel,
every ID exactly once, and label consistency without retaining decoded batches.
Identical cross-implementation shuffle order is not required.

The largest decoded tensor payload alone is `4096 * 128 * 128 * 3 * 4` bytes,
or **768 MiB**. This is a calculation, not measured usage or an upper bound.
TensorFlow and temporary allocations add memory. The runner checks for at least
3 GiB available RAM before each full worker, uses one worker at a time and a
300-second per-worker timeout. That headroom check is not a hard memory cap.

## Reproducing the comparison

Use the pinned Python 3.12 CPU runtime described in the
[research workflow](research-workflow.md). From the repository root, prepare a
new experiment using the preserved baseline file included with this repository:

```powershell
python tools/benchmark_input_memory.py prepare --baseline docs/evidence/baseline-data.py --output experiments/input-memory-new
python experiments/input-memory-new/benchmark_input_memory.py run experiments/input-memory-new
```

The baseline is a byte-exact copy from the recorded public commit; the harness
checks its SHA-256 before preparation. The command freezes the current revised
implementation and installed runtime into the new plan. It reproduces the
protocol, not the exact memory values from another execution. See the original
[portable plan](evidence/input-memory-plan.json) and
[observations](evidence/input-memory-results.json) for the recorded experiment.

Optional run flags record context: `--power-mode`, `--power-source`,
`--background`. Omitted fields remain `not recorded`. Keep power mode consistent
and record substantial background activity. The program prints progress for
30 workers, then a summary and result directory. No browser, dataset download or
camera is required. Use a fresh destination for each plan; do not replace an
existing plan after seeing its measurements.

The plan seals copied scripts, source implementations and an input manifest;
each PNG has its own hash. Hashes detect accidental drift, not malicious rewriting
of all receipts. Each run gets a fresh results directory with context, commands,
exit codes, logs, worker JSON, raw records and summary JSON/text. A failed run
retains a failure receipt. A hard-killed parent can leave `.run.lock`; inspect the
unfinished run before manually clearing a stale lock. Never delete failed results
just to keep successful ones.

## Summaries and interpretation

Report each size separately: mean, median, sample SD, sample variance, minimum
and maximum of its five process peaks. SD/variance use `ddof=1`. Paired results
are `indexed - baseline` in MiB and `indexed / baseline` as a ratio, calculated
per seed before aggregation. Negative deltas and ratios below one indicate a
lower observed indexed-process peak. All valid observations are retained.

The spread describes these fresh-process observations on one fixed synthetic
workload and machine. It is not uncertainty about real-world mask images,
population generalization, individual request tails or model accuracy. RSS can
vary with operating-system memory management and shared-library residency.

An eight-image/four-worker smoke plan checks the harness only. Its outputs are
labelled as smoke evidence and must not be quoted as benchmark results. Tests
also exercise summaries with known numbers, invalid records, changed sources,
version drift, overwrite protection and failed/timeout workers.

Real classification accuracy is still unmeasured. The historical corpus's exact
provenance and annotation coordinate convention remain unresolved; neither is
needed for these generated input-pipeline fixtures. A later accuracy study must
resolve those details and report the three mask categories separately.

Metric references: [Microsoft process memory counters](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters),
[GetProcessMemoryInfo](https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-getprocessmemoryinfo),
and [Python resource interface](https://docs.python.org/3/library/resource.html).
