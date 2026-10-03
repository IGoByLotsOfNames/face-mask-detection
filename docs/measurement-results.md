# Input-pipeline memory results

These results come from the completed run on 3 October 2026. The 30 worker
receipts were checked and the summary statistics independently recalculated.
The benchmark was not rerun during that review.

## Observed result

At 4,096 synthetic images, the revised pipeline's mean process peak
resident memory was **272.972 MiB**, versus **1,036.840 MiB** for the preserved
baseline. The mean paired reduction was **73.67%** across five seeds. The revised
pipeline had lower peaks in all 15 case/seed pairs, with no excluded observations.

These are process-lifetime peaks including interpreter/TensorFlow startup,
input verification and one shuffled pass. They do not measure only a shuffle
buffer, training memory, latency, mask-classification accuracy or camera quality.

| Images | Implementation | n | Mean MiB | Median MiB | Sample SD MiB | Sample variance MiB^2 |
|---:|---|---:|---:|---:|---:|---:|
| 256 | baseline | 5 | 310.324 | 310.312 | 0.149 | 0.022324 |
| 256 | indexed | 5 | 267.887 | 267.902 | 0.142 | 0.020168 |
| 1,024 | baseline | 5 | 455.370 | 455.336 | 0.364 | 0.132472 |
| 1,024 | indexed | 5 | 268.655 | 268.805 | 0.296 | 0.087605 |
| 4,096 | baseline | 5 | 1036.840 | 1036.633 | 0.581 | 0.337379 |
| 4,096 | indexed | 5 | 272.972 | 272.953 | 0.330 | 0.108760 |

| Images | Mean indexed - baseline MiB | Sample SD MiB | Mean paired reduction |
|---:|---:|---:|---:|
| 256 | -42.437 | 0.289 | 13.67% |
| 1,024 | -186.716 | 0.409 | 41.00% |
| 4,096 | -763.868 | 0.448 | 73.67% |

All standard deviations and variances use `ddof=1` across five fresh processes
per implementation and case. Dispersion describes these process observations,
not request-tail latency or uncertainty over real image populations. Percentage
reduction is the mean of `100 * (1 - indexed_peak / baseline_peak)` over matched
seeds. It is calculated per pair before averaging. Cases are not pooled.

## Why the implementation can use less memory

The baseline decodes images into float32 tensors before shuffling
with a buffer equal to the dataset size. The revision shuffles integer indices,
then decodes images as needed, batches them and prefetches one batch. This avoids
holding a dataset-sized shuffle buffer of decoded images.

The large decoded-buffer term scales with image count, height, width and channels;
the index array and manifest still scale with image count. Therefore this is not
a claim of constant total memory. Decoding still occurs and runtime overhead
remains. The experiment compares the two whole implementations, so it does not
isolate the contribution of each changed operation.

## Conditions and reproducibility

The recorded runtime was Windows 11 build 26200, AMD64, 16 logical CPUs;
Python 3.12.14, TensorFlow 2.20.0, Keras 3.15.1, NumPy 2.2.6, Pillow 12.0.0.
CPU only, two intra-op threads and one inter-op thread. OneDNN optimizations off.
All 30 worker exits were zero; worker stderr logs were empty.

The author reported performance power mode and no substantial background
applications. Plugged-in versus battery status was not supplied.
The original runtime context says "not recorded" because optional flags were
omitted. The later author-reported context is recorded separately; raw receipts
were not edited.

The frozen plan used three sizes (256/1,024/4,096), seeds 17/29/43/71/101,
a fixed paired schedule seed of 1729, 32-image batches and one pass.
Synthetic 16x16 constant-RGB PNGs encode unique IDs and are resized to 128x128x3
float32 tensors. Labels are ID modulo 2. Both implementations use the shared
binary input mode because the baseline supports two classes. This experiment
does not compare the upgraded three-category classifier.

Each paired run is adjacent; first positions are balanced 3/2 or 2/3 per size.
Filesystem caches were not flushed. The parent required at least 3 GiB available
RAM before each worker. This is one run of the complete protocol on one machine.

- Plan SHA-256: `ccdfef19e18c0f5d2a49e7a9907efded99a7a685146c905d9de9e20edd0b40de`.
- Baseline commit: `5f2e2ba9796b00255a3d93808f7ecdefdb6c5ecc`.
- Baseline data.py SHA-256: `270ca39a449477078cca3d2aa72a40cb7ecb8c1c914d1ad48eec5ddaf20da4cc`.
- Revised data.py SHA-256: `dbe71ddaeab3983721fd1312a0dea30aa1dd33416ee22a00f3f4e77d9d3f47d6`.
- [Portable frozen plan](evidence/input-memory-plan.json).
- [All observations and summary statistics](evidence/input-memory-results.json).
- [Preserved baseline implementation](evidence/baseline-data.py).

The audit checked all 30 individual worker receipts against the
plan, commands, zero exits, sources, runtime, CPU settings, sample identities
and batch counts. It verified all 4,096 fixture hashes, reproduced the complete
summary with the frozen module, independently recalculated the statistics with
Python's standard library, and hashed all 155 result files before and after review.
All checks passed. The reviewer did not rerun the benchmark.

The public [result record](evidence/input-memory-results.json) retains the
observations, extrema and paired ratio/delta statistics without local-machine
paths. See the [measurement protocol](measurement.md) for exact scope and
reproduction commands.

## Interpretation and limits

The synthetic sizes are useful probes of pipeline scaling; they
are not representative mask images. Results may differ on other machines,
runtime versions, decoding workloads and operating-system memory conditions.
Do not extrapolate the small observed increase in revised peaks into an
unbounded constant-memory guarantee or claim training/inference speedup.

The measured result can be stated as: "Reduced input-pipeline peak process
memory by 73.67% on a controlled 4,096-image synthetic workload, comparing five
paired fresh-process runs on Windows."

The original COVID-era project history remains intact. This revision demonstrates
software engineering and measured memory use; real three-category accuracy,
camera reliability and historical dataset provenance/coordinates remain separate
unresolved or unmeasured items. The original archive and the raw measurement
records remain unchanged.
