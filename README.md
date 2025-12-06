# VLA



## Benchmarks

### Video benchmarks
* Video-MME (w/o sub)
* MVBench
* MMBench-Video
* MLVU

### Multi-Image
* BLINK
* MuirBench
* Mantis-Eval
* MileBench
* MIRB

## Model Comparison

Criteria: 
* Parameters <= 2B (hard limit due to GPU VRAM)
* Completely open model released in 2025 (for obvious reasons :))

Following models are selected for comparisons.

* Qwen3-vl-2B-Instruct
* InternVL-3.5-2B
* LFM2-1.6B
* SmolVLM2 - 2.2B

Metric score comparison on video understanding benchmarks:

| Models/Benchmark | Qwen3-vl-2B-Instruct | InternVL-3.5-2B | LFM2-1.6B | SmolVLM2 - 2.2B|
|----------|----------|----------|----------|------------|
| Video-MME   | 61.9     | 58.4     | -     |     52.1       |
| MVBench    | 61.7     | 65.9     | -     |      46.3      |
| MMBench-Video    | -     | 1.56     | -     |-|
| MLVU    | 68.3     | 64.4     | -     |55.2|
| Video-MMMU    | 41.9     | -     | -     |-|
| LVBench    | 47.4     | -     | -     |-|


Metric score comparison on multi-image benchmarks:

| Models/Benchmarks |Qwen3-vl-2B-Instruct | InternVL-3.5-2B | LFM2-1.6B | SmolVLM2 - 2.2B|
|----------|----------|----------|----------|------------|
| BLINK    | 53.8     | 51.3     | 44.50     |-|
| MuirBench    | 47.4     | 44.0     | -     |-|
| MantisEval    | -     | 58.5     | -     |-|
| MileBench    | -     | -     | -     |-|
| MIRB    | -     | 45.9     | -     |-|

