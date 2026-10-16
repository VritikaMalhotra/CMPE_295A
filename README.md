# Measuring and Repairing Quantization Damage in Small LLM Agents on Commodity Hardware

**A research testbed for measuring tool-call accuracy, output validity, and inference cost across local model configurations.**

This repository contains the team's Workbook 1 pilot code and experiment scaffolding for CMPE 295A, Fall 2026. The current runnable work focuses on BFCL tool-calling evaluation with quantized Qwen3 models, schema-derived grammar constraints, and GPU energy profiling. Additional spikes explore inference setup and checkpoint/resume behavior.

> **Research status:** Files and measurements in this repository are pilot/development artifacts, not final study results. The October 8 pilot ran on a shared Google Colab A100-SXM4-40GB using llama.cpp v0.6.0 (`d81235049384534c167caea52b85a694f6103d14`). The team's planned evaluation platform is the SJSU CoE HPC cluster; see the meeting notes for current access and execution status.

## Questions This Testbed Helps Measure

- How do model quantization formats affect local tool-call accuracy and failure types?
- Can grammar-constrained decoding improve syntactic and schema validity without forcing a tool call when a task does not need one?
- What inference throughput and GPU energy use accompany these configurations?
- How does a small checkpoint/resume workflow behave when training or execution is interrupted?

These are experiment directions represented by the current scripts, not claims that the questions have been conclusively answered. The formal problem statement, hypotheses, controls, and semester plan are maintained in Workbook 1.

## Repository Map

| Path | Contents |
| --- | --- |
| [`testbed/`](testbed/) | BFCL v2 adapter, task-specific grammar generation, and the pilot runner. |
| [`spikes/`](spikes/) | Focused experiments: llama.cpp build/serve and tool calls, grammar constraints, BFCL single-turn evaluation, NVML energy sampling, and checkpoint/resume. |
| [`spikes/vendor/`](spikes/vendor/) | Pinned vendored JSON-schema-to-GBNF converter and provenance note. |
| [`analysis/`](analysis/) | CSV-to-figure generation and slide-number extraction utilities. |
| [`configs/`](configs/) | Reserved for experiment configurations. |
| [`train/`](train/) | Reserved for training workflows. |
| [`slurm/`](slurm/) | Reserved for cluster job scripts. |
| [`meetings/`](meetings/) | Advisor and team meeting notes, decisions, and action items. |
| [`CONTRIBUTORS.md`](CONTRIBUTORS.md) | Contributor attribution and historical code-to-contribution mapping. |

The directory structure follows Workbook 1, Chapter 8. Experiment spikes are kept as scripts under `spikes/` so each investigation has a concrete, reviewable entry point.

## Experiment Entry Points

| Script | Purpose |
| --- | --- |
| [`testbed/run_v2.sh`](testbed/run_v2.sh) | Starts a local llama-server, runs BFCL v2 generation, then evaluates the results. |
| [`testbed/bfcl_v2.py`](testbed/bfcl_v2.py) | Adds pilot protocol controls to the BFCL Qwen function-calling handler: thinking mode, prompt caching, grammar, sampling, and seed. |
| [`testbed/grammar.py`](testbed/grammar.py) | Converts BFCL function schemas into per-task GBNF grammars, including free-text and forced-call variants. |
| [`spikes/s3_grammar_r1.py`](spikes/s3_grammar_r1.py) | Compares unconstrained and grammar-constrained raw completion for BFCL tasks. |
| [`spikes/s6_nvml_energy.py`](spikes/s6_nvml_energy.py) | Samples GPU power, energy, memory, and clocks during generation; writes sample and summary CSVs. |
| [`spikes/s7_toy_qad_resume.py`](spikes/s7_toy_qad_resume.py) | Exercises checkpoint/resume behavior on a small QAD training run. |
| [`analysis/make_figs.py`](analysis/make_figs.py) | Builds pilot figures from result CSVs in a progress directory. |
| [`analysis/slide_numbers.py`](analysis/slide_numbers.py) | Extracts slide-number references for analysis and reporting. |

Other spike scripts document the individual build, serving, benchmark, and resume procedures in their module docstrings.

## Running the Pilot Harness

The checked-in runner is designed for the team's prepared Colab environment, not a generic workstation. It currently assumes these paths exist:

- `/content/work/llama.cpp/build-cuda/bin/llama-server`
- `/content/bfcl-env/bin/python`
- `/content/work/v2/bfcl_v2.py`
- `/content/work/Qwen3-0.6B`

It also expects the model path and BFCL category to be supplied by the caller. Its positional interface is:

```text
run_v2.sh TAG MODEL CATS NP PORT TEMP GRAMMAR THINKING SAMPLING_JSON SEED THREADS
```

The runner writes `protocol.json`, server/generation/evaluation logs, and status timestamps under `/content/work/bfcl/v2/<TAG>/`. The adapter accepts `V2_GRAMMAR=none|free|forced`; `V2_THINKING`, `V2_SAMPLING` (JSON), and `V2_SEED` control the corresponding protocol options. Review the script before adapting it: paths and server settings are currently environment-specific.

For the standalone grammar spike, see the invocation documented in [`spikes/s3_grammar_r1.py`](spikes/s3_grammar_r1.py). The energy script requires a CUDA-capable NVIDIA GPU, NVML support, a compatible llama.cpp build, and prompt token IDs in JSON format.

## Generate Figures

Install the analysis dependencies (`pandas` and `matplotlib`) in your Python environment, then place the pilot CSVs and `RESULTS.md` in a progress directory. From the repository root:

```bash
python analysis/make_figs.py --progress /path/to/progress
```

The script reads `bfcl_pilot.csv` and, when present, `bfcl_nonlive_pilot.csv`, `nvml_summary.csv`, `bfcl_v2_pilot.csv`, `grammar_pilot.csv`, and `RESULTS.md`. It creates a `figs/` directory under the progress path. These inputs and generated figures are not checked into this repository by default.

## Reproducibility Notes

- The recorded pilot used llama.cpp v0.6.0 and BFCL evaluation package version `2026.3.23`; see individual spike docstrings for their specific setup and workload.
- In the BFCL v2 adapter, prompt caching is disabled for requests; grammar, thinking behavior, sampling parameters, and seed are set by environment variables.
- The energy spike subtracts measured idle power and reports both counter-based and sampled energy per token. GPU sharing and clock behavior can affect measurements.
- Preserve each run's model/quantization, dataset and category, protocol settings, seed, software versions, hardware, and raw outputs alongside derived figures.
- Do not interpret pilot charts as final results. Confirm the final methodology and analysis against the project workbook before drawing conclusions.

## Team

CMPE 295A, Fall 2026, advised by Dr. Bertin Cordova-Diba.

- Ekant Kapgate
- Vritika Malhotra
- Yashashav Devalapalli Kamalraj

See [`meetings/README.md`](meetings/README.md) for the meeting log and [`CONTRIBUTORS.md`](CONTRIBUTORS.md) for contribution details.
