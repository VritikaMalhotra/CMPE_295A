# Abstract

Small open language models of 0.6 to 4 billion parameters now run as tool-calling agents on laptops and phones, and they reach users almost only as 4-bit quantized files. Published evaluations judge these files by aggregate benchmark scores, which can stay flat while an agent fails more often at individual steps, and the three available repairs (grammar-constrained decoding, added bits and quantization-aware distillation) have never been compared on agents at this scale. This project builds an open testbed that runs BFCL multi-turn and τ²-bench-verified against llama.cpp's real 4-bit kernels on its CUDA and CPU backends, logs and classifies every agent step, and compares each 4-bit configuration with its 16-bit reference on the same tasks and seeds. It measures where 4-bit damage concentrates by step type and conversation depth, compares the three repairs with their controls (including a train-longer control for distillation), tests whether repairs fix the underlying failures or only the score, and measures each repair's cost in memory, latency, extra tokens and energy. Development runs on six models (Qwen3-0.6B, 1.7B and 4B, SmolLM3-3B, LFM2.5-1.2B and 2.6B) in three 4-bit formats, about 2,500 episodes per configuration on the SJSU CoE HPC cluster, already show uneven damage: per-step failures rise by up to 1.33 times under Q4_0, Q4_0 lowers episode success in every model, and some formats leave the step-failure rate unchanged while success still drops. The confirmatory analysis is pre-registered.

# Glossary

**Agent step.** One model output in an episode: a tool call, a text reply to the user, or the end of a turn.
**Amplification ratio.** The rate of a failure type per step under a 4-bit configuration divided by its rate under the 16-bit reference on the same tasks and seeds.
**BF16.** The 16-bit brain floating-point format, used here for the full-precision reference models.
**BFCL.** The Berkeley Function-Calling Leaderboard; this project uses its multi-turn categories.
**GBNF.** llama.cpp's grammar format for constraining generated text.
**GGUF.** The model file format of llama.cpp.
**IQ4_XS, Q4_0, Q4_K_M.** llama.cpp 4-bit weight formats: non-uniform levels with an importance matrix; one scale per block of 32 weights; and k-quant super-blocks with some tensors at higher precision.
**KL divergence.** Kullback-Leibler divergence between the next-token distributions of a quantized model and its 16-bit reference.
**llama.cpp.** The open-source C/C++ inference engine through which small models run on devices, with CUDA and CPU backends.
**QAD.** Quantization-aware distillation: training a quantized student to match the output distribution of its full-precision teacher.
**QAT.** Quantization-aware training: training with simulated quantization in the forward pass.
**Simulated user.** A language model that plays the customer in τ²-bench, following hidden task instructions.
**Slurm.** The job scheduler of the SJSU CoE HPC cluster.
**τ²-bench-verified.** A corrected version of the τ²-bench customer-service benchmark (airline, retail and telecom domains).
