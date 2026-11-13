# Our agents and our data

This page answers two questions from the 2026-11-09 feedback (Gao; Bertin on the data): what kind of agent we
test and what it is for, and exactly which domains and data the study uses. Counts come from the bfcl-eval
2026.3.23 and tau2-bench-verified (864350a) data in the testbed environment, read on 2026-11-10, and from the
experiment configs (`configs/experiment/ref_*.yaml`). The counts are in
`results/m3/benchmark_domains.json` (`scripts/benchmark_domains.py`, CPU job 111946).

## 1. The agent

**Category: a single tool-calling (function-calling) LLM agent in a ReAct-style loop.** Each step, the model
reads the conversation and the tool schemas, then either calls tools or answers in text. A called tool runs, and
its result goes back into the conversation for the next step. This is the reason → act → observe loop from
Workbook 2 Ch. 1.

**Purpose.** The study looks at small models (0.6B to 4B parameters) that could run on a laptop or phone as an
assistant that does things through APIs: managing files, booking travel, trading, and handling an airline,
retail or telecom customer. The question is where 4-bit quantization breaks such an agent, and which repairs fix
it.

**What is in the loop.** There is one agent. The other parts are the environment, not agents:

| Part | Role | Who or what |
|---|---|---|
| Agent | Decides every action: which tool to call with which arguments, or what to say | The small model under test (Qwen3, SmolLM3, LFM2.5), served by llama-server (llama.cpp d812350) from a GGUF file, CUDA or CPU |
| Tool executors | Run a call deterministically and return its result | BFCL: instances of the benchmark's Python API classes (e.g. `GorillaFileSystem`), run by bfcl-eval's executor. τ²: the domain environment and its database |
| User | Supplies the requests | BFCL: scripted user turns from the task file. τ²: an **LLM simulated user** (gpt-oss-20b MXFP4, temperature 0, low reasoning effort, one cluster job serving all agents), which follows the task's hidden instructions. In telecom it also runs tools on "its phone" |
| Judge | Scores the episode | The official checkers, unmodified: bfcl-eval's multi-turn state and response checker; τ²'s database-state evaluator. FR-E2: our re-scoring agrees on 7,506 of 7,506 Qwen3 reference episodes |

So **we do not run a multi-agent system.** The simulated user is a second LLM, but it is part of the τ²
benchmark's environment and is the same for every model and format. Only the agent under test changes between
configurations.

**Per-step mechanics** (C2 runner, C3 backend). The prompt is the model's own chat template with the tool schemas
in the system turn (Qwen3: bfcl-eval's handler; SmolLM3, LFM2.5: their chat templates, `testbed/prompt_formats.py`).
Non-thinking mode is used where the template has one. Context is 40,960 tokens, with at most 4,096 new tokens per
step. A turn ends when the model answers in text; bfcl's limit of 20 steps per turn ends a loop. Every step is
logged (C5) and gets one failure class (C6, `docs/codebook_v0.md`).

## 2. BFCL v4 multi-turn: API domains

Three categories, a fixed-seed sample of **100 tasks each** (seed 295, `testbed/adapter.py`) out of 200 per
category:
- **base:** every function is available, and every request has what it needs.
- **miss_func:** a needed function is withheld at first and revealed in a later turn. The right behaviour is to
  say the call can't be made yet.
- **miss_param:** a request leaves out a required value. The right behaviour is to ask the user.

Each task gives the agent one or two API classes (97 tasks have one, 203 have two). Turns per task: base mean
3.7 (1–7), miss_func 4.7 (2–8), miss_param 4.7 (2–8).

| API class (domain) | Functions | Tasks using it in our sample: base / miss_func / miss_param | Total |
|---|---|---|---|
| VehicleControlAPI (car: engine, brakes, doors, climate, fuel, navigation, tyres) | 22 | 26 / 26 / 27 | 79 |
| GorillaFileSystem (shell-like file system: `ls`, `cd`, `mv`, `cat`, `grep`, …) | 18 | 20 / 28 / 27 | 75 |
| TravelAPI (flight booking and cancellation, credit cards, budget, insurance, invoices) | 18 | 25 / 28 / 21 | 74 |
| TradingBot (stock lookup, orders, watchlist, account funding) | 20 | 29 / 18 / 25 | 72 |
| TwitterAPI (posts, comments, retweets, follows, search) | 14 | 19 / 21 / 19 | 59 |
| MessageAPI (contacts, sending, searching and deleting messages) | 10 | 16 / 21 / 20 | 57 |
| TicketAPI (create, edit, resolve and close support tickets) | 9 | 16 / 16 / 14 | 46 |
| MathAPI (arithmetic, unit conversion, mean, standard deviation) | 17 | 12 / 14 / 15 | 41 |

Counts in a row add up to more than 100 per category because two-class tasks count once for each class.

## 3. τ²-bench-verified: customer-service domains

τ²-bench-verified (Amazon AGI's verified release of τ²-bench, commit 864350a), `base` split. **We keep only tasks whose gold agent actions
write to the database** (decision 6): an agent that does nothing would pass the others.

| Domain | What the user wants | Base tasks | Write tasks we run | Agent tools | User tools | Write tools |
|---|---|---|---|---|---|---|
| airline | Book, change or cancel flights; baggage; compensation, under the airline policy | 50 | **26** | 14 | 0 | 6 |
| retail | Cancel, modify, exchange or return orders; update address or payment | 114 | **104** | 16 | 0 | 7 |
| telecom | Technical support (mobile data, roaming, suspended lines, bills); the user acts on their own phone | 114 | **71** | 13 | 30 | 6 |
| **Total** | | 278 | **201** | | | |

Each run covers **501 tasks** (300 BFCL + 201 τ²) × (4 sampled trials at T 0.7 + 1 greedy pass) = **2,505
episodes** per model, format and backend.

## 4. Text corpora used outside the agent runs

| Use | Data | Source | Filter |
|---|---|---|---|
| KL check of every quantized file against BF16 (M2, M3) | WikiText-2 raw, **test** split (SHA-256 `173c87a5…`), ctx 512, 50 chunks | Merity et al. 2016, via llama.cpp's `scripts/get-wikitext-2.sh` (`scripts/get_eval_text.sh`) | none (never used for calibration) |
| Importance matrix for IQ4_XS (M3) | WikiText-2 raw, **train** split, 200 × 512 tokens | same | every line sharing a 13-gram with any benchmark text is dropped (FR-E5): 0 of 36,719 lines hit (`results/m3/calib_ngram_check.json`) |
| R2b layer-sensitivity ranking (J1) | 2,000 held-out Toucan multi-turn conversations with tool replies cut to 256 tokens (`rank_text_t256.txt`, 4,632,056 tokens, SHA-256 `dc06ce1f…`); the ranking reads the first 512,000 tokens | `Agent-Ark/Toucan-1.5M` (Apache-2.0) | 13-gram filter; disjoint from the QAD training set |
| QAD training mix (R3, K) | 41,545 records, 50.0M Qwen3 tokens: Toucan 15M, Nemotron-Agentic tool_calling 5M, ToolACE 5M, FineWeb-Edu 20M, Cosmopedia v2 5M | Hugging Face, revisions pinned (`docs/qad_data.md`) | 13-gram filter against BFCL and τ² text, then dedupe |
| Parity prompts (FR-E17) | 200 fixed prompts per model, replayed from the Oct 28 BF16 episodes | our own logs | — |

The **benchmark text** behind every 13-gram filter is everything `scripts/benchmark_ngrams.py dump` writes:
the BFCL multi-turn data and function docs, the τ² tasks and policies, and the default prompts of BFCL and τ².
