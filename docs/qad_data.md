# QAD training mix and R2b calibration conversations (Yashashav; Workbook 2 package K)

Workbook 2 Ch. 3: "The training mix is public function-calling and general text, filtered to share no 13-gram
with any benchmark task." Planning figure: about 50 million tokens per run (Ch. 3, R3 sizing). The R2b
ranking needs "calibration conversations that are disjoint from both benchmarks" (Ch. 3, Layer sensitivity).
Both come from one build: `train/build_qad_mix.py`, job `slurm/qad_mix.sbatch`. Sources chosen by Yashashav on
2026-11-09 from a shortlist whose licences were read from the Hugging Face dataset cards and API.

## Sources (revisions pinned in `SOURCES` in the script)

| Source | Hugging Face id, revision | Licence | Budget (Qwen3-0.6B tokens) | Used |
|---|---|---|---|---|
| Toucan | `Agent-Ark/Toucan-1.5M` SFT config, `0df3cf3` | Apache-2.0 | 15M | `multi-turn` + `single-turn-diversify` subsets; 2,000 multi-turn conversations held out for R2b |
| Nemotron | `nvidia/Nemotron-Agentic-v1` `tool_calling`, `650d590` | CC-BY-4.0 | 5M | `reasoning_content` dropped (non-thinking mode) |
| ToolACE | `Team-ACE/ToolACE`, `6bda777` | Apache-2.0 | 5M | Python-style calls parsed into structured calls; its system prompt dropped |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` sample-10BT shard `000_00000`, `87f0914` | ODC-By 1.0 (+ CommonCrawl terms) | 20M | general text |
| Cosmopedia v2 | `HuggingFaceTB/smollm-corpus` `cosmopedia-v2` shard 0, `3ba9d60` | ODC-By 1.0 | 5M | general text |

Not used: `Salesforce/APIGen-MT-5k` (CC-BY-NC-4.0, built on tau-bench airline/retail, and its card forbids
training models that compete with OpenAI); `HuggingFaceTB/smoltalk` magpie-ultra (Llama 3.1 outputs: the Llama
licence would require "Llama" in the published model names); Nemotron `interactive_agent` (its records copy
tau2's agent prompt verbatim, so the 13-gram filter would drop them all).

**Obligations when the QAD checkpoints are published** (Workbook 2 release plan): credit NVIDIA's
Nemotron-Agentic-v1 (CC-BY-4.0) and the ODC-By sources in every model card. Toucan's tool responses are real MCP
output collected Jun–Sep 2025; its card says personal data was removed on a best-effort basis only.

## How a record gets in

1. **Normalise** to OpenAI-style `messages` (system / user / assistant / tool; assistant tool calls as
   `{"name", "arguments"}`) plus the record's tool schemas. Toucan's `tool_call` / `tool_response` turns become
   assistant tool calls and tool turns.
   ToolACE's tool list is read from its system prompt, and the prompt itself is dropped (it is BFCL's own prompt
   wording). Its `type: dict` becomes `type: object`.
2. **13-gram filter (FR-E5).** The record is dropped if any 13-gram of its messages or tool schemas is in the
   benchmark text. The benchmark text is everything `scripts/benchmark_ngrams.py dump` writes: BFCL multi-turn
   data and function docs, tau2 tasks and policies, plus BFCL's default prompts and tau2's agent prompts.
3. **Dedupe** on a content hash across all sources.
4. **Budget.** The order inside each source is a seeded shuffle (seed 295, salted per source). Records are taken
   until the source's budget is reached.
   Tokens are counted with the **Qwen3-0.6B tokenizer** on the plain rendering plus the tool schemas. This is the
   planning unit only: vocabularies differ, so the pack step reports each model's own count under its chat
   template.
5. **R2b holdout** is drawn first: 2,000 Toucan multi-turn conversations. Its hashes block the training draw, and
   the manifest checks that the two id sets are disjoint.

## Outputs (`$CMPE295_GROUP/data/qad_mix/`; summary copied to `results/m3/qad_mix_manifest.json`)
- `train.jsonl`: one record per line (`id`, `source`, `kind` chat|text, `messages` + `tools` or `text`,
  `n_tokens_qwen3`), mixed in seeded order.
- `rank_conversations.jsonl`, and `rank_text.txt` for `RANK_TEXT` in the R2b ranking.
- `manifest.json`: SHA-256 of every raw file and output, filter counts per source, token totals. `pass` means the
  whole rank text shares no 13-gram with the benchmarks, and the holdout is disjoint from training.

## Calls made in the build (confirmed 11-09 by Yashashav)

All three are accepted as they are: the role-prefixed ranking text, dropping the ToolACE records in the two
other call formats, and packing in sprint 7.
- **The R2b ranking text is plain role-prefixed text** (`user: …`, `assistant: <tool_call>{json}</tool_call>`,
  `tool: …`), one file for every model:
  - `llama-perplexity` reads chat-template markers as ordinary text, so each model's own template would add junk
    tokens.
  - One shared text keeps the rankings comparable across models.
  - Tool schemas are left out of the ranking text, so the 20 × 512-token window covers more conversations.
- **ToolACE records in two other call formats** (`APIName=>{…}` and `{<apiname>-{…}}`, 518 of 11,300) count as
  parse failures and are dropped. A 500-record check parsed 95.6%.
- **No packing yet.** Turning `train.jsonl` into `[N, seq_len]` token tensors with each model's chat template, and
  the loss mask for QAD-KL+CE, belong to the training pipeline (Sprint 7).
