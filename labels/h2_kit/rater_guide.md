# Rater guide: labelling agent steps (H2 kit, codebook v0)

**For the human raters only.** Fill the `label` column of your own copy of a sheet, alone, without talking to
the other rater, and without looking at the key files or the classifier. Write one class from the list below in
every row. Use `note` for anything unclear. Your labels are compared with the second rater's (Cohen's kappa,
target ≥ 0.80) and with the classifier's.

Sheets in this folder (labels empty):
- `rater_sample_30.csv`: 30 steps, the codebook v0 check (M2 plan item 14).
- `pilot_300.csv`: 300 steps for the κ study (M3 exit: two independent raters, κ ≥ 0.80).

Steps come from development runs of Qwen3-0.6B on the same 501 tasks, in BF16 and three 4-bit formats. The sheet
hides which format and run each step came from, so judge only what is in the row. The rules are those of `docs/codebook_v0.md`. If this
guide and the codebook disagree, the codebook wins; write the case in `note`.

## What one row is

One **step** is one generation by the agent inside one turn of one episode.
- **BFCL:** a turn is one user request of the task.
- **τ²:** a turn is the agent's answer to one simulated-user message.

| Column | Meaning |
|---|---|
| `benchmark` | `bfcl_multi_turn_base` / `_miss_func` / `_miss_param`, or `tau2_verified_<domain>` |
| `turn`, `step` | Turn index in the episode; step index inside the turn (0 = the first answer to the request) |
| `turn_has_ground_truth` | BFCL: true if the reference solution makes at least one call in this turn |
| `is_last_step_of_turn`, `forced_quit` | The turn ended after this step; the step went past bfcl's 20-step limit |
| `stop_reason` | `eos` = the model stopped; `limit` = it hit the token budget |
| `n_tool_call_blocks` | How many tool-call blocks the output contains |
| `decoded_calls` | The calls that parsed, as Python calls |
| `known_tools` | False if a parsed call names a tool that is not in the task's tool list |
| `schema_errors` | Schema problems of the parsed calls (missing required, unknown argument, wrong type, value outside an enum) |
| `execution_results` | What each call returned |
| `raw_text` | The model's full output |

## The classes: apply the first rule that fits, top to bottom

| # | Label | Failure? | Use it when |
|---|---|---|---|
| 1 | `truncated` | yes | `stop_reason` is `limit`: the output was cut off |
| 2 | `malformed` | yes | The output tries to call a tool but a call block does not parse: more call blocks than parsed calls, or a `<tool_call>` tag with nothing parsed |
| 3 | `forced_quit` | yes | `forced_quit` is true |
| 4 | `unknown_tool` | yes | A parsed call names a tool outside the task's list (`known_tools` false) |
| 5 | `invalid_args` | yes | `schema_errors` is not empty |
| 6 | `exec_error` | yes | A result starts with `Error during execution` (the tool raised) |
| 7 | `tool_error_payload` | yes | A valid call ran, and a result is a JSON object with a non-empty `"error"` key, e.g. `{"error": "cd: temp: No such directory"}`. Plain text that mentions an error does not count |
| 8 | `ok_call` | no | Parsed, known, schema-valid calls that ran without either kind of error |
| 9 | `reply` | no | τ² only: no call, a text message to the simulated user |
| 10 | `abstain_ok` | no | BFCL, no call, and `turn_has_ground_truth` is false (e.g. a miss_param turn where asking is right) |
| 11 | `missing_call_direct_answer` | yes | BFCL, no call, step 0, `turn_has_ground_truth` true, **and** the visible text (ignore anything inside `<think>…</think>`) is not empty, has no `?`, and does not announce a call ("I will / let me … check / call / use …") |
| 12 | `missing_call` | yes | BFCL, no call, step 0, `turn_has_ground_truth` true, and rule 11 does not fit (empty, asks a question, or announces a call) |
| 13 | `end_turn` | no | BFCL, no call after one or more calls earlier in the same turn (step ≥ 1): the model closes the turn |

Worked examples (from `docs/codebook_v0.md`):
- `ok_call`: `cd(folder='documents')` → `{"current_working_directory": "documents"}`.
- `tool_error_payload`: `cd(folder='temp')` → `{"error": "cd: temp: No such directory"}`.
- `missing_call_direct_answer`: turn "What is the total size of the files in documents?", step 0, output
  "The total size is 2.4 MB.", no call.
- `missing_call`: same turn, output "Which folder do you mean?" or "Let me check the folder first.", no call.
- `abstain_ok`: miss_param turn "Book me a flight" with no destination; the model asks "Where to?".
- `malformed`: `<tool_call>[{"name": "ls", "arguments": {}}]</tool_call>` (a list, not one call object).

## Edge cases
- A τ² reply that contains a broken `<tool_call>` block is `malformed`, not `reply` (rule 2 comes first).
- A step can make several calls: one schema error makes it `invalid_args`, one raised error `exec_error`.
- Thinking text (`<think>…</think>`) never counts as the visible answer.
- Some failures can hide inside `ok_call` and `end_turn`: a wrong tool, a wrong value, a policy violation. These
  need the reference trajectory and are **not** classes in v0. Label the row by the table, and write what you
  saw in `note`, e.g. `note: wrong file moved`. Those notes feed codebook v1.

## When you are done
Save your copy as `<sheet>_<your initials>.csv`. Don't edit any column other than `label` and `note`. Then
compute agreement:
`python scripts/label_kit.py kappa pilot_300_AB.csv pilot_300_CD.csv --key <key file> --out kappa_pilot_300.json`.
Only whoever runs this step reads the key file, and only after both raters have finished.
