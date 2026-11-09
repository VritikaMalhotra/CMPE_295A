# Step-failure codebook v0

Status: **v0, for registration at M2.** It describes `testbed/classifier.py` (C6 rules v0) as the code
stands; the classes below are the classes that code assigns. Two further classes are written down as
**added by decisions 3a and 3b** (Ekant, 2026-11-08; advisor to confirm). They are on by default;
`classify_step(row, proposed=False)` gives the labels from before the decision, to compare with older logs.
v1 is frozen at M3 with the κ report (Workbook FR-E8, Table 23).

## Unit and inputs
- **Unit:** one model step, i.e. one generation inside one turn of one episode (one `StepRow`, C5).
  BFCL: a turn is one user turn of the task; τ²: a turn is the agent's answer to one simulated-user message.
- **Every step gets exactly one class**, failures and non-failures alike, so rates have a denominator.
- **Inputs** are only fields logged in the step row: `stop_reason`, `raw_text`, `n_tool_call_blocks`,
  `decoded_calls`, `known_tools`, `schema_errors`, `execution_results`, `forced_quit`, `step`,
  `turn_has_ground_truth`, `benchmark`. No model is rerun; changing the codebook relabels frozen logs.
- **Precedence:** rules are checked in the order of the table; the first rule that fires gives the class.
  Format problems therefore win over content problems, and the step limit wins over call errors.

## Classes (v0, as implemented)

| # | Class | Failure? | Decision rule (first match wins) | Table 23 |
|---|---|---|---|---|
| 1 | `truncated` | yes | `stop_reason == "limit"`: generation hit `n_predict` | part of F2 |
| 2 | `malformed` | yes | more `<tool_call>` blocks than parsed calls, or `<tool_call>` in the text and no call parsed | F2 |
| 3 | `forced_quit` | yes | `forced_quit`: the step that exceeds bfcl's 20-step turn limit (`MAXIMUM_STEP_LIMIT`) | F7 (limit form) |
| 4 | `unknown_tool` | yes | at least one parsed call and `known_tools` is false (a name outside the task's tool list) | F1 |
| 5 | `invalid_args` | yes | `schema_errors` not empty: missing required argument, unknown argument, wrong JSON type, value outside an enum | F3 |
| 6 | `exec_error` | yes | a result starts with `Error during execution` (bfcl's executor raised; τ² results with `error=True` get the same prefix) | — |
| 7 | `ok_call` | no | parsed, known, schema-valid call(s) that executed | — |
| 8 | `reply` | no | τ² (`benchmark` starts with `tau2`): no call, i.e. a text message to the simulated user | (F9 is judged per episode) |
| 9 | `abstain_ok` | no | BFCL, no call, on a turn whose ground truth is empty (correct refusal, e.g. miss_param) | — |
| 10 | `missing_call` | yes | BFCL, no call, step 0 of a turn whose ground truth is not empty | F6 (turn opening) |
| 11 | `end_turn` | no | BFCL, no call after one or more calls in the turn: the model closes the turn | (F6 if calls are still pending) |

Notes on the rules:
- `malformed` covers what Table 23 calls F2 apart from cut-off output, which is `truncated` (rule 1), because a
  cut-off can come from the token budget rather than the model.
- `forced_quit` is checked before the call-content rules because the 21st step is usually a call inside an
  error loop; the loop itself (F7, three identical calls) is not detected step by step in v0.
- A τ² reply that contains a broken `<tool_call>` block is `malformed`, not `reply` (rule 2 comes first).
- Episode success is not decided here: bfcl's `multi_turn_checker` (state and response checks) and τ²'s
  evaluator (reward) judge the episode. These classes say *how* each step behaved.

### Examples
BFCL examples use GorillaFileSystem (multi_turn_base); the τ² example uses airline.

1. `truncated`: `<tool_call>\n{"name": "cat", "arguments": {"file_name": "rep` and `stop_reason = "limit"`.
2. `malformed`: `<tool_call>\n[{"name": "ls", "arguments": {}}]\n</tool_call>` (a list, not one
   `{"name", "arguments"}` object); bfcl scores the episode `multi_turn:inference_error`.
3. `forced_quit`: the 21st `ls()` in a turn where the model keeps listing the same folder.
4. `unknown_tool`: `{"name": "list_files", "arguments": {}}` when the tools are `ls`, `cd`, `cat`, ...
5. `invalid_args`: `{"name": "cd", "arguments": {}}`: `cd` requires `folder`.
6. `exec_error`: a schema-valid BFCL call whose class method raises, so bfcl's executor returns
   `Error during execution: <exception text>`. τ²: `get_user_details` with an unknown user id, which tau2 flags as
   a tool error: `Error during execution: User not found`.
7. `ok_call`: `{"name": "cd", "arguments": {"folder": "documents"}}` → `{"current_working_directory": "documents"}`.
8. `reply`: τ² airline: "Could you give me your user id?"
9. `abstain_ok`: miss_param turn "Book me a flight" with no destination; the model asks "Where to?" and the
   turn's ground truth is empty.
10. `missing_call`: turn "Move final_report.pdf into temp", step 0, the model answers "Done, the file is in temp."
11. `end_turn`: after `mv(...)` executed, the model writes "I moved the file." and stops.

## Classes added by decisions 3a/3b

Both are on by default and checked at the positions shown; nothing else moves.

### `tool_error_payload`: the tool returned an error without raising
**Status: in v0 (decision 3b, Ekant 2026-11-08; advisor to confirm).**
- **Why:** many BFCL tool methods do not raise; they *return* an error object, e.g. GorillaFileSystem's
  `{"error": "cd: temp: No such directory"}` (about 200 such returns in bfcl's function source). v0 counts these
  steps as `ok_call`, so failed actions look like successes in the damage map.
- **Decision rule:** after rule 6 (`exec_error`) and before rule 7 (`ok_call`): the step has at least one parsed,
  known, schema-valid call, and at least one execution result parses as a JSON object with a non-empty
  `"error"` key. Plain strings that mention errors, empty `"error"` values and other keys do not count.
- **Example:** `{"name": "cd", "arguments": {"folder": "temp"}}` → `{"error": "cd: temp: No such directory"}`.
- **Open question:** should it count as a failure in H1 (it is in `FAILURES`), or be reported next to `exec_error`
  as one "tool rejected the call" group?

### `missing_call_direct_answer`: missing call with a direct answer
**Status: in v0 (decision 3a, Ekant 2026-11-08; advisor to confirm).**
- **Why:** in the Oct 8 pilot, Q4_0 `--pure` lost 13 points on BFCL simple_python, mostly by not calling the tool.
  Of the 58 no-call outputs hand-labelled, 49 answered the question directly (26 with wrong or invented facts),
  5 asked the user for details, 3 said they would call the tool but did not, 1 was an unparseable call
  (Workbook v3, pilot findings). Answering directly is the quantization-specific pattern; asking the user is not.
- **Decision rule:** refines rule 10 only. A step that rule 10 would label `missing_call` becomes
  `missing_call_direct_answer` when its visible text (after removing `<think>...</think>`) is not empty,
  contains no `?`, and does not announce a call (a phrase such as "I will / let me / I'm going to ... call /
  use / check / run / fetch / search ..."). Otherwise it stays `missing_call`.
- **Example:** turn "What is the total size of the files in documents?", step 0: "The total size is 2.4 MB."
  Counter-examples (stay `missing_call`): "Which folder do you mean?", "Let me check the folder first."
- **Open question:** the rule is lexical; the 30-step rater sample (plan item 14) should check its precision
  before v1.

## Not covered by v0 (Table 23 classes that need the reference trajectory)
F4 wrong tool, F5 wrong argument value, F8 policy violation and F9 user-facing response error need alignment to
BFCL's ground truth or τ²'s expected actions, and F7 needs the repetition count; F0 "other" is the fallback.
None is implemented in `classifier.py` v0. Until they are, `ok_call` can hide F4/F5 and `end_turn` can hide F6.
The κ check (Workbook Ch. 7: two raters, 300 stratified steps, κ ≥ 0.80) applies to the classes above.

## Change log
- v0 (2026-11-08): written from `testbed/classifier.py` as implemented (11 classes); two proposed classes added
  behind `proposed=True`.
- v0 (2026-11-08, later): decisions 3a/3b add `tool_error_payload` and `missing_call_direct_answer` to v0, on
  by default; both count as failures (also in `analysis/mixed_effects.R`).
