# R1 grammars per tool-call format (B2, sprint 6)

R1 constrains decoding with a GBNF grammar built from the task's tool schemas (`testbed/grammar.py`). There is one
grammar per call syntax, picked from the model's prompt format (`grammars_for(fmt.syntax)` in `testbed/run.py`):

| Call syntax | Models | Block | Builders |
|---|---|---|---|
| JSON | Qwen3, SmolLM3 | `<tool_call>\n{"name": …, "arguments": {…}}\n</tool_call>`, one or more | `task_grammar` (forced), `task_grammar_free` (free) |
| Pythonic | LFM2.5 | `<|tool_call_start|>[f(a=…), g(b=…)]<|tool_call_end|>`, one block | `pythonic_task_grammar`, `pythonic_task_grammar_free` |

- **Forced:** schema-valid calls only.
- **Free:** the same calls, or text that does not start with `<` or whitespace. Free is the R1 arm; forced is the
  labelled ablation.

**Argument values.** In both grammars, required arguments come first, then optional ones, each in schema order.
Unknown names and extra keys are rejected. Pythonic values are written the way the chat template writes them:
strings in single quotes with `\\ \' \n \r` escapes; numbers and `True`/`False` as Python literals; lists and
dicts as JSON, through llama.cpp's own converter. For LFM2.5-2.6B, whose answer starts inside `<think>`, a
`think` prefix allows reasoning before `</think>`.

## Checks
- **Property tests** (`tests/test_r1_property.py`, hypothesis, against llama.cpp's `test-gbnf-validator` at the
  pinned commit), for both syntaxes:
  - every schema-valid call is accepted, and a Pythonic call also parses back unchanged;
  - an unknown name, a missing required key and an extra key are rejected;
  - the free branch accepts text unless it starts with `<` or whitespace.
  - Cluster suite 111882: 260 passed, 1 skipped, validator found.
  - A later cluster suite (112511) failed one Pythonic property test: hypothesis found `cd(folder='\x00')`, which
    the grammar accepts but the parser did not read back (Python rejects a NUL byte in source). Fixed in 47d41f7,
    with the input pinned as an example.
  - After the fix: suite 112513 gave 263 passed, 1 skipped; property tests under 5 more hypothesis seeds (112514)
    gave 10/10 on each.
  - Counts are from the job logs (cluster only, not committed).
- **Smoke, grammar on** (job 111954, P100; step and token counts from its log, not committed; one BFCL base task, greedy, Q4_0-pure, each with its no-grammar
  control):
  - **JSON, Qwen3-0.6B:** 10 steps, every step a schema-valid call or text. Identical to the control: greedy
    Qwen3 output already fits the grammar here.
  - **Pythonic, LFM2.5-1.2B:** 11 steps, every step a call or text, and the grammar is visibly applied: the
    control wrote `startEngine(ignitionMode="START")`, R1 wrote `startEngine(ignitionMode='START')`. Each call
    marker is still one token (15 completion tokens for the first call under both), so the grammar does not make
    the model spell the markers out.
- Runs at scale are not part of this week (M4).
