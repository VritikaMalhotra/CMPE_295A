"""bfcl-eval runner for the pilot's v2 protocol, without editing the installed package.

Applies, in-process, to QwenFCHandler:
  - non-thinking prompt suffix (Qwen3 enable_thinking=false) unless V2_THINKING=1
  - cache_prompt=false on every request (prompt-cache reuse changes greedy outputs)
  - optional per-task GBNF grammar: V2_GRAMMAR=none|free|forced
  - optional sampling params (V2_SAMPLING JSON, e.g. {"top_k": 20}) and V2_SEED
Usage: python bfcl_v2.py generate|evaluate <normal bfcl arguments>
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "spikes", "vendor"))
import grammar as p3_grammar  # noqa: E402
from bfcl_eval.model_handler.local_inference import qwen_fc  # noqa: E402

GRAMMAR = os.environ.get("V2_GRAMMAR", "none")
THINKING = os.environ.get("V2_THINKING", "0") == "1"
SAMPLING = json.loads(os.environ.get("V2_SAMPLING", "{}"))
SEED = os.environ.get("V2_SEED")

_format_prompt_orig = qwen_fc.QwenFCHandler._format_prompt


def _format_prompt(self, messages, function):
    prompt = _format_prompt_orig(self, messages, function)
    return prompt if THINKING else prompt + "<think>\n\n</think>\n\n"


def _query_prompting(self, inference_data):
    # same as bfcl's OSSHandler._query_prompting, plus extra_body
    function = inference_data["function"]
    message = inference_data["message"]
    formatted_prompt = self._format_prompt(message, function)
    inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}
    input_token_count = len(self.tokenizer.tokenize(formatted_prompt))
    if self.max_context_length < input_token_count + 2:
        leftover_tokens_count = 1000
    else:
        leftover_tokens_count = min(4096, self.max_context_length - input_token_count - 2)
    extra_body = {"cache_prompt": False, **SAMPLING}
    if SEED is not None:
        extra_body["seed"] = int(SEED)
    if GRAMMAR == "free":
        extra_body["grammar"] = p3_grammar.task_grammar_free(function)
    elif GRAMMAR == "forced":
        extra_body["grammar"] = p3_grammar.task_grammar(function)
    start_time = time.time()
    api_response = self.client.completions.create(
        model=self.model_path_or_id, temperature=self.temperature, prompt=formatted_prompt,
        max_tokens=leftover_tokens_count, extra_body=extra_body, timeout=72000)
    return api_response, time.time() - start_time


qwen_fc.QwenFCHandler._format_prompt = _format_prompt
qwen_fc.QwenFCHandler._query_prompting = _query_prompting

if __name__ == "__main__":
    from bfcl_eval.__main__ import cli
    cli()
