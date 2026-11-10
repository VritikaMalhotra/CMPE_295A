"""QAD mix builder (train/build_qad_mix.py): parsing, normalisation, rendering and filtering, no downloads."""
import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("build_qad_mix", os.path.join(ROOT, "train", "build_qad_mix.py"))
qm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qm)


def test_py_calls_with_spaces_in_names_and_several_calls():
    s = '[Market Trends API(trend_type="MARKET_INDEXES", country="us"), getWeather(city="Paris, France", days=3)]'
    assert qm.parse_py_calls(s) == [
        {"name": "Market Trends API", "arguments": {"trend_type": "MARKET_INDEXES", "country": "us"}},
        {"name": "getWeather", "arguments": {"city": "Paris, France", "days": 3}}]


def test_py_calls_nested_values_json_words_and_no_arguments():
    s = '[Search(filters={"tags": ["a", "b"], "open": true}, limit=null), Ping()]'
    assert qm.parse_py_calls(s) == [
        {"name": "Search", "arguments": {"filters": {"tags": ["a", "b"], "open": True}, "limit": None}},
        {"name": "Ping", "arguments": {}}]


def test_py_calls_returns_none_for_plain_text_and_lists():
    assert qm.parse_py_calls("The market is up today.") is None
    assert qm.parse_py_calls("[1, 2, 3]") is None
    assert qm.parse_py_calls('["a", "b"]') is None


def test_py_calls_rejects_malformed_call_lists():
    with pytest.raises(qm.ParseError):
        qm.parse_py_calls('[f(x="unterminated)]')
    with pytest.raises(qm.ParseError):
        qm.parse_py_calls('[f(1, 2)]')                        # positional arguments


def test_toolace_record_keeps_tools_and_drops_the_system_prompt():
    rec = {"system": "You are an expert in composing functions. ...\n" + qm.TOOLACE_TOOLS_MARK + "\n"
                     '[{"name": "Market Trends API", "parameters": {"type": "dict", "properties": {}}}]\nMore text.',
           "conversations": [{"from": "user", "value": "Top US trends?"},
                             {"from": "assistant", "value": '[Market Trends API(country="us")]'},
                             {"from": "tool", "value": '{"trends": []}'},
                             {"from": "assistant", "value": "No trends right now."}]}
    msgs, tools = qm.toolace_record(rec)
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert msgs[1]["tool_calls"] == [{"name": "Market Trends API", "arguments": {"country": "us"}}]
    assert tools == [{"type": "function", "function": {"name": "Market Trends API",
                                                         "parameters": {"type": "object", "properties": {}}}}]
    assert all("expert in composing" not in m["content"] for m in msgs)


def test_toucan_tool_calls_attach_to_the_assistant_turn():
    raw = [{"role": "user", "content": "Find i18n components"},
           {"role": "assistant", "content": "Searching."},
           {"role": "tool_call", "content": "{'name': 'search', 'arguments': '{\"query\": \"i18n\"}'}"},
           {"role": "tool_call", "content": "{'name': 'search', 'arguments': '{\"query\": \"l10n\"}'}"},
           {"role": "tool_response", "content": "found 2"},
           {"role": "tool_response", "content": "found 1"},
           {"role": "tool_call", "content": "{'name': 'get', 'arguments': '{\"id\": 7}'}"},
           {"role": "tool_response", "content": "ok"},
           {"role": "assistant", "content": "Done."}]
    msgs, tools = qm.toucan_record({"messages": json.dumps(raw), "tools": json.dumps([{"type": "function"}])})
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "tool", "assistant", "tool", "assistant"]
    assert msgs[1]["tool_calls"] == [{"name": "search", "arguments": {"query": "i18n"}},
                                     {"name": "search", "arguments": {"query": "l10n"}}]
    assert msgs[4] == {"role": "assistant", "content": "", "tool_calls": [{"name": "get", "arguments": {"id": 7}}]}
    assert tools == [{"type": "function"}]


def test_nemotron_drops_reasoning_and_empty_system():
    obj = {"messages": [{"role": "system", "content": ""},
                        {"role": "user", "content": "Balance?"},
                        {"role": "assistant", "content": "", "reasoning_content": "Let me think",
                         "tool_calls": [{"id": "c1", "type": "function",
                                         "function": {"name": "bal", "arguments": {"addr": "1A"}}}]},
                        {"role": "tool", "content": "{'balance': 1}"},
                        {"role": "assistant", "content": "It is 1.", "tool_calls": None}],
           "tools": [{"type": "function", "function": {"name": "bal"}}]}
    msgs, tools = qm.nemotron_record(obj)
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert msgs[1] == {"role": "assistant", "content": "", "tool_calls": [{"name": "bal", "arguments": {"addr": "1A"}}]}
    assert "reasoning_content" not in json.dumps(msgs) and "Let me think" not in json.dumps(msgs)
    assert tools == obj["tools"]


def test_render_plain_has_roles_and_tool_calls_but_no_special_tokens():
    msgs = [{"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "f", "arguments": {"x": 1}}]},
            {"role": "tool", "content": "2"}]
    assert qm.render_plain(msgs) == ('user: Hi\nassistant: <tool_call>{"name": "f", "arguments": {"x": 1}}'
                                     '</tool_call>\ntool: 2')


def test_overlap_check_covers_messages_and_tool_schemas():
    bench = qm.ng.ngram_set("change the current working directory to the specified folder in the file system now")
    clean_msgs = [{"role": "user", "content": "please list my files"}]
    tools_hit = [{"type": "function", "function": {"name": "cd", "description":
                  "Change the current working directory to the specified folder in the file system now."}}]
    assert not qm.overlaps(qm.record_check_text(clean_msgs, []), bench)
    assert qm.overlaps(qm.record_check_text(clean_msgs, tools_hit), bench)


def test_seeded_order_is_reproducible_and_salted():
    assert qm.seeded_order(50, salt="a") == qm.seeded_order(50, salt="a")
    assert qm.seeded_order(50, salt="a") != qm.seeded_order(50, salt="b")
    assert sorted(qm.seeded_order(50)) == list(range(50))


def test_content_hash_ignores_key_order():
    assert qm.content_hash({"a": 1, "b": [1, 2]}) == qm.content_hash({"b": [1, 2], "a": 1})
