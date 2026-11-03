"""scripts/kl_report.py parses llama-perplexity logs (format of tools/perplexity at the pinned d812350)."""
import importlib.util
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("kl_report", os.path.join(ROOT, "scripts", "kl_report.py"))
kl_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kl_report)

BASE_LOG = """\
print_info: file type   = BF16
print_info: file size   = 1.40 GiB (16.01 BPW)
perplexity: calculating perplexity over 50 chunks, n_ctx=512, batch_size=2048, n_seq=4
Final estimate: PPL = 20.1234 +/- 0.45678
"""


def variant_log(kld, same_top, bpw, size="336.22 MiB", ftype="Q4_0"):
    return f"""\
print_info: file type   = {ftype}
print_info: file size   = {size} ({bpw:.2f} BPW)
kl_divergence: computing over 50 chunks, n_ctx=512, batch_size=2048, n_seq=4
====== Perplexity statistics ======
Mean PPL(Q)                   :  24.567890 ±   0.512345
Mean PPL(base)                :  20.123400 ±   0.456780
Cor(ln(PPL(Q)), ln(PPL(base))):  97.12%
Mean ln(PPL(Q)/PPL(base))     :   0.199700 ±   0.004321
Mean PPL(Q)/PPL(base)         :   1.221000 ±   0.005276
Mean PPL(Q)-PPL(base)         :   4.444490 ±   0.112233

====== KL divergence statistics ======
Mean    KLD: {kld:10.6f} ±   0.001234
Maximum KLD:  12.345678
99.9%   KLD:   5.432100
99.0%   KLD:   2.109876
95.0%   KLD:   0.876543
90.0%   KLD:   0.543210
Median  KLD:   0.098765
10.0%   KLD:   0.001234
 5.0%   KLD:   0.000456
 1.0%   KLD:   0.000012
 0.1%   KLD:  -0.000001
Minimum KLD:  -0.000010

====== Token probability statistics ======
Mean    Δp: -3.210 ± 0.123 %
Maximum Δp: 98.765%
Median  Δp: -0.012%
Minimum Δp: -99.123%
RMS Δp    : 12.345 ± 0.234 %
Same top p: {same_top:6.3f} ± 0.321 %
"""


def test_parse_variant_log():
    r = kl_report.parse_log(variant_log(0.212345, 81.234, 4.53))
    assert r["kld_mean"] == 0.212345 and r["kld_mean_unc"] == 0.001234
    assert r["kld_max"] == 12.345678 and r["kld_p999"] == 5.4321 and r["kld_p99"] == 2.109876
    assert r["kld_median"] == 0.098765
    assert r["ppl_q"] == 24.56789 and r["ppl_base"] == 20.1234 and r["ppl_ratio"] == 1.221
    assert r["same_top_p_pct"] == 81.234 and r["dp_rms_pct"] == 12.345 and r["dp_mean_pct"] == -3.21
    assert r["bpw"] == 4.53 and r["file_size_mib"] == 336.22 and r["file_type"] == "Q4_0"
    assert r["chunks_scored"] == 50 and r["n_ctx"] == 512


def test_parse_base_log():
    r = kl_report.parse_log(BASE_LOG)
    assert r["ppl_final"] == 20.1234 and r["ppl_final_unc"] == 0.45678
    assert r["bpw"] == 16.01 and r["file_size_mib"] == round(1.40 * 1024, 2) and r["file_type"] == "BF16"


def test_build_report_and_ordering_checks(tmp_path):
    d = tmp_path / "kl"
    d.mkdir()
    (d / "BF16.log").write_text(BASE_LOG)
    (d / "Q4_0-pure.log").write_text(variant_log(0.212, 80.1, 4.53))
    (d / "Q4_K_M.log").write_text(variant_log(0.090, 88.0, 5.15, ftype="Q4_K - Medium"))
    (d / "Q8_0.log").write_text(variant_log(0.004, 98.7, 8.50))
    m = tmp_path / "models"
    m.mkdir()
    (m / "M-Q8_0.gguf.sha256").write_text("ab" * 32 + "  M-Q8_0.gguf\n")
    text = tmp_path / "wiki.test.raw"
    text.write_text("hello world\n")
    rep = kl_report.build_report("M", str(d), str(m), str(text), 512, 50)
    assert set(rep["variants"]) == {"Q4_0-pure", "Q4_K_M", "Q8_0"}
    assert rep["checks"] == {"q8_0_closer_than_q4_k_m": True, "q8_0_closer_than_q4_0_pure": True}
    assert rep["complete"] is True
    assert rep["variants"]["Q8_0"]["sha256"] == "ab" * 32 and rep["variants"]["Q4_K_M"]["sha256"] is None
    assert rep["eval_text_sha256"] and rep["bf16"]["ppl_final"] == 20.1234
    json.dumps(rep)


def test_incomplete_log_is_flagged(tmp_path):
    d = tmp_path / "kl"
    d.mkdir()
    (d / "Q4_0-pure.log").write_text("print_info: file type   = Q4_0\nCUDA error: out of memory\n")
    rep = kl_report.build_report("M", str(d), None, None, 512, 50)
    assert rep["variants"]["Q4_0-pure"]["complete"] is False and rep["complete"] is False
