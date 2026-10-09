"""Spikes S1/S2: build llama.cpp (CUDA + CPU), convert and quantize Qwen3-0.6B,
serve each file, and send one tool call through the raw /completion endpoint.

Pilot run: Google Colab A100-SXM4-40GB, llama.cpp v0.6.0 (d812350), 2026-10-08 PT.
Pilot / development data, not study results.

Usage (on a CUDA machine with cmake, git, and the HF model downloaded):
    python s1_s2_build_serve_toolcall.py --work /content/work --model-dir /content/work/Qwen3-0.6B
"""
import argparse
import collections
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

import requests

TAG = "v0.6.0"
CTX = 8192
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City and state, e.g. San Jose, CA"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"]}}}]
MESSAGES = [{"role": "user", "content": "What is the weather in San Jose, CA right now? Answer in celsius."}]
TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def run(cmd, log=None):
    print("+", cmd)
    out = open(log, "w") if log else None
    subprocess.run(cmd, shell=True, check=True, stdout=out, stderr=subprocess.STDOUT)


def build(work):
    src = f"{work}/llama.cpp"
    if not os.path.isdir(src):
        run(f"git clone --depth 1 --branch {TAG} https://github.com/ggml-org/llama.cpp {src}")
    run(f"cmake -S {src} -B {src}/build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release")
    run(f"cmake --build {src}/build-cuda -j 12 --target llama-server llama-cli llama-quantize llama-bench")
    run(f"cmake -S {src} -B {src}/build-cpu -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release")
    run(f"cmake --build {src}/build-cpu -j 12 --target llama-server llama-cli llama-quantize")
    return src


def convert_and_quantize(src, model_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    bf16 = f"{out_dir}/Qwen3-0.6B-BF16.gguf"
    files = {"BF16": bf16, "Q4_0": f"{out_dir}/Qwen3-0.6B-Q4_0-pure.gguf", "Q4_K_M": f"{out_dir}/Qwen3-0.6B-Q4_K_M.gguf"}
    run(f"{sys.executable} {src}/convert_hf_to_gguf.py {model_dir} --outtype bf16 --outfile {bf16}")
    run(f"{src}/build-cuda/bin/llama-quantize --pure {bf16} {files['Q4_0']} Q4_0")
    run(f"{src}/build-cuda/bin/llama-quantize {bf16} {files['Q4_K_M']} Q4_K_M")
    return files


def bits_per_weight(src, files):
    sys.path.insert(0, f"{src}/gguf-py")
    import gguf
    rows = []
    for fmt, path in files.items():
        reader = gguf.GGUFReader(path)
        n = sum(int(t.n_elements) for t in reader.tensors)
        census = collections.Counter(t.tensor_type.name for t in reader.tensors)
        size = os.path.getsize(path)
        rows.append({"format": fmt, "bytes": size, "n_params": n, "bpw_whole_file": round(size * 8 / n, 4),
                     "tensor_types": ";".join(f"{k}:{v}" for k, v in sorted(census.items())),
                     "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()})
    return rows


def start_server(src, model, build_dir, port):
    ngl = 99 if build_dir == "build-cuda" else 0
    cmd = (f"{src}/{build_dir}/bin/llama-server -m {model} -ngl {ngl} -c {CTX} -np 1 "
           f"--host 127.0.0.1 --port {port} --seed 0 --no-webui")
    proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(300):
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return proc
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server on port {port} did not start")


def stop_server(proc):
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    proc.wait(timeout=30)


class PeakGpuMemory:
    """Samples device memory every 100 ms with NVML and keeps the maximum."""

    def __enter__(self):
        import pynvml
        pynvml.nvmlInit()
        self._nvml, self._h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        self.peak, self._stop = 0, False
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def _loop(self):
        while not self._stop:
            self.peak = max(self.peak, self._nvml.nvmlDeviceGetMemoryInfo(self._h).used)
            time.sleep(0.1)

    def __exit__(self, *exc):
        self._stop = True
        self._t.join()


def tool_call_request(model_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    prompt = tok.apply_chat_template(MESSAGES, tools=TOOLS, add_generation_prompt=True,
                                     enable_thinking=False, tokenize=False)
    ids = tok.encode(prompt, add_special_tokens=False)
    return {"prompt": ids, "n_predict": 128, "temperature": 0.0, "top_k": 1, "seed": 0,
            "cache_prompt": False, "return_tokens": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/content/work")
    ap.add_argument("--model-dir", default="/content/work/Qwen3-0.6B")
    args = ap.parse_args()

    src = build(args.work)
    files = convert_and_quantize(src, args.model_dir, f"{args.work}/models")
    print(json.dumps(bits_per_weight(src, files), indent=1))

    body = tool_call_request(args.model_dir)
    results = {}
    for fmt, path in files.items():
        with PeakGpuMemory() as mem:
            proc = start_server(src, path, "build-cuda", 18080)
            runs = [requests.post("http://127.0.0.1:18080/completion", json=body).json() for _ in range(5)]
        stop_server(proc)
        proc = start_server(src, path, "build-cpu", 18081)
        cpu = requests.post("http://127.0.0.1:18081/completion", json=body).json()
        stop_server(proc)
        first = runs[0]
        results[fmt] = {
            "content": first["content"],
            "parsed": [json.loads(m) for m in TOOLCALL_RE.findall(first["content"])],
            "decode_tok_s": [round(r["timings"]["predicted_per_second"], 1) for r in runs],
            "prompt_tok_s": [round(r["timings"]["prompt_per_second"], 1) for r in runs],
            "peak_gpu_mem_mib": mem.peak // 2**20,
            "cpu_tokens_identical": cpu.get("tokens") == first.get("tokens"),
        }
        print(fmt, results[fmt])
    json.dump(results, open(f"{args.work}/p1_toolcall.json", "w"), indent=1)


if __name__ == "__main__":
    main()
