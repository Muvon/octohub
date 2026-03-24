# Qwen3.5-9B on Modal via vLLM (AWQ Quantized)
#
# Model:    QuantTrio/Qwen3.5-9B-AWQ
#           Dense 9B-param model, AWQ 4-bit quantized (~5GB vs ~18GB BF16)
#           Supports reasoning (thinking mode) + tool calling
#           Fastest Qwen3.5 variant, ideal for high-throughput scenarios
#
# Hardware: 1x L4 (24GB) — AWQ model fits easily with KV cache headroom
#           BF16 requires H100 (80GB), AWQ saves 3x memory
# Cost:      L4: ~$0.80/hr | A10G: ~$1.10/hr
#            $0 when idle (serverless)
#
# Deploy:   modal deploy qwen35-9b.py
# Test:     modal run qwen35-9b.py
# Weights:  downloaded automatically on first cold start, cached in volume

import json
import subprocess
import time
import modal

# ---------------------------------------------------------------------------
# Container image
# Use latest vllm/vllm-openai — supports Qwen3.5 architecture.
# HF_XET_HIGH_PERFORMANCE=1 — enables XET protocol for fast weight downloads.
# ---------------------------------------------------------------------------
IMAGE = (
    modal.Image.from_registry("vllm/vllm-openai:latest", add_python="3.12")
    .entrypoint([])
    .run_commands("pip3 install 'huggingface-hub==0.36.0'")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

HF_CACHE_DIR = "/root/.cache/huggingface"
VLLM_CACHE_DIR = "/root/.cache/vllm"

hf_cache_vol = modal.Volume.from_name("qwen9b-awq-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("qwen9b-awq-vllm-cache", create_if_missing=True)

MODEL_ID = "QuantTrio/Qwen3.5-9B-AWQ"
SERVED_NAME = "qwen3.5-9b"

# GPU selection: L4 is cheapest and sufficient for AWQ
# Change to "A10G" for slightly better throughput
GPU_TYPE = "L4"

app = modal.App("qwen3.5-9b")


# ---------------------------------------------------------------------------
# vLLM inference server
#
# Key flags:
#   --tensor-parallel-size 1     single GPU is enough for 5GB AWQ model
#   --reasoning-parser qwen3     strips <think>...</think> into separate field
#   --enable-auto-tool-choice    enables tool calling
#   --tool-call-parser qwen3_coder  Qwen3.5 tool call format
#   --enable-prefix-caching      reuse KV cache across requests with shared prefix
#   --gpu-memory-utilization 0.90  leave headroom for CUDA overhead
#   --max-model-len 16384        fits in 24GB VRAM with AWQ + KV cache
# ---------------------------------------------------------------------------
@app.cls(
    image=IMAGE,
    gpu=GPU_TYPE,
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        VLLM_CACHE_DIR: vllm_cache_vol,
    },
    timeout=10 * 60,
    scaledown_window=5 * 60,
)
@modal.concurrent(max_inputs=64)
class QwenServer:
    @modal.enter()
    def start_server(self):
        cmd = [
            "vllm", "serve",
            MODEL_ID,
            "--served-model-name", SERVED_NAME,
            "--host", "0.0.0.0",
            "--port", "8000",
            "--tensor-parallel-size", "1",
            "--reasoning-parser", "qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
            "--enable-prefix-caching",
            "--gpu-memory-utilization", "0.90",
            "--max-model-len", "16384",
        ]
        print("Starting vLLM:", " ".join(cmd))
        self._proc = subprocess.Popen(" ".join(cmd), shell=True)
        self._wait_for_server()

    def _wait_for_server(self, timeout: int = 600):
        import urllib.error
        import urllib.request

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                urllib.request.urlopen("http://localhost:8000/health", timeout=2)
                print("vLLM ready.")
                return
            except (urllib.error.URLError, OSError):
                time.sleep(5)
        raise TimeoutError("vLLM did not start within timeout")

    @modal.exit()
    def stop_server(self):
        self._proc.terminate()
        self._proc.wait(timeout=30)

    @modal.web_server(port=8000, startup_timeout=600)
    def serve(self):
        pass


# ---------------------------------------------------------------------------
# Local test — run with: modal run qwen35-9b.py
# Sends a reasoning request and prints the thinking + answer separately.
# ---------------------------------------------------------------------------
@app.local_entrypoint()
async def test():
    import aiohttp

    url = await QwenServer.serve.get_web_url.aio()
    endpoint = f"{url}/v1/chat/completions"

    payload = {
        "model": SERVED_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a Python function that checks if a number is prime. Be concise."},
        ],
        "max_tokens": 1024,
        "temperature": 0.6,
        "stream": True,
    }

    print(f"\nSending request to {endpoint}\n{'─' * 60}")
    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, json=payload) as resp:
            resp.raise_for_status()
            in_thinking = False
            async for line in resp.content:
                line = line.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    choice = chunk["choices"][0]
                    delta = choice["delta"]

                    # Reasoning content (thinking tokens)
                    if delta.get("reasoning_content"):
                        if not in_thinking:
                            print("\n[THINKING]\n")
                            in_thinking = True
                        print(delta["reasoning_content"], end="", flush=True)

                    # Regular content (answer)
                    if delta.get("content"):
                        if in_thinking:
                            print("\n\n[ANSWER]\n")
                            in_thinking = False
                        print(delta["content"], end="", flush=True)

    print(f"\n{'─' * 60}\nDone.")