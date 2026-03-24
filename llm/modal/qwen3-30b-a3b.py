# Qwen3-30B-A3B on Modal via vLLM
#
# Model:    Qwen/Qwen3-30B-A3B-Instruct-2507
#           MoE architecture: 30B total params, 3B active per token
#           128 experts, 8 active experts per token
#           Native 131K context (YaRN extended to 256K)
#           Excellent for agentic tasks, reasoning, code generation
#
# Hardware: 1x L40S (48GB) — FP8 quantization fits with KV cache headroom
#           Alternative: A100-40GB for BF16
# Cost:      L40S: ~$1.10/hr | A100-40G: ~$2.00/hr
#            $0 when idle (serverless)
#
# Deploy:   modal deploy qwen3-30b-a3b.py
# Test:     modal run qwen3-30b-a3b.py
# Weights:  downloaded automatically on first cold start, cached in volume

import json
import subprocess
import time
import modal

# ---------------------------------------------------------------------------
# Container image
# Use latest vllm/vllm-openai — supports Qwen3 architecture
# HF_XET_HIGH_PERFORMANCE=1 — enables XET protocol for fast weight downloads
# ---------------------------------------------------------------------------
IMAGE = (
    modal.Image.from_registry("vllm/vllm-openai:latest", add_python="3.12")
    .entrypoint([])
    .run_commands("pip3 install 'huggingface-hub==0.36.0'")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

HF_CACHE_DIR = "/root/.cache/huggingface"
VLLM_CACHE_DIR = "/root/.cache/vllm"

hf_cache_vol = modal.Volume.from_name("qwen3-30b-a3b-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("qwen3-30b-a3b-vllm-cache", create_if_missing=True)

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
SERVED_NAME = "qwen3-30b-a3b"

# GPU selection: L40S (48GB) for FP8, A100-40G for BF16
# L40S is cheaper and sufficient for FP8 quantization
GPU_TYPE = "L40S"

app = modal.App("qwen3-30b-a3b")


# ---------------------------------------------------------------------------
# vLLM inference server
#
# Key flags:
#   --tensor-parallel-size 1     single GPU sufficient for MoE with 3B active
#   --reasoning-parser qwen3     strips <think>...</think> into separate field
#   --enable-auto-tool-choice    enables tool calling
#   --tool-call-parser qwen3_coder  Qwen3 tool call format
#   --enable-prefix-caching      reuse KV cache across requests with shared prefix
#   --gpu-memory-utilization 0.90  leave headroom for CUDA overhead
#   --max-model-len 131072       131K context (YaRN extended)
# ---------------------------------------------------------------------------
@app.cls(
    image=IMAGE,
    gpu=GPU_TYPE,
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        VLLM_CACHE_DIR: vllm_cache_vol,
    },
    timeout=15 * 60,
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
            "--max-model-len", "131072",
        ]
        print("Starting vLLM:", " ".join(cmd))
        self._proc = subprocess.Popen(" ".join(cmd), shell=True)
        self._wait_for_server()

    def _wait_for_server(self, timeout: int = 900):
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

    @modal.web_server(port=8000, startup_timeout=900)
    def serve(self):
        pass


# ---------------------------------------------------------------------------
# Local test — run with: modal run qwen3-30b-a3b.py
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