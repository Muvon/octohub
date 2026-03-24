# Kimi-K2.5-NVFP4 on Modal via vLLM
#
# Model:    nvidia/Kimi-K2.5-NVFP4
#           NVFP4-quantized 1T-param MoE (32B active), ~300GB weights
#           Tested by NVIDIA on B200 only — requires Blackwell for native FP4 tensor cores
#
# Hardware: 4x B200 (192GB each = 768GB total, ~300GB for weights, rest for KV cache)
# Cost:     ~$8/hr per B200 → ~$32/hr at peak, $0 when idle (serverless)
#
# Deploy:   modal deploy kimi-k25.py
# Test:     modal run kimi-k25.py
# Weights:  downloaded automatically on first cold start via XET, cached in volume

import json
import subprocess
import time
import modal

# ---------------------------------------------------------------------------
# Container image
# NVIDIA requires vllm/vllm-openai:v0.15.0 specifically for this NVFP4 model.
# We build on top of it and add huggingface-hub for weight downloads.
# ---------------------------------------------------------------------------
IMAGE = (
    modal.Image.from_registry("vllm/vllm-openai:v0.15.0", add_python="3.12")
    .entrypoint([])
    .run_commands("pip3 install 'huggingface-hub==0.36.0'")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

HF_CACHE_DIR = "/root/.cache/huggingface"
VLLM_CACHE_DIR = "/root/.cache/vllm"

hf_cache_vol = modal.Volume.from_name("kimi-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("kimi-vllm-cache", create_if_missing=True)

MODEL_ID = "nvidia/Kimi-K2.5-NVFP4"
MODEL_REVISION = None
SERVED_NAME = "kimi-k2.5"

app = modal.App("kimi-k2.5")



# ---------------------------------------------------------------------------
# vLLM inference server
#
# Key flags:
#   --tensor-parallel-size 4     split across 4x B200
#   --tool-call-parser kimi_k2   Kimi-specific tool call format
#   --reasoning-parser kimi_k2   Kimi-specific thinking tags
#   --trust-remote-code          required by nvidia/Kimi-K2.5-NVFP4
#   --gpu-memory-utilization 0.90  leave headroom for CUDA overhead
#   --max-model-len 256000       256K context; 4x B200 has ~468GB free after weights
#   --enable-prefix-caching      reuse KV cache across requests with shared prefix
# ---------------------------------------------------------------------------
@app.cls(
    image=IMAGE,
    gpu="B200:4",
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        VLLM_CACHE_DIR: vllm_cache_vol,
    },
    timeout=15 * 60,
    scaledown_window=5 * 60,
)
@modal.concurrent(max_inputs=32)
class KimiServer:
    @modal.enter()
    def start_server(self):
        cmd = [
            "vllm", "serve",
            MODEL_ID,
            "--served-model-name", SERVED_NAME,
            "--host", "0.0.0.0",
            "--port", "8000",
            "--tensor-parallel-size", "4",
            "--tool-call-parser", "kimi_k2",
            "--reasoning-parser", "kimi_k2",
            "--enable-auto-tool-choice",
            "--trust-remote-code",
            "--gpu-memory-utilization", "0.90",
            "--max-model-len", "256000",
            "--enable-prefix-caching",
        ]
        if MODEL_REVISION:
            cmd += ["--revision", MODEL_REVISION]
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
# Local test — run with: modal run kimi.py
# ---------------------------------------------------------------------------
@app.local_entrypoint()
async def test():
    import aiohttp

    url = await KimiServer.serve.get_web_url.aio()
    endpoint = f"{url}/v1/chat/completions"

    payload = {
        "model": SERVED_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a Python function that checks if a number is prime. Be concise."},
        ],
        "max_tokens": 512,
        "temperature": 0.6,
        "stream": True,
    }

    print(f"\nSending request to {endpoint}\n{'─' * 60}")
    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.content:
                line = line.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    print(delta, end="", flush=True)
    print(f"\n{'─' * 60}\nDone.")
