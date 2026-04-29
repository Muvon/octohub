# Kimi-K2.6 on Modal via vLLM
#
# Model:    moonshotai/Kimi-K2.6
#           1T-param MoE (32B active), NATIVE INT4 weights (QAT, same as K2-Thinking),
#           ~594GB on disk. Multimodal (text + vision), thinking mode by default.
#           Same architecture as Kimi-K2.5, refreshed weights with INT4 QAT
#           — Moonshot claims ~2x inference speed and 50% less GPU memory vs FP16
#             with negligible quality loss.
#
# Hardware: 8x H200 (141GB each = 1128GB total) — official Moonshot reference,
#           verified config in vLLM recipes. Native INT4 weights ~594GB → ~530GB
#           free for KV cache → comfortable 256K context.
#
# Cost:     ~$4.54/hr per H200 → ~$36.32/hr at peak, $0 when idle (serverless)
#
# Why H200 over alternatives:
#   - 8x H100 (640GB) fits INT4 weights but KV cache is tight → lower max context
#   - 4x B200 NVFP4 is faster/cheaper BUT no official nvidia/Kimi-K2.6-NVFP4 yet
#     (only K2.5 has NVFP4); FP4 also has measurable (small) quality drop
#   - 8x H200 = official reference, native INT4, full quality, single node
#
# Deploy:   modal deploy kimi-k26.py
# Test:     modal run kimi-k26.py
# Weights:  downloaded automatically on first cold start via XET, cached in volume

import json
import subprocess
import time
import modal

# ---------------------------------------------------------------------------
# Container image
# Moonshot recommends vLLM 0.19.1 (manually verified) for K2.6 stable production.
# Nightly wheels work but are experimental. We pin to v0.19.1 image.
# ---------------------------------------------------------------------------
IMAGE = (
    modal.Image.from_registry("vllm/vllm-openai:v0.19.1", add_python="3.12")
    .entrypoint([])
    .run_commands("pip3 install 'huggingface-hub==0.36.0'")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

HF_CACHE_DIR = "/root/.cache/huggingface"
VLLM_CACHE_DIR = "/root/.cache/vllm"

hf_cache_vol = modal.Volume.from_name("kimi-k26-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("kimi-k26-vllm-cache", create_if_missing=True)

MODEL_ID = "moonshotai/Kimi-K2.6"
MODEL_REVISION = None
SERVED_NAME = "kimi-k2.6"

app = modal.App("kimi-k2.6")


# ---------------------------------------------------------------------------
# vLLM inference server
#
# Key flags (per Moonshot's official deploy_guidance.md for K2.6):
#   --tensor-parallel-size 8       split across all 8x H200s
#   --mm-encoder-tp-mode data      vision encoder in data-parallel (small encoder,
#                                  TP communication overhead > compute gain)
#   --tool-call-parser kimi_k2     Kimi-specific tool call format
#   --reasoning-parser kimi_k2     thinking mode is ON by default in K2.6
#   --enable-auto-tool-choice      let model decide when to call tools
#   --trust-remote-code            required by Kimi architecture
#   --gpu-memory-utilization 0.90  leave headroom for CUDA overhead
#   --max-model-len 256000         256K context; 8x H200 has ~530GB free after weights
#   --enable-prefix-caching        reuse KV cache across requests with shared prefix
# ---------------------------------------------------------------------------
@app.cls(
    image=IMAGE,
    gpu="H200:8",
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        VLLM_CACHE_DIR: vllm_cache_vol,
    },
    timeout=20 * 60,
    scaledown_window=5 * 60,
)
@modal.concurrent(max_inputs=32)
class KimiK26Server:
    @modal.enter()
    def start_server(self):
        cmd = [
            "vllm", "serve",
            MODEL_ID,
            "--served-model-name", SERVED_NAME,
            "--host", "0.0.0.0",
            "--port", "8000",
            "--tensor-parallel-size", "8",
            "--mm-encoder-tp-mode", "data",
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

    def _wait_for_server(self, timeout: int = 1200):
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

    @modal.web_server(port=8000, startup_timeout=1200)
    def serve(self):
        pass


# ---------------------------------------------------------------------------
# Local test — run with: modal run kimi-k26.py
# Sends a reasoning request and prints thinking + answer separately.
# ---------------------------------------------------------------------------
@app.local_entrypoint()
async def test():
    import aiohttp

    url = await KimiK26Server.serve.get_web_url.aio()
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

                    if delta.get("reasoning_content"):
                        if not in_thinking:
                            print("\n[THINKING]\n")
                            in_thinking = True
                        print(delta["reasoning_content"], end="", flush=True)

                    if delta.get("content"):
                        if in_thinking:
                            print("\n\n[ANSWER]\n")
                            in_thinking = False
                        print(delta["content"], end="", flush=True)

    print(f"\n{'─' * 60}\nDone.")
