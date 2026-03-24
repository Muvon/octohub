# MiMo-V2-Flash on Modal via vLLM
#
# Model:    XiaomiMiMo/MiMo-V2-Flash
#           MoE with 309B total / 15B active parameters, FP8 quantized
#           Hybrid attention (SWA+GA) + Multi-Token Prediction (MTP)
#           Supports reasoning (thinking mode) + tool calling
#           SOTA on reasoning and agentic benchmarks
#
# Hardware: 4x H100 (80GB each = 320GB total)
#           Weights ~150GB FP8, plenty of KV cache headroom
# Cost:     ~$15.88/hr at peak (4x H100), $0 when idle (serverless)
#
# Deploy:   modal deploy mimo-v2-flash.py
# Test:     modal run mimo-v2-flash.py
# Weights:  downloaded automatically on first cold start via XET, cached in volume

import json
import subprocess
import time
import modal

# ---------------------------------------------------------------------------
# Container image
# Use latest vllm/vllm-openai — supports MiMo-V2-Flash architecture.
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

hf_cache_vol = modal.Volume.from_name("mimo-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("mimo-vllm-cache", create_if_missing=True)

MODEL_ID = "XiaomiMiMo/MiMo-V2-Flash"
MODEL_REVISION = None  # pin to a commit hash to avoid surprise repo updates
SERVED_NAME = "mimo-v2-flash"

app = modal.App("mimo-v2-flash")


# ---------------------------------------------------------------------------
# vLLM inference server
#
# Key flags:
#   --tensor-parallel-size 4     split across 4x H100 (309B MoE needs distributed)
#   --reasoning-parser qwen3     strips </think>... into separate field
#   --tool-call-parser qwen3_xml  MiMo tool call format (XML-based)
#   --enable-auto-tool-choice    enables tool calling
#   --trust-remote-code          required for MiMo architecture
#   --generation-config vllm      use vLLM's generation config
#   --gpu-memory-utilization 0.90  leave headroom for CUDA overhead
#   --max-model-len 65536        recommended for most scenarios (max 256k)
# ---------------------------------------------------------------------------
@app.cls(
    image=IMAGE,
    gpu="H100:4",
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        VLLM_CACHE_DIR: vllm_cache_vol,
    },
    timeout=15 * 60,
    scaledown_window=5 * 60,
)
@modal.concurrent(max_inputs=32)
class MiMoServer:
    @modal.enter()
    def start_server(self):
        cmd = [
            "vllm", "serve",
            MODEL_ID,
            "--served-model-name", SERVED_NAME,
            "--host", "0.0.0.0",
            "--port", "8000",
            "--tensor-parallel-size", "4",
            "--reasoning-parser", "qwen3",
            "--tool-call-parser", "qwen3_xml",
            "--enable-auto-tool-choice",
            "--trust-remote-code",
            "--generation-config", "vllm",
            "--gpu-memory-utilization", "0.90",
            "--max-model-len", "65536",
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
# Local test — run with: modal run mimo-v2-flash.py
# Sends a reasoning request and prints the thinking + answer separately.
# ---------------------------------------------------------------------------
@app.local_entrypoint()
async def test():
    import aiohttp

    url = await MiMoServer.serve.get_web_url.aio()
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
        "chat_template_kwargs": {"enable_thinking": True},
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
