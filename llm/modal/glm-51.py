# GLM-5.1-FP8 on Modal via vLLM
#
# Model:    zai-org/GLM-5.1-FP8
#           754B total / ~32B active MoE (8+1 experts/token), FP8 quantized, ~370GB weights
#           GlmMoeDSA arch: Gated DeltaNet linear attention + DeepSeek Sparse Attention + MTP head
#           Refreshed version of GLM-5, best-in-class open-source on coding/agentic tasks
#
# Hardware: 8x H200 (141GB each = 1128GB total)
#           Official zai-org reference setup. 8x H100 (640GB) also fits but tighter KV cache.
#
# Cost:     ~$4.54/hr per H200 → ~$36.32/hr at peak, $0 when idle (serverless)
#
# NOTE on NVFP4:
#   No official `nvidia/GLM-5.1-NVFP4` checkpoint exists yet — only unverified
#   community quants (lukealonso, koushd, mconcat). Skipping NVFP4 deploy until
#   NVIDIA publishes a vendor-verified checkpoint.
#
# NOTE: Requires vllm/vllm-openai:glm51-cu130 — special image with GLM-5.1 arch
#       support + DeepGEMM kernels for FP8 MoE.
#
# Deploy:   modal deploy glm-51.py
# Test:     modal run glm-51.py
# Weights:  downloaded automatically on first cold start via XET, cached in volume

import json
import subprocess
import time
import modal

# ---------------------------------------------------------------------------
# Container image
# MUST use vllm/vllm-openai:glm51-cu130 — standard latest image lacks GLM-5.1 arch
# and DeepGEMM kernels required for FP8 MoE inference.
# ---------------------------------------------------------------------------
IMAGE = (
    modal.Image.from_registry("vllm/vllm-openai:glm51-cu130", add_python="3.12")
    .entrypoint([])
    .run_commands("pip3 install 'huggingface-hub==0.36.0'")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

HF_CACHE_DIR = "/root/.cache/huggingface"
VLLM_CACHE_DIR = "/root/.cache/vllm"

hf_cache_vol = modal.Volume.from_name("glm51-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("glm51-vllm-cache", create_if_missing=True)

MODEL_ID = "zai-org/GLM-5.1-FP8"
MODEL_REVISION = None
SERVED_NAME = "glm-5.1"

app = modal.App("glm-5.1")


# ---------------------------------------------------------------------------
# vLLM inference server
#
# Key flags (per zai-org official deploy guide + vLLM GLM5 recipe):
#   --tensor-parallel-size 8       split across all 8x H200s
#   --speculative-config.method mtp        use Multi-Token Prediction head
#   --speculative-config.num_speculative_tokens 3   3 draft tokens per step
#   --tool-call-parser glm47       GLM-5.x tool call format
#   --reasoning-parser glm45       GLM-5.x reasoning/thinking format (thinking ON by default)
#   --chat-template-content-format=string  required for GLM-5.1 chat template
#   --enable-auto-tool-choice      allow model to decide when to call tools
#   --gpu-memory-utilization 0.85  zai-org recommendation (MTP needs extra mem)
#   --max-model-len 200000         200K context; plenty of KV cache headroom
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
class GLM51Server:
    @modal.enter()
    def start_server(self):
        cmd = [
            "vllm", "serve",
            MODEL_ID,
            "--served-model-name", SERVED_NAME,
            "--host", "0.0.0.0",
            "--port", "8000",
            "--tensor-parallel-size", "8",
            "--speculative-config.method", "mtp",
            "--speculative-config.num_speculative_tokens", "3",
            "--tool-call-parser", "glm47",
            "--reasoning-parser", "glm45",
            "--chat-template-content-format=string",
            "--enable-auto-tool-choice",
            "--trust-remote-code",
            "--gpu-memory-utilization", "0.85",
            "--max-model-len", "200000",
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
# Local test — run with: modal run glm-51.py
# Sends a reasoning request and prints thinking + answer separately.
# ---------------------------------------------------------------------------
@app.local_entrypoint()
async def test():
    import aiohttp

    url = await GLM51Server.serve.get_web_url.aio()
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
        # Disable thinking mode for a quick test — remove to enable reasoning
        "chat_template_kwargs": {"enable_thinking": False},
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
