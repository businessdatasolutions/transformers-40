Great news — your M2 MacBook Pro is actually one of the better-supported platforms for Transformers.js v4 with WebGPU. I found real benchmark data from an M2 Pro user in a GitHub issue that gives us concrete numbers to work with.

## M2 + WebGPU: How it works

On macOS, Chrome's WebGPU implementation runs on top of Metal via ANGLE. Apple Silicon's unified memory architecture is an advantage here because the GPU shares the same memory pool as the CPU, meaning model weights don't need to be copied between separate memory banks. The v4 release was explicitly tested across ~200 model architectures with this new WebGPU C++ runtime, and the Metal backend on macOS is one of the primary targets.

Use Chrome (Chromium-based browsers have the most mature WebGPU support). Safari also supports WebGPU but via a feature flag and may have compatibility quirks.

## Real M2 performance data

From GitHub issue #1599, a user benchmarked on an M2 Pro running macOS 14.3 with Chrome 145:

**Qwen3-4B (q4f16)** achieved ~22.3 tokens/second decode speed with ~828ms time-to-first-token (TTFT). That's very usable for an orchestrator agent. The model ONNX file is ~2.4 GB at q4f16.

**Qwen3.5-4B (q4f16)** was significantly slower (~14.6 t/s, ~16s TTFT) due to its larger vocabulary and hybrid attention architecture not being fully optimized in ONNX Runtime's WebGPU kernels yet. So avoid the 3.5 variants for now — stick with Qwen3 or SmolLM3 where the attention patterns are better supported.

For the smaller **Qwen3-0.6B (q4f16)**, you can expect substantially faster inference — likely 60+ tokens/second on your M2 given the proportionally smaller compute load.

## Recommended setup for your M2

Given that the M2 MacBook Pro has either 8GB, 16GB, or 24GB unified memory (shared between CPU and GPU), the key constraint is how many models you can keep loaded simultaneously.

### If you have 16GB+ RAM (most likely scenario)

Use **Qwen3-0.6B** (`onnx-community/Qwen3-0.6B-ONNX`) as your single model for all agents. At q4f16, this model is very compact — well under 500MB — meaning you could theoretically load it multiple times or pair it with other models. The Qwen3 family supports thinking/no-thinking mode, so your orchestrator agent can use `/think` mode for complex reasoning while worker agents run in `/no_think` mode for speed. This is the simplest and most reliable architecture: one model, multiple roles via system prompts.

If you want a stronger orchestrator, pair **SmolLM3-3B** (`HuggingFaceTB/SmolLM3-3B-ONNX`) at q4f16 (~1.5 GB in memory) as the orchestrator with **Qwen3-0.6B** for the worker agents. SmolLM3-3B has excellent tool calling (92.3 BFCL score) and instruction following, which matters a lot for an orchestrator that needs to parse agent outputs and decide next steps. But don't keep both loaded simultaneously — load the orchestrator, run it, dispose it, then load the worker agent model. Use `ModelRegistry.is_pipeline_cached` to make subsequent loads near-instant from cache.

### If you have 8GB RAM

Stick with **Qwen3-0.6B** exclusively. With only 8GB shared between OS, browser, and GPU, a 3B model at q4 would be tight. The 0.6B model will give you comfortable headroom and still support the thinking/no-thinking toggle for your orchestrator vs worker agent distinction.

## Optimal configuration code for M2

The key settings for your M2:

```js
const generator = await pipeline(
  "text-generation",
  "onnx-community/Qwen3-0.6B-ONNX",
  { device: "webgpu", dtype: "q4f16" }
);
```

Use `q4f16` specifically (not `q4` or `q8`). The `q4f16` format keeps activations in float16, which Apple Silicon's GPU handles natively and efficiently. The `q4` format (int4 weights with fp32 activations) is less optimal for GPU execution on Metal. And `q8` doubles the model size for marginal quality improvement that won't matter for most educational app agent tasks.

## Architecture recommendation for your agentic system

Rather than loading multiple different models, I'd suggest a pattern where you use **one Qwen3-0.6B instance** with **role-differentiated system prompts** and thinking mode toggling:

The orchestrator agent uses thinking mode enabled, with a system prompt defining its role as the task planner and coordinator. It reasons through what educational content to generate, how to break tasks down, etc.

The worker agents (content generator, quiz builder, feedback agent, etc.) use no-think mode on the same model with task-specific system prompts. This gives you low-latency responses because skipping the thinking phase dramatically reduces output length and time.

This avoids the memory pressure of loading multiple models, eliminates model-swap latency, and the `q4f16` Qwen3-0.6B on your M2 should give you fast enough throughput that the user experience feels responsive.

If you find the 0.6B model's reasoning is insufficient for the orchestrator (it is a small model after all), you could explore an upgrade path to **Qwen3-1.7B** (`onnx-community/Qwen3-1.7B-ONNX`) which roughly triples the reasoning capability while still staying well within M2 memory limits. That would put you in a similar performance bracket to what the issue reporter saw with the 4B model — probably 30-40 t/s on your M2, which is still quite usable.