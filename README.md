# Transformers.js Playground

Two browser-based ML demos powered by [Transformers.js v4](https://huggingface.co/docs/transformers.js) and WebGPU. No server, no Python — everything runs locally in the browser.

## Demos

### Object Detection (`index.html`)
Upload any image and get bounding boxes drawn around detected objects. Uses the [DETR ResNet-50](https://huggingface.co/Xenova/detr-resnet-50) model with a 50% confidence threshold.

### Chat with RAG (`chat.html`)
A full chat interface with:
- **Model switching** — Qwen3 0.6B (with thinking mode) or Llama 3.2 1B
- **Thinking mode** — toggle visible chain-of-thought reasoning (Qwen3 only)
- **Document RAG** — upload `.txt` or `.pdf` files; relevant chunks are retrieved and injected into the prompt using semantic search
- **Streaming** — tokens stream to the UI as they are generated
- **Markdown + LaTeX rendering** — responses are rendered with `marked`, sanitized with `DOMPurify`, and math is rendered with KaTeX

## Requirements

- A browser with **WebGPU support** (Chrome 113+, Edge 113+)
- Node.js (for local dev server only)

## Getting Started

```bash
npm install
npm run dev
```

Then open `http://localhost:5173` in your browser.

## How It Works

### Object Detection
`index.js` loads a `pipeline("object-detection", ...)` pipeline directly from the Hugging Face Hub using a CDN import. When you upload an image, it runs inference and overlays CSS-positioned bounding boxes on the image.

### Chat RAG Pipeline
1. On model load, both the chat model and an embedding model (`all-MiniLM-L6-v2`) are loaded in parallel via WebGPU.
2. Uploaded documents are chunked with overlap, embedded, and stored in memory as `Float32Array` vectors.
3. On each user message, the top-k most relevant chunks are retrieved via cosine similarity and prepended to the system prompt as context.
4. A `ThinkingParser` state machine separates `<think>...</think>` tokens from the final answer during streaming, displaying them in a collapsible block.

## Models

| Model | Size | Backend | Thinking |
|---|---|---|---|
| [Qwen3 0.6B](https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX) | 0.6B | WebGPU (q4f16) | Yes |
| [Llama 3.2 1B](https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct-ONNX) | 1B | WebGPU (q4f16) | No |
| [DETR ResNet-50](https://huggingface.co/Xenova/detr-resnet-50) | — | WASM | — |
| [all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2) | — | WASM (fp32) | — |
