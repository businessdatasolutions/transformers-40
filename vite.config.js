import { defineConfig } from "vite";

export default defineConfig({
  optimizeDeps: {
    exclude: ["@huggingface/transformers"], // don't pre-bundle — ONNX uses dynamic imports at runtime
  },
});
