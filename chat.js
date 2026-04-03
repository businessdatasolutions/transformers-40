import { pipeline, TextStreamer } from "@huggingface/transformers";

// ── Model registry ────────────────────────────────────────────────────────────
const MODELS = [
  {
    id:       "onnx-community/Qwen3-0.6B-ONNX",
    label:    "Qwen3 0.6B",
    tag:      "0.6B · q4f16 · WebGPU",
    dtype:    "q4f16",
    thinking: true,
  },
  {
    id:       "onnx-community/Llama-3.2-1B-Instruct-ONNX",
    label:    "Llama 3.2 1B",
    tag:      "1B · q4f16 · WebGPU",
    dtype:    "q4f16",
    thinking: false,
  },
];

let currentModel = MODELS[0];

// ── DOM refs ──────────────────────────────────────────────────────────────────
const messagesEl  = document.getElementById("messages");
const inputEl     = document.getElementById("input");
const sendBtn     = document.getElementById("send-btn");
const thinkBtn    = document.getElementById("think-btn");
const statusEl    = document.getElementById("status");
const progressBar = document.getElementById("progress-bar");
const progressWrap= document.getElementById("progress-wrap");
const modelSelect = document.getElementById("model-select");
const modelTag    = document.getElementById("model-tag");
const loadBtn     = document.getElementById("load-btn");
const fileInput   = document.getElementById("file-input");
const docsBadge   = document.getElementById("docs-badge");

// ── PDF.js worker ─────────────────────────────────────────────────────────────
if (window.pdfjsLib) {
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    "https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.worker.min.js";
}

// ── Populate model dropdown ───────────────────────────────────────────────────
MODELS.forEach((m, i) => {
  const opt       = document.createElement("option");
  opt.value       = i;
  opt.textContent = m.label;
  modelSelect.appendChild(opt);
});

// ── State ─────────────────────────────────────────────────────────────────────
let generator       = null;
let embedder        = null;  // feature-extraction pipeline; reused across chat model switches
let generating      = false;
let thinkingEnabled = true;
let messages        = [];    // cleared on model switch

// ── RAG: chunk store ──────────────────────────────────────────────────────────
const chunkStore = []; // { text: string, embedding: Float32Array, source: string }

// ── Thinking-mode toggle ──────────────────────────────────────────────────────
thinkBtn.addEventListener("click", () => {
  thinkingEnabled = !thinkingEnabled;
  thinkBtn.textContent    = thinkingEnabled ? "Thinking ON" : "Thinking OFF";
  thinkBtn.dataset.active = thinkingEnabled;
});

// ── Progress tracking ─────────────────────────────────────────────────────────
let fileProgress = {};
let fileDone     = new Set();
let totalFiles   = 0;

function resetProgress() {
  fileProgress = {};
  fileDone     = new Set();
  totalFiles   = 0;
}

function updateProgress(info) {
  if (info.status === "initiate") totalFiles++;

  if (info.status === "progress") {
    progressBar.classList.remove("compiling");
    fileProgress[info.file] = info.progress ?? 0;
    const avg = Object.values(fileProgress).reduce((a, b) => a + b, 0) / Object.values(fileProgress).length;
    progressBar.style.width = `${avg.toFixed(1)}%`;
    statusEl.textContent    = `Downloading… ${avg.toFixed(0)}%`;
  }

  if (info.status === "done") {
    fileDone.add(info.file);
    if (totalFiles > 0 && fileDone.size >= totalFiles) {
      progressBar.style.width = "100%";
      progressBar.classList.add("compiling");
      statusEl.textContent = "Compiling WebGPU shaders…";
    }
  }
}

// ── Model load ────────────────────────────────────────────────────────────────
async function loadModel(modelDef) {
  currentModel = modelDef;

  resetProgress();
  setUIEnabled(false);
  modelSelect.disabled    = true;
  loadBtn.disabled        = true;
  progressWrap.hidden     = false;
  progressBar.style.width = "100%";
  progressBar.classList.add("compiling");
  statusEl.textContent    = "Fetching model files…";
  modelTag.textContent    = modelDef.tag;
  thinkBtn.hidden         = !modelDef.thinking;

  try {
    // Load chat model + embedding model in parallel.
    // embedder ?? pipeline(...) reuses the already-loaded embedder on model switch.
    [generator, embedder] = await Promise.all([
      pipeline("text-generation", modelDef.id, {
        device:            "webgpu",
        dtype:             modelDef.dtype,
        progress_callback: updateProgress,
      }),
      embedder ?? pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
        dtype:             "fp32",
        progress_callback: updateProgress,
      }),
    ]);

    progressWrap.hidden  = true;
    statusEl.textContent = "Ready";
    setUIEnabled(true);
    modelSelect.disabled = false;
    loadBtn.disabled     = false;
  } catch (err) {
    statusEl.textContent = `Failed to load: ${err.message}`;
    console.error(err);
    modelSelect.disabled = false;
    loadBtn.disabled     = false;
  }
}

// ── Load button ───────────────────────────────────────────────────────────────
loadBtn.addEventListener("click", async () => {
  if (generating) return;
  const selected = MODELS[modelSelect.value];

  if (generator && selected.id === currentModel.id) return; // already loaded

  if (generator) {
    await generator.dispose(); // free WebGPU memory before loading next
    generator = null;
    messages  = [];
    messagesEl.replaceChildren();
  }

  await loadModel(selected);
});

// ── RAG helpers ───────────────────────────────────────────────────────────────

function chunkText(text, size = 500, overlap = 100) {
  const chunks = [];
  let i = 0;
  while (i < text.length) {
    let end = Math.min(i + size, text.length);
    if (end < text.length) {
      const dot = text.lastIndexOf(".", end);
      if (dot > i + size * 0.5) end = dot + 1;
    }
    const chunk = text.slice(i, end).trim();
    if (chunk.length > 50) chunks.push(chunk);
    if (end >= text.length) break;
    i = end - overlap;
  }
  return chunks;
}

function cosineSim(a, b) {
  let dot = 0, nA = 0, nB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]; nA += a[i] * a[i]; nB += b[i] * b[i];
  }
  return dot / (Math.sqrt(nA) * Math.sqrt(nB));
}

async function retrieve(query, k = 3, minScore = 0.25) {
  const qTensor = await embedder(query, { pooling: "mean", normalize: true });
  const qEmb    = qTensor.data;
  return chunkStore
    .map(c  => ({ ...c, score: cosineSim(qEmb, c.embedding) }))
    .filter(c => c.score >= minScore)
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

async function extractPdfText(file) {
  const buffer = await file.arrayBuffer();
  const pdf    = await pdfjsLib.getDocument({ data: buffer }).promise;
  const pages  = await Promise.all(
    Array.from({ length: pdf.numPages }, (_, i) =>
      pdf.getPage(i + 1)
        .then(p  => p.getTextContent())
        .then(tc => tc.items.map(it => it.str).join(" "))
    )
  );
  return pages.join("\n\n");
}

function updateDocsBadge() {
  const sources = new Set(chunkStore.map(c => c.source)).size;
  docsBadge.textContent = `${chunkStore.length} chunks · ${sources} doc${sources !== 1 ? "s" : ""}`;
  docsBadge.hidden      = chunkStore.length === 0;
}

// ── File upload handler ───────────────────────────────────────────────────────
fileInput.addEventListener("change", async (e) => {
  const prevStatus = statusEl.textContent;
  setUIEnabled(false);

  for (const file of e.target.files) {
    try {
      const text   = file.name.toLowerCase().endsWith(".pdf")
        ? await extractPdfText(file)
        : await file.text();

      const chunks = chunkText(text);
      for (let i = 0; i < chunks.length; i++) {
        statusEl.textContent = `Embedding chunk ${i + 1}/${chunks.length} from "${file.name}"…`;
        const tensor = await embedder(chunks[i], { pooling: "mean", normalize: true });
        chunkStore.push({
          text:      chunks[i],
          embedding: new Float32Array(tensor.data),
          source:    file.name,
        });
      }
    } catch (err) {
      statusEl.textContent = `Error processing "${file.name}": ${err.message}`;
      console.error(err);
    }
  }

  e.target.value       = "";
  statusEl.textContent = prevStatus;
  updateDocsBadge();
  setUIEnabled(true);
});

// ── ThinkingParser ────────────────────────────────────────────────────────────
const THINK_OPEN  = "<think>";
const THINK_CLOSE = "</think>";

class ThinkingParser {
  constructor(onThinkUpdate, onAnswerToken) {
    this.onThinkUpdate = onThinkUpdate;
    this.onAnswerToken = onAnswerToken;
    this.state         = "PREFIX";
    this.buffer        = "";
    this.thinkingText  = "";
  }

  feed(token) {
    switch (this.state) {
      case "PREFIX":   this._feedPrefix(token);   break;
      case "THINKING": this._feedThinking(token); break;
      case "ANSWER":   this.onAnswerToken(token); break;
    }
  }

  _feedPrefix(token) {
    this.buffer += token;
    if (this.buffer === THINK_OPEN) { this.state = "THINKING"; this.buffer = ""; return; }
    if (THINK_OPEN.startsWith(this.buffer)) return;
    this.state = "ANSWER";
    this.onAnswerToken(this.buffer);
    this.buffer = "";
  }

  _feedThinking(token) {
    this.thinkingText += token;
    const closeIdx = this.thinkingText.indexOf(THINK_CLOSE);
    if (closeIdx === -1) {
      const safeLen = Math.max(0, this.thinkingText.length - THINK_CLOSE.length);
      this.onThinkUpdate(this.thinkingText.slice(0, safeLen));
      return;
    }
    this.onThinkUpdate(this.thinkingText.slice(0, closeIdx));
    const remainder = this.thinkingText.slice(closeIdx + THINK_CLOSE.length).trimStart();
    this.state = "ANSWER";
    if (remainder) this.onAnswerToken(remainder);
  }
}

// ── Chat send ─────────────────────────────────────────────────────────────────
async function send() {
  const text = inputEl.value.trim();
  if (!text || generating || !generator) return;

  generating = true;
  setUIEnabled(false);
  inputEl.value = "";

  appendMessage("user", text);
  messages.push({ role: "user", content: text });

  const { thinkDetails, thinkPre, answerDiv } = appendAssistantBubble();

  let answerText   = "";
  let thinkStarted = false;

  const parser = new ThinkingParser(
    (thinkSoFar) => {
      if (!thinkStarted) { thinkStarted = true; thinkDetails.hidden = false; }
      thinkPre.textContent = thinkSoFar;
      messagesEl.scrollTop = messagesEl.scrollHeight;
    },
    (token) => {
      answerText           += token;
      answerDiv.textContent = answerText;
      messagesEl.scrollTop  = messagesEl.scrollHeight;
    },
  );

  // ── RAG: retrieve relevant chunks and build context block ─────────────────
  let contextBlock = "";
  if (chunkStore.length > 0 && embedder) {
    const hits = await retrieve(text);
    if (hits.length > 0) {
      contextBlock =
        "\n\nCONTEXT (from uploaded documents — use this to answer when relevant):\n\n" +
        hits.map(h => `[${h.source}]\n${h.text}`).join("\n\n---\n\n") +
        "\n\nIf the answer is not in the context, say so clearly.";
    }
  }

  const thinkingDirective = currentModel.thinking
    ? (thinkingEnabled ? "/think " : "/no_think ")
    : "";

  const systemMsg = {
    role:    "system",
    content: `${thinkingDirective}You are a helpful assistant.${contextBlock}\n\nFormat responses in Markdown. For mathematical expressions use LaTeX: $...$ for inline math and $$...$$ for display equations.`,
  };

  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt:         true,
    skip_special_tokens: true,
    callback_function:   (token) => parser.feed(token),
  });

  await generator([systemMsg, ...messages], {
    max_new_tokens: 4096,
    temperature:    0.7,
    do_sample:      true,
    streamer,
  });

  if (parser.state === "PREFIX" && parser.buffer) {
    answerText           += parser.buffer;
    answerDiv.textContent = answerText;
  }

  renderAnswer(answerDiv, answerText);
  messages.push({ role: "assistant", content: answerText.trim() });

  generating = false;
  setUIEnabled(true);
  inputEl.focus();
}

sendBtn.addEventListener("click", send);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});

// ── DOM helpers ───────────────────────────────────────────────────────────────
function appendMessage(role, text) {
  const div       = document.createElement("div");
  div.className   = `message ${role}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function appendAssistantBubble() {
  const bubble           = document.createElement("div");
  bubble.className       = "message assistant";

  const thinkDetails     = document.createElement("details");
  thinkDetails.hidden    = true;
  thinkDetails.className = "think-block";
  const summary          = document.createElement("summary");
  summary.textContent    = "Thinking…";
  thinkDetails.appendChild(summary);
  const thinkPre         = document.createElement("pre");
  thinkPre.className     = "think-text";
  thinkDetails.appendChild(thinkPre);

  const answerDiv        = document.createElement("div");
  answerDiv.className    = "answer-text";

  bubble.appendChild(thinkDetails);
  bubble.appendChild(answerDiv);
  messagesEl.appendChild(bubble);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  return { bubble, thinkDetails, thinkPre, answerDiv };
}

function renderAnswer(el, text) {
  const rawHtml  = window.marked.parse(text);
  const safeHtml = window.DOMPurify.sanitize(rawHtml);
  el.innerHTML   = safeHtml;
  el.classList.add("rendered");
  if (window.renderMathInElement) {
    renderMathInElement(el, {
      delimiters: [
        { left: "$$", right: "$$", display: true  },
        { left: "$",  right: "$",  display: false },
      ],
      throwOnError: false,
    });
  }
}

function setUIEnabled(enabled) {
  inputEl.disabled   = !enabled;
  sendBtn.disabled   = !enabled;
  thinkBtn.disabled  = !enabled;
  fileInput.disabled = !enabled;
}
