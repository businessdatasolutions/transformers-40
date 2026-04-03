import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers";

// Only load models from Hugging Face Hub (not local filesystem)
env.allowLocalModels = false;

const fileUpload = document.getElementById("file-upload");
const imageContainer = document.getElementById("image-container");
const status = document.getElementById("status");

status.textContent = "Loading model...";

// Top-level await: page is interactive only after the model is ready
const detector = await pipeline("object-detection", "Xenova/detr-resnet-50");

status.textContent = "Ready";

fileUpload.addEventListener("change", function (e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();

  reader.onload = function (e2) {
    imageContainer.replaceChildren(); // safe DOM clear (no innerHTML)
    const image = document.createElement("img");
    image.src = e2.target.result;
    imageContainer.appendChild(image);
    detect(image);
  };

  reader.readAsDataURL(file);
});

async function detect(img) {
  status.textContent = "Analysing...";

  const output = await detector(img.src, {
    threshold: 0.5,   // only show detections with ≥50% confidence
    percentage: true, // return box coordinates as fractions (0–1) for CSS %
  });

  status.textContent = "";
  output.forEach(renderBox);
}

function renderBox({ box, label }) {
  const { xmin, ymin, xmax, ymax } = box;

  // Random color per box
  const color = "#" + Math.floor(Math.random() * 0xffffff).toString(16).padStart(6, "0");

  const boxElement = document.createElement("div");
  boxElement.className = "bounding-box";
  Object.assign(boxElement.style, {
    borderColor: color,
    left:   `${100 * xmin}%`,
    top:    `${100 * ymin}%`,
    width:  `${100 * (xmax - xmin)}%`,
    height: `${100 * (ymax - ymin)}%`,
  });

  const labelElement = document.createElement("span");
  labelElement.textContent = label;
  labelElement.className = "bounding-box-label";
  labelElement.style.backgroundColor = color;

  boxElement.appendChild(labelElement);
  imageContainer.appendChild(boxElement);
}
