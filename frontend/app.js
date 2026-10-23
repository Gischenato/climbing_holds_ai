import { clampView, pickHold, polygonArea, zoomAt } from "./geometry.js";

const $ = (id) => document.getElementById(id);
const fileInput = $("file");
const detectBtn = $("detect");
const selectAllBtn = $("select-all");
const clearBtn = $("clear");
const zoomBtn = $("zoom-reset");
const counter = $("counter");
const statusEl = $("status");
const stage = $("stage");
const placeholder = $("placeholder");
const loading = $("loading");
const viewport = $("viewport");
const canvas = $("canvas");
const ctx = canvas.getContext("2d");

const MAX_ZOOM = 20;
const STAGE_PADDING = 16;   // padding do .stage (8px de cada lado)
const DRAG_THRESHOLD = 4;   // px: abaixo disso, o gesto é um clique, não um arrasto

// Estado. Todas as coordenadas de `holds` estão no espaço da imagem processada pelo servidor.
let file = null;
let image = null;      // HTMLImageElement com a foto
let holds = [];        // [{ id, polygon, bbox, score, area, path }]
let selected = new Set();
let hovered = null;
let busy = false;

// Visualização: `fit` ajusta a imagem à tela; `view` é o zoom/deslocamento em cima disso.
// O canvas guarda a imagem em resolução cheia e é escalado por CSS transform.
let fit = 1;
const view = { scale: 1, x: 0, y: 0 };
let drag = null;
let suppressClick = false;

function setStatus(message, isError = false) {
  statusEl.hidden = !message;
  statusEl.textContent = message || "";
  statusEl.classList.toggle("error", isError);
}

function updateUI() {
  detectBtn.disabled = !file || busy;
  selectAllBtn.disabled = busy || holds.length === 0;
  clearBtn.disabled = busy || selected.size === 0;
  zoomBtn.disabled = !image || view.scale === 1;
  zoomBtn.textContent = `Zoom ${Math.round(view.scale * 100)}%`;
  counter.textContent = holds.length
    ? `${selected.size} de ${holds.length} selecionadas`
    : "Nenhuma agarra detectada";
  loading.hidden = !busy;
}

function draw() {
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  // Espessuras em pixels de imagem, para ficarem constantes na tela qualquer que seja o zoom
  const unit = 1 / (fit * view.scale);

  for (const hold of holds) {
    const isSelected = selected.has(hold.id);
    if (isSelected) {
      ctx.fillStyle = "rgba(255, 122, 61, 0.55)";
      ctx.fill(hold.path);
    } else if (hold === hovered) {
      ctx.fillStyle = "rgba(255, 255, 255, 0.35)";
      ctx.fill(hold.path);
    }
    ctx.lineJoin = "round";
    ctx.lineWidth = (isSelected ? 3 : 2.5) * unit;       // halo escuro: contorno visível em fundo claro e escuro
    ctx.strokeStyle = "rgba(0, 0, 0, 0.55)";
    ctx.stroke(hold.path);
    ctx.lineWidth = (isSelected ? 1.6 : 1) * unit;
    ctx.strokeStyle = isSelected ? "#ffb48a" : "rgba(255, 255, 255, 0.9)";
    ctx.stroke(hold.path);
  }
}

// Aplica zoom/deslocamento: mantém a imagem cobrindo a moldura e redesenha
function applyView() {
  const vw = viewport.clientWidth;
  const vh = viewport.clientHeight;
  Object.assign(view, clampView(view, vw, vh));
  canvas.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.scale})`;
  canvas.style.touchAction = view.scale > 1 ? "none" : "manipulation"; // com zoom, o arrasto move a imagem
  canvas.style.cursor = view.scale > 1 ? "grab" : "pointer";
  draw();
  updateUI();
}

// Calcula o tamanho em que a imagem cabe na tela (sem ampliar além de 100%)
function layout() {
  if (!image) return;
  const availW = stage.clientWidth - STAGE_PADDING;
  const availH = window.innerHeight * 0.8;
  fit = Math.min(1, availW / canvas.width, availH / canvas.height);
  const w = `${canvas.width * fit}px`;
  const h = `${canvas.height * fit}px`;
  viewport.style.width = canvas.style.width = w;
  viewport.style.height = canvas.style.height = h;
  applyView();
}

function resetView() {
  view.scale = 1;
  view.x = view.y = 0;
}

function setImage(img, width = img.naturalWidth, height = img.naturalHeight) {
  if (width !== canvas.width || height !== canvas.height) resetView();
  image = img;
  canvas.width = width;
  canvas.height = height;
  viewport.hidden = false;
  placeholder.hidden = true;
  layout();
}

function loadFile(newFile) {
  if (!newFile || !newFile.type.startsWith("image/")) {
    setStatus("Escolha um arquivo de imagem.", true);
    return;
  }
  file = newFile;
  holds = [];
  selected = new Set();
  hovered = null;
  resetView();
  setStatus("");
  const url = URL.createObjectURL(newFile);
  const img = new Image();
  img.onload = () => {
    setImage(img);
    URL.revokeObjectURL(url);
    updateUI();
  };
  img.onerror = () => {
    URL.revokeObjectURL(url);
    file = null;
    setStatus("Não foi possível abrir essa imagem.", true);
    updateUI();
  };
  img.src = url;
}

async function detect() {
  if (!file || busy) return;
  busy = true;
  setStatus("");
  updateUI();
  try {
    const form = new FormData();
    form.append("file", file);
    const resp = await fetch("/detect", { method: "POST", body: form });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      const detail = typeof body.detail === "string" ? body.detail : `erro ${resp.status}`;
      throw new Error(detail);
    }
    const data = await resp.json();
    holds = data.holds.map((h) => {
      const path = new Path2D();
      h.polygon.forEach(([x, y], i) => (i === 0 ? path.moveTo(x, y) : path.lineTo(x, y)));
      path.closePath();
      return { ...h, area: polygonArea(h.polygon), path };
    });
    selected = new Set();
    hovered = null;
    // o servidor pode ter reduzido a imagem: o canvas passa a usar o tamanho que ele devolveu
    setImage(image, data.width, data.height);
    setStatus(holds.length ? "" : "Nenhuma agarra encontrada nesta foto.");
  } catch (err) {
    setStatus(`Falha na detecção: ${err.message}`, true);
  } finally {
    busy = false;
    updateUI();
  }
}

// Posição do mouse/dedo -> coordenadas da imagem. O rect do canvas já inclui o zoom e o deslocamento.
function toImageCoords(event) {
  const rect = canvas.getBoundingClientRect();
  return [
    ((event.clientX - rect.left) * canvas.width) / rect.width,
    ((event.clientY - rect.top) * canvas.height) / rect.height,
  ];
}

// Zoom com a roda do mouse (ou pinça no trackpad), ancorado no ponto sob o cursor
viewport.addEventListener(
  "wheel",
  (event) => {
    if (!image) return;
    event.preventDefault();
    const delta = event.deltaY * (event.deltaMode === 1 ? 16 : 1);
    const rect = viewport.getBoundingClientRect();
    const next = zoomAt(
      view,
      event.clientX - rect.left,
      event.clientY - rect.top,
      Math.exp(-delta * 0.0015),
      viewport.clientWidth,
      viewport.clientHeight,
      MAX_ZOOM,
    );
    if (next.scale === view.scale) return;
    Object.assign(view, next);
    applyView();
  },
  { passive: false },
);

canvas.addEventListener("pointerdown", (event) => {
  suppressClick = false;
  if (event.button !== 0 || view.scale <= 1) return;
  drag = { x: event.clientX, y: event.clientY, vx: view.x, vy: view.y, moved: false };
  canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener("pointermove", (event) => {
  if (drag) {
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    if (!drag.moved && Math.hypot(dx, dy) > DRAG_THRESHOLD) {
      drag.moved = true;
      canvas.style.cursor = "grabbing";
    }
    if (drag.moved) {
      view.x = drag.vx + dx;
      view.y = drag.vy + dy;
      applyView();
    }
    return;
  }
  if (busy || !holds.length) return;
  const hold = pickHold(holds, ...toImageCoords(event));
  if (hold !== hovered) {
    hovered = hold;
    canvas.style.cursor = hold ? "pointer" : view.scale > 1 ? "grab" : "default";
    draw();
  }
});

function endDrag() {
  if (!drag) return;
  suppressClick = drag.moved; // soltar depois de arrastar não deve selecionar a agarra embaixo
  drag = null;
  canvas.style.cursor = view.scale > 1 ? "grab" : "pointer";
}
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);

canvas.addEventListener("pointerleave", () => {
  if (hovered) {
    hovered = null;
    draw();
  }
});

canvas.addEventListener("click", (event) => {
  if (suppressClick) {
    suppressClick = false;
    return;
  }
  if (busy || !holds.length) return;
  const hold = pickHold(holds, ...toImageCoords(event));
  if (!hold) return;
  if (selected.has(hold.id)) selected.delete(hold.id);
  else selected.add(hold.id);
  draw();
  updateUI();
});

fileInput.addEventListener("change", () => loadFile(fileInput.files[0]));
detectBtn.addEventListener("click", detect);
zoomBtn.addEventListener("click", () => {
  resetView();
  applyView();
});
selectAllBtn.addEventListener("click", () => {
  selected = new Set(holds.map((h) => h.id));
  draw();
  updateUI();
});
clearBtn.addEventListener("click", () => {
  selected = new Set();
  draw();
  updateUI();
});
window.addEventListener("resize", layout);

// Arrastar e soltar uma imagem
for (const type of ["dragenter", "dragover"]) {
  stage.addEventListener(type, (e) => { e.preventDefault(); stage.classList.add("dragging"); });
}
for (const type of ["dragleave", "drop"]) {
  stage.addEventListener(type, (e) => { e.preventDefault(); stage.classList.remove("dragging"); });
}
stage.addEventListener("drop", (e) => loadFile(e.dataTransfer.files[0]));

updateUI();
