// Pendant Swap Studio: UI controller.
// Posts to the FastAPI backend; the API key is sent in the X-API-Key header and never stored.
// Scrolling is NATIVE (no JS scroll hijacking). Motion (Framer team's vanilla engine) is
// loaded lazily and only enhances dynamic moments; the app works fully without it.

const API_BASE = window.location.port === "5500" ? "http://localhost:8000" : "";
const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Mark <html> so entrance elements start hidden (revealed by IntersectionObserver below).
document.documentElement.classList.add("js-anim");

// --- Lazy Motion (non-blocking; resolves long before any result comes back) ---
let MO = null;
const motionLoaded = import("https://cdn.jsdelivr.net/npm/motion@11/+esm")
  .then((m) => { MO = m; }).catch(() => {});
const delay = (ms) => new Promise((r) => setTimeout(r, ms));

// --- Lenis smooth scroll (studio-grade inertia; native scroll if it fails / reduced-motion) ---
let lenis = null;
if (!reduce) {
  import("https://cdn.jsdelivr.net/npm/lenis@1/+esm")
    .then(({ default: Lenis }) => {
      lenis = new Lenis({
        duration: 1.05,
        easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
        smoothWheel: true,
        wheelMultiplier: 1,
        touchMultiplier: 1.6,
      });
      const raf = (time) => { lenis.raf(time); requestAnimationFrame(raf); };
      requestAnimationFrame(raf);
    })
    .catch(() => {});
}
function smoothScrollTo(el) {
  if (lenis) lenis.scrollTo(el, { offset: -16 });
  else el.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
}

const EASE = [0.16, 1, 0.3, 1];
function fxFade(el, fromTransform = "translateY(8px)") {
  if (!MO || reduce || !el) return;
  MO.animate(el, { opacity: [0, 1], transform: [fromTransform, "none"] }, { duration: 0.4, easing: EASE });
}
function fxPop(el) {
  if (!MO || reduce || !el) return;
  MO.animate(el, { transform: ["scale(0.82)", "scale(1)"] }, { duration: 0.36, easing: [0.34, 1.56, 0.64, 1] });
}
function fxStagger(els) {
  if (!MO || reduce || !els.length) return;
  MO.animate(els, { opacity: [0, 1], transform: ["translateY(10px)", "none"] },
    { duration: 0.4, delay: MO.stagger(0.045), easing: EASE });
}

// --- Element refs ---
const form = document.getElementById("swapForm");
const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const resultImg = document.getElementById("resultImg");
const downloadLink = document.getElementById("downloadLink");
const keySection = document.getElementById("keySection");
const retriesLabel = document.getElementById("retriesLabel");
const attemptLabel = document.getElementById("attemptLabel");
const attemptBadge = document.getElementById("attemptBadge");
const attemptQa = document.getElementById("attemptQa");
const attemptList = document.getElementById("attemptList");
const showBox = document.getElementById("showBox");

let attempts = [];
let current = 0;

// --- Carousel ---
document.getElementById("prevBtn").addEventListener("click", () => { showAttempt(current - 1); renderAttemptList(); });
document.getElementById("nextBtn").addEventListener("click", () => { showAttempt(current + 1); renderAttemptList(); });
showBox.addEventListener("change", () => renderCurrentImage());

function showAttempt(i) {
  if (!attempts.length) return;
  current = (i + attempts.length) % attempts.length;
  renderCurrentImage();
  const a = attempts[current];
  attemptLabel.textContent = "Attempt " + (current + 1) + " / " + attempts.length;
  attemptBadge.textContent = (a.passed ? "PASS" : "FAIL") + "  ·  " + a.height_mm + "mm  ·  aspect " + a.aspect + "  ·  score " + a.score;
  attemptBadge.className = "badge " + (a.passed ? "pass" : "fail");
  attemptQa.textContent = a.summary;
  fxPop(attemptBadge);
}

function renderCurrentImage() {
  const a = attempts[current];
  if (!a) return;
  const useBox = showBox.checked && a.annotated;
  resultImg.src = "data:image/jpeg;base64," + (useBox ? a.annotated : a.image);
  downloadLink.href = "data:image/jpeg;base64," + a.image;
  downloadLink.download = "pendant-attempt-" + (current + 1) + ".jpg";
  fxFade(resultImg, "scale(1.03)");
}

function renderAttemptList() {
  attemptList.innerHTML = "";
  attempts.forEach((a, i) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "attempt-row " + (a.passed ? "pass" : "fail") + (i === current ? " active" : "");
    if (a.isFinal) {
      row.textContent = "★ FINAL size-locked · " + (a.passed ? "pass" : "review");
    } else {
      row.textContent = "#" + (i + 1) + " · " + (a.passed ? "✓" : "✗") + " · " + a.height_mm + "mm · asp " + a.aspect + " · score " + a.score;
    }
    row.addEventListener("click", () => { showAttempt(i); renderAttemptList(); });
    attemptList.appendChild(row);
  });
  fxStagger([...attemptList.children]);
}

// --- Mode toggle ---
document.querySelectorAll("input[name='mode']").forEach((radio) => {
  radio.addEventListener("change", () => {
    const isGenerate = radio.value === "generate" && radio.checked;
    keySection.classList.toggle("hidden", !isGenerate);
    retriesLabel.classList.toggle("hidden", !isGenerate);
  });
});

// --- Submit ---
form.addEventListener("submit", async (event) => { event.preventDefault(); await runSwap(); });

async function runSwap() {
  const mode = document.querySelector("input[name='mode']:checked").value;
  const apiKey = document.getElementById("apiKey").value.trim();
  const modelFile = document.getElementById("model").files[0];
  const pendantFile = document.getElementById("pendant").files[0];

  if (!modelFile || !pendantFile) { showStatus("Please select both a model photo and a pendant photo.", "error"); return; }
  if (mode === "generate" && !apiKey) { showStatus("A Gemini API key is required for Generate mode.", "error"); return; }

  const fd = new FormData();
  fd.append("model", modelFile);
  fd.append("pendant", pendantFile);
  fd.append("mode", mode);
  fd.append("target_mm", document.getElementById("targetMm").value);
  fd.append("ref_px", document.getElementById("refPx").value);
  fd.append("ref_mm", document.getElementById("refMm").value);
  fd.append("rotate_deg", document.getElementById("rotate").value);
  fd.append("top_crop_px", document.getElementById("topCrop").value);

  const hangX = document.getElementById("hangX").value;
  const hangY = document.getElementById("hangY").value;
  if (hangX) fd.append("hang_x", hangX);
  if (hangY) fd.append("hang_y", hangY);

  if (mode === "generate") {
    fd.append("max_retries", document.getElementById("maxRetries").value);
    fd.append("model_id", document.getElementById("modelId").value);
    fd.append("extra_prompt", document.getElementById("extraPrompt").value.trim());
    fd.append("composite_finish", document.getElementById("compositeFinish").checked ? "true" : "false");
    fd.append("replace_chain", document.getElementById("replaceChain").checked ? "true" : "false");
  }

  const headers = {};
  if (apiKey) headers["X-API-Key"] = apiKey;

  showStatus(mode === "composite" ? "Running composite…" : "Running generate loop. This may take a moment…");
  setRunning(true);

  try {
    const resp = await fetch(`${API_BASE}/swap`, { method: "POST", headers, body: fd });
    const data = await resp.json();
    if (!resp.ok) { showStatus("Error: " + (data.detail || resp.statusText), "error"); return; }

    if (data.attempts && data.attempts.length > 0) {
      attempts = [];
      if (data.final_qa) {
        attempts.push({
          image: data.result_image, annotated: null,
          summary: "FINAL (size-locked)\n\n" + data.final_qa,
          passed: data.final_qa.indexOf("PASSED") !== -1,
          height_mm: "final", aspect: "n/a", score: "n/a", isFinal: true,
        });
      }
      data.attempts.forEach((a) => attempts.push(a));
    } else {
      attempts = [{ image: data.result_image, annotated: null, summary: "Composite mode (no QA)", passed: true, height_mm: "n/a", aspect: "n/a", score: "n/a" }];
    }

    current = 0;
    showAttempt(0);
    renderAttemptList();

    const sizeNote = data.gen_image_size ? "  ·  Gemini output " + data.gen_image_size[0] + "×" + data.gen_image_size[1] + "px" : "";
    resultsEl.classList.remove("hidden");
    resultsEl.classList.add("is-in");
    showStatus("Done." + sizeNote);
    fxFade(resultsEl, "translateY(16px)");
    smoothScrollTo(resultsEl);
  } catch (err) {
    showStatus("Network error: " + err.message, "error");
  } finally {
    setRunning(false);
  }
}

function setRunning(isRunning) {
  runBtn.disabled = isRunning;
  document.body.classList.toggle("is-running", isRunning);
}

function showStatus(msg, type) {
  statusEl.textContent = msg;
  statusEl.className = type === "error" ? "error" : "";
  statusEl.classList.remove("hidden");
  fxFade(statusEl, "translateY(6px)");
}

// --- Settings persistence (everything except API key + files) ---
const STORE_KEY = "pendant-swap-settings";
const PERSIST = {
  targetMm: "value", refPx: "value", refMm: "value", hangX: "value", hangY: "value",
  rotate: "value", topCrop: "value", maxRetries: "value", modelId: "value",
  extraPrompt: "value", replaceChain: "checked",
};

function saveSettings() {
  const data = {};
  for (const [id, prop] of Object.entries(PERSIST)) {
    const el = document.getElementById(id);
    if (el) data[id] = el[prop];
  }
  const mode = document.querySelector("input[name='mode']:checked");
  if (mode) data.mode = mode.value;
  try { localStorage.setItem(STORE_KEY, JSON.stringify(data)); } catch (e) {}
}

function restoreSettings() {
  let data;
  try { data = JSON.parse(localStorage.getItem(STORE_KEY)); } catch (e) { return; }
  if (!data) return;
  for (const [id, prop] of Object.entries(PERSIST)) {
    if (data[id] == null) continue;
    const el = document.getElementById(id);
    if (el) el[prop] = data[id];
  }
  if (data.mode) {
    const radio = document.querySelector(`input[name='mode'][value='${data.mode}']`);
    if (radio) { radio.checked = true; radio.dispatchEvent(new Event("change")); }
  }
}

// --- File name display ---
function bindFileName(inputId, labelId) {
  const input = document.getElementById(inputId);
  const label = document.getElementById(labelId);
  if (!input || !label) return;
  input.addEventListener("change", () => {
    const file = input.files[0];
    label.textContent = file ? file.name : "Choose file";
    input.closest(".upload-card")?.classList.toggle("has-file", !!file);
  });
}

// --- Entrance reveal: Motion spring when available, CSS fallback. Never leaves content hidden. ---
function revealEl(el, idx) {
  if (MO && !reduce) {
    MO.animate(el, { opacity: [0, 1], transform: ["translateY(20px)", "none"] },
      { type: "spring", stiffness: 88, damping: 18, delay: Math.min(idx * 0.07, 0.24) });
  } else {
    el.style.transitionDelay = Math.min(idx * 80, 200) + "ms";
    el.classList.add("is-in");
  }
}

async function initReveal() {
  const els = [...document.querySelectorAll("[data-enter]")];
  const revealAll = () => els.forEach((el) => el.classList.add("is-in"));
  if (reduce || !("IntersectionObserver" in window)) { revealAll(); return; }

  // Give Motion a brief moment so the first paint uses springs, not the CSS fallback.
  await Promise.race([motionLoaded, delay(240)]);

  const io = new IntersectionObserver((entries, obs) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      revealEl(entry.target, els.indexOf(entry.target));
      obs.unobserve(entry.target);
    });
  }, { threshold: 0.12 });
  els.forEach((el) => io.observe(el));

  // Safety net: if anything is still hidden after 1.6s, reveal it.
  setTimeout(revealAll, 1600);
}

// --- Boot ---
form.addEventListener("input", saveSettings);
form.addEventListener("change", saveSettings);
restoreSettings();
bindFileName("model", "modelFileName");
bindFileName("pendant", "pendantFileName");
initReveal();
