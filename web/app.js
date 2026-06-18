// pendant-swap minimal UI
// Posts to the FastAPI backend at the same origin (or localhost:8000).
// The API key is sent in the X-API-Key header — never stored, never logged.

const API_BASE = window.location.port === "5500" ? "http://localhost:8000" : "";

const form       = document.getElementById("swapForm");
const runBtn     = document.getElementById("runBtn");
const statusEl   = document.getElementById("status");
const resultsEl  = document.getElementById("results");
const resultImg  = document.getElementById("resultImg");
const downloadLink = document.getElementById("downloadLink");
const keySection = document.getElementById("keySection");
const retriesLabel = document.getElementById("retriesLabel");

// Carousel state
let attempts = [];
let current = 0;
const attemptLabel = document.getElementById("attemptLabel");
const attemptBadge = document.getElementById("attemptBadge");
const attemptQa    = document.getElementById("attemptQa");
const attemptList  = document.getElementById("attemptList");
const showBox      = document.getElementById("showBox");

document.getElementById("prevBtn").addEventListener("click", () => showAttempt(current - 1));
document.getElementById("nextBtn").addEventListener("click", () => showAttempt(current + 1));
showBox.addEventListener("change", () => renderCurrentImage());

function showAttempt(i) {
  if (!attempts.length) return;
  current = (i + attempts.length) % attempts.length;
  renderCurrentImage();
  const a = attempts[current];
  attemptLabel.textContent = "Attempt " + (current + 1) + " / " + attempts.length;
  attemptBadge.textContent = (a.passed ? "PASS" : "FAIL")
    + "  ·  " + a.height_mm + "mm  ·  aspect " + a.aspect + "  ·  score " + a.score;
  attemptBadge.className = "badge " + (a.passed ? "pass" : "fail");
  attemptQa.textContent = a.summary;
}

function renderCurrentImage() {
  const a = attempts[current];
  const useBox = showBox.checked && a.annotated;
  const src = "data:image/jpeg;base64," + (useBox ? a.annotated : a.image);
  resultImg.src = src;
  downloadLink.href = "data:image/jpeg;base64," + a.image;
  downloadLink.download = "pendant-attempt-" + (current + 1) + ".jpg";
}

function renderAttemptList() {
  attemptList.innerHTML = "";
  attempts.forEach((a, i) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "attempt-row " + (a.passed ? "pass" : "fail")
      + (i === current ? " active" : "");
    if (a.isFinal) {
      row.textContent = "★ FINAL (size-locked)  " + (a.passed ? "✓" : "✗");
    } else {
      row.textContent = "#" + (i) + "  " + (a.passed ? "✓" : "✗")
        + "  " + a.height_mm + "mm  asp " + a.aspect + "  (score " + a.score + ")";
    }
    row.addEventListener("click", () => { showAttempt(i); renderAttemptList(); });
    attemptList.appendChild(row);
  });
}

// Show/hide key field and retries based on mode
document.querySelectorAll("input[name='mode']").forEach(radio => {
  radio.addEventListener("change", () => {
    const isGenerate = radio.value === "generate" && radio.checked;
    keySection.classList.toggle("hidden", !isGenerate);
    retriesLabel.classList.toggle("hidden", !isGenerate);
  });
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  await runSwap();
});

async function runSwap() {
  const mode    = document.querySelector("input[name='mode']:checked").value;
  const apiKey  = document.getElementById("apiKey").value.trim();
  const modelFile   = document.getElementById("model").files[0];
  const pendantFile = document.getElementById("pendant").files[0];

  if (!modelFile || !pendantFile) {
    showStatus("Please select both model and pendant images.", "error");
    return;
  }
  if (mode === "generate" && !apiKey) {
    showStatus("A Gemini API key is required for Generate mode.", "error");
    return;
  }

  const fd = new FormData();
  fd.append("model",       modelFile);
  fd.append("pendant",     pendantFile);
  fd.append("mode",        mode);
  fd.append("target_mm",   document.getElementById("targetMm").value);
  fd.append("ref_px",      document.getElementById("refPx").value);
  fd.append("ref_mm",      document.getElementById("refMm").value);
  fd.append("rotate_deg",  document.getElementById("rotate").value);
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
  if (apiKey) {
    headers["X-API-Key"] = apiKey;  // preferred — never in URL or logs
  }

  showStatus(mode === "composite" ? "Running composite…" : "Running generate loop — this may take a moment…");
  runBtn.disabled = true;
  resultsEl.classList.add("hidden");

  try {
    const resp = await fetch(`${API_BASE}/swap`, {
      method: "POST",
      headers,
      body: fd,
    });

    const data = await resp.json();

    if (!resp.ok) {
      showStatus("Error: " + (data.detail || resp.statusText), "error");
      return;
    }

    if (data.attempts && data.attempts.length > 0) {
      // Generate mode. Headline = the size-locked FINAL result; then raw attempts.
      attempts = [];
      if (data.final_qa) {
        attempts.push({
          image: data.result_image,
          annotated: null,
          summary: "FINAL (size-locked)\n\n" + data.final_qa,
          passed: data.final_qa.indexOf("PASSED") !== -1,
          height_mm: "final", aspect: "—", score: "—",
          isFinal: true,
        });
      }
      data.attempts.forEach(a => attempts.push(a));
      current = 0;
      showAttempt(0);
      renderAttemptList();
    } else {
      // Composite mode: single image, no attempts
      attempts = [{
        image: data.result_image,
        annotated: null,
        summary: "(composite mode — no QA)",
        passed: true, height_mm: "-", aspect: "-", score: "-",
      }];
      current = 0;
      showAttempt(0);
      renderAttemptList();
    }

    const sizeNote = data.gen_image_size
      ? "  (Gemini output: " + data.gen_image_size[0] + "×" + data.gen_image_size[1] + "px)" : "";
    resultsEl.classList.remove("hidden");
    showStatus("Done." + sizeNote);
  } catch (err) {
    showStatus("Network error: " + err.message, "error");
  } finally {
    runBtn.disabled = false;
  }
}

function showStatus(msg, type) {
  statusEl.textContent = msg;
  statusEl.className = type === "error" ? "error" : "";
  statusEl.classList.remove("hidden");
}

// ---------------------------------------------------------------------------
// Settings persistence (localStorage). Everything EXCEPT the API key and the
// file pickers — the key is a secret (never stored) and browsers forbid
// pre-filling file inputs.
// ---------------------------------------------------------------------------
const STORE_KEY = "pendant-swap-settings";

// id -> "value" | "checked"; plus the mode radio handled separately.
// NOTE: compositeFinish (Force exact size) is intentionally NOT persisted — it's
// an occasional override that should always default OFF, never get stuck on.
const PERSIST = {
  targetMm: "value", refPx: "value", refMm: "value",
  hangX: "value", hangY: "value", rotate: "value", topCrop: "value",
  maxRetries: "value", modelId: "value", extraPrompt: "value",
  replaceChain: "checked",
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

// Save on any change within the form; restore once on load.
form.addEventListener("input", saveSettings);
form.addEventListener("change", saveSettings);
restoreSettings();
