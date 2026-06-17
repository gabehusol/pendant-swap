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
const qaReport   = document.getElementById("qaReport");
const attemptCount = document.getElementById("attemptCount");
const keySection = document.getElementById("keySection");
const retriesLabel = document.getElementById("retriesLabel");

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

    // Render result image
    const imgSrc = "data:image/jpeg;base64," + data.result_image;
    resultImg.src = imgSrc;
    downloadLink.href = imgSrc;

    // Final composited QA (what actually matters)
    const finalQaEl = document.getElementById("finalQa");
    if (data.final_qa) {
      finalQaEl.textContent = data.final_qa
        + (data.gen_image_size ? "\n\n(Gemini output: " + data.gen_image_size[0] + "×" + data.gen_image_size[1] + "px)" : "");
    } else {
      finalQaEl.textContent = data.mode === "composite" ? "(composite mode)" : "(composite finish off)";
    }

    // AI attempt QA reports
    if (data.qa_reports && data.qa_reports.length > 0) {
      qaReport.textContent = data.qa_reports.join("\n\n---\n\n");
      attemptCount.textContent = "Attempts: " + data.qa_reports.length
        + (data.chosen_attempt != null
           ? "  |  Best: attempt " + (data.chosen_attempt + 1)
           : "");
    } else {
      qaReport.textContent = "(no QA report — composite mode)";
      attemptCount.textContent = "";
    }

    resultsEl.classList.remove("hidden");
    showStatus("Done.");
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
