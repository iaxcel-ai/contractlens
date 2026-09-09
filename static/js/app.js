// Frontend logic for ContractLens. This file:
//   1. Handles the "paste text" vs "upload .txt" tabs
//   2. Sends the document text to the Flask backend (POST /analyze)
//   3. Renders the JSON response (score cards + expandable clause cards)
//   4. Builds a CSV download client-side from the last analysis result

// Keep the most recent analysis result around so the CSV download button
// doesn't need to re-request anything from the server.
let lastAnalysisResult = null;

const contractTextArea = document.getElementById("contract-text");
const fileInput = document.getElementById("file-input");
const analyzeButton = document.getElementById("analyze-button");
const errorMessage = document.getElementById("error-message");
const resultsSection = document.getElementById("results-section");
const clauseList = document.getElementById("clause-list");
const flaggedOnlyCheckbox = document.getElementById("flagged-only-checkbox");
const downloadCsvButton = document.getElementById("download-csv-button");

let activeTab = "paste"; // "paste" or "upload"

// --- Tab switching (paste text vs upload file) -----------------------------

document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => {
    activeTab = button.dataset.tab;

    document.querySelectorAll(".tab-button").forEach((b) => b.classList.toggle("active", b === button));
    document.querySelectorAll(".tab-panel").forEach((panel) => {
      panel.classList.toggle("hidden", panel.dataset.panel !== activeTab);
    });
  });
});

// Show the chosen filename in the drop zone instead of the browser's
// default "no file chosen" text.
const fileDropLabel = document.getElementById("file-drop-label");
fileInput.addEventListener("change", () => {
  fileDropLabel.textContent = fileInput.files[0] ? fileInput.files[0].name : "Choose a .txt file, or drop it here";
});

// --- Getting the document text, whichever tab is active ---------------------

function getDocumentText() {
  if (activeTab === "paste") {
    return Promise.resolve(contractTextArea.value);
  }

  // "upload" tab: read the selected .txt file as plain text in the browser
  // (no need to send the raw file to the server - just its text content).
  const file = fileInput.files[0];
  if (!file) {
    return Promise.resolve("");
  }
  return file.text();
}

// --- Analyze button ----------------------------------------------------------

analyzeButton.addEventListener("click", async () => {
  const rawText = (await getDocumentText()).trim();

  errorMessage.classList.add("hidden");

  if (!rawText) {
    errorMessage.textContent = "Please paste some text or choose a .txt file first.";
    errorMessage.classList.remove("hidden");
    return;
  }

  analyzeButton.disabled = true;
  analyzeButton.textContent = "Analyzing...";

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: rawText }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Something went wrong analyzing this document.");
    }

    lastAnalysisResult = data;
    renderResults(data);
  } catch (err) {
    errorMessage.textContent = err.message;
    errorMessage.classList.remove("hidden");
    resultsSection.classList.add("hidden");
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.textContent = "Analyze document";
  }
});

// --- Rendering the results ----------------------------------------------------

// Maps the backend's risk level text to a CSS data-attribute value, which
// controls the verdict ring's color (see .verdict-ring[data-level=...] in
// style.css).
const RISK_LEVEL_TO_RING = {
  "Minimal Risk": "minimal",
  "Low Risk": "low",
  "Moderate Risk": "moderate",
  "High Risk": "high",
};

// One flat color per category, used for the small pill tag on each clause
// card. Kept muted/desaturated to sit comfortably on the paper background.
const CATEGORY_COLORS = {
  fees: "#b8862b",
  "auto-renewal": "#6a5ea8",
  cancellation: "#2e6fa7",
  "data sharing": "#2f8f83",
  liability: "#8a4a3d",
  "dispute resolution": "#7a4fa0",
  other: "#6b6a5f",
};

function renderResults(data) {
  resultsSection.classList.remove("hidden");

  document.getElementById("doc-score").textContent = data.document.score;
  document.getElementById("doc-level").textContent = data.document.level;
  document.getElementById("doc-flagged").textContent =
    `${data.document.flagged_clauses} of ${data.document.total_clauses} clauses flagged`;

  const ring = document.getElementById("verdict-ring");
  ring.dataset.level = RISK_LEVEL_TO_RING[data.document.level] || "minimal";

  renderClauseList(data.clauses);
}

function renderClauseList(clauses) {
  const showFlaggedOnly = flaggedOnlyCheckbox.checked;
  const visibleClauses = showFlaggedOnly ? clauses.filter((c) => c.flagged) : clauses;

  clauseList.innerHTML = "";

  visibleClauses.forEach((clause) => {
    clauseList.appendChild(buildClauseCard(clause));
  });
}

// Inline SVG icons (lucide-style: 1.6px stroke, 24x24 viewBox) so the UI
// never relies on emoji, which render inconsistently across platforms and
// read as "chat", not "product UI".
const ICON_ALERT_TRIANGLE = `<svg class="clause-status-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M10.3 3.9 1.9 18a1.6 1.6 0 0 0 1.4 2.4h17.4a1.6 1.6 0 0 0 1.4-2.4L13.7 3.9a1.6 1.6 0 0 0-2.8 0Z"/><path d="M12 9v4"/><path d="M12 16.5h.01"/></svg>`;
const ICON_CHECK_CIRCLE = `<svg class="clause-status-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="m8.5 12.5 2.3 2.3 4.7-5.1"/></svg>`;
const ICON_CHEVRON = `<svg class="clause-card-caret" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m9 6 6 6-6 6"/></svg>`;

function buildClauseCard(clause) {
  const card = document.createElement("div");
  card.className = `clause-card ${clause.flagged ? "flagged" : ""}`;

  const categoryTitle = clause.category.replace(/\b\w/g, (ch) => ch.toUpperCase());
  const categoryColor = CATEGORY_COLORS[clause.category] || CATEGORY_COLORS.other;

  const riskBadge = clause.flagged
    ? `<span class="risk-badge" data-level="${clause.risk_level.toLowerCase()}">${clause.risk_level} risk</span>`
    : "";

  const header = document.createElement("div");
  header.className = "clause-card-header";
  header.innerHTML = `
    ${clause.flagged ? ICON_ALERT_TRIANGLE : ICON_CHECK_CIRCLE}
    <span class="clause-card-title">Clause ${clause.index}</span>
    <span class="category-tag" style="background:${categoryColor}">${escapeHtml(categoryTitle)}</span>
    ${riskBadge}
    <span class="clause-card-spacer"></span>
    ${ICON_CHEVRON}
  `;
  header.addEventListener("click", () => card.classList.toggle("expanded"));

  const body = document.createElement("div");
  body.className = "clause-card-body";

  let bodyHtml = `
    <div class="field">
      <span class="field-label">Original text</span>
      ${escapeHtml(clause.text)}
    </div>
    <div class="field">
      <span class="field-label">Classification confidence</span>
      ${Math.round(clause.confidence * 100)}%
    </div>
  `;

  if (clause.flagged) {
    bodyHtml += `
      <div class="field">
        <span class="field-label">Risk score</span>
        ${clause.risk_score}
      </div>
      <div class="reasons-box">
        <strong>Why flagged:</strong> ${escapeHtml(clause.reasons.join("; "))}
      </div>
      <div class="plain-english-box">
        <strong>Plain English:</strong> ${escapeHtml(clause.plain_english)}
      </div>
    `;
  } else {
    bodyHtml += `<div class="field">No red flags detected for this clause.</div>`;
  }

  body.innerHTML = bodyHtml;

  card.appendChild(header);
  card.appendChild(body);
  return card;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// --- "Show flagged clauses only" checkbox --------------------------------------

flaggedOnlyCheckbox.addEventListener("change", () => {
  if (lastAnalysisResult) {
    renderClauseList(lastAnalysisResult.clauses);
  }
});

// --- CSV download (built client-side from the last analysis result) -----------

downloadCsvButton.addEventListener("click", () => {
  if (!lastAnalysisResult) return;

  const header = ["Clause", "Category", "Flagged", "Risk Score", "Reasons", "Plain English"];
  const rows = lastAnalysisResult.clauses.map((c) => [
    c.text,
    c.category,
    c.flagged,
    c.risk_score,
    c.reasons.join("; "),
    c.plain_english,
  ]);

  const csvContent = [header, ...rows]
    .map((row) => row.map(csvEscape).join(","))
    .join("\n");

  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.href = url;
  link.download = "contractlens_analysis.csv";
  link.click();

  URL.revokeObjectURL(url);
});

function csvEscape(value) {
  const str = String(value ?? "");
  // Wrap in quotes and double up any internal quotes if the value contains
  // a comma, quote, or newline - standard CSV escaping.
  if (/[",\n]/.test(str)) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}
