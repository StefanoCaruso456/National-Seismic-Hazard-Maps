const chatThread = document.getElementById("chatThread");
const hero = document.getElementById("hero");
const heroTitle = hero.querySelector("h3");
const heroCopy = hero.querySelector("p");
const input = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const micBtn = document.getElementById("micBtn");
const attachBtn = document.getElementById("attachBtn");
const fileInput = document.getElementById("fileInput");
const newChatBtn = document.getElementById("newChatBtn");
const recentChatsList = document.getElementById("recentChatsList");
const sidebar = document.getElementById("sidebar");
const toggleSidebar = document.getElementById("toggleSidebar");
const template = document.getElementById("messageTemplate");
const modeButtons = Array.from(document.querySelectorAll(".mode-btn[data-mode]"));
const auditLaunchBtn = document.getElementById("auditLaunchBtn");
const refreshBtn = document.getElementById("refreshBtn");
const debugToggleBtn = document.getElementById("debugToggleBtn");
const settingsBtn = document.getElementById("settingsBtn");
const exportBtn = document.getElementById("exportBtn");
const modePill = document.getElementById("modePill");
const evidencePill = document.getElementById("evidencePill");
const scopeBothBtn = document.getElementById("scopeBothBtn");
const scopeRepoBtn = document.getElementById("scopeRepoBtn");
const scopeUploadsBtn = document.getElementById("scopeUploadsBtn");
const topKSelect = document.getElementById("topKSelect");
const topKRange = document.getElementById("topKRange");
const topKValue = document.getElementById("topKValue");
const scoreInfoBtn = document.getElementById("scoreInfoBtn");
const scoreExplainer = document.getElementById("scoreExplainer");
const statusLine = document.getElementById("statusLine");
const attachmentList = document.getElementById("attachmentList");
const uploadLibrarySection = document.getElementById("uploadLibrarySection");
const uploadLibraryList = document.getElementById("uploadLibraryList");
const refreshUploadsBtn = document.getElementById("refreshUploadsBtn");
const pinnedSourcesSection = document.getElementById("pinnedSourcesSection");
const pinnedSources = document.getElementById("pinnedSources");
const clearPinsBtn = document.getElementById("clearPinsBtn");
const debugPanel = document.getElementById("debugPanel");
const debugContent = document.getElementById("debugContent");
const debugStateLabel = document.getElementById("debugStateLabel");
const overlay = document.getElementById("overlay");
const suggestionOverlay = document.getElementById("suggestionOverlay");
const suggestionBtn = document.getElementById("suggestionBtn");
const promptSuggestionModal = document.getElementById("promptSuggestionModal");
const closeSuggestionModalBtn = document.getElementById("closeSuggestionModalBtn");
const promptModalBody = document.getElementById("promptModalBody");
const exploreCapabilitiesLink = document.getElementById("exploreCapabilitiesLink");
const settingsDrawer = document.getElementById("settingsDrawer");
const closeSettingsBtn = document.getElementById("closeSettingsBtn");
const showSnippetsToggle = document.getElementById("showSnippetsToggle");
const compactCitationsToggle = document.getElementById("compactCitationsToggle");
const debugTraceToggle = document.getElementById("debugTraceToggle");
const persistUploadsToggle = document.getElementById("persistUploadsToggle");
const clearSessionBtn = document.getElementById("clearSessionBtn");

const MODE_CONFIG = {
  chat: {
    label: "Chat",
    placeholder: "Ask anything about National-Seismic-Hazard-Maps...",
    heroTitle: "What do you want to understand in this legacy codebase?",
    heroCopy: "Ask about entry points, file I/O, data flow, business rules, and dependencies.",
  },
  search: {
    label: "Search",
    placeholder: "Search for files, routines, common blocks, or keywords...",
    heroTitle: "Inspect retrieval hits before asking for synthesis",
    heroCopy:
      "Search mode returns raw ranked chunks so you can verify relevance before generating an answer.",
  },
  patterns: {
    label: "Code Patterns",
    placeholder: "Find recurring coding patterns, common blocks, and routine styles...",
    heroTitle: "Mine recurring implementation patterns",
    heroCopy:
      "Pattern mode extracts repeated structures from retrieved chunks, including common blocks and routine signatures.",
  },
  dependencies: {
    label: "Dependencies",
    placeholder: "Trace call/use/common dependencies across retrieved code...",
    heroTitle: "Trace dependency signals quickly",
    heroCopy:
      "Dependency mode surfaces call sites, module usage, and shared common blocks from the top retrieved chunks.",
  },
  audit: {
    label: "Audit",
    placeholder: "Select an audit type or describe the area you want audited...",
    heroTitle: "Run Structured Engineering Audits",
    heroCopy:
      "Audit mode generates scannable reports with findings, evidence, recommended fixes, and prioritized next actions.",
  },
};

const AUDIT_REPORT_CONTRACT = `Return the report in this exact section order:
Overview
Key Findings
Evidence (files/lines)
Recommendations
Next Actions

For every finding include:
- Priority: High | Medium | Low
- Why it matters
- File/function evidence with file paths and line ranges when available
- Recommended fix (incremental, no rewrites)

If the repository is large, start with a fast scan and add a "Deeper Pass Targets" subsection in Next Actions.`;

const AUDIT_WORKFLOWS = {
  architecture: {
    title: "Architecture Audit (System Map + Critical Paths)",
    reportType: "Architecture Audit",
    followUps: [
      "Deeper pass: trace dependency chains from run_all_hazard.sh",
      "Deeper pass: analyze top 3 high-risk coupling areas",
      "Deeper pass: map runtime flow for the ingestion pipeline",
    ],
    prompt: `You are a senior staff engineer performing an Architecture Audit on this codebase.

Objectives:
- Identify true entry points (CLI, scripts, services, jobs, UI bootstraps, schedulers).
- Build a component map of major domains/modules and responsibilities.
- Trace critical execution paths for top 3-5 core flows.
- Surface hidden coupling, circular dependencies, tight integrations, and high-risk areas.
- Highlight global state, side effects, concurrency hazards, IO-heavy sections, and brittle integrations.

Method:
- Start with a repository scan, then trace from entry points inward.
- Use evidence-based reporting with concrete files/functions.
- Prefer concrete call chains over vague descriptions.
- No rewrites; prioritize incremental, safe refactors.

Required sections:
- Executive Summary (10 lines max)
- System Map (component purpose, inputs/outputs, key files)
- Entry Points (how invoked, what triggered, key call chain)
- Critical Paths (top 3-5, step-by-step with file/function refs)
- Dependency and Coupling Risks
- Architecture Recommendations (5-10 actions by ROI)
- Immediate Next Steps ("If I had 2 days, I'd do X")

${AUDIT_REPORT_CONTRACT}`,
  },
  stack: {
    title: "Stack & Dependency Audit (Tech + Runtime + Build)",
    reportType: "Stack & Dependency Audit",
    followUps: [
      "Deeper pass: dependency risk register with patch plan",
      "Deeper pass: build and environment reproducibility audit",
      "Deeper pass: runtime topology with external integration risks",
    ],
    prompt: `You are a senior platform engineer performing a Stack and Dependency Audit.

Objectives:
- Identify full stack: languages, frameworks, libraries, and major subsystems.
- Document build/run pipeline with install, config, test, build, and start steps.
- Map runtime architecture: services, workers, scripts, jobs, databases, caches, external APIs.
- Audit dependency risks (outdated versions, vulnerable/unmaintained libs, overlap).
- Identify config and environment risks: secrets, env var sprawl, non-reproducible builds.

Method:
- Enumerate manifests, lockfiles, build scripts, CI/CD descriptors, and runtime configs.
- Distinguish verified facts vs inferred assumptions.
- Prefer minimal-change recommendations that reduce risk quickly.

Required sections:
- Stack Snapshot (table: category, technology, where found)
- Build and Run Golden Path (commands, env vars, common failures/fixes)
- Runtime Topology (components and communication map)
- Dependency Risk Register (top 10 risks with impact and effort)
- Security and Secrets Review
- Recommended Standardization (5-10 improvements)
- Next Actions (prioritized checklist)

${AUDIT_REPORT_CONTRACT}`,
  },
  maintainability: {
    title: "Maintainability & Risk Audit (Hotspots + Debt + Test Health)",
    reportType: "Maintainability & Risk Audit",
    followUps: [
      "Deeper pass: top 10 hotspot modules with refactor sequence",
      "Deeper pass: critical-path test gap analysis",
      "Deeper pass: reliability and observability risk drill-down",
    ],
    prompt: `You are a senior engineering lead running a Maintainability and Risk Audit.

Objectives:
- Identify change-risk hotspots: complexity, coupling, high-churn zones, TODO/FIXME heavy areas.
- Find maintainability smells: duplication, god modules, unclear boundaries, hidden side effects.
- Assess test health: missing tests on critical paths, flaky patterns, integration gaps.
- Assess operational risk: error handling gaps, weak observability, retry/idempotency issues.
- Deliver a phased refactor and test plan with quick wins and medium-term actions.

Method:
- Start with directory-level triage, then deep-dive top 10 central modules.
- Use objective signals (size, complexity approximation, fan-in/fan-out, dependency depth).
- Avoid rewrites; optimize for safe incremental progress.

Required sections:
- Executive Summary (10 lines max)
- Hotspot Leaderboard (ranked with risk reason and ROI)
- Technical Debt Themes (3-7 recurring issues with examples)
- Test and Quality Assessment
- Reliability and Observability Gaps
- Refactor Plan (Phase 0-3)
- Definition of Done with measurable outcomes

${AUDIT_REPORT_CONTRACT}`,
  },
};

const AUDIT_PROMPT_TO_TYPE = Object.fromEntries(
  Object.entries(AUDIT_WORKFLOWS).map(([type, workflow]) => [workflow.title, type]),
);

const PROMPT_MODE_CATEGORIES = [
  {
    key: "system-overview",
    title: "SYSTEM OVERVIEW",
    mode: "chat",
    prompts: [
      "Explain the hazard curve workflow",
      "What is the main entry point for this codebase?",
      "Describe the data flow from inputs to outputs",
    ],
  },
  {
    key: "find-code",
    title: "FIND CODE",
    mode: "search",
    prompts: [
      "Where is hazard curve calculation implemented?",
      "Find files referencing ground motion models",
      "Show code that reads configuration files",
    ],
  },
  {
    key: "code-analysis",
    title: "CODE ANALYSIS",
    mode: "patterns",
    prompts: [
      "Show loops iterating over seismic sources",
      "Identify patterns used for reading input tables",
      "Find functions similar to hazgridX",
    ],
  },
  {
    key: "dependency-graph",
    title: "DEPENDENCY GRAPH",
    mode: "dependencies",
    prompts: [
      "Trace dependencies from run_all_hazard.sh",
      "Which modules depend on hazard curve routines?",
      "Show the call chain for hazgridX",
    ],
  },
  {
    key: "audit-workflows",
    title: "AUDIT",
    mode: "audit",
    prompts: [
      AUDIT_WORKFLOWS.architecture.title,
      AUDIT_WORKFLOWS.stack.title,
      AUDIT_WORKFLOWS.maintainability.title,
    ],
  },
];

const DEFAULT_UPLOAD_LIMITS = {
  maxFiles: 8,
  maxFileBytes: 1_500_000,
  maxTotalBytes: 6_000_000,
};

const state = {
  mode: "chat",
  scope: "both",
  topK: 5,
  showSnippets: true,
  compactCitations: false,
  listening: false,
  recognition: null,
  attachments: [],
  lastRequest: null,
  history: [],
  loading: false,
  retrievalInfo: null,
  repoUrl: "https://github.com/StefanoCaruso456/National-Seismic-Hazard-Maps",
  projectId: "nshmp-main",
  persistUploads: false,
  uploadLibrary: [],
  pinnedSources: [],
  debugPanelOpen: false,
  debugTraceEnabled: false,
  lastDebugPayload: null,
  recentChats: [],
  currentSessionTitle: null,
  contextFile: "",
  suggestionModalOpen: false,
  activeAuditType: "architecture",
};

function autoResize() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
}

function currentTopKMax() {
  const max = Number(topKRange.max);
  if (!Number.isFinite(max) || max < 1) return 20;
  return Math.round(max);
}

function clampTopK(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return 5;
  return Math.min(currentTopKMax(), Math.max(1, Math.round(num)));
}

function uploadLimits() {
  return {
    maxFiles: Number(state.retrievalInfo?.upload_max_files) || DEFAULT_UPLOAD_LIMITS.maxFiles,
    maxFileBytes: Number(state.retrievalInfo?.upload_max_file_bytes) || DEFAULT_UPLOAD_LIMITS.maxFileBytes,
    maxTotalBytes: Number(state.retrievalInfo?.upload_max_total_bytes) || DEFAULT_UPLOAD_LIMITS.maxTotalBytes,
  };
}

function normalizedEvidenceLabel(value) {
  const raw = String(value || "").trim().toLowerCase();
  if (raw === "high") return "high";
  if (raw === "medium") return "medium";
  if (raw === "low") return "low";
  return "unknown";
}

function updateEvidencePill(evidence = null) {
  const label = normalizedEvidenceLabel(evidence?.label);
  const score = typeof evidence?.score === "number" ? ` (${Math.round(evidence.score * 100)}%)` : "";
  evidencePill.className = `evidence-pill ${label}`;
  const title = evidence?.reason || "Evidence strength unavailable";
  evidencePill.title = title;
  evidencePill.textContent =
    label === "unknown" ? "Evidence: n/a" : `Evidence: ${label[0].toUpperCase()}${label.slice(1)}${score}`;
}

function formatBytes(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value <= 0) return "0 B";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(2)} MB`;
}

function setScope(scope) {
  const normalized = ["repo", "uploads", "both"].includes(scope) ? scope : "both";
  state.scope = normalized;
  scopeBothBtn.classList.toggle("active", normalized === "both");
  scopeRepoBtn.classList.toggle("active", normalized === "repo");
  scopeUploadsBtn.classList.toggle("active", normalized === "uploads");
  setStatus(`Scope: ${normalized}`);
}

function encodeRepoPath(path) {
  return String(path || "")
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/");
}

function sourceLink(item) {
  const file = item?.file_path || "";
  if (!file || file.startsWith("uploaded/")) return "";
  const start = Number(item.line_start) || 1;
  const end = Number(item.line_end) || start;
  const repoUrl = state.repoUrl || "https://github.com/StefanoCaruso456/National-Seismic-Hazard-Maps";
  return `${repoUrl}/blob/main/${encodeRepoPath(file)}#L${start}-L${end}`;
}

function sourceKey(item) {
  return `${item.file_path || "unknown"}:${item.line_start || "?"}-${item.line_end || "?"}`;
}

function renderPinnedSources() {
  pinnedSources.innerHTML = "";
  if (!state.pinnedSources.length) {
    pinnedSourcesSection.classList.add("hidden");
    return;
  }

  pinnedSourcesSection.classList.remove("hidden");
  for (const item of state.pinnedSources) {
    const node = document.createElement("span");
    node.className = "pinned-item";
    const link = sourceLink(item);
    if (link) {
      const anchor = document.createElement("a");
      anchor.href = link;
      anchor.target = "_blank";
      anchor.rel = "noopener noreferrer";
      anchor.textContent = sourceKey(item);
      node.appendChild(anchor);
    } else {
      node.textContent = sourceKey(item);
    }
    pinnedSources.appendChild(node);
  }
}

function pinSource(item) {
  const key = sourceKey(item);
  const exists = state.pinnedSources.some((pinned) => sourceKey(pinned) === key);
  if (exists) {
    setStatus("Source already pinned");
    return;
  }
  state.pinnedSources.push(item);
  if (item?.file_path && !String(item.file_path).startsWith("uploaded/")) {
    state.contextFile = String(item.file_path);
  }
  renderPinnedSources();
  setStatus("Source pinned");
}

function renderUploadLibrary() {
  uploadLibraryList.innerHTML = "";
  if (!Array.isArray(state.uploadLibrary) || !state.uploadLibrary.length) {
    const empty = document.createElement("div");
    empty.className = "upload-meta";
    empty.textContent = "No persisted uploads yet.";
    uploadLibraryList.appendChild(empty);
    return;
  }

  for (const entry of state.uploadLibrary) {
    const row = document.createElement("article");
    row.className = "upload-row";

    const main = document.createElement("div");
    main.className = "upload-main";

    const name = document.createElement("span");
    name.className = "upload-name";
    name.textContent = entry.file_name || entry.file_sha;

    const meta = document.createElement("span");
    meta.className = "upload-meta";
    meta.textContent = `${entry.file_sha} • ${formatBytes(entry.file_size)} • ${entry.chunk_count} chunks`;

    main.append(name, meta);

    const actions = document.createElement("div");
    actions.className = "upload-actions";

    const pinBtn = document.createElement("button");
    pinBtn.type = "button";
    pinBtn.className = "tiny-btn";
    pinBtn.textContent = entry.pinned ? "Unpin" : "Pin";
    pinBtn.dataset.uploadAction = "pin";
    pinBtn.dataset.fileSha = entry.file_sha;
    pinBtn.dataset.currentPinned = entry.pinned ? "1" : "0";

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "tiny-btn";
    deleteBtn.textContent = "Delete";
    deleteBtn.dataset.uploadAction = "delete";
    deleteBtn.dataset.fileSha = entry.file_sha;

    actions.append(pinBtn, deleteBtn);
    row.append(main, actions);
    uploadLibraryList.appendChild(row);
  }
}

async function refreshUploadLibrary() {
  try {
    const response = await fetch(`/api/uploads?project_id=${encodeURIComponent(state.projectId)}`);
    if (!response.ok) {
      setStatus("Failed to refresh upload library", "warn");
      return;
    }
    const data = await response.json();
    state.uploadLibrary = Array.isArray(data.files) ? data.files : [];
    renderUploadLibrary();
  } catch (_err) {
    setStatus("Failed to refresh upload library", "warn");
  }
}

async function persistCurrentAttachments() {
  if (!state.attachments.length || !state.persistUploads) return;

  const formData = new FormData();
  formData.append("project_id", state.projectId);
  formData.append("persist_uploads", "true");
  for (const attachment of state.attachments) {
    formData.append("files", attachment.file, attachment.name);
  }

  const response = await fetch("/api/uploads/ingest", {
    method: "POST",
    body: formData,
  });
  let data = null;
  try {
    data = await response.json();
  } catch (_err) {
    data = null;
  }
  if (!response.ok) {
    const detail = data && data.detail ? data.detail : `Upload ingestion failed (${response.status})`;
    throw new Error(detail);
  }

  const statuses = Array.isArray(data.files) ? data.files : [];
  const persisted = statuses.filter((row) => row.status === "persisted").length;
  const skipped = statuses.filter((row) => row.status !== "persisted").length;
  setStatus(`Uploads ingested: ${persisted} persisted, ${skipped} skipped`);
  await refreshUploadLibrary();
}

function setDebugPanelState(open) {
  state.debugPanelOpen = Boolean(open);
  debugPanel.classList.toggle("hidden", !state.debugPanelOpen);
  debugToggleBtn.classList.toggle("active", state.debugPanelOpen);
  debugStateLabel.textContent = state.debugPanelOpen ? "on" : "off";
}

function renderDebugPayload(payload) {
  state.lastDebugPayload = payload || null;
  if (!payload) {
    debugContent.textContent = "Debug trace not returned for this response.";
    return;
  }
  debugContent.textContent = JSON.stringify(payload, null, 2);
}

function formatDuration(ms) {
  if (!Number.isFinite(ms)) return "-";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function normalizeScore(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return Math.min(1, Math.max(0, value));
}

function scorePercent(score) {
  if (score === null) return "n/a";
  return `${Math.round(score * 100)}%`;
}

function confidenceLabel(score) {
  if (score === null) return "Unknown";
  if (score >= 0.8) return "High";
  if (score >= 0.6) return "Medium";
  return "Low";
}

function setBusy(isBusy) {
  state.loading = Boolean(isBusy);
  sendBtn.disabled = state.loading;
  refreshBtn.disabled = state.loading;
  suggestionBtn.disabled = state.loading;
  if (auditLaunchBtn) {
    auditLaunchBtn.disabled = state.loading;
  }
}

function setStatus(text, tone = "") {
  statusLine.textContent = text;
  statusLine.dataset.tone = tone;
}

function summarizePrompt(text, limit = 68) {
  const firstLine = String(text || "")
    .split("\n")[0]
    .replace(/\s+/g, " ")
    .trim();
  if (!firstLine) return "";
  if (firstLine.length <= limit) return firstLine;
  return `${firstLine.slice(0, Math.max(0, limit - 1)).trim()}...`;
}

function renderRecentChats() {
  recentChatsList.innerHTML = "";
  if (!state.recentChats.length) {
    const empty = document.createElement("span");
    empty.className = "recent-chat-empty";
    empty.textContent = "No recent chats yet.";
    recentChatsList.appendChild(empty);
    return;
  }

  for (const item of state.recentChats) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "recent-chat-item";
    button.textContent = item.title;
    button.dataset.mode = item.mode;
    button.dataset.prompt = item.title;
    button.addEventListener("click", () => {
      setMode(item.mode);
      input.value = item.title;
      autoResize();
      setStatus("Prompt loaded from recent chat");
    });
    recentChatsList.appendChild(button);
  }
}

function archiveCurrentSession() {
  if (!state.history.length) return;
  const firstUser = state.history.find((entry) => entry.role === "user");
  const sourceTitle = state.currentSessionTitle || firstUser?.text || "";
  const title = summarizePrompt(sourceTitle);
  if (!title) return;

  const next = {
    title,
    mode: state.mode,
    at: new Date().toISOString(),
  };
  state.recentChats = [next, ...state.recentChats.filter((entry) => entry.title !== title)].slice(0, 8);
  renderRecentChats();
}

function deriveContextFile() {
  const explicit = String(state.contextFile || "").trim();
  if (explicit) return explicit;

  const fromPinned = state.pinnedSources.find((item) => item?.file_path && !String(item.file_path).startsWith("uploaded/"));
  if (fromPinned?.file_path) {
    return String(fromPinned.file_path);
  }

  const fromAttachment = state.attachments[0]?.name;
  if (fromAttachment) {
    return String(fromAttachment);
  }

  return "";
}

function contextSuggestions(contextFile) {
  if (!contextFile) return [];
  const base = contextFile.split("/").pop() || contextFile;
  const stem = base.replace(/\.[^.]+$/, "") || base;
  return [
    { text: `Explain what ${base} does`, mode: "chat" },
    { text: `Show functions called by ${stem}`, mode: "dependencies" },
    { text: "Find similar hazard curve implementations", mode: "patterns" },
  ];
}

function inferAuditType(text) {
  const normalized = String(text || "").trim().toLowerCase();
  if (!normalized) return state.activeAuditType || "architecture";
  if (normalized.includes("stack") || normalized.includes("dependency")) return "stack";
  if (
    normalized.includes("maintainability")
    || normalized.includes("risk")
    || normalized.includes("hotspot")
    || normalized.includes("debt")
    || normalized.includes("test")
  ) {
    return "maintainability";
  }
  return "architecture";
}

function auditFollowUpsForType(auditType) {
  const workflow = AUDIT_WORKFLOWS[auditType];
  if (!workflow) return [];
  return workflow.followUps.slice(0, 3);
}

function buildAuditWorkflowPrompt(auditType, userIntent = "") {
  const resolvedType = AUDIT_WORKFLOWS[auditType] ? auditType : inferAuditType(userIntent);
  const workflow = AUDIT_WORKFLOWS[resolvedType];
  const intent = String(userIntent || "").trim();
  const contextLine = intent
    ? `Audit focus request from user: ${intent}\nConstrain recommendations to this focus where possible.`
    : "Audit focus request from user: full repository baseline scan.";

  return `${workflow.prompt}

Execution requirements:
- Start with a fast scan baseline.
- If repository size/complexity is high, identify specific deeper-pass targets.
- Mark uncertainty explicitly as Hypothesis + verification steps.
- Keep recommendations incremental and safe.

Report format contract:
Overview -> Key Findings -> Evidence (files/lines) -> Recommendations -> Next Actions

${contextLine}`;
}

function buildAuditDispatch(promptText) {
  const mappedType = AUDIT_PROMPT_TO_TYPE[promptText];
  const auditType = mappedType || inferAuditType(promptText);
  return {
    auditType,
    displayText: AUDIT_WORKFLOWS[auditType].title,
    workflowPrompt: buildAuditWorkflowPrompt(auditType, promptText),
  };
}

function apiModeForMode(mode) {
  return mode === "audit" ? "chat" : mode;
}

function PromptChip(promptText, mode, onSelect) {
  const chip = document.createElement("button");
  chip.type = "button";
  chip.className = "prompt-chip";
  chip.textContent = promptText;
  chip.addEventListener("click", () => {
    onSelect(promptText, mode);
  });
  return chip;
}

function PromptCategory(category, onSelect) {
  const section = document.createElement("section");
  section.className = "prompt-category";
  if (category.key) {
    section.dataset.categoryKey = category.key;
  }

  const title = document.createElement("h4");
  const uniqueModes = [...new Set(category.prompts.map((item) => item.mode))];
  if (uniqueModes.length === 1 && MODE_CONFIG[uniqueModes[0]]) {
    title.textContent = `${category.title} (${MODE_CONFIG[uniqueModes[0]].label})`;
  } else {
    title.textContent = category.title;
  }
  section.appendChild(title);

  const list = document.createElement("div");
  list.className = "prompt-chip-list";
  category.prompts.slice(0, 3).forEach((item) => {
    list.appendChild(PromptChip(item.text, item.mode, onSelect));
  });
  section.appendChild(list);
  return section;
}

function PromptSuggestionModal(modeCategories, onSelect, contextFile, options = {}) {
  promptModalBody.innerHTML = "";
  const filterKeys = Array.isArray(options.filterKeys) ? new Set(options.filterKeys) : null;

  const normalizedCategories = modeCategories
    .filter((category) => !filterKeys || filterKeys.has(category.key))
    .map((category) => ({
      key: category.key,
      title: category.title,
      prompts: category.prompts.slice(0, 3).map((text) => ({
        text,
        mode: category.mode,
      })),
    }));

  normalizedCategories.forEach((category) => {
    promptModalBody.appendChild(PromptCategory(category, onSelect));
  });

  const contextPromptItems = contextSuggestions(contextFile);
  if (contextPromptItems.length && !filterKeys) {
    promptModalBody.appendChild(
      PromptCategory({
        key: "context-suggestions",
        title: "CONTEXT SUGGESTIONS",
        prompts: contextPromptItems,
      }, onSelect),
    );
  }
}

function openSuggestionModal(options = {}) {
  if (state.loading) {
    setStatus("Wait for the current request to finish", "warn");
    return;
  }
  closeSettings();
  const contextFile = deriveContextFile();
  PromptSuggestionModal(PROMPT_MODE_CATEGORIES, onPromptSelected, contextFile, options);
  state.suggestionModalOpen = true;
  suggestionOverlay.classList.remove("hidden");
  promptSuggestionModal.classList.remove("hidden");

  const focusKey = String(options.focusKey || "").trim();
  if (focusKey) {
    requestAnimationFrame(() => {
      const target = promptModalBody.querySelector(`[data-category-key="${focusKey}"]`);
      if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }
}

function closeSuggestionModal() {
  state.suggestionModalOpen = false;
  suggestionOverlay.classList.add("hidden");
  promptSuggestionModal.classList.add("hidden");
}

async function onPromptSelected(promptText, mode) {
  closeSuggestionModal();
  setMode(mode);
  let dispatchPrompt = promptText;
  let displayPrompt = promptText;
  let auditType = null;

  if (mode === "audit") {
    const dispatch = buildAuditDispatch(promptText);
    auditType = dispatch.auditType;
    state.activeAuditType = dispatch.auditType;
    dispatchPrompt = dispatch.workflowPrompt;
    displayPrompt = dispatch.displayText;
  }

  input.value = displayPrompt;
  autoResize();
  updateEvidencePill(null);
  setStatus(`Mode: ${MODE_CONFIG[mode].label} • Evidence: n/a • TopK: ${state.topK}`);
  await submitQuestion({
    modeOverride: mode,
    questionOverride: dispatchPrompt,
    displayOverride: displayPrompt,
    skipAuditWrap: mode === "audit",
    auditType,
  });
}

function bumpCounter(counter, key) {
  if (!key) return;
  const normalized = key.toLowerCase().trim();
  if (!normalized) return;
  counter.set(normalized, (counter.get(normalized) || 0) + 1);
}

function collectMatches(matches, regex, counter) {
  for (const item of matches) {
    const source = item.snippet || "";
    regex.lastIndex = 0;
    let hit;
    while ((hit = regex.exec(source)) !== null) {
      bumpCounter(counter, hit[1]);
    }
  }
}

function topEntries(counter, limit = 5) {
  return [...counter.entries()].sort((a, b) => b[1] - a[1]).slice(0, limit);
}

function formatEntryList(counter, limit = 5) {
  const entries = topEntries(counter, limit);
  if (!entries.length) return "none detected";
  return entries.map(([name, count]) => `${name} (${count})`).join(", ");
}

function syncTopKInputs(nextValue) {
  const safe = clampTopK(nextValue);
  state.topK = safe;
  topKSelect.value = String(safe);
  topKRange.value = String(safe);
  topKValue.value = String(safe);
}

function populateTopKSelect(maxValue) {
  const safeMax = Math.max(1, Number(maxValue) || 20);
  topKSelect.innerHTML = "";

  for (let value = 1; value <= safeMax; value += 1) {
    const option = document.createElement("option");
    option.value = String(value);
    option.textContent = String(value);
    if (value === state.topK) {
      option.selected = true;
    }
    topKSelect.appendChild(option);
  }
}

function applyDisplayPreferences() {
  document.body.classList.toggle("compact-mode", state.compactCitations);
  document.querySelectorAll(".citation-snippet").forEach((node) => {
    node.hidden = !state.showSnippets;
  });
  document.querySelectorAll(".toggle-snippet-btn").forEach((btn) => {
    btn.textContent = state.showSnippets ? "Hide snippet" : "Show snippet";
  });
}

function updateExplainerText() {
  if (!state.retrievalInfo) {
    scoreExplainer.querySelector("p").textContent =
      "Relevance is a hybrid score from semantic similarity + lexical overlap, normalized to 0-100. Top K controls how many chunks are retrieved per request.";
    return;
  }

  const info = state.retrievalInfo;
  const lexicalPct = Math.round(info.lexical_weight * 100);
  const semanticPct = 100 - lexicalPct;
  const minPct = Math.round(info.min_hybrid_score * 100);
  const maxFiles = Number(info.upload_max_files) || DEFAULT_UPLOAD_LIMITS.maxFiles;
  const maxFileKb = Math.round(
    (Number(info.upload_max_file_bytes) || DEFAULT_UPLOAD_LIMITS.maxFileBytes) / 1024,
  );
  scoreExplainer.querySelector("p").textContent =
    `Hybrid relevance uses ${semanticPct}% semantic similarity + ${lexicalPct}% lexical overlap. Results below ~${minPct}% are filtered unless nothing stronger exists. Candidate pool expands by x${info.candidate_multiplier} before reranking. Attachments are chunked in temporary scope (up to ${maxFiles} files, ~${maxFileKb}KB each).`;
}

function setMode(mode) {
  if (!MODE_CONFIG[mode]) return;
  state.mode = mode;

  for (const item of modeButtons) {
    item.classList.toggle("active", item.dataset.mode === mode);
  }

  const config = MODE_CONFIG[mode];
  modePill.textContent = `Mode: ${config.label}`;
  input.placeholder = config.placeholder;
  heroTitle.textContent = config.heroTitle;
  heroCopy.textContent = config.heroCopy;

  setStatus(`${config.label} mode ready`);

  if (window.matchMedia("(max-width: 980px)").matches) {
    sidebar.classList.remove("open");
  }
}

function resetSession() {
  archiveCurrentSession();
  chatThread.innerHTML = "";
  hero.style.display = "block";
  input.value = "";
  autoResize();
  state.attachments = [];
  state.pinnedSources = [];
  state.history = [];
  state.lastRequest = null;
  state.lastDebugPayload = null;
  state.currentSessionTitle = null;
  state.contextFile = "";
  state.activeAuditType = "architecture";
  closeSuggestionModal();
  renderAttachmentList();
  renderPinnedSources();
  renderDebugPayload(null);
  updateEvidencePill(null);
  setStatus("Session cleared");
}

function roleLabel(role) {
  return role === "assistant" ? "Assistant" : "You";
}

function buildMetaText(role, meta) {
  const parts = [roleLabel(role)];
  if (meta.modeLabel) parts.push(meta.modeLabel);
  if (meta.resultType) parts.push(meta.resultType);
  if (typeof meta.resultCount === "number") parts.push(`${meta.resultCount} sources`);
  if (meta.evidenceLabel) parts.push(`evidence ${meta.evidenceLabel}`);
  if (typeof meta.elapsedMs === "number") parts.push(formatDuration(meta.elapsedMs));
  parts.push(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
  return parts.join(" • ");
}

function defaultResultTypeForMode(mode) {
  if (mode === "audit") return "Audit Report";
  if (mode === "dependencies") return "Dependency Graph";
  if (mode === "patterns") return "Pattern Examples";
  if (mode === "search") return "Ranked Chunks";
  return "Answer";
}

function normalizeFollowUps(items) {
  if (!Array.isArray(items)) return [];
  const seen = new Set();
  const out = [];
  for (const raw of items) {
    const next = String(raw || "").trim();
    if (!next) continue;
    const key = next.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(next);
    if (out.length >= 6) break;
  }
  return out;
}

function citationRange(item) {
  const file = item.file_path || "unknown";
  const start = item.line_start ?? "?";
  const end = item.line_end ?? "?";
  return `${file}:${start}-${end}`;
}

async function copyText(text, successMessage = "Copied to clipboard") {
  if (!text) return;
  try {
    if (!navigator.clipboard || !navigator.clipboard.writeText) {
      throw new Error("Clipboard unavailable");
    }
    await navigator.clipboard.writeText(text);
    setStatus(successMessage);
  } catch (_err) {
    setStatus("Clipboard not available in this browser", "warn");
  }
}

function renderCitation(citationsWrap, item, index) {
  const score = normalizeScore(item.score);
  const rangeText = citationRange(item);
  const sourceType = String(item.source_type || "repo");
  const sourceLabel = sourceType === "repo" ? "Repo" : "Upload";

  if (state.compactCitations) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "citation-chip";
    chip.textContent = `[${sourceLabel}] #${index} ${rangeText} • ${scorePercent(score)}`;
    chip.title = "Copy citation";
    chip.addEventListener("click", () => {
      copyText(rangeText, "Citation copied");
    });
    citationsWrap.appendChild(chip);
    return;
  }

  const card = document.createElement("article");
  card.className = "citation-card";

  const header = document.createElement("div");
  header.className = "citation-head";

  const lead = document.createElement("div");
  lead.className = "citation-lead";

  const rank = document.createElement("span");
  rank.className = "citation-rank";
  rank.textContent = `#${index}`;

  const path = document.createElement("span");
  path.className = "citation-path";
  path.textContent = rangeText;

  const scorePill = document.createElement("span");
  scorePill.className = "citation-score";
  scorePill.textContent = `${scorePercent(score)} ${confidenceLabel(score)}`;

  const sourceBadge = document.createElement("span");
  sourceBadge.className = `source-badge ${sourceType}`;
  sourceBadge.textContent = sourceLabel;

  lead.append(rank, sourceBadge, path, scorePill);
  header.appendChild(lead);

  const actions = document.createElement("div");
  actions.className = "citation-actions";

  const copyBtn = document.createElement("button");
  copyBtn.type = "button";
  copyBtn.className = "tiny-btn";
  copyBtn.textContent = "Copy ref";
  copyBtn.addEventListener("click", () => {
    copyText(rangeText, "Citation copied");
  });
  actions.appendChild(copyBtn);

  const openUrl = sourceLink(item);
  const openBtn = document.createElement("button");
  openBtn.type = "button";
  openBtn.className = "tiny-btn";
  openBtn.textContent = "Open file";
  if (openUrl) {
    openBtn.addEventListener("click", () => {
      if (item?.file_path && !String(item.file_path).startsWith("uploaded/")) {
        state.contextFile = String(item.file_path);
      }
      window.open(openUrl, "_blank", "noopener,noreferrer");
    });
  } else {
    openBtn.disabled = true;
    openBtn.title = "Open file unavailable for uploaded temporary sources";
  }
  actions.appendChild(openBtn);

  const pinBtn = document.createElement("button");
  pinBtn.type = "button";
  pinBtn.className = "tiny-btn";
  pinBtn.textContent = "Pin source";
  pinBtn.addEventListener("click", () => {
    pinSource(item);
  });
  actions.appendChild(pinBtn);

  const scoreTrack = document.createElement("div");
  scoreTrack.className = "score-track";
  const scoreFill = document.createElement("div");
  scoreFill.className = "score-fill";
  scoreFill.style.width = score === null ? "0%" : `${Math.round(score * 100)}%`;
  scoreTrack.appendChild(scoreFill);

  card.append(header, actions, scoreTrack);

  const snippetText = (item.snippet || "").trim();
  if (snippetText) {
    const snippet = document.createElement("pre");
    snippet.className = "citation-snippet";
    snippet.hidden = !state.showSnippets;
    snippet.textContent = snippetText;

    const toggleBtn = document.createElement("button");
    toggleBtn.type = "button";
    toggleBtn.className = "tiny-btn toggle-snippet-btn";
    toggleBtn.textContent = state.showSnippets ? "Hide snippet" : "Show snippet";
    toggleBtn.addEventListener("click", () => {
      snippet.hidden = !snippet.hidden;
      toggleBtn.textContent = snippet.hidden ? "Show snippet" : "Hide snippet";
    });

    actions.appendChild(toggleBtn);
    card.appendChild(snippet);
  }

  citationsWrap.appendChild(card);
}

function addMessage(role, text, citations = [], meta = {}) {
  const node = template.content.cloneNode(true);
  const article = node.querySelector(".message");
  const bubble = node.querySelector(".bubble");
  const bubbleWrap = node.querySelector(".bubble-wrap");
  const citationsWrap = node.querySelector(".citations");
  const metaRow = node.querySelector(".message-meta");

  article.classList.add(role);
  bubble.textContent = text;
  metaRow.textContent = buildMetaText(role, meta);

  if (role === "assistant") {
    if (meta.resultType) {
      const resultType = document.createElement("div");
      resultType.className = "result-type-head";
      resultType.textContent = meta.resultType;
      bubbleWrap.insertBefore(resultType, bubble);
    }

    const copyAnswerBtn = document.createElement("button");
    copyAnswerBtn.type = "button";
    copyAnswerBtn.className = "tiny-btn";
    copyAnswerBtn.textContent = "Copy answer";
    copyAnswerBtn.addEventListener("click", () => {
      copyText(text, "Answer copied");
    });
    metaRow.appendChild(copyAnswerBtn);

    if (meta.debugPayload) {
      const debugBtn = document.createElement("button");
      debugBtn.type = "button";
      debugBtn.className = "tiny-btn";
      debugBtn.textContent = "Show retrieval debug";
      debugBtn.addEventListener("click", () => {
        setDebugPanelState(true);
        renderDebugPayload(meta.debugPayload);
        setStatus("Debug trace opened");
      });
      metaRow.appendChild(debugBtn);
    }
  }

  if (citations.length) {
    citations.forEach((item, index) => renderCitation(citationsWrap, item, index + 1));
  } else {
    citationsWrap.remove();
  }

  if (role === "assistant" && Array.isArray(meta.followUps) && meta.followUps.length) {
    const followRow = document.createElement("div");
    followRow.className = "followup-row";

    const label = document.createElement("span");
    label.className = "followup-label";
    label.textContent = "Next:";
    followRow.appendChild(label);

    meta.followUps.forEach((suggestion) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "followup-chip";
      chip.textContent = suggestion;
      chip.addEventListener("click", () => {
        submitQuestion({
          modeOverride: meta.modeValue || state.mode,
          questionOverride: suggestion,
        });
      });
      followRow.appendChild(chip);
    });

    bubbleWrap.appendChild(followRow);
  }

  chatThread.appendChild(node);
  chatThread.scrollTop = chatThread.scrollHeight;

  if (role === "user" && !state.currentSessionTitle) {
    state.currentSessionTitle = summarizePrompt(text);
  }

  state.history.push({
    role,
    text,
    citations,
    meta,
    timestamp: new Date().toISOString(),
  });
}

function buildSearchSummary(question, matches) {
  if (!matches.length) {
    return `No retrieval matches found for "${question}". Try specific routine names, file names, or common block identifiers.`;
  }

  const top = matches.slice(0, 3).map((item, idx) => {
    const score = scorePercent(normalizeScore(item.score));
    return `${idx + 1}. ${citationRange(item)} • ${score}`;
  });

  return `Retrieved ${matches.length} ranked chunk(s) for "${question}".\nTop hits:\n${top.join("\n")}`;
}

function generatePatternInsights(question, matches) {
  if (!matches.length) {
    return `No pattern insights could be generated for "${question}" because no chunks were retrieved.`;
  }

  const commonBlocks = new Map();
  const subroutines = new Map();
  const functions = new Map();
  const modules = new Map();
  const files = new Map();
  let dimensionMentions = 0;

  for (const item of matches) {
    bumpCounter(files, item.file_path || "unknown");
    const snippet = item.snippet || "";

    const dimHits = snippet.match(/\bdimension\s*\(/gi);
    if (dimHits) dimensionMentions += dimHits.length;
  }

  collectMatches(matches, /common\s*\/\s*([a-z0-9_]+)/gi, commonBlocks);
  collectMatches(matches, /\bsubroutine\s+([a-z0-9_]+)/gi, subroutines);
  collectMatches(matches, /\bfunction\s+([a-z0-9_]+)/gi, functions);
  collectMatches(matches, /\bmodule\s+([a-z0-9_]+)/gi, modules);

  const lines = [
    `Pattern insight for "${question}" from ${matches.length} retrieved chunk(s):`,
    `- Frequent common blocks: ${formatEntryList(commonBlocks)}`,
    `- Subroutine signatures: ${formatEntryList(subroutines)}`,
    `- Function signatures: ${formatEntryList(functions)}`,
    `- Modules: ${formatEntryList(modules)}`,
    `- Array dimension declarations found: ${dimensionMentions}`,
    `- Most represented files: ${formatEntryList(files)}`,
  ];

  return lines.join("\n");
}

function generateDependencyInsights(question, matches) {
  if (!matches.length) {
    return `No dependency insight could be generated for "${question}" because no chunks were retrieved.`;
  }

  const calls = new Map();
  const uses = new Map();
  const includes = new Map();
  const commonBlocks = new Map();

  collectMatches(matches, /\bcall\s+([a-z0-9_]+)/gi, calls);
  collectMatches(matches, /\buse\s+([a-z0-9_]+)/gi, uses);
  collectMatches(matches, /\binclude\s+['\"]?([^'\"\n\s]+)/gi, includes);
  collectMatches(matches, /common\s*\/\s*([a-z0-9_]+)/gi, commonBlocks);

  const lines = [
    `Dependency scan for "${question}" across ${matches.length} chunk(s):`,
    `- CALL targets: ${formatEntryList(calls)}`,
    `- USE modules: ${formatEntryList(uses)}`,
    `- INCLUDE files: ${formatEntryList(includes)}`,
    `- Shared COMMON blocks: ${formatEntryList(commonBlocks)}`,
    "- Next step: ask a focused follow-up like 'where is <target> defined?' to expand one dependency edge.",
  ];

  return lines.join("\n");
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  let data = null;
  try {
    data = await response.json();
  } catch (_err) {
    // Ignore parse failure; handled below by status text.
  }

  if (!response.ok) {
    const detail = data && data.detail ? data.detail : `Request failed with status ${response.status}`;
    throw new Error(detail);
  }

  return data || {};
}

async function postMultipart(url, payload) {
  const formData = new FormData();
  formData.append("question", payload.question);
  formData.append("top_k", String(payload.topK));
  formData.append("debug", payload.debug ? "true" : "false");
  formData.append("mode", payload.mode || state.mode);
  formData.append("scope", payload.scope || state.scope);
  formData.append("project_id", payload.projectId || state.projectId);
  formData.append("persist_uploads", payload.persistUploads ? "true" : "false");

  for (const file of payload.files || []) {
    formData.append("files", file, file.name);
  }

  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });

  let data = null;
  try {
    data = await response.json();
  } catch (_err) {
    // Ignore parse failure; handled below by status text.
  }

  if (!response.ok) {
    const detail = data && data.detail ? data.detail : `Request failed with status ${response.status}`;
    throw new Error(detail);
  }

  return data || {};
}

async function runModeQuery(mode, question, files = []) {
  const hasUploads = Array.isArray(files) && files.length > 0;
  const debug = state.debugTraceEnabled;
  const scope = state.scope;
  const projectId = state.projectId;
  const persistUploads = state.persistUploads;
  const uiMode = mode || state.mode;
  const apiMode = apiModeForMode(uiMode);
  const searchRequest = async (topK) => {
    if (hasUploads) {
      return postMultipart("/api/search/upload", {
        question,
        topK,
        files,
        debug,
        mode: apiMode,
        scope,
        projectId,
        persistUploads,
      });
    }

    return postJson("/api/search", {
      question,
      top_k: topK,
      debug,
      mode: apiMode,
      scope,
      project_id: projectId,
    });
  };

  if (uiMode === "chat" || uiMode === "audit") {
    const requestTopK = uiMode === "audit" ? Math.min(20, Math.max(state.topK, 8)) : state.topK;
    const data = hasUploads
      ? await postMultipart("/api/query/upload", {
          question,
          topK: requestTopK,
          files,
          debug,
          mode: apiMode,
          scope,
          projectId,
          persistUploads,
        })
      : await postJson("/api/query", {
          question,
          top_k: requestTopK,
          debug,
          mode: apiMode,
          scope,
          project_id: projectId,
        });

    return {
      text: data.answer || "No answer returned.",
      citations: data.citations || [],
      evidence: data.evidence_strength || {},
      debug: data.debug || null,
      resultType: defaultResultTypeForMode(uiMode),
      followUps: uiMode === "audit" ? auditFollowUpsForType(state.activeAuditType) : [],
    };
  }

  if (uiMode === "search") {
    const topK = state.topK;
    const data = await searchRequest(topK);
    const matches = data.matches || [];
    const summary = data.summary || buildSearchSummary(question, matches);
    return {
      text: summary,
      citations: matches,
      evidence: data.evidence_strength || {},
      debug: data.debug || null,
      resultType: data.result_type || defaultResultTypeForMode(uiMode),
      followUps: normalizeFollowUps(data.follow_ups || []),
    };
  }

  if (uiMode === "patterns") {
    const expandedTopK = Math.min(20, Math.max(state.topK, 6));
    const data = await searchRequest(expandedTopK);
    const matches = data.matches || [];
    return {
      text: data.summary || generatePatternInsights(question, matches),
      citations: matches,
      evidence: data.evidence_strength || {},
      debug: data.debug || null,
      resultType: data.result_type || defaultResultTypeForMode(uiMode),
      followUps: normalizeFollowUps(data.follow_ups || []),
    };
  }

  const expandedTopK = Math.min(20, Math.max(state.topK, 6));
  const data = await searchRequest(expandedTopK);
  const matches = data.matches || [];
  return {
    text: data.summary || generateDependencyInsights(question, matches),
    citations: matches,
    evidence: data.evidence_strength || {},
    debug: data.debug || null,
    resultType: data.result_type || defaultResultTypeForMode(uiMode),
    followUps: normalizeFollowUps(data.follow_ups || []),
  };
}

async function submitQuestion(options = {}) {
  const currentMode = options.modeOverride || state.mode;
  if (currentMode !== state.mode) {
    setMode(currentMode);
  }

  const rawQuestion = (options.questionOverride || input.value).trim();
  if (!rawQuestion) {
    setStatus("Type a question first", "warn");
    return;
  }

  let question = rawQuestion;
  if (currentMode === "audit" && !options.skipAuditWrap) {
    const auditType = options.auditType || state.activeAuditType || inferAuditType(rawQuestion);
    state.activeAuditType = auditType;
    question = buildAuditWorkflowPrompt(auditType, rawQuestion);
  }

  const displaySeed = String(options.displayOverride || rawQuestion).trim() || rawQuestion;
  const displayQuestion =
    state.attachments.length && !options.questionOverride
      ? `${displaySeed}\n[Attached: ${state.attachments.map((item) => item.name).join(", ")}]`
      : displaySeed;

  hero.style.display = "none";
  addMessage("user", displayQuestion, [], { modeLabel: MODE_CONFIG[state.mode].label });
  if (!options.questionOverride) {
    input.value = "";
    autoResize();
  }

  setBusy(true);
  state.lastRequest = {
    mode: state.mode,
    question,
    display: displaySeed,
    skipAuditWrap: currentMode === "audit",
    auditType: state.activeAuditType || null,
  };
  setStatus(
    `Running ${MODE_CONFIG[state.mode].label.toLowerCase()} request${
      state.attachments.length ? ` with ${state.attachments.length} upload(s)` : ""
    }...`,
  );

  const started = performance.now();

  try {
    if (state.persistUploads && state.attachments.length) {
      await persistCurrentAttachments();
    }
    const filesToSend = state.persistUploads ? [] : state.attachments.map((item) => item.file);
    const result = await runModeQuery(
      state.mode,
      question,
      filesToSend,
    );
    const elapsedMs = performance.now() - started;

    addMessage("assistant", result.text, result.citations, {
      modeLabel: MODE_CONFIG[state.mode].label,
      modeValue: state.mode,
      resultType: result.resultType || defaultResultTypeForMode(state.mode),
      followUps: normalizeFollowUps(result.followUps || []),
      debugPayload: result.debug || null,
      elapsedMs,
      resultCount: result.citations.length,
      evidenceLabel: normalizedEvidenceLabel(result.evidence?.label),
    });
    updateEvidencePill(result.evidence || null);

    if (state.debugPanelOpen || state.debugTraceEnabled) {
      renderDebugPayload(result.debug || null);
      if (!state.debugPanelOpen && result.debug) {
        setDebugPanelState(true);
      }
    }

    const contextHit = result.citations.find(
      (item) => item?.file_path && !String(item.file_path).startsWith("uploaded/"),
    );
    if (contextHit?.file_path) {
      state.contextFile = String(contextHit.file_path);
    }

    const count = result.citations.length;
    setStatus(
      `${MODE_CONFIG[state.mode].label}: ${count} source${count === 1 ? "" : "s"} in ${formatDuration(
        elapsedMs,
      )}`,
    );

    if (state.attachments.length) {
      state.attachments = [];
      renderAttachmentList();
    }
  } catch (err) {
    addMessage("assistant", `Error: ${err.message}`);
    setStatus("Request failed", "warn");
  } finally {
    setBusy(false);
  }
}

function renderAttachmentList() {
  attachmentList.innerHTML = "";

  for (let i = 0; i < state.attachments.length; i += 1) {
    const item = state.attachments[i];
    const chip = document.createElement("span");
    chip.className = "attachment-chip";

    const label = document.createElement("span");
    label.textContent = item.name;

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "attachment-remove";
    remove.textContent = "x";
    remove.dataset.attachmentIndex = String(i);
    remove.title = "Remove attachment";

    chip.append(label, remove);
    attachmentList.appendChild(chip);
  }
}

function openSettings() {
  closeSuggestionModal();
  settingsDrawer.classList.remove("hidden");
  overlay.classList.remove("hidden");
}

function closeSettings() {
  settingsDrawer.classList.add("hidden");
  overlay.classList.add("hidden");
}

async function refreshLastPrompt() {
  if (!state.lastRequest) {
    setStatus("No previous prompt to refresh", "warn");
    return;
  }

  await submitQuestion({
    modeOverride: state.lastRequest.mode,
    questionOverride: state.lastRequest.question,
    displayOverride: state.lastRequest.display,
    skipAuditWrap: Boolean(state.lastRequest.skipAuditWrap),
    auditType: state.lastRequest.auditType || null,
  });
}

function exportSession() {
  if (!state.history.length) {
    setStatus("Session is empty", "warn");
    return;
  }

  const payload = {
    exported_at: new Date().toISOString(),
    mode: state.mode,
    top_k: state.topK,
    debug_trace_enabled: state.debugTraceEnabled,
    pinned_sources: state.pinnedSources,
    retrieval_info: state.retrievalInfo,
    messages: state.history,
  };

  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const stamp = new Date().toISOString().replace(/[T:.]/g, "-").slice(0, 19);
  link.href = url;
  link.download = `legacylens-session-${stamp}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);

  setStatus("Session exported");
}

async function loadRetrievalInfo() {
  try {
    const response = await fetch("/api/retrieval-info");
    if (!response.ok) {
      populateTopKSelect(currentTopKMax());
      updateExplainerText();
      return;
    }
    state.retrievalInfo = await response.json();
    if (state.retrievalInfo.repo_url) {
      state.repoUrl = String(state.retrievalInfo.repo_url);
    }
    if (state.retrievalInfo.default_project_id) {
      state.projectId = String(state.retrievalInfo.default_project_id);
    }
    if (state.retrievalInfo.query_top_k_max) {
      topKRange.max = String(Math.max(1, Number(state.retrievalInfo.query_top_k_max)));
    }
    populateTopKSelect(currentTopKMax());
    syncTopKInputs(state.topK);
    updateExplainerText();
    await refreshUploadLibrary();
  } catch (_err) {
    populateTopKSelect(currentTopKMax());
    updateExplainerText();
  }
}

function initSpeech() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    micBtn.disabled = true;
    micBtn.title = "Speech recognition unavailable in this browser";
    return;
  }

  state.recognition = new SpeechRecognition();
  state.recognition.continuous = false;
  state.recognition.lang = "en-US";

  state.recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    input.value = `${input.value} ${transcript}`.trim();
    autoResize();
  };

  state.recognition.onend = () => {
    state.listening = false;
    micBtn.textContent = "🎙";
  };

  state.recognition.onerror = () => {
    state.listening = false;
    micBtn.textContent = "🎙";
    setStatus("Microphone input failed", "warn");
  };
}

modeButtons.forEach((item) => {
  item.addEventListener("click", () => {
    const mode = item.dataset.mode;
    setMode(mode);
    if (mode === "audit") {
      openSuggestionModal({
        filterKeys: ["audit-workflows"],
        focusKey: "audit-workflows",
      });
    }
  });
});

input.addEventListener("input", autoResize);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    submitQuestion();
  }
});

sendBtn.addEventListener("click", () => submitQuestion());
newChatBtn.addEventListener("click", resetSession);
clearSessionBtn.addEventListener("click", () => {
  resetSession();
  closeSettings();
});

refreshBtn.addEventListener("click", refreshLastPrompt);
scopeBothBtn.addEventListener("click", () => setScope("both"));
scopeRepoBtn.addEventListener("click", () => setScope("repo"));
scopeUploadsBtn.addEventListener("click", () => setScope("uploads"));
debugToggleBtn.addEventListener("click", () => {
  setDebugPanelState(!state.debugPanelOpen);
  if (state.debugPanelOpen) {
    renderDebugPayload(state.lastDebugPayload);
  }
});
settingsBtn.addEventListener("click", openSettings);
closeSettingsBtn.addEventListener("click", closeSettings);
overlay.addEventListener("click", closeSettings);
suggestionBtn.addEventListener("click", openSuggestionModal);
closeSuggestionModalBtn.addEventListener("click", closeSuggestionModal);
suggestionOverlay.addEventListener("click", closeSuggestionModal);
exploreCapabilitiesLink.addEventListener("click", (event) => {
  event.preventDefault();
  closeSuggestionModal();
  setMode("chat");
  input.value = "Show example questions I can ask in Chat, Search, Code Patterns, Dependencies, and Audit modes.";
  autoResize();
  setStatus("Capabilities example inserted");
});
exportBtn.addEventListener("click", exportSession);
refreshUploadsBtn.addEventListener("click", refreshUploadLibrary);
clearPinsBtn.addEventListener("click", () => {
  state.pinnedSources = [];
  renderPinnedSources();
  setStatus("Pinned sources cleared");
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeSettings();
    closeSuggestionModal();
    sidebar.classList.remove("open");
  }
});

scoreInfoBtn.addEventListener("click", () => {
  scoreExplainer.classList.toggle("hidden");
});

topKSelect.addEventListener("change", () => {
  syncTopKInputs(topKSelect.value);
  setStatus(`Top K set to ${state.topK}`);
});

topKRange.addEventListener("input", () => {
  syncTopKInputs(topKRange.value);
});

topKRange.addEventListener("change", () => {
  setStatus(`Top K set to ${state.topK}`);
});

showSnippetsToggle.addEventListener("change", () => {
  state.showSnippets = showSnippetsToggle.checked;
  applyDisplayPreferences();
  setStatus(state.showSnippets ? "Snippets enabled" : "Snippets hidden");
});

compactCitationsToggle.addEventListener("change", () => {
  state.compactCitations = compactCitationsToggle.checked;
  applyDisplayPreferences();
  setStatus(state.compactCitations ? "Compact citation view enabled" : "Detailed citation view enabled");
});

debugTraceToggle.addEventListener("change", () => {
  state.debugTraceEnabled = debugTraceToggle.checked;
  setStatus(state.debugTraceEnabled ? "Debug trace enabled" : "Debug trace disabled");
  if (state.debugTraceEnabled) {
    setDebugPanelState(true);
  }
});

persistUploadsToggle.addEventListener("change", () => {
  state.persistUploads = persistUploadsToggle.checked;
  setStatus(state.persistUploads ? "Upload persistence enabled" : "Upload persistence disabled");
});

attachBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (!fileInput.files || fileInput.files.length === 0) return;

  const limits = uploadLimits();
  let totalBytes = state.attachments.reduce((sum, item) => sum + (item.size || 0), 0);

  for (const file of Array.from(fileInput.files)) {
    if (state.attachments.length >= limits.maxFiles) {
      setStatus(`Upload limit reached (${limits.maxFiles} files max)`, "warn");
      break;
    }
    if (file.size > limits.maxFileBytes) {
      setStatus(`"${file.name}" is too large (${Math.round(limits.maxFileBytes / 1024)} KB max)`, "warn");
      continue;
    }
    if (totalBytes + file.size > limits.maxTotalBytes) {
      setStatus(`Total upload size exceeds limit (${Math.round(limits.maxTotalBytes / 1024)} KB)`, "warn");
      continue;
    }

    const signature = `${file.name}|${file.size}|${file.lastModified}`;
    const exists = state.attachments.some((item) => item.signature === signature);
    if (!exists) {
      state.attachments.push({
        signature,
        name: file.name,
        size: file.size,
        file,
      });
      totalBytes += file.size;
    }
  }

  fileInput.value = "";
  renderAttachmentList();
  setStatus(`${state.attachments.length} attachment(s) ready`);
});

attachmentList.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;

  const indexRaw = target.dataset.attachmentIndex;
  if (typeof indexRaw === "undefined") return;

  const index = Number(indexRaw);
  if (!Number.isInteger(index) || index < 0 || index >= state.attachments.length) return;

  state.attachments.splice(index, 1);
  renderAttachmentList();
  setStatus("Attachment removed");
});

uploadLibraryList.addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  const action = target.dataset.uploadAction;
  const fileSha = target.dataset.fileSha;
  if (!action || !fileSha) return;

  if (action === "delete") {
    try {
      const response = await fetch(
        `/api/uploads/${encodeURIComponent(fileSha)}?project_id=${encodeURIComponent(state.projectId)}`,
        {
          method: "DELETE",
        },
      );
      if (!response.ok) {
        throw new Error(`Delete failed (${response.status})`);
      }
      setStatus("Upload deleted");
      await refreshUploadLibrary();
    } catch (_err) {
      setStatus("Failed to delete upload", "warn");
    }
    return;
  }

  if (action === "pin") {
    const currentPinned = target.dataset.currentPinned === "1";
    try {
      const response = await fetch(`/api/uploads/${encodeURIComponent(fileSha)}/pin`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_id: state.projectId,
          pinned: !currentPinned,
        }),
      });
      if (!response.ok) {
        throw new Error(`Pin failed (${response.status})`);
      }
      setStatus(currentPinned ? "Upload unpinned" : "Upload pinned");
      await refreshUploadLibrary();
    } catch (_err) {
      setStatus("Failed to pin upload", "warn");
    }
  }
});

toggleSidebar.addEventListener("click", () => {
  sidebar.classList.toggle("open");
});

micBtn.addEventListener("click", () => {
  if (!state.recognition) return;

  if (!state.listening) {
    state.listening = true;
    micBtn.textContent = "■";
    state.recognition.start();
    setStatus("Listening...");
  } else {
    state.listening = false;
    micBtn.textContent = "🎙";
    state.recognition.stop();
    setStatus("Voice input stopped");
  }
});

populateTopKSelect(currentTopKMax());
syncTopKInputs(state.topK);
state.debugTraceEnabled = debugTraceToggle.checked;
state.persistUploads = persistUploadsToggle.checked;
setScope("both");
setMode("chat");
initSpeech();
autoResize();
updateEvidencePill(null);
renderPinnedSources();
renderUploadLibrary();
renderRecentChats();
setDebugPanelState(false);
renderDebugPayload(null);
closeSuggestionModal();
loadRetrievalInfo();
setStatus("Ready");
