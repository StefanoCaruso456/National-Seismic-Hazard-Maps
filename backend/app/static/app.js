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
  hybrid: {
    label: "Hybrid",
    placeholder: "Architecture-first analysis with graph-constrained citations...",
    heroTitle: "Fuse structure graph with line-level evidence",
    heroCopy:
      "Hybrid mode uses GitNexus process/context/impact signals to constrain Pinecone retrieval, then returns architecture-first evidence with citations.",
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
  diagrams: {
    label: "Diagrams",
    placeholder: "Select a diagram type to map architecture, execution, data flow, or dependencies...",
    heroTitle: "Generate Repository Diagrams",
    heroCopy:
      "Diagram mode generates Mermaid diagrams from repository evidence so engineers can quickly understand system structure and runtime flow.",
  },
  audit: {
    label: "Audit",
    placeholder: "Run a full codebase audit or describe a focus area...",
    heroTitle: "Run A Unified Codebase Audit",
    heroCopy:
      "Run one fast baseline audit across architecture, stack/dependencies, and maintainability, then drill deeper on selected areas.",
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

const AUDIT_WORKFLOW = {
  title: "Run Codebase Audit",
  reportType: "Audit Report",
  followUps: [
    "Deeper pass: architecture map + critical execution paths",
    "Deeper pass: stack/build/runtime dependency risk register",
    "Deeper pass: maintainability hotspots + test gap plan",
  ],
  prompt: `You are a senior staff engineer performing a unified Codebase Audit for a legacy or brownfield repository.

Primary goals:
- Build a system map: entry points, major modules, critical call chains, and high-risk coupling.
- Inventory stack and runtime: languages, frameworks, build/run commands, manifests, runtime assumptions, infra integrations.
- Identify maintainability and change-risk hotspots: complexity, duplication, weak boundaries, test gaps, reliability concerns.

Method:
- Use a trace-driven approach: start from entry points and follow real execution/data flow.
- Start with a fast scan baseline for the full repo, then mark best candidates for deeper pass.
- Distinguish verified evidence vs inference; label uncertain claims as "Hypothesis" with verification steps.
- Prioritize incremental fixes and safe refactors; do not recommend rewrites.

Required content inside the report:
- Architecture: component map, where execution starts, top critical paths, coupling risks.
- Stack and Dependencies: stack snapshot, build/run golden path, dependency risks, config/env gaps.
- Maintainability and Risk: hotspot leaderboard, test/quality gaps, reliability/observability issues.
- Priority for every finding: High | Medium | Low, with why it matters.

${AUDIT_REPORT_CONTRACT}`,
};

const DIAGRAM_WORKFLOWS = {
  systemArchitecture: {
    title: "System Architecture: National Hazard Run",
    reportType: "System Architecture Diagram",
    followUps: [
      "Refine architecture diagram around ingestion and compute boundaries",
      "Add external systems and deployment boundary nodes",
      "Generate a zoomed-in diagram for the hazard engine subsystem",
    ],
    prompt: `You are a senior software architect analyzing a code repository.

Your task is to generate a HIGH-LEVEL SYSTEM ARCHITECTURE DIAGRAM for this repository.

Follow these steps carefully:
1) Scan the entire repository structure.
2) Identify major system components including entry scripts, core computational modules, data/config folders, build tools, external dependencies, and output artifacts.
3) Group related components into logical subsystems such as orchestration scripts, compute engine, model logic, configuration, data sources, outputs, and build system.
4) Identify directional relationships between components.
5) Produce a clean architecture diagram.

Primary scope for this repository:
- System boundaries centered on run orchestration, regional hazard scripts, core hazard binaries, config/data inputs, and outputs.

OUTPUT FORMAT:
- Return ONLY a Mermaid diagram.
- Use flowchart TD.
- Keep the diagram high level.
- Use logical system components instead of individual files.
- Limit the diagram to 8-12 nodes.
- Prefer clarity over completeness.
- At least 6 nodes must be traceable to real repository components/scripts/binaries.
- No explanatory prose outside Mermaid.`,
  },
  executionPipeline: {
    title: "Execution Pipeline: run_all_hazard.sh",
    reportType: "Execution Pipeline Diagram",
    followUps: [
      "Trace startup path from the primary run script in more detail",
      "Add optional and error branches in the pipeline",
      "Generate pipeline focused only on output generation stages",
    ],
    prompt: `You are a software engineer investigating how a repository executes.

Your task is to produce an EXECUTION PIPELINE DIAGRAM.

Follow these steps:
1) Identify all entry points such as shell scripts, main programs, run scripts, and Makefile targets.
2) Determine execution order.
3) Identify intermediate steps such as preprocessing, model initialization, simulation, aggregation, and output generation.
4) Trace the runtime sequence.
5) Generate a pipeline diagram.

Primary scope for this repository:
- run_all_hazard.sh sequence: environment checks -> make -> getmeanrjf.v2 preprocessing -> hazrun_casc_2014.sh / hazrun_ceus_2014.sh / hazrun_wus_2014.sh -> logs/out outputs.

OUTPUT FORMAT:
- Return ONLY a Mermaid diagram.
- Use flowchart LR.
- Must represent execution order.
- Use left-to-right flow.
- Max 10 nodes.
- Avoid low-level function names.
- Include the orchestration and regional execution stages when evidence exists.
- No explanatory prose outside Mermaid.`,
  },
  dataFlow: {
    title: "Data Flow: conf/* to out/*",
    reportType: "Data Flow Diagram",
    followUps: [
      "Add more detail for configuration and lookup-table lineage",
      "Show intermediate datasets and transformation boundaries",
      "Generate data flow only for one critical output artifact",
    ],
    prompt: `You are a data systems architect.

Your goal is to produce a DATA FLOW DIAGRAM for the repository.

Steps:
1) Identify major data inputs including configuration files, scientific datasets, model parameters, and lookup tables.
2) Identify transformations that process the data.
3) Identify outputs including generated maps, simulation results, and processed datasets.
4) Trace how data moves through the system.
5) Create a clean data lineage diagram.

Primary scope for this repository:
- conf/WUS, conf/CEUS, conf/CASC input files and scripts/GR tables flowing into hazard executables and then to logs/out artifacts.

OUTPUT FORMAT:
- Return ONLY a Mermaid diagram.
- Use flowchart TD.
- Show transformations clearly.
- Use descriptive node names.
- Max 12 nodes.
- Include input -> transform -> output lineage with concrete repository anchors.
- No explanatory prose outside Mermaid.`,
  },
  dependencyGraph: {
    title: "Dependency Graph: Core Hazard Engines",
    reportType: "Module Dependency Graph",
    followUps: [
      "Expand dependency graph around the most central module",
      "Map dependency chains starting from run scripts",
      "Highlight possible circular dependencies and shared utility overload",
    ],
    prompt: `You are analyzing the internal code dependencies of a repository.

Your task is to create a MODULE DEPENDENCY GRAPH.

Steps:
1) Scan the source code.
2) Identify modules, packages, or major files.
3) Detect dependency relationships including imports, module usage, and function calls.
4) Identify the most central modules.
5) Focus only on the most important modules.

Primary scope for this repository:
- Core engine set: hazgridXnga13l, hazFXnga13l, hazSUBX, hazpoint, hazinterpnga and utility/build dependencies tied to Makefile and scripts.

OUTPUT FORMAT:
- Return ONLY a Mermaid graph.
- Use graph TD.
- Show directional dependencies.
- Focus on the top 10 most important modules.
- Do not include every file.
- Prefer modules that are demonstrably central in entry scripts or build targets.
- No explanatory prose outside Mermaid.`,
  },
  buildRuntime: {
    title: "Build & Runtime: Makefile + gfortran",
    reportType: "Build and Runtime Environment Diagram",
    followUps: [
      "Add CI/CD and deployment stages to the environment diagram",
      "Expand runtime dependency nodes for configs and data mounts",
      "Generate a reproducibility-focused build/run diagram",
    ],
    prompt: `You are a build systems engineer.

Your task is to create a BUILD AND RUNTIME ENVIRONMENT DIAGRAM.

Steps:
1) Identify build system details such as Makefile, shell scripts, and compiler commands.
2) Identify compiler and toolchain.
3) Identify runtime dependencies such as datasets, environment variables, and config files.
4) Identify the final executable/runtime target and outputs.

Primary scope for this repository:
- Makefile/gfortran build chain producing bin/* executables, then runtime scripts consuming conf/* and producing out/logs.

OUTPUT FORMAT:
- Return ONLY a Mermaid diagram.
- Use flowchart TD.
- Focus on build and runtime environment.
- Avoid low-level file detail.
- Keep the diagram readable.
- Include both compile-time and run-time stages.
- No explanatory prose outside Mermaid.`,
  },
};

const DIAGRAM_PROMPT_TO_TYPE = Object.fromEntries(
  Object.entries(DIAGRAM_WORKFLOWS).map(([type, workflow]) => [workflow.title, type]),
);

const NSHMP_DIAGRAM_REPO_FACTS = [
  "Entry orchestrator: run_all_hazard.sh",
  "Regional runners: scripts/hazrun_casc_2014.sh, scripts/hazrun_ceus_2014.sh, scripts/hazrun_wus_2014.sh",
  "Build file: Makefile (gfortran targets to bin/*)",
  "Core executables: bin/hazgridXnga13l, bin/hazFXnga13l, bin/hazSUBX, bin/hazpoint, bin/hazinterpnga",
  "Input/config roots: conf/CASC, conf/CEUS, conf/WUS, scripts/GR",
  "Outputs/logs: out/, out/combine/, logs/",
];

const NSHMP_DIAGRAM_SCOPE_HINTS = {
  systemArchitecture:
    "Map orchestration and subsystem boundaries around run_all_hazard.sh, regional scripts, core binaries, and output directories.",
  executionPipeline:
    "Trace the concrete runtime sequence from run_all_hazard.sh through make/getmeanrjf.v2, then regional scripts, then outputs.",
  dataFlow:
    "Trace data lineage from conf/* and scripts/GR inputs into hazard binaries and then logs/out artifacts.",
  dependencyGraph:
    "Center dependency graph on hazgridXnga13l, hazFXnga13l, hazSUBX, hazpoint, hazinterpnga and their build/run relationships.",
  buildRuntime:
    "Model Makefile + gfortran compile chain and runtime dependence on conf/*, scripts/*, and generated outputs.",
};

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
    key: "diagram-types",
    title: "DIAGRAMS",
    mode: "diagrams",
    prompts: [
      DIAGRAM_WORKFLOWS.systemArchitecture.title,
      DIAGRAM_WORKFLOWS.executionPipeline.title,
      DIAGRAM_WORKFLOWS.dataFlow.title,
    ],
  },
  {
    key: "diagram-types-advanced",
    title: "DIAGRAMS ADVANCED",
    mode: "diagrams",
    prompts: [
      DIAGRAM_WORKFLOWS.dependencyGraph.title,
      DIAGRAM_WORKFLOWS.buildRuntime.title,
    ],
  },
  {
    key: "audit",
    title: "RUN AUDIT",
    mode: "audit",
    prompts: [AUDIT_WORKFLOW.title],
  },
];

const DEFAULT_UPLOAD_LIMITS = {
  maxFiles: 8,
  maxFileBytes: 1_500_000,
  maxTotalBytes: 6_000_000,
};

const BACKEND_QUERY_SOFT_MAX = 7600;

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
  activeDiagramType: "systemArchitecture",
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
  modeButtons.forEach((btn) => {
    btn.disabled = state.loading;
  });
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

function auditFollowUps() {
  return AUDIT_WORKFLOW.followUps.slice(0, 3);
}

function buildAuditWorkflowPrompt(userIntent = "") {
  const intent = String(userIntent || "").trim();
  const contextLine = intent
    ? `Audit focus request from user: ${intent}\nConstrain recommendations to this focus where possible.`
    : "Audit focus request from user: full repository baseline scan.";

  return `${AUDIT_WORKFLOW.prompt}

Execution requirements:
- Start with a fast scan baseline.
- If repository size/complexity is high, identify specific deeper-pass targets.
- Mark uncertainty explicitly as Hypothesis + verification steps.
- Keep recommendations incremental and safe.

Report format contract:
Overview -> Key Findings -> Evidence (files/lines) -> Recommendations -> Next Actions

${contextLine}`;
}

function formatApiErrorDetail(detail, fallback) {
  if (!detail) return fallback;
  if (typeof detail === "string") return detail;

  if (Array.isArray(detail)) {
    const parts = detail
      .map((item) => {
        if (!item) return "";
        if (typeof item === "string") return item;
        if (typeof item === "object") {
          const loc = Array.isArray(item.loc) ? item.loc.join(".") : "";
          const msg = typeof item.msg === "string" ? item.msg : JSON.stringify(item);
          return loc ? `${loc}: ${msg}` : msg;
        }
        return String(item);
      })
      .filter(Boolean);
    return parts.length ? parts.join(" | ") : fallback;
  }

  if (typeof detail === "object") {
    if (typeof detail.message === "string") return detail.message;
    return JSON.stringify(detail);
  }

  return String(detail);
}

function compactPromptWhitespace(text) {
  return String(text || "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]{2,}/g, " ")
    .trim();
}

function trimQuestionForBackend(question) {
  const normalized = compactPromptWhitespace(question);
  if (normalized.length <= BACKEND_QUERY_SOFT_MAX) {
    return { value: normalized, trimmed: false };
  }

  const suffix = "\n\n[Prompt trimmed to fit backend request limits.]";
  const maxBodyLength = Math.max(32, BACKEND_QUERY_SOFT_MAX - suffix.length);
  const body = normalized.slice(0, maxBodyLength).trimEnd();
  return { value: `${body}${suffix}`, trimmed: true };
}

function buildAuditDispatch(promptText) {
  const isDefaultPrompt = String(promptText || "").trim() === AUDIT_WORKFLOW.title;
  return {
    displayText: AUDIT_WORKFLOW.title,
    workflowPrompt: buildAuditWorkflowPrompt(isDefaultPrompt ? "" : promptText),
  };
}

function inferDiagramType(text) {
  const normalized = String(text || "").trim().toLowerCase();
  if (!normalized) return state.activeDiagramType || "systemArchitecture";
  if (normalized.includes("execution") || normalized.includes("pipeline") || normalized.includes("runtime flow")) {
    return "executionPipeline";
  }
  if (normalized.includes("data flow") || normalized.includes("lineage")) {
    return "dataFlow";
  }
  if (normalized.includes("dependency") || normalized.includes("module graph") || normalized.includes("import")) {
    return "dependencyGraph";
  }
  if (normalized.includes("build") || normalized.includes("runtime environment") || normalized.includes("compiler")) {
    return "buildRuntime";
  }
  return "systemArchitecture";
}

function diagramFollowUpsForType(diagramType) {
  const workflow = DIAGRAM_WORKFLOWS[diagramType];
  if (!workflow) return [];
  return workflow.followUps.slice(0, 3);
}

function buildDiagramWorkflowPrompt(diagramType, userIntent = "") {
  const resolvedType = DIAGRAM_WORKFLOWS[diagramType] ? diagramType : inferDiagramType(userIntent);
  const workflow = DIAGRAM_WORKFLOWS[resolvedType];
  const intent = String(userIntent || "").trim();
  const focusLine = intent
    ? `Focus request from user: ${intent}\nConstrain node labels and scope around this focus where possible.`
    : "Focus request from user: full repository baseline.";
  const scopeHint = NSHMP_DIAGRAM_SCOPE_HINTS[resolvedType] || NSHMP_DIAGRAM_SCOPE_HINTS.systemArchitecture;
  const repoFacts = NSHMP_DIAGRAM_REPO_FACTS.map((fact) => `- ${fact}`).join("\n");

  return `${workflow.prompt}

Global guardrails:
- First scan repo, then reason, then draw.
- Use evidence from discovered files/modules; avoid invented components.
- Keep node labels concise and engineering-relevant.
- Include concrete repository anchors whenever evidence exists.
- Do not emit generic placeholder nodes like "Component A/B" or "Module X".
- If evidence is insufficient for a node, annotate the label with "(hypothesis)".
- Keep deterministic structure: stable top-to-bottom or left-to-right flow with no disconnected nodes.

Verified repository anchors to prioritize:
${repoFacts}

Scope directive:
${scopeHint}

${focusLine}`;
}

function buildDiagramDispatch(promptText) {
  const mappedType = DIAGRAM_PROMPT_TO_TYPE[promptText];
  const diagramType = mappedType || inferDiagramType(promptText);
  return {
    diagramType,
    displayText: DIAGRAM_WORKFLOWS[diagramType].title,
    workflowPrompt: buildDiagramWorkflowPrompt(diagramType, promptText),
  };
}

function apiModeForMode(mode) {
  return mode === "audit" || mode === "diagrams" ? "chat" : mode;
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
  let diagramType = null;

  if (mode === "audit") {
    const dispatch = buildAuditDispatch(promptText);
    dispatchPrompt = dispatch.workflowPrompt;
    displayPrompt = dispatch.displayText;
  } else if (mode === "diagrams") {
    const dispatch = buildDiagramDispatch(promptText);
    diagramType = dispatch.diagramType;
    state.activeDiagramType = dispatch.diagramType;
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
    skipDiagramWrap: mode === "diagrams",
    diagramType,
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
  state.activeDiagramType = "systemArchitecture";
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

function defaultResultTypeForMode(mode, detail = null) {
  if (mode === "audit") return AUDIT_WORKFLOW.reportType;
  if (mode === "diagrams") {
    const diagramType = detail || state.activeDiagramType;
    return DIAGRAM_WORKFLOWS[diagramType]?.reportType || "Mermaid Diagram";
  }
  if (mode === "hybrid") return "Hybrid Architecture + Evidence";
  if (mode === "graph") return "Graph Architecture";
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

function stripMarkdownBold(text) {
  return String(text || "")
    .replace(/\*\*([^*]+?)\*\*/g, "$1")
    .replace(/\*\*/g, "");
}

function normalizeDiagramText(text) {
  const raw = String(text || "").trim();
  if (!raw) return "";
  const fencedMatch = raw.match(/```(?:mermaid)?\s*([\s\S]*?)```/i);
  let next = fencedMatch ? fencedMatch[1].trim() : raw;
  next = next
    .replace(/^`{3,}\s*/gm, "")
    .replace(/`{3,}$/gm, "")
    .replace(/^["']{1,3}\s*/, "")
    .replace(/\s*["']{1,3}$/, "")
    .trim();
  return next || raw;
}

const HYBRID_GRAPH_COLORS = {
  query: "#d45757",
  process: "#3f7d95",
  symbol: "#6472c9",
  entrypoint: "#8c5fb5",
  file: "#2f9f84",
  impact: "#e48f3b",
  default: "#7a8a99",
  edge: "rgba(70, 92, 105, 0.35)",
};

function buildHybridGraphModel(graphPayload) {
  const graph = graphPayload && typeof graphPayload === "object" ? graphPayload : {};
  const canvas = graph.canvas && typeof graph.canvas === "object" ? graph.canvas : {};
  const rawNodes = Array.isArray(canvas.nodes) ? canvas.nodes : [];
  const rawEdges = Array.isArray(canvas.edges) ? canvas.edges : [];

  const nodes = [];
  const edges = [];
  const seenNodes = new Set();
  const seenEdges = new Set();

  const addNode = (id, label, kind = "default", size = 1, path = "") => {
    const nodeId = String(id || "").trim();
    if (!nodeId || seenNodes.has(nodeId)) return;
    seenNodes.add(nodeId);
    nodes.push({
      id: nodeId,
      label: String(label || nodeId).slice(0, 90),
      kind: String(kind || "default"),
      size: Math.max(Number(size) || 1, 0.5),
      path: String(path || ""),
    });
  };

  const addEdge = (source, target, kind = "edge") => {
    const src = String(source || "").trim();
    const dst = String(target || "").trim();
    if (!src || !dst) return;
    const edgeKey = `${src}|${dst}|${kind}`;
    if (seenEdges.has(edgeKey)) return;
    seenEdges.add(edgeKey);
    edges.push({ source: src, target: dst, kind: String(kind || "edge") });
  };

  if (rawNodes.length) {
    rawNodes.forEach((node) => {
      if (!node || typeof node !== "object") return;
      addNode(node.id, node.label, node.kind, node.size, node.path);
    });
    rawEdges.forEach((edge) => {
      if (!edge || typeof edge !== "object") return;
      addEdge(edge.source, edge.target, edge.kind);
    });
  }

  if (!nodes.length) {
    const queryId = "query:hybrid";
    addNode(queryId, "Hybrid Query", "query", 1.5);
    const processes = Array.isArray(graph.processes) ? graph.processes : [];
    processes.slice(0, 12).forEach((process, idx) => {
      const label = String(process?.summary || process?.id || `Process ${idx + 1}`);
      const nodeId = `process:${idx}:${label.toLowerCase().replace(/[^a-z0-9_]+/g, "_")}`;
      addNode(nodeId, label, "process", 1.1);
      addEdge(queryId, nodeId, "process");
    });
    const entrypoints = Array.isArray(graph.entrypoints) ? graph.entrypoints : [];
    entrypoints.slice(0, 10).forEach((entry, idx) => {
      const label = String(entry || `Entrypoint ${idx + 1}`);
      const nodeId = `entry:${idx}:${label.toLowerCase().replace(/[^a-z0-9_]+/g, "_")}`;
      addNode(nodeId, label, "entrypoint", 1.0);
      addEdge(queryId, nodeId, "entrypoint");
    });
    const candidateFiles = Array.isArray(graph.candidate_files) ? graph.candidate_files : [];
    candidateFiles.slice(0, 18).forEach((file, idx) => {
      const path = String(file || "").trim();
      if (!path) return;
      const fileName = path.split("/").pop() || path;
      const nodeId = `file:${idx}:${path.toLowerCase().replace(/[^a-z0-9_./-]+/g, "_")}`;
      addNode(nodeId, fileName, "file", 0.95, path);
      addEdge(queryId, nodeId, "candidate");
    });
  }

  return { nodes: nodes.slice(0, 120), edges: edges.slice(0, 220) };
}

function drawHybridGraphSnapshot(canvas, model) {
  if (!canvas || !model || !Array.isArray(model.nodes) || !model.nodes.length) return;
  const host = canvas.parentElement;
  const width = Math.max(300, Math.min(900, (host?.clientWidth || 640) - 6));
  const height = 300;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const queryNode = model.nodes.find((node) => node.kind === "query") || model.nodes[0];
  const centerX = width * 0.5;
  const centerY = height * 0.5;
  const positions = new Map();

  model.nodes.forEach((node) => {
    if (node.id === queryNode.id) {
      positions.set(node.id, { x: centerX, y: centerY });
    }
  });

  const orderedKinds = ["process", "symbol", "entrypoint", "impact", "file", "default"];
  const ringRadius = {
    process: Math.min(width, height) * 0.2,
    symbol: Math.min(width, height) * 0.3,
    entrypoint: Math.min(width, height) * 0.37,
    impact: Math.min(width, height) * 0.43,
    file: Math.min(width, height) * 0.48,
    default: Math.min(width, height) * 0.52,
  };
  const grouped = {};
  orderedKinds.forEach((kind) => { grouped[kind] = []; });
  model.nodes.forEach((node) => {
    if (node.id === queryNode.id) return;
    const kind = orderedKinds.includes(node.kind) ? node.kind : "default";
    grouped[kind].push(node);
  });

  let angleOffset = -Math.PI / 2;
  orderedKinds.forEach((kind) => {
    const group = grouped[kind];
    if (!group.length) return;
    const radius = ringRadius[kind];
    group.forEach((node, idx) => {
      const angle = angleOffset + ((Math.PI * 2 * idx) / group.length);
      positions.set(node.id, {
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius,
      });
    });
    angleOffset += 0.18;
  });

  ctx.strokeStyle = HYBRID_GRAPH_COLORS.edge;
  ctx.lineWidth = 1;
  model.edges.forEach((edge) => {
    const source = positions.get(edge.source);
    const target = positions.get(edge.target);
    if (!source || !target) return;
    ctx.beginPath();
    ctx.moveTo(source.x, source.y);
    ctx.lineTo(target.x, target.y);
    ctx.stroke();
  });

  const maxLabels = model.nodes.length > 40 ? 24 : 999;
  let labelsShown = 0;
  model.nodes.forEach((node) => {
    const pos = positions.get(node.id);
    if (!pos) return;
    const color = HYBRID_GRAPH_COLORS[node.kind] || HYBRID_GRAPH_COLORS.default;
    const radius = node.id === queryNode.id ? 10 : Math.max(4, Math.min(9, 3 + node.size * 2.2));
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.globalAlpha = node.id === queryNode.id ? 1 : 0.92;
    ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.lineWidth = 0.8;
    ctx.stroke();

    if (labelsShown >= maxLabels) return;
    if (node.kind === "file" && model.nodes.length > 26) return;
    labelsShown += 1;
    ctx.fillStyle = "rgba(31, 46, 58, 0.86)";
    ctx.font = "11px 'JetBrains Mono', 'IBM Plex Mono', monospace";
    ctx.textBaseline = "middle";
    ctx.fillText(String(node.label || "").slice(0, 30), pos.x + radius + 4, pos.y);
  });
}

function hybridNodeKindColor(kind) {
  return HYBRID_GRAPH_COLORS[kind] || HYBRID_GRAPH_COLORS.default;
}

function hybridGraphNodeLabel(node) {
  const label = String(node?.label || node?.id || "node");
  const kind = String(node?.kind || "node");
  const path = String(node?.path || "");
  if (path) {
    return `${label}\n${kind}\n${path}`;
  }
  return `${label}\n${kind}`;
}

function openGraphNodeSource(node) {
  const path = String(node?.path || "").trim();
  if (!path) {
    setStatus(`Selected graph node: ${String(node?.label || node?.id || "node")}`);
    return;
  }
  const citation = { file_path: path, line_start: 1, line_end: 1 };
  const link = sourceLink(citation);
  state.contextFile = path;
  if (!link) {
    setStatus(`Graph file selected: ${path}`);
    return;
  }
  window.open(link, "_blank", "noopener,noreferrer");
}

function renderHybridGraphCanvas(panel, graphPayload) {
  const model = buildHybridGraphModel(graphPayload);
  const wrap = document.createElement("div");
  wrap.className = "hybrid-graph-wrap";

  if (!model.nodes.length) {
    const empty = document.createElement("p");
    empty.className = "hybrid-panel-empty";
    empty.textContent = "No graph nodes returned.";
    wrap.appendChild(empty);
    panel.appendChild(wrap);
    return;
  }

  const toolbar = document.createElement("div");
  toolbar.className = "hybrid-graph-toolbar";

  const title = document.createElement("span");
  title.className = "hybrid-graph-title";
  title.textContent = "Interactive Graph";
  toolbar.appendChild(title);

  const status = document.createElement("span");
  status.className = "hybrid-graph-status";
  status.textContent = `${model.nodes.length} nodes • ${model.edges.length} edges`;
  toolbar.appendChild(status);

  const viewport = document.createElement("div");
  viewport.className = "hybrid-graph-viewport";

  const details = document.createElement("div");
  details.className = "hybrid-graph-details";
  details.textContent = "Click a node to inspect details and open linked files.";

  const legend = document.createElement("div");
  legend.className = "hybrid-graph-legend";
  legend.textContent = "query process symbol entrypoint impact file";

  wrap.append(toolbar, viewport, details, legend);
  panel.appendChild(wrap);

  const GraphFactory = window.ForceGraph3D;
  if (typeof GraphFactory !== "function") {
    const canvas = document.createElement("canvas");
    canvas.className = "hybrid-graph-canvas";
    viewport.appendChild(canvas);
    drawHybridGraphSnapshot(canvas, model);
    details.textContent = "3D graph library unavailable. Showing static snapshot fallback.";
    return;
  }

  const graphData = {
    nodes: model.nodes.map((node) => ({
      ...node,
      val: Math.max(1, Math.min(12, Number(node.size || 1) * 3.2)),
    })),
    links: model.edges.map((edge, idx) => ({
      id: `edge-${idx}`,
      source: edge.source,
      target: edge.target,
      kind: edge.kind || "edge",
    })),
  };

  const graph3d = GraphFactory({
    controlType: "orbit",
    rendererConfig: { antialias: true, alpha: true },
  })(viewport)
    .backgroundColor("rgba(0,0,0,0)")
    .graphData(graphData)
    .nodeColor((node) => hybridNodeKindColor(String(node.kind || "default")))
    .nodeLabel((node) => hybridGraphNodeLabel(node))
    .nodeVal((node) => Number(node.val || 2))
    .linkColor(() => HYBRID_GRAPH_COLORS.edge)
    .linkOpacity(0.42)
    .linkWidth((link) => (String(link.kind || "").startsWith("impact_") ? 1.8 : 1.15))
    .onNodeHover((node) => {
      viewport.style.cursor = node ? "pointer" : "grab";
    })
    .onNodeClick((node) => {
      const kind = String(node?.kind || "node");
      const label = String(node?.label || node?.id || "node");
      const path = String(node?.path || "");
      details.textContent = path ? `${kind}: ${label} (${path})` : `${kind}: ${label}`;
      openGraphNodeSource(node);
    })
    .onLinkClick((link) => {
      const source = typeof link?.source === "object" ? link.source?.label || link.source?.id : link?.source;
      const target = typeof link?.target === "object" ? link.target?.label || link.target?.id : link?.target;
      details.textContent = `edge: ${String(link?.kind || "related")} • ${source} -> ${target}`;
    });

  try {
    graph3d.d3Force("charge").strength(-95);
    graph3d.d3VelocityDecay(0.32);
  } catch (_err) {
    // Ignore if force engine internals differ.
  }

  const controls = graph3d.controls?.();
  if (controls) {
    controls.enableDamping = true;
    controls.dampingFactor = 0.14;
    controls.rotateSpeed = 0.7;
    controls.zoomSpeed = 0.8;
  }

  requestAnimationFrame(() => {
    try {
      graph3d.zoomToFit(900, 60);
    } catch (_err) {
      // Ignore initial layout timing errors.
    }
  });
}

function renderHybridArchitecturePanel(panel, graphPayload) {
  panel.innerHTML = "";
  const title = document.createElement("h5");
  title.textContent = "Architecture Context";
  panel.appendChild(title);

  const graph = graphPayload && typeof graphPayload === "object" ? graphPayload : {};
  renderHybridGraphCanvas(panel, graph);
  const processes = Array.isArray(graph.processes) ? graph.processes : [];
  const entrypoints = Array.isArray(graph.entrypoints) ? graph.entrypoints : [];
  const candidateFiles = Array.isArray(graph.candidate_files) ? graph.candidate_files : [];
  const errors = Array.isArray(graph.errors) ? graph.errors : [];
  const hybridDebug = graph.hybrid_debug && typeof graph.hybrid_debug === "object" ? graph.hybrid_debug : {};
  const graphMeta = hybridDebug.graph_metadata && typeof hybridDebug.graph_metadata === "object"
    ? hybridDebug.graph_metadata
    : {};

  const lines = [];
  lines.push(`Repo: ${String(graph.repo || "n/a")}`);
  lines.push(`Processes: ${processes.length}`);
  if (entrypoints.length) {
    lines.push(`Entrypoints: ${entrypoints.slice(0, 6).join(", ")}`);
  }
  if (candidateFiles.length) {
    lines.push(`Candidate files (${candidateFiles.length}): ${candidateFiles.slice(0, 8).join(", ")}${candidateFiles.length > 8 ? " ..." : ""}`);
  } else {
    lines.push("Candidate files: none (fallback retrieval used)");
  }
  if (errors.length) {
    lines.push(`Graph errors: ${errors.slice(0, 3).join(" | ")}`);
  }
  if (Object.prototype.hasOwnProperty.call(hybridDebug, "graph_index_present")) {
    lines.push(`Graph index present: ${Boolean(hybridDebug.graph_index_present)}`);
  }
  if (hybridDebug.fallback_reason) {
    lines.push(`Fallback reason: ${String(hybridDebug.fallback_reason)}`);
  }
  if (graphMeta.commit_hash) {
    lines.push(`Graph commit: ${String(graphMeta.commit_hash)}`);
  }

  const impact = graph.impact && typeof graph.impact === "object" ? graph.impact : {};
  const upstream = impact.upstream && typeof impact.upstream === "object" ? impact.upstream : {};
  const downstream = impact.downstream && typeof impact.downstream === "object" ? impact.downstream : {};
  const upstreamCount = Number(upstream.impactedCount || 0);
  const downstreamCount = Number(downstream.impactedCount || 0);
  if (upstreamCount || downstreamCount) {
    lines.push(`Impact: upstream=${upstreamCount}, downstream=${downstreamCount}`);
  }

  const body = document.createElement("pre");
  body.className = "hybrid-panel-body";
  body.textContent = lines.join("\n");
  panel.appendChild(body);
}

function renderHybridEvidencePanel(panel, evidenceRows = []) {
  panel.innerHTML = "";
  const title = document.createElement("h5");
  title.textContent = "Evidence & Citations";
  panel.appendChild(title);

  if (!Array.isArray(evidenceRows) || !evidenceRows.length) {
    const empty = document.createElement("p");
    empty.className = "hybrid-panel-empty";
    empty.textContent = "No evidence rows returned.";
    panel.appendChild(empty);
    return;
  }

  const list = document.createElement("ul");
  list.className = "hybrid-evidence-list";
  for (const row of evidenceRows.slice(0, 8)) {
    const item = document.createElement("li");
    const idx = Number(row.citation_index || 0);
    const file = String(row.file_path || "unknown");
    const start = Number(row.line_start || 1);
    const end = Number(row.line_end || start);
    const snippet = String(row.snippet || "").trim();
    item.textContent = `[${idx}] ${file}:${start}-${end}${snippet ? ` — ${snippet}` : ""}`;
    list.appendChild(item);
  }
  panel.appendChild(list);
}

function isLikelyMermaidDiagram(text) {
  const normalized = normalizeDiagramText(text).trim().toLowerCase();
  return normalized.startsWith("flowchart ") || normalized.startsWith("graph ");
}

function addMessage(role, text, citations = [], meta = {}) {
  const node = template.content.cloneNode(true);
  const article = node.querySelector(".message");
  const bubble = node.querySelector(".bubble");
  const bubbleWrap = node.querySelector(".bubble-wrap");
  const citationsWrap = node.querySelector(".citations");
  const metaRow = node.querySelector(".message-meta");
  let renderedText = role === "assistant" ? stripMarkdownBold(text) : text;
  if (role === "assistant" && meta.modeValue === "diagrams") {
    renderedText = normalizeDiagramText(renderedText);
  }

  article.classList.add(role);
  bubble.textContent = renderedText;
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
      copyText(renderedText, "Answer copied");
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

    if (meta.modeValue === "hybrid") {
      const hybridPanels = document.createElement("div");
      hybridPanels.className = "hybrid-panels";

      const architecturePanel = document.createElement("section");
      architecturePanel.className = "hybrid-panel architecture";
      renderHybridArchitecturePanel(architecturePanel, meta.hybridGraph || {});

      const evidencePanel = document.createElement("section");
      evidencePanel.className = "hybrid-panel evidence";
      renderHybridEvidencePanel(evidencePanel, meta.hybridEvidence || []);

      hybridPanels.append(architecturePanel, evidencePanel);
      bubbleWrap.appendChild(hybridPanels);
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
    text: renderedText,
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

function isTransientHttpStatus(status) {
  return status === 408 || status === 425 || status === 429 || status >= 500;
}

function sleep(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

async function fetchWithRetry(url, options, retries = 1) {
  let lastError = null;
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      const response = await fetch(url, options);
      if (attempt < retries && isTransientHttpStatus(response.status)) {
        await sleep(300 * (attempt + 1));
        continue;
      }
      return response;
    } catch (err) {
      lastError = err;
      if (attempt >= retries) {
        throw err;
      }
      await sleep(300 * (attempt + 1));
    }
  }
  throw lastError || new Error("Request failed");
}

async function postJson(url, payload) {
  const response = await fetchWithRetry(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }, 2);

  let data = null;
  try {
    data = await response.json();
  } catch (_err) {
    // Ignore parse failure; handled below by status text.
  }

  if (!response.ok) {
    const fallback = `Request failed with status ${response.status}`;
    const detail = data && "detail" in data ? data.detail : null;
    throw new Error(formatApiErrorDetail(detail, fallback));
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

  const response = await fetchWithRetry(url, {
    method: "POST",
    body: formData,
  }, 2);

  let data = null;
  try {
    data = await response.json();
  } catch (_err) {
    // Ignore parse failure; handled below by status text.
  }

  if (!response.ok) {
    const fallback = `Request failed with status ${response.status}`;
    const detail = data && "detail" in data ? data.detail : null;
    throw new Error(formatApiErrorDetail(detail, fallback));
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

  if (uiMode === "hybrid") {
    const hybridDefaultTopK = Number(state.retrievalInfo?.hybrid_top_k_default || 12);
    const requestTopK = Math.min(20, Math.max(state.topK, hybridDefaultTopK));
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
      text: data.answer || "No hybrid answer returned.",
      citations: data.citations || [],
      evidence: data.evidence_strength || {},
      debug: data.debug || null,
      resultType: defaultResultTypeForMode(uiMode),
      followUps: [],
      hybridGraph: data.graph || {},
      hybridEvidence: data.evidence || [],
    };
  }

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
      resultType: defaultResultTypeForMode(
        uiMode,
        uiMode === "diagrams" ? state.activeDiagramType : null,
      ),
      followUps:
        uiMode === "audit"
          ? auditFollowUps()
          : [],
      hybridGraph: data.graph || {},
      hybridEvidence: data.evidence || [],
    };
  }

  if (uiMode === "diagrams") {
    const requestTopK = Math.min(20, Math.max(state.topK, 8));
    let data = hasUploads
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

    if (!isLikelyMermaidDiagram(data.answer || "")) {
      const strictQuestion = `${question}

Validation retry:
- Return ONLY raw Mermaid syntax (no markdown fences, no prose).
- Start with "flowchart" or "graph".
- Use concrete repository anchors and deterministic edges.`;
      data = hasUploads
        ? await postMultipart("/api/query/upload", {
            question: strictQuestion,
            topK: requestTopK,
            files,
            debug,
            mode: apiMode,
            scope,
            projectId,
            persistUploads,
          })
        : await postJson("/api/query", {
            question: strictQuestion,
            top_k: requestTopK,
            debug,
            mode: apiMode,
            scope,
            project_id: projectId,
          });
    }

    return {
      text: data.answer || "No answer returned.",
      citations: data.citations || [],
      evidence: data.evidence_strength || {},
      debug: data.debug || null,
      resultType: defaultResultTypeForMode(
        uiMode,
        uiMode === "diagrams" ? state.activeDiagramType : null,
      ),
      followUps: diagramFollowUpsForType(state.activeDiagramType),
      hybridGraph: data.graph || {},
      hybridEvidence: data.evidence || [],
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
      hybridGraph: {},
      hybridEvidence: [],
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
      hybridGraph: {},
      hybridEvidence: [],
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
    hybridGraph: {},
    hybridEvidence: [],
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
    question = buildAuditWorkflowPrompt(rawQuestion);
  } else if (currentMode === "diagrams" && !options.skipDiagramWrap) {
    const diagramType = options.diagramType || state.activeDiagramType || inferDiagramType(rawQuestion);
    state.activeDiagramType = diagramType;
    question = buildDiagramWorkflowPrompt(diagramType, rawQuestion);
  }
  const preparedQuestion = trimQuestionForBackend(question);
  question = preparedQuestion.value;

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
    skipDiagramWrap: currentMode === "diagrams",
    diagramType: state.mode === "diagrams" ? state.activeDiagramType : null,
  };
  setStatus(
    `Running ${MODE_CONFIG[state.mode].label.toLowerCase()} request${
      state.attachments.length ? ` with ${state.attachments.length} upload(s)` : ""
    }${preparedQuestion.trimmed ? " (trimmed for backend limit)" : ""}...`,
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
      hybridGraph: result.hybridGraph || {},
      hybridEvidence: result.hybridEvidence || [],
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
    const rawMessage = String(err?.message || "Request failed");
    const friendlyMessage = /failed to fetch|networkerror|load failed/i.test(rawMessage)
      ? "Network error: unable to reach the backend service. Please retry in a few seconds."
      : rawMessage;
    addMessage("assistant", `Error: ${friendlyMessage}`);
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
    skipDiagramWrap: Boolean(state.lastRequest.skipDiagramWrap),
    diagramType: state.lastRequest.diagramType || null,
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
    if (mode === "diagrams") {
      setMode("diagrams");
      openSuggestionModal({
        filterKeys: ["diagram-types", "diagram-types-advanced"],
        focusKey: "diagram-types",
      });
      return;
    }
    if (mode === "audit") {
      const dispatch = buildAuditDispatch(AUDIT_WORKFLOW.title);
      closeSuggestionModal();
      input.value = dispatch.displayText;
      autoResize();
      updateEvidencePill(null);
      setStatus(`Mode: ${MODE_CONFIG.audit.label} • Evidence: n/a • TopK: ${state.topK}`);
      submitQuestion({
        modeOverride: "audit",
        questionOverride: dispatch.workflowPrompt,
        displayOverride: dispatch.displayText,
        skipAuditWrap: true,
      });
      return;
    }
    setMode(mode);
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
  input.value = "Show example questions I can ask in Chat, Hybrid, Search, Code Patterns, Dependencies, Diagrams, and Run Audit modes.";
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
