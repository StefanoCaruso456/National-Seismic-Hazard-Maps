const chatThread = document.getElementById("chatThread");
const hero = document.getElementById("hero");
const input = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const micBtn = document.getElementById("micBtn");
const attachBtn = document.getElementById("attachBtn");
const fileInput = document.getElementById("fileInput");
const newChatBtn = document.getElementById("newChatBtn");
const sidebar = document.getElementById("sidebar");
const toggleSidebar = document.getElementById("toggleSidebar");
const template = document.getElementById("messageTemplate");

let listening = false;
let recognition;

function autoResize() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
}

function citationText(c) {
  const file = c.file_path || "unknown";
  const start = c.line_start ?? "?";
  const end = c.line_end ?? "?";
  const score = typeof c.score === "number" ? c.score.toFixed(3) : "n/a";
  return `${file}:${start}-${end} (score ${score})`;
}

function renderCitation(citationsWrap, item) {
  const badge = document.createElement("span");
  badge.textContent = citationText(item);
  citationsWrap.appendChild(badge);

  if (!item.snippet) return;

  const snippet = document.createElement("pre");
  snippet.className = "citation-snippet";
  snippet.textContent = item.snippet;
  citationsWrap.appendChild(snippet);
}

function addMessage(role, text, citations = []) {
  const node = template.content.cloneNode(true);
  const article = node.querySelector(".message");
  const bubble = node.querySelector(".bubble");
  const citationsWrap = node.querySelector(".citations");

  article.classList.add(role);
  bubble.textContent = text;

  citations.forEach((item) => renderCitation(citationsWrap, item));

  chatThread.appendChild(node);
  chatThread.scrollTop = chatThread.scrollHeight;
}

async function submitQuestion() {
  const question = input.value.trim();
  if (!question) return;

  hero.style.display = "none";
  addMessage("user", question);
  input.value = "";
  autoResize();
  sendBtn.disabled = true;

  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 5 }),
    });

    if (!res.ok) {
      throw new Error(`Request failed with status ${res.status}`);
    }

    const data = await res.json();
    addMessage("assistant", data.answer || "No answer returned.", data.citations || []);
  } catch (err) {
    addMessage("assistant", `Error: ${err.message}`);
  } finally {
    sendBtn.disabled = false;
  }
}

function initSpeech() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    micBtn.disabled = true;
    micBtn.title = "Speech recognition unavailable in this browser";
    return;
  }

  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.lang = "en-US";

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    input.value = `${input.value} ${transcript}`.trim();
    autoResize();
  };

  recognition.onend = () => {
    listening = false;
    micBtn.textContent = "🎙";
  };

  recognition.onerror = () => {
    listening = false;
    micBtn.textContent = "🎙";
  };
}

input.addEventListener("input", autoResize);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    submitQuestion();
  }
});

sendBtn.addEventListener("click", submitQuestion);

attachBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) {
    addMessage("assistant", `Attached file: ${fileInput.files[0].name} (upload handling not implemented yet).`);
  }
});

newChatBtn.addEventListener("click", () => {
  chatThread.innerHTML = "";
  hero.style.display = "block";
  input.value = "";
  autoResize();
});

toggleSidebar.addEventListener("click", () => {
  sidebar.classList.toggle("open");
});

micBtn.addEventListener("click", () => {
  if (!recognition) return;

  if (!listening) {
    listening = true;
    micBtn.textContent = "■";
    recognition.start();
  } else {
    listening = false;
    micBtn.textContent = "🎙";
    recognition.stop();
  }
});

initSpeech();
autoResize();
