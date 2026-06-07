const state = {
  graph: { nodes: [], edges: [], centerId: "", cy: null, evidence: [], evidenceVisible: 10 },
  user: { poemText: "", symbols: [], emotions: [], relations: [], nodes: [], edges: [], similarPoems: [], cy: null },
  options: { symbols: [], emotions: [] },
  selectedSymbols: new Set(),
  selectedEmotions: new Set(),
};

const colors = {
  symbol: "#f05a9d",
  emotion: "#b5165e",
  poem: "#d75fb8",
  author: "#a35d7a",
  node: "#9c7184",
};

const API_BASE = window.location.protocol === "file:" ? "http://127.0.0.1:8000" : "";
const EVIDENCE_PAGE_SIZE = 10;
const AUTOCOMPLETE_LIMIT = 10;

function $(id) {
  return document.getElementById(id);
}

function toast(message) {
  const el = $("toast");
  el.textContent = displayText(message);
  el.classList.add("show");
  window.setTimeout(() => el.classList.remove("show"), 2800);
}

async function postJson(url, body) {
  const response = await fetch(apiUrl(url), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function apiUrl(path) {
  if (/^https?:\/\//i.test(path)) return path;
  return `${API_BASE}${path}`;
}

function apiErrorMessage(message) {
  if (API_BASE) return `${message} Start FastAPI at ${API_BASE} or open the app at ${API_BASE}/.`;
  return `${message} Check the server console.`;
}

function setLoading(button, loading, label = "Working") {
  if (!button) return;
  if (loading) {
    button.dataset.label ||= button.innerHTML;
    button.disabled = true;
    button.setAttribute("aria-busy", "true");
    button.innerHTML = `<span class="loader" aria-hidden="true"></span><span>${escapeHtml(label)}</span>`;
    return;
  }
  button.disabled = false;
  button.removeAttribute("aria-busy");
  if (button.dataset.label) button.innerHTML = button.dataset.label;
}

function formatNodeLabel(value, maxLength = 30) {
  const text = String(value || "").trim();
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength - 3)}...`;
}

function nodeElements(nodes) {
  return nodes.map((node) => ({
    data: {
      id: node.id,
      label: formatNodeLabel(node.label || node.id),
      rawLabel: node.label || node.id,
      type: node.type || "node",
      frequency: Number(node.frequency || 1),
    },
  }));
}

function edgeElements(edges) {
  const weights = edges.map((edge) => Number(edge.weight || 1));
  const minWeight = weights.length ? Math.min(...weights) : 1;
  const maxWeight = weights.length ? Math.max(...weights) : 1;
  const spread = Math.max(1, maxWeight - minWeight);

  return edges.map((edge, index) => {
    const weight = Number(edge.weight || 1);
    const normalizedWeight = maxWeight === minWeight ? 0.55 : (weight - minWeight) / spread;
    return {
      data: {
        id: `${edge.source}|${edge.target}|${edge.type || "edge"}|${edge.poem_id || index}`,
        source: edge.source,
        target: edge.target,
        type: edge.type || "edge",
        weight,
        normalizedWeight,
      },
    };
  });
}

function renderGraph(containerId, nodes, edges, centerId, onTap) {
  const el = $(containerId);
  const cy = cytoscape({
    container: el,
    elements: [...nodeElements(nodes), ...edgeElements(edges)],
    style: [
      {
        selector: "node",
        style: {
          "background-color": (item) => colors[item.data("type")] || colors.node,
          label: "data(label)",
          color: "#221018",
          "font-family": "Aptos, Segoe UI, Helvetica Neue, Arial, sans-serif",
          "font-size": 11,
          "font-weight": 700,
          "text-wrap": "wrap",
          "text-max-width": 92,
          "text-margin-y": -10,
          "text-background-color": "#fffafd",
          "text-background-opacity": 0.92,
          "text-background-padding": 3,
          "text-background-shape": "roundrectangle",
          width: (item) => Math.max(22, Math.min(58, 18 + Math.sqrt(Number(item.data("frequency") || 1)))),
          height: (item) => Math.max(22, Math.min(58, 18 + Math.sqrt(Number(item.data("frequency") || 1)))),
          "border-color": "#fffafd",
          "border-width": 3,
          "overlay-opacity": 0,
        },
      },
      {
        selector: "node.center-node",
        style: {
          "border-color": "#221018",
          "border-width": 5,
        },
      },
      {
        selector: "node:selected",
        style: {
          "border-color": "#e5337e",
          "border-width": 5,
        },
      },
      {
        selector: "edge",
        style: {
          width: (item) => 1.8 + Number(item.data("normalizedWeight") || 0) * 8.4,
          "line-color": "#df9db6",
          "source-arrow-shape": "none",
          "target-arrow-shape": "none",
          "curve-style": "bezier",
          opacity: (item) => 0.48 + Number(item.data("normalizedWeight") || 0) * 0.42,
        },
      },
      {
        selector: "edge:selected",
        style: {
          "line-color": "#e5337e",
          opacity: 1,
        },
      },
    ],
    layout: {
      name: "cose",
      animate: false,
      fit: true,
      padding: 46,
      nodeRepulsion: 7800,
      idealEdgeLength: 118,
      edgeElasticity: 92,
      gravity: 0.28,
    },
    wheelSensitivity: 0.16,
    minZoom: 0.18,
    maxZoom: 2.4,
  });

  if (centerId) cy.getElementById(centerId).addClass("center-node");
  cy.on("tap", "node", (event) => onTap(event.target.id(), event.target.data()));
  return cy;
}

function renderEvidence(rows, reset = true) {
  state.graph.evidence = rows || [];
  if (reset) state.graph.evidenceVisible = EVIDENCE_PAGE_SIZE;
  renderEvidencePage();
}

function renderEvidencePage() {
  const el = $("evidenceList");
  const button = $("loadMoreEvidence");
  const rows = state.graph.evidence.slice(0, state.graph.evidenceVisible);

  if (!rows.length) {
    el.innerHTML = `<div class="is-empty">Relation evidence appears when graph edges include corpus excerpts.</div>`;
    if (button) button.hidden = true;
    return;
  }

  el.innerHTML = rows
    .map((row) => `
      <article class="example">
        <strong>${escapeHtml(row.source_symbol)} <span class="relation-word">to</span> ${escapeHtml(row.target_emotion)}</strong>
        <span class="muted">${escapeHtml(row.title || "Untitled")} by ${escapeHtml(row.author || "Unknown")}</span>
        <div>${escapeHtml(row.context_snippet || "")}</div>
      </article>
    `)
    .join("");

  if (button) {
    const remaining = Math.max(0, state.graph.evidence.length - state.graph.evidenceVisible);
    button.hidden = remaining === 0;
    button.textContent = remaining > EVIDENCE_PAGE_SIZE ? `Load ${EVIDENCE_PAGE_SIZE} more` : `Load ${remaining} more`;
  }
}

function loadMoreEvidence() {
  state.graph.evidenceVisible += EVIDENCE_PAGE_SIZE;
  renderEvidencePage();
}

function updateGraphStats() {
  $("graphNodeCount").textContent = `${state.graph.nodes.length} nodes`;
  $("graphEdgeCount").textContent = `${state.graph.edges.length} edges`;
}

function updateConnectionValue() {
  const value = $("topK").value;
  const output = $("topKValue");
  if (output) output.textContent = value;
}

function updateGraphSubtitle(text) {
  const el = $("graphSubtitle");
  if (el) el.textContent = displayText(text);
}

async function searchGraph() {
  const button = $("searchButton");
  setLoading(button, true, "Searching");
  updateGraphSubtitle("Searching the corpus");
  try {
    const topK = Math.max(5, Math.min(40, Number($("topK").value) || 18));
    $("topK").value = topK;
    const data = await postJson("/api/graph/search", {
      search_type: $("searchType").value,
      query: $("searchQuery").value,
      top_k: topK,
    });
    const nodes = data.nodes || [];
    const edges = data.edges || [];
    state.graph = { ...state.graph, nodes, edges, centerId: data.center_id || "" };
    if (state.graph.cy) state.graph.cy.destroy();
    state.graph.cy = renderGraph("graph", nodes, edges, state.graph.centerId, expandGraph);
    $("selectedNode").textContent = state.graph.centerId ? displayText(state.graph.centerId) : "No matching node found.";
    renderEvidence(data.evidence || [], true);
    updateGraphStats();
    updateGraphSubtitle(state.graph.centerId ? `Centered on ${state.graph.centerId}` : "No graph results");
  } catch (error) {
    updateGraphSubtitle("Search failed");
    toast(apiErrorMessage("Search failed."));
  } finally {
    setLoading(button, false);
  }
}

async function expandGraph(nodeId, nodeData) {
  $("selectedNode").textContent = displayText(`${nodeData.rawLabel || nodeData.label} (${nodeData.type})`);
  updateGraphSubtitle(`Expanded ${nodeData.rawLabel || nodeData.label}`);
  try {
    const data = await postJson("/api/graph/expand", {
      current_nodes: state.graph.nodes,
      current_edges: state.graph.edges,
      node_id: nodeId,
      top_k: 12,
    });
    state.graph = {
      ...state.graph,
      nodes: data.nodes || [],
      edges: data.edges || [],
      centerId: data.center_id || nodeId,
    };
    if (state.graph.cy) state.graph.cy.destroy();
    state.graph.cy = renderGraph("graph", state.graph.nodes, state.graph.edges, state.graph.centerId, expandGraph);
    renderEvidence(data.evidence || [], true);
    updateGraphStats();
  } catch (error) {
    toast(apiErrorMessage("Could not expand this node."));
  }
}

function chip(text, kind = "", options = {}) {
  const value = options.value ? ` data-value="${escapeAttr(options.value)}"` : "";
  const remove = options.removable ? `<span class="chip-remove" aria-hidden="true">x</span>` : "";
  return `<span class="chip ${kind}"${value}>${escapeHtml(text)}${remove}</span>`;
}

function uniqueValues(rows, key, limit = 25) {
  const seen = new Set();
  const values = [];
  for (const row of rows || []) {
    const value = row[key];
    if (value && !seen.has(value)) {
      seen.add(value);
      values.push(value);
    }
    if (values.length === limit) break;
  }
  return values;
}

function walkableUserEdges() {
  const nodeIds = new Set((state.user.nodes || []).map((node) => node.id));
  return (state.user.edges || []).filter((edge) =>
    edge.source &&
    edge.target &&
    edge.source !== edge.target &&
    nodeIds.has(edge.source) &&
    nodeIds.has(edge.target)
  );
}

function hasWalkableUserGraph() {
  return walkableUserEdges().length > 0;
}

function walkableUserNodes() {
  const connectedIds = new Set();
  walkableUserEdges().forEach((edge) => {
    connectedIds.add(edge.source);
    connectedIds.add(edge.target);
  });
  return (state.user.nodes || []).filter((node) => connectedIds.has(node.id));
}

function updateWalkAvailability() {
  const button = $("walkButton");
  if (!button) return;
  const ready = hasWalkableUserGraph();
  button.disabled = !ready;
  button.classList.toggle("is-disabled", !ready);
  button.title = ready
    ? "Generate from a connected path in your poem graph"
    : "Analyze a poem with at least one symbol-emotion connection first";
}

function renderAnalyzeResults(data) {
  state.user = { ...state.user, ...data, poemText: $("poemText").value };
  const symbols = uniqueValues(data.symbols, "symbol");
  const emotions = uniqueValues(data.emotions, "emotion_category");
  const similarPoems = data.similar_poems || [];

  $("symbolChips").innerHTML = symbols.length
    ? symbols.map((value) => chip(value, "symbol")).join("")
    : `<span class="is-empty">No clear symbols found.</span>`;
  $("emotionChips").innerHTML = emotions.length
    ? emotions.map((value) => chip(value, "emotion")).join("")
    : `<span class="is-empty">No lexicon emotions found.</span>`;
  $("similarPoems").innerHTML = similarPoems.length
    ? similarPoems.map((poem) => `
      <article class="poem-card">
        <strong>${escapeHtml(poem.title || "Untitled")}</strong>
        <span class="muted">${escapeHtml(poem.author || "Unknown")}</span>
        <div>${escapeHtml(poem.poem_text || "").slice(0, 700)}</div>
      </article>
    `).join("")
    : `<div class="is-empty">Similar poems appear when embeddings are available.</div>`;

  if (state.user.cy) state.user.cy.destroy();
  state.user.cy = renderGraph("userGraph", data.nodes || [], data.edges || [], data.nodes?.[0]?.id || "", () => {});
  updateWalkAvailability();
  if (!hasWalkableUserGraph()) {
    $("walkResult").innerHTML = `
      <strong>No connected path yet</strong>
      <div>Random walk needs at least one symbol close to an emotion in your poem.</div>
    `;
  }
  setAnalysisResultsVisible(true);
}

async function analyzePoem() {
  const button = $("analyzeButton");
  setLoading(button, true, "Analyzing");
  $("walkResult").textContent = "";
  try {
    const data = await postJson("/api/analyze", { poem_text: $("poemText").value });
    renderAnalyzeResults(data);
  } catch (error) {
    toast(apiErrorMessage("Analysis failed. spaCy or model setup may need attention."));
  } finally {
    setLoading(button, false);
  }
}

async function randomWalk() {
  if (!hasWalkableUserGraph()) {
    updateWalkAvailability();
    $("walkResult").innerHTML = `
      <strong>No connected path yet</strong>
      <div>Random walk needs at least one symbol-emotion connection. Try placing an image word near an emotion word, then analyze again.</div>
    `;
    return;
  }

  const button = $("walkButton");
  setLoading(button, true, "Walking");
  try {
    const data = await postJson("/api/random-walk", {
      poem_text: $("poemText").value,
      nodes: walkableUserNodes(),
      edges: walkableUserEdges(),
      relations: state.user.relations,
      steps: 5,
    });
    $("walkResult").innerHTML = `
      <strong>${data.path?.length ? escapeHtml(data.path.join(" to ")) : "Random walk"}</strong>
      <div>${escapeHtml(data.poem || data.message || "")}</div>
    `;
  } catch (error) {
    toast(apiErrorMessage("Random walk generation failed."));
  } finally {
    setLoading(button, false);
    updateWalkAvailability();
  }
}

async function loadOptions() {
  const response = await fetch(apiUrl("/api/options"));
  if (!response.ok) throw new Error(await response.text());
  state.options = await response.json();
  renderSelectedTerms();
}

function renderSelectedTerms() {
  $("selectedSymbols").innerHTML = state.selectedSymbols.size
    ? [...state.selectedSymbols].map((item) => chip(item, "symbol", { value: item, removable: true })).join("")
    : `<span class="is-empty">No symbols selected.</span>`;
  $("selectedEmotions").innerHTML = state.selectedEmotions.size
    ? [...state.selectedEmotions].map((item) => chip(item, "emotion", { value: item, removable: true })).join("")
    : `<span class="is-empty">No emotions selected.</span>`;
}

function addTermFromInput(input, set, options) {
  const value = input.value.trim().toLowerCase();
  if (!value) return;
  const exact = options.find((item) => item.toLowerCase() === value) || value;
  set.add(exact);
  input.value = "";
  renderSelectedTerms();
  hideAutocompleteMenus();
}

function autocompleteMatches(value, options, selectedSet) {
  const query = value.trim().toLowerCase();
  const available = options.filter((item) => !selectedSet.has(item));
  if (!query) return available.slice(0, AUTOCOMPLETE_LIMIT);
  return available
    .filter((item) => item.toLowerCase().includes(query))
    .sort((first, second) => {
      const a = first.toLowerCase();
      const b = second.toLowerCase();
      const aStarts = a.startsWith(query) ? 0 : 1;
      const bStarts = b.startsWith(query) ? 0 : 1;
      return aStarts - bStarts || a.localeCompare(b);
    })
    .slice(0, AUTOCOMPLETE_LIMIT);
}

function hideAutocompleteMenus() {
  document.querySelectorAll(".autocomplete-menu").forEach((menu) => {
    menu.hidden = true;
  });
  document.querySelectorAll(".autocomplete-field.is-open").forEach((field) => {
    field.classList.remove("is-open");
  });
  document.querySelectorAll("[role='combobox']").forEach((input) => {
    input.setAttribute("aria-expanded", "false");
  });
}

function renderAutocompleteMenu(input, menu, options, selectedSet) {
  const field = input.closest(".autocomplete-field");
  const matches = autocompleteMatches(input.value, options, selectedSet);
  if (!matches.length) {
    menu.hidden = true;
    field?.classList.remove("is-open");
    input.setAttribute("aria-expanded", "false");
    return;
  }
  menu.innerHTML = matches
    .map((item) => `<button class="autocomplete-option" type="button" role="option" data-value="${escapeAttr(item)}">${escapeHtml(item)}</button>`)
    .join("");
  menu.hidden = false;
  field?.classList.add("is-open");
  input.setAttribute("aria-expanded", "true");
}

function bindAutocomplete(inputId, menuId, selectedSet, optionsGetter) {
  const input = $(inputId);
  const menu = $(menuId);
  if (!input || !menu) return;

  const render = () => renderAutocompleteMenu(input, menu, optionsGetter(), selectedSet);
  input.addEventListener("input", render);
  input.addEventListener("focus", render);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      addTermFromInput(input, selectedSet, optionsGetter());
    }
    if (event.key === "Escape") hideAutocompleteMenus();
  });
  menu.addEventListener("mousedown", (event) => event.preventDefault());
  menu.addEventListener("click", (event) => {
    const option = event.target.closest(".autocomplete-option");
    if (!option) return;
    input.value = option.dataset.value || "";
    addTermFromInput(input, selectedSet, optionsGetter());
  });
}

async function generatePoem() {
  const button = $("generateButton");
  setLoading(button, true, "Writing");
  $("generatedPoem").textContent = "Writing from selected graph signals...";
  try {
    const data = await postJson("/api/generate", {
      symbols: [...state.selectedSymbols],
      emotions: [...state.selectedEmotions],
      style: $("styleInput").value,
      length: $("lengthInput").value,
    });
    $("generatedPoem").textContent = displayText(data.poem || data.message || "No poem returned.");
  } catch (error) {
    $("generatedPoem").textContent = apiErrorMessage("Generation failed.");
  } finally {
    setLoading(button, false);
  }
}

function escapeHtml(value) {
  return displayText(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttr(value) {
  return escapeHtml(value).replaceAll("`", "&#096;");
}

function setAnalysisResultsVisible(visible) {
  $("analysisWorkspace")?.classList.toggle("pending", !visible);
  document.querySelectorAll(".analysis-results").forEach((section) => {
    section.classList.toggle("revealed", visible);
    section.setAttribute("aria-hidden", visible ? "false" : "true");
  });
  if (visible) {
    [0, 180, 640].forEach((delay) => {
      window.setTimeout(() => {
        state.user.cy?.resize();
        state.user.cy?.fit(undefined, 46);
      }, delay);
    });
  }
}

function displayText(value) {
  return String(value ?? "")
    .replaceAll("\u2014", " - ")
    .replaceAll("\u2013", " - ")
    .replaceAll("\u2192", " to ")
    .replaceAll("\u2190", " from ")
    .replaceAll("\u2191", " up ")
    .replaceAll("\u2193", " down ")
    .replaceAll("\u2026", "...");
}

function bindEvents() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab, .page").forEach((el) => el.classList.remove("active"));
      tab.classList.add("active");
      $(tab.dataset.page).classList.add("active");
      window.requestAnimationFrame(() => {
        state.graph.cy?.resize();
        state.user.cy?.resize();
        state.graph.cy?.fit(undefined, 46);
        state.user.cy?.fit(undefined, 46);
      });
    });
  });

  $("searchButton").addEventListener("click", searchGraph);
  $("searchQuery").addEventListener("keydown", (event) => {
    if (event.key === "Enter") searchGraph();
  });
  $("topK").addEventListener("input", updateConnectionValue);
  $("loadMoreEvidence").addEventListener("click", loadMoreEvidence);
  $("analyzeButton").addEventListener("click", analyzePoem);
  $("walkButton").addEventListener("click", randomWalk);
  $("generateButton").addEventListener("click", generatePoem);

  bindAutocomplete("symbolFilter", "symbolMenu", state.selectedSymbols, () => state.options.symbols);
  bindAutocomplete("emotionFilter", "emotionMenu", state.selectedEmotions, () => state.options.emotions);
  document.addEventListener("pointerdown", (event) => {
    if (!event.target.closest(".autocomplete-field")) hideAutocompleteMenus();
  });

  $("selectedSymbols").addEventListener("click", (event) => {
    const selectedChip = event.target.closest(".chip");
    if (!selectedChip) return;
    const value = selectedChip.dataset.value;
    if (value) state.selectedSymbols.delete(value);
    renderSelectedTerms();
  });
  $("selectedEmotions").addEventListener("click", (event) => {
    const selectedChip = event.target.closest(".chip");
    if (!selectedChip) return;
    const value = selectedChip.dataset.value;
    if (value) state.selectedEmotions.delete(value);
    renderSelectedTerms();
  });
}

bindEvents();
updateConnectionValue();
setAnalysisResultsVisible(false);
updateWalkAvailability();
loadOptions().catch(() => toast(apiErrorMessage("Could not load term options.")));
searchGraph();
