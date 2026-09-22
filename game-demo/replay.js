"use strict";

(() => {
  const INDEX_SCHEMA = "worthify-game-index-v1";
  const REPLAY_SCHEMA = "worthify-game-replay-v1";
  const TICK_MS = 1200;
  const REPLAY_ROOT = new URL(".", document.currentScript.src);
  const INDEX_URL = new URL("runs/index.json", REPLAY_ROOT);

  const ui = {
    episodeSelect: document.querySelector("#episode-select"),
    gameBadge: document.querySelector("#game-badge"),
    smokeWarning: document.querySelector("#smoke-warning"),
    status: document.querySelector("#status"),
    statusTitle: document.querySelector("#replay-title"),
    statusDetail: document.querySelector("#status-detail"),
    player: document.querySelector("#player"),
    frame: document.querySelector("#game-frame"),
    frameNote: document.querySelector("#frame-note"),
    stepLabel: document.querySelector("#step-label"),
    scoreLabel: document.querySelector("#score-label"),
    restartButton: document.querySelector("#restart-button"),
    playButton: document.querySelector("#play-button"),
    timeline: document.querySelector("#timeline"),
    positionLabel: document.querySelector("#position-label"),
    speedSelect: document.querySelector("#speed-select"),
    actionMode: document.querySelector("#action-mode"),
    actionTitle: document.querySelector("#action-title"),
    stateText: document.querySelector("#state-text"),
    optionList: document.querySelector("#option-list"),
    factGame: document.querySelector("#fact-game"),
    factSeed: document.querySelector("#fact-seed"),
    factController: document.querySelector("#fact-controller"),
    modelFact: document.querySelector("#model-fact"),
    factModel: document.querySelector("#fact-model"),
    adapterFact: document.querySelector("#adapter-fact"),
    factAdapter: document.querySelector("#fact-adapter"),
    factReward: document.querySelector("#fact-reward"),
    factEnded: document.querySelector("#fact-ended"),
    interpretationNote: document.querySelector("#interpretation-note"),
    limitationsList: document.querySelector("#limitations-list")
  };

  let episode = null;
  let episodeUrl = null;
  let position = 0;
  let timer = null;
  let loadSequence = 0;

  function isRecord(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  function isFiniteNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function requireCondition(condition, message) {
    if (!condition) throw new Error(message);
  }

  function safeEpisodeHref(href) {
    if (typeof href !== "string" || href.length > 240 || href.includes("\\") || href.includes("%") || href.includes("?") || href.includes("#")) return false;
    const parts = href.split("/");
    return parts.length >= 2 && parts.every((part) => /^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(part) && part !== "." && part !== "..") && parts.at(-1) === "episode.json";
  }

  function safeFramePath(path) {
    return typeof path === "string" && /^frames\/[A-Za-z0-9][A-Za-z0-9._-]*\.(?:png|jpe?g|webp)$/.test(path);
  }

  function validateIndex(data) {
    requireCondition(isRecord(data) && data.schema === INDEX_SCHEMA, "Unsupported recording index.");
    requireCondition(Array.isArray(data.episodes), "Recording index has no episode list.");
    return data.episodes.map((item, index) => {
      requireCondition(isRecord(item), `Episode entry ${index + 1} is malformed.`);
      requireCondition(typeof item.title === "string" && item.title.trim().length > 0 && item.title.length <= 120, `Episode entry ${index + 1} has an invalid title.`);
      requireCondition(item.game === "doom" || item.game === "tetris", `Episode entry ${index + 1} has an unknown game.`);
      requireCondition(safeEpisodeHref(item.href), `Episode entry ${index + 1} has an unsafe path.`);
      return { title: item.title, game: item.game, href: item.href };
    });
  }

  function validateReplay(data, indexEntry) {
    requireCondition(isRecord(data) && data.schema === REPLAY_SCHEMA, "Unsupported episode format.");
    requireCondition(data.game === "doom" || data.game === "tetris", "Episode has an unknown game.");
    requireCondition(data.game === indexEntry.game, "Episode game does not match the recording index.");
    requireCondition(Number.isInteger(data.seed), "Episode seed is invalid.");
    requireCondition(isRecord(data.controller), "Episode controller metadata is missing.");
    requireCondition(data.controller.kind === "model" || data.controller.kind === "smoke", "Episode controller kind is invalid.");
    const modelRun = data.controller.kind === "model";
    if (modelRun) {
      requireCondition(data.controller.selection === "argmax", "Model selection method is unsupported.");
      requireCondition(isRecord(data.controller.model), "Episode model metadata is missing.");
      for (const field of ["source", "revision", "adapter", "adapter_revision", "adapter_sha256"]) {
        requireCondition(typeof data.controller.model[field] === "string" && data.controller.model[field].length > 0, `Episode model ${field} is invalid.`);
      }
    } else {
      requireCondition(data.controller.selection === "seeded_random", "Smoke-test selection method is unsupported.");
      requireCondition(data.controller.model === null, "Smoke tests cannot claim model provenance.");
    }
    requireCondition(isRecord(data.environment), "Episode environment metadata is missing.");
    requireCondition(safeFramePath(data.initial_frame), "Episode initial frame path is invalid.");
    requireCondition(Array.isArray(data.decisions) && data.decisions.length > 0, "Episode has no decisions.");
    requireCondition(data.decisions.length <= 100000, "Episode is too large to replay safely.");

    let expectedBefore = data.initial_frame;
    let sawTermination = false;
    data.decisions.forEach((decision, index) => {
      const label = `Decision ${index}`;
      requireCondition(isRecord(decision), `${label} is malformed.`);
      requireCondition(decision.step === index, `${label} has an invalid step number.`);
      requireCondition(typeof decision.state === "string" && decision.state.length <= 100000, `${label} state is invalid.`);
      requireCondition(typeof decision.question === "string" && decision.question.length > 0 && decision.question.length <= 10000, `${label} question is invalid.`);
      requireCondition(Array.isArray(decision.options) && decision.options.length >= 2 && decision.options.length <= 16, `${label} options are invalid.`);
      const optionIds = new Set();
      decision.options.forEach((option) => {
        requireCondition(isRecord(option) && typeof option.id === "string" && option.id.length > 0 && option.id.length <= 200, `${label} has an invalid option ID.`);
        requireCondition(typeof option.description === "string" && option.description.length <= 10000, `${label} has an invalid option description.`);
        requireCondition(!optionIds.has(option.id), `${label} has duplicate option IDs.`);
        optionIds.add(option.id);
      });
      requireCondition(safeFramePath(decision.frame_before) && safeFramePath(decision.frame_after), `${label} has an invalid frame path.`);
      requireCondition(decision.frame_before === expectedBefore, `${label} does not continue from the preceding frame.`);
      expectedBefore = decision.frame_after;
      requireCondition(optionIds.has(decision.selected_option_id), `${label} selected option is missing.`);
      if (modelRun) {
        requireCondition(Array.isArray(decision.probabilities) && decision.probabilities.length === decision.options.length, `${label} probabilities do not match its options.`);
        requireCondition(decision.probabilities.every((value) => isFiniteNumber(value) && value >= 0 && value <= 1), `${label} has invalid probabilities.`);
        const probabilityTotal = decision.probabilities.reduce((total, value) => total + value, 0);
        requireCondition(Math.abs(probabilityTotal - 1) <= 0.02, `${label} probabilities do not sum to one.`);
        requireCondition(Array.isArray(decision.option_logits) && decision.option_logits.length === decision.options.length, `${label} logits do not match its options.`);
        requireCondition(decision.option_logits.every(isFiniteNumber), `${label} has invalid logits.`);
        requireCondition(isFiniteNumber(decision.inference_seconds) && decision.inference_seconds >= 0, `${label} inference time is invalid.`);
        const selectedIndex = decision.options.findIndex((option) => option.id === decision.selected_option_id);
        requireCondition(decision.probabilities[selectedIndex] === Math.max(...decision.probabilities), `${label} selected action is not a model argmax.`);
      } else {
        requireCondition(Array.isArray(decision.probabilities) && decision.probabilities.length === 0, `${label} smoke test claims model probabilities.`);
        requireCondition(Array.isArray(decision.option_logits) && decision.option_logits.length === 0, `${label} smoke test claims model logits.`);
        requireCondition(decision.inference_seconds === null, `${label} smoke test claims model timing.`);
      }
      requireCondition(isFiniteNumber(decision.reward), `${label} reward is invalid.`);
      requireCondition(typeof decision.terminated === "boolean", `${label} termination flag is invalid.`);
      requireCondition(isRecord(decision.summary), `${label} summary is invalid.`);
      requireCondition(!sawTermination, `${label} appears after a terminated decision.`);
      sawTermination = decision.terminated;
    });

    requireCondition(isRecord(data.summary), "Episode summary is invalid.");
    requireCondition(data.ended_reason === "terminated" || data.ended_reason === "step_limit", "Episode end reason is invalid.");
    const finalDecision = data.decisions.at(-1);
    const boundedReason = finalDecision.summary.termination_reason === "max_steps" || finalDecision.summary.termination_reason === "step_limit";
    const expectedEnd = !finalDecision.terminated || boundedReason ? "step_limit" : "terminated";
    requireCondition(data.ended_reason === expectedEnd, "Episode end reason conflicts with the final decision.");
    requireCondition(Array.isArray(data.limitations) && data.limitations.every((item) => typeof item === "string" && item.length <= 10000), "Episode limitations are invalid.");
    return data;
  }

  function setStatus(title, detail) {
    stopPlayback();
    ui.player.hidden = true;
    ui.status.hidden = false;
    ui.statusTitle.textContent = title;
    ui.statusDetail.textContent = detail;
  }

  function setPlaying(playing) {
    ui.playButton.setAttribute("aria-label", playing ? "Pause replay" : "Play replay");
    ui.playButton.firstChild.textContent = playing ? "❚❚ " : "▶ ";
    ui.playButton.querySelector("span").textContent = playing ? "Pause" : "Play";
  }

  function stopPlayback() {
    if (timer !== null) window.clearTimeout(timer);
    timer = null;
    setPlaying(false);
  }

  function scheduleNext() {
    if (!episode || position >= episode.decisions.length) {
      stopPlayback();
      return;
    }
    const speed = Number(ui.speedSelect.value);
    timer = window.setTimeout(() => {
      timer = null;
      position += 1;
      renderPosition();
      scheduleNext();
    }, TICK_MS / speed);
  }

  function summaryScore(summary) {
    for (const key of ["score", "game_score", "total_score", "lines", "kills"]) {
      if (typeof summary[key] === "string" || isFiniteNumber(summary[key])) {
        const label = key.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());
        return `${label} ${String(summary[key])}`;
      }
    }
    return "Score unavailable";
  }

  function readableReason(reason) {
    if (typeof reason !== "string" || reason.length === 0) return null;
    return reason.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());
  }

  function formatEndReason(replay, summary) {
    const recordedReason = readableReason(summary.termination_reason);
    if (recordedReason) return recordedReason;
    return replay.ended_reason === "step_limit" ? "Step limit" : "Environment terminated";
  }

  function renderOptions(decision) {
    ui.optionList.replaceChildren();
    decision.options.forEach((option, index) => {
      const row = document.createElement("div");
      row.className = "option";
      if (option.id === decision.selected_option_id) {
        row.classList.add("selected");
        row.setAttribute("aria-current", "true");
      }

      const name = document.createElement("div");
      name.className = "option-name";
      const id = document.createElement("span");
      id.className = "option-id";
      id.textContent = option.id;
      const description = document.createElement("span");
      description.className = "option-description";
      description.textContent = option.description;
      name.append(id, description);

      const score = document.createElement("span");
      score.className = "option-score";
      score.textContent = episode.controller.kind === "model"
        ? `${(decision.probabilities[index] * 100).toFixed(1)}% · logit ${decision.option_logits[index].toFixed(3)}`
        : (option.id === decision.selected_option_id ? "scripted selection" : "no model score");
      row.append(name, score);
      ui.optionList.append(row);
    });
  }

  function frameUrl(path) {
    return new URL(path, episodeUrl).href;
  }

  function renderPosition() {
    if (!episode) return;
    const total = episode.decisions.length;
    const initial = position === 0;
    const decision = initial ? episode.decisions[0] : episode.decisions[position - 1];
    const shownFrame = initial ? episode.initial_frame : decision.frame_after;
    const shownSummary = initial ? {} : decision.summary;
    const preterminal = shownSummary.last_frame_is_preterminal === true;

    ui.frame.src = frameUrl(shownFrame);
    ui.frame.alt = `${episode.game === "doom" ? "Doom" : "Tetris"} recorded frame at replay position ${position} of ${total}`;
    ui.stepLabel.textContent = initial ? "Initial state · next: step 0" : `After step ${decision.step}`;
    ui.scoreLabel.textContent = position === total && preterminal
      ? "Score — final telemetry unavailable"
      : summaryScore(position === total ? episode.summary : shownSummary);
    ui.frameNote.hidden = !preterminal;
    ui.frameNote.textContent = preterminal
      ? "Last available frame is preterminal; displayed telemetry may also be preterminal."
      : "";
    ui.positionLabel.textContent = `${position} / ${total}`;
    ui.timeline.value = String(position);
    ui.actionMode.textContent = initial ? "Next decision" : "Action taken";
    const selected = decision.options.find((option) => option.id === decision.selected_option_id);
    ui.actionTitle.textContent = `${decision.question} — ${selected.id}: ${selected.description}`;
    ui.stateText.textContent = decision.state;
    renderOptions(decision);
    ui.factReward.textContent = initial ? "—" : String(decision.reward);
    ui.factEnded.textContent = position === total ? formatEndReason(episode, decision.summary) : "In progress";
    ui.playButton.disabled = false;
  }

  function renderEpisode() {
    stopPlayback();
    position = 0;
    ui.status.hidden = true;
    ui.player.hidden = false;
    ui.gameBadge.hidden = false;
    ui.gameBadge.textContent = episode.game;
    ui.smokeWarning.hidden = episode.controller.kind === "model";
    ui.timeline.max = String(episode.decisions.length);
    ui.factGame.textContent = episode.game === "doom" ? "Doom" : "Tetris";
    ui.factSeed.textContent = String(episode.seed);
    ui.factController.textContent = episode.controller.kind === "model" ? "Model · argmax" : "Scripted smoke test";
    ui.modelFact.hidden = episode.controller.kind !== "model";
    ui.adapterFact.hidden = episode.controller.kind !== "model";
    if (episode.controller.kind === "model") {
      const model = episode.controller.model;
      ui.factModel.textContent = `${model.source} @ ${model.revision}`;
      ui.factAdapter.textContent = `${model.adapter} @ ${model.adapter_revision} · sha256 ${model.adapter_sha256}`;
    } else {
      ui.factModel.textContent = "—";
      ui.factAdapter.textContent = "—";
    }
    ui.interpretationNote.textContent = episode.controller.kind === "model"
      ? "Option probabilities are conditional and uncalibrated. The controller received text game state only; frames are shown for human review."
      : "This scripted smoke test has no model probabilities or logits. Its controller received text game state only; frames are shown for plumbing review.";
    ui.limitationsList.replaceChildren();
    episode.limitations.forEach((text) => {
      const item = document.createElement("li");
      item.textContent = text;
      ui.limitationsList.append(item);
    });
    renderPosition();
  }

  async function fetchJson(url) {
    const response = await fetch(url, { credentials: "same-origin" });
    if (!response.ok) throw new Error(`Request failed (${response.status}).`);
    return response.json();
  }

  async function loadEpisode(indexEntry) {
    const sequence = ++loadSequence;
    setStatus("Loading recorded episode", "Validating frames, decisions, and controller metadata…");
    ui.smokeWarning.hidden = true;
    ui.gameBadge.hidden = true;
    try {
      const url = new URL(indexEntry.href, INDEX_URL);
      const data = validateReplay(await fetchJson(url), indexEntry);
      if (sequence !== loadSequence) return;
      episodeUrl = url;
      episode = data;
      renderEpisode();
    } catch (error) {
      if (sequence !== loadSequence) return;
      episode = null;
      setStatus("Recording unavailable", error instanceof Error ? error.message : "The episode could not be validated.");
    }
  }

  async function initialize() {
    try {
      const entries = validateIndex(await fetchJson(INDEX_URL));
      ui.episodeSelect.replaceChildren();
      if (entries.length === 0) {
        const option = document.createElement("option");
        option.textContent = "No recordings published";
        ui.episodeSelect.append(option);
        setStatus("Recordings are coming", "Training and recording are still in progress. No completed episodes are published yet.");
        return;
      }
      entries.forEach((entry, index) => {
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = entry.title;
        ui.episodeSelect.append(option);
      });
      ui.episodeSelect.disabled = false;
      ui.episodeSelect.addEventListener("change", () => loadEpisode(entries[Number(ui.episodeSelect.value)]));
      await loadEpisode(entries[0]);
    } catch (error) {
      setStatus("Replay archive unavailable", error instanceof Error ? error.message : "The recording index could not be read.");
    }
  }

  ui.playButton.addEventListener("click", () => {
    if (!episode) return;
    if (timer !== null) {
      stopPlayback();
      return;
    }
    if (position >= episode.decisions.length) position = 0;
    renderPosition();
    setPlaying(true);
    scheduleNext();
  });

  ui.restartButton.addEventListener("click", () => {
    if (!episode) return;
    stopPlayback();
    position = 0;
    renderPosition();
  });

  ui.timeline.addEventListener("input", () => {
    if (!episode) return;
    stopPlayback();
    position = Number(ui.timeline.value);
    renderPosition();
  });

  ui.speedSelect.addEventListener("change", () => {
    if (timer === null) return;
    window.clearTimeout(timer);
    timer = null;
    setPlaying(true);
    scheduleNext();
  });

  ui.frame.addEventListener("error", () => {
    if (!episode) return;
    setStatus("Recording frame unavailable", "A referenced frame could not be loaded. Playback stopped.");
  });

  initialize();
})();
