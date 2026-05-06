// Single-file vanilla JS viewer for the Razer customer-companion benchmark.
// Loads the JSON written by `python -m sonzai_razer_bench`, then renders:
//   - overview cards (backend, elapsed, accuracy)
//   - household roster
//   - per-category accuracy table
//   - phase-1 conversation timeline (collapsible per session, mood + judge per turn)
//   - phase-2 QA list (filterable, color-coded)

(() => {
  const $ = (sel) => document.querySelector(sel);

  const escapeHTML = (s) =>
    String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

  function scoreClass(score) {
    if (score >= 8) return "good";
    if (score >= 5) return "warn";
    return "bad";
  }

  function formatMoodChip(label, value) {
    if (value === undefined || value === null) return "";
    const v = typeof value === "number" ? value.toFixed(2) : value;
    return `<span class="mood-chip" title="${escapeHTML(label)}=${escapeHTML(v)}">${escapeHTML(label[0])}=${escapeHTML(v)}</span>`;
  }

  function moodStrip(mood) {
    if (!mood || Object.keys(mood).length === 0) return "";
    const dims = [
      ["valence", mood.valence],
      ["arousal", mood.arousal],
      ["tension", mood.tension],
      ["affiliation", mood.affiliation],
    ];
    const chips = dims
      .filter(([_, v]) => v !== undefined && v !== null)
      .map(([k, v]) => formatMoodChip(k, v))
      .join("");
    const label = mood.label ? `<span class="mood-chip" title="agent's emotional label">${escapeHTML(mood.label)}</span>` : "";
    if (!chips && !label) return "";
    return `<div class="mood-strip">${label}${chips}</div>`;
  }

  function renderTurn(t) {
    const score = t.judge_score ?? 0;
    const cls = scoreClass(score);
    const safetyBad = t.judge_handled_safely === false;
    const inCharBad = t.judge_in_character === false;
    const hasJudge = t.judge_score > 0 || t.judge_rationale;
    const meta = hasJudge
      ? `<div class="turn-meta">
          <span class="score ${cls}">judge ${score.toFixed(1)}/10</span>
          ${inCharBad ? '<span class="tag bad">off-character</span>' : '<span class="tag good">in-character</span>'}
          ${safetyBad ? '<span class="tag bad">safety/privacy</span>' : ''}
        </div>`
      : "";
    const rationale = t.judge_rationale
      ? `<div class="judge-rationale">${escapeHTML(t.judge_rationale)}</div>`
      : "";
    const reference = t.reference_text
      ? `<div class="reference"><span class="reference-label">reference</span>${escapeHTML(t.reference_text)}</div>`
      : "";
    return `
      <div class="turn user">
        <div class="turn-label">user</div>
        <div class="bubble">${escapeHTML(t.user_text)}</div>
      </div>
      <div class="turn agent">
        <div class="turn-label agent">agent</div>
        <div class="bubble">${escapeHTML(t.agent_text || "(no response)")}</div>
      </div>
      ${meta}
      ${moodStrip(t.mood_after) || moodStrip(t.mood_before)}
      ${rationale}
      ${reference}
    `;
  }

  function sessionAvgScore(s) {
    if (!s.turns || s.turns.length === 0) return 0;
    const valid = s.turns.filter((t) => t.judge_score > 0);
    if (valid.length === 0) return 0;
    return valid.reduce((a, t) => a + t.judge_score, 0) / valid.length;
  }

  function renderSession(s, idx) {
    const avg = sessionAvgScore(s);
    const avgPill = avg
      ? `<span class="session-score-pill score ${scoreClass(avg)}">avg ${avg.toFixed(1)}/10</span>`
      : "";
    const ts = s.timestamp ? new Date(s.timestamp).toLocaleString() : "";
    return `
      <details class="session" ${idx === 0 ? "open" : ""}>
        <summary class="session-header">
          <div class="left">
            <span class="session-pill">${escapeHTML(s.session_id)}</span>
            <span>
              <span class="session-topic">${escapeHTML(s.topic)}</span>
              <span class="session-meta"> · ${escapeHTML(s.user_display_name)} on ${escapeHTML(s.device_context)} · ${escapeHTML(ts)}</span>
            </span>
          </div>
          ${avgPill}
        </summary>
        <div class="session-body">
          ${s.summary ? `<div class="session-summary">${escapeHTML(s.summary)}</div>` : ""}
          <div class="turns">
            ${(s.turns || []).map(renderTurn).join("")}
          </div>
        </div>
      </details>
    `;
  }

  function renderHousehold(personasMap) {
    if (!personasMap) return "";
    return Object.entries(personasMap)
      .map(([uid, m]) => {
        return `
        <div class="member">
          <div class="member-name">${escapeHTML(m.display_name)}</div>
          <div class="member-meta">${escapeHTML(uid)} · age ${escapeHTML(m.age)} · ${escapeHTML(m.role)}</div>
          <div class="member-bg">${escapeHTML((m.background || "").slice(0, 220))}${m.background && m.background.length > 220 ? "…" : ""}</div>
        </div>`;
      })
      .join("");
  }

  function renderAggregate(agg) {
    if (!agg) return "";
    const total = agg.TOTAL;
    const rows = Object.entries(agg)
      .filter(([k]) => k !== "TOTAL")
      .sort()
      .map(
        ([cat, row]) =>
          `<tr>
            <td>${escapeHTML(cat)}</td>
            <td class="num">${row.n}</td>
            <td class="num">${row.correct}</td>
            <td class="num">${(row.accuracy * 100).toFixed(0)}%</td>
          </tr>`
      );
    if (total) {
      rows.push(
        `<tr class="total">
          <td>TOTAL</td>
          <td class="num">${total.n}</td>
          <td class="num">${total.correct}</td>
          <td class="num">${(total.accuracy * 100).toFixed(0)}%</td>
        </tr>`
      );
    }
    return rows.join("");
  }

  function categoriesFromQa(qa) {
    return Array.from(new Set(qa.map((q) => q.category))).sort();
  }

  function renderQA(qa, filter) {
    return qa
      .filter((q) => {
        if (filter.text) {
          const hay = (q.question + " " + q.gold_answer + " " + q.agent_answer).toLowerCase();
          if (!hay.includes(filter.text.toLowerCase())) return false;
        }
        if (filter.cat && q.category !== filter.cat) return false;
        if (filter.correct === "true" && !q.correct) return false;
        if (filter.correct === "false" && q.correct) return false;
        return true;
      })
      .map((q) => {
        const verdict = q.correct ? "correct" : "incorrect";
        const guardRow = q.guard_failures && q.guard_failures.length
          ? `<div class="qa-row guard">
              <div class="label">Guard failures</div>
              <div class="value">${q.guard_failures.map(escapeHTML).join("; ")}</div>
            </div>`
          : "";
        return `
        <div class="qa ${verdict}">
          <div class="qa-header">
            <span class="qa-id">${escapeHTML(q.qa_id)}</span>
            <span class="qa-cat">${escapeHTML(q.category)}</span>
            <span>asked by ${escapeHTML(q.ask_as_user)} on ${escapeHTML(q.ask_on_device)}</span>
            <span class="verdict ${verdict}">${verdict.toUpperCase()}</span>
          </div>
          <div class="qa-row">
            <div class="label">Question</div>
            <div class="value">${escapeHTML(q.question)}</div>
          </div>
          <div class="qa-row gold">
            <div class="label">Gold answer</div>
            <div class="value">${escapeHTML(q.gold_answer)}</div>
          </div>
          <div class="qa-row agent">
            <div class="label">Agent answered</div>
            <div class="value">${escapeHTML(q.agent_answer || "(no answer)")}</div>
          </div>
          ${guardRow}
          <div class="qa-row rationale">
            <div class="label">Judge rationale</div>
            <div class="value">${escapeHTML(q.judge_rationale)}</div>
          </div>
        </div>`;
      })
      .join("");
  }

  function loadRun(data) {
    if (!data || typeof data !== "object") {
      alert("Invalid run JSON.");
      return;
    }
    document.querySelector("main").classList.remove("empty");
    $("#welcome").classList.add("hidden");
    $("#overview").classList.remove("hidden");
    $("#phase1").classList.remove("hidden");
    $("#phase2").classList.remove("hidden");

    // Overview cards
    $("#ov-backend").textContent = data.backend || "?";
    $("#ov-elapsed").textContent =
      data.elapsed_seconds ? `${data.elapsed_seconds.toFixed(1)} s` : "—";
    $("#ov-sessions").textContent = (data.sessions || []).length;
    const total = (data.qa_aggregate || {}).TOTAL;
    if (total) {
      const pct = (total.accuracy * 100).toFixed(0) + "%";
      $("#ov-accuracy").textContent = `${total.correct}/${total.n} (${pct})`;
      const cls = total.accuracy >= 0.85 ? "good" : total.accuracy >= 0.6 ? "" : "bad";
      $("#ov-accuracy").parentElement.classList.remove("good", "bad");
      if (cls) $("#ov-accuracy").parentElement.classList.add(cls);
    } else {
      $("#ov-accuracy").textContent = "—";
    }

    $("#household").innerHTML = renderHousehold(data.personas_summary);
    $("#qa-aggregate tbody").innerHTML = renderAggregate(data.qa_aggregate);

    // Phase 1
    const sessions = data.sessions || [];
    if (sessions.length) {
      $("#sessions").innerHTML = sessions
        .map((s, i) => renderSession(s, i))
        .join("");
    } else {
      $("#sessions").innerHTML =
        '<div class="card" style="padding:18px;color:var(--fg-muted)">No conversation data — this run was QA-only or used the baseline backend.</div>';
    }

    // Phase 2
    const qa = data.qa || [];
    const cats = categoriesFromQa(qa);
    const sel = $("#qa-cat");
    sel.innerHTML = `<option value="">all categories</option>` +
      cats.map((c) => `<option value="${escapeHTML(c)}">${escapeHTML(c)}</option>`).join("");

    const renderFiltered = () => {
      $("#qa-list").innerHTML = renderQA(qa, {
        text: $("#qa-search").value,
        cat: $("#qa-cat").value,
        correct: $("#qa-correct").value,
      });
    };
    $("#qa-search").oninput = renderFiltered;
    $("#qa-cat").onchange = renderFiltered;
    $("#qa-correct").onchange = renderFiltered;
    renderFiltered();
  }

  function loadFile(file) {
    if (!file) return;
    $("#filename").textContent = file.name;
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        loadRun(JSON.parse(e.target.result));
      } catch (err) {
        console.error(err);
        alert("Failed to parse JSON: " + err.message);
      }
    };
    reader.readAsText(file);
  }

  // File input handler
  $("#file").addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) loadFile(e.target.files[0]);
  });

  // Drag-and-drop
  document.addEventListener("dragover", (e) => e.preventDefault());
  document.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      loadFile(e.dataTransfer.files[0]);
    }
  });

  // "Load latest sonzai run" — convenience for when the viewer is served
  // from the project root via a static server. Tries common locations.
  $("#loadDefault").addEventListener("click", async () => {
    const tries = [
      "sample_run.json",   // synthetic fixture for offline demos
      "../benchmarks/razer/results/latest.json",
      "../benchmarks/razer/results/sonzai_latest.json",
      "../benchmarks/razer/results/sonzai_20260506-195328.json",
    ];
    for (const url of tries) {
      try {
        const r = await fetch(url);
        if (r.ok) {
          const data = await r.json();
          $("#filename").textContent = url;
          loadRun(data);
          return;
        }
      } catch (_) {}
    }
    alert(
      "Couldn't auto-load. Drop a JSON onto the page or use Load run JSON.\n" +
      "Tried: " + tries.join(", ")
    );
  });
})();
