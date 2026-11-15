# Claude Code project instructions

At the start of every session, before doing anything else, read these three files in full:

1. [CONTEXT.md](CONTEXT.md) — stable project facts, dataset, metric, architecture, source hierarchy.
2. [DECISIONS.md](DECISIONS.md) — append-only record of modeling/workflow decisions and why.
3. [PROGRESS.md](PROGRESS.md) — current session state, latest score, and the single next action.

These three files are the project's memory. Do not re-derive facts they already contain by re-reading notebooks or re-running analysis — trust them unless the repo's current code/data output contradicts them, per CONTEXT.md's verification rule.

End every working session by updating PROGRESS.md per its own "Session-end rule."

## Working with the learner (portable preferences)

These travel with the repo, so they apply on any PC (the per-PC auto-memory does not).

- **Language:** reply in Banglish (Bengali in Latin script). Keep small everyday words in English too, not only technical terms: total, average, ratio, sales, week, row, column, check, forecast, error. Bengali only for the connecting grammar.
- **Teaching style:** tiny worked example first (3 series, 5 weeks), then the real numbers. The learner writes the code; give concept, skeleton and hints, then review the saved notebook.
- **Pace:** fast. Once a concept is understood, move on; no extra drills. At most one prediction question per lesson, only when it teaches something new.
- **Code style:** teach the professional, vectorized numpy/pandas way, unpacked piece by piece on a tiny matrix. Do not fall back to beginner loops.
- **Notebook review:** read the saved file (not what the learner says ran); flag unsaved/empty cells; use the `vn1` kernel (`D:/conda_envs/vn1` on the home PC).
- **Honesty:** if an earlier statement of yours was wrong, say so plainly and correct it. Check numbers against the data before stating them.
- **Skeptical sources:** every rule taken from a book, article, webinar or our own result is tagged with its evidence and cross-checked against an independent source. Four of the five books in `references/` are by Nicolas Vandeput (SupChains is his company), so they do not cross-check each other. Labels: 🟢 strong (2+ independent sources and tested on our data), 🟡 moderate (one source or one dataset), 🔴 hypothesis. Record disagreements; never copy book text, paraphrase with chapter/page.
- **Long-term goal:** a demand forecasting + inventory optimization SaaS and AI agent the learner can sell to clients, built on VN1 (forecasting) and VN2 (inventory) knowledge. "Beat VN1 1st place" is a post-competition replay (D016) and must never be used as a claim without that caveat.

## Current plan (2026-11-15)

1. `learn/` course, shortened: Lesson 0 metric (done) → 1 data + EDA (Step 3 in progress) → 2 benchmark ladder → 3 backtest and leakage → 12 quantile forecast → safety stock → order quantity. Lessons 4–11 (AMSI, Theta, LightGBM, Tweedie, dormancy, ensemble, pipeline, ablation) are paused and learned just-in-time during VN2.
2. Skills from VN1 lessons, built with the skeptical protocol above: `forecast-eval`, `demand-data-health`, then `agent-design` (SupChains Way) and `inventory-policy` (Inventory Optimization book + VN2 winners' webinar).
3. **VN2 blind attempt:** open only the competition data, rules and scoring code. Keep the VN2 winners' webinar, leaderboard and winners' files sealed until our attempt is pre-registered, frozen and committed. VN1 lessons are fair to use (they were public before VN2). Claude may carry VN2 knowledge from training; flag any idea that looks VN2-specific and disclose it in the report.
4. MVP after VN2, in a separate private repo: Streamlit UI over a UI-free `engine/` package (forecast, backtest, health, inventory) plus a Claude API explain layer; move to FastAPI + React only once there are paying clients.
