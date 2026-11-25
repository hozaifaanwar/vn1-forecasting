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
- **Code and notebook text in English:** code comments, docstrings and markdown cells inside code/notebooks are always written in English (explanations in chat stay Banglish).
- **Teaching style:** tiny worked example first (3 series, 5 weeks), then the real numbers. The learner writes the code; give concept, skeleton and hints, then review the saved notebook.
- **Pace:** fast. Once a concept is understood, move on; no extra drills. At most one prediction question per lesson, only when it teaches something new.
- **Code style:** teach the professional, vectorized numpy/pandas way, unpacked piece by piece on a tiny matrix. Do not fall back to beginner loops.
- **Notebook review:** read the saved file (not what the learner says ran); flag unsaved/empty cells; use the `vn1` kernel (`D:/conda_envs/vn1` on the home PC).
- **Honesty:** if an earlier statement of yours was wrong, say so plainly and correct it. Check numbers against the data before stating them.
- **Skeptical sources:** every rule taken from a book, article, webinar or our own result is tagged with its evidence and cross-checked against an independent source. Four of the five books in `references/` are by Nicolas Vandeput (SupChains is his company), so they do not cross-check each other. Labels: 🟢 strong (2+ independent sources and tested on our data), 🟡 moderate (one source or one dataset), 🔴 hypothesis. Record disagreements; never copy book text, paraphrase with chapter/page.
- **Long-term goal:** a demand forecasting + inventory optimization SaaS and AI agent the learner can sell to clients, built on VN1 (forecasting) and VN2 (inventory) knowledge. "Beat VN1 1st place" is a post-competition replay (D016) and must never be used as a claim without that caveat.

## Time budget and phases (2026-11-19)

The learner has a full-time job: plan on **~10 hours a week** (more when possible). Split per week: ~7 h learning/building, ~2 h client work (small Upwork jobs, pilots), ~1 h reviewing what Claude built. Consistency over intensity. Claude builds UI/infra; the learner writes the core forecasting/inventory logic.

| Phase | Content | Learner time | Weeks at 10 h |
|---|---|---|---|
| 1 | VN1 short course (Lessons 2, 3, 12) + `forecast-eval` skill; DemandPilot Fast/Best mode (port `vn1_fresh`) | ~20 h | ~2 |
| 2 | VN2 blind attempt (inventory: lead time, censored demand, cost); DemandPilot v0.2: censored demand, cost-based orders, MOQ/case pack | ~45 h | ~4–5 |
| 3 | AI agent layer: only the needed parts of Ed Donner's courses (LLM API, tool calling, agents, RAG); explanations and exceptions in DemandPilot | ~30 h | ~3 |
| 4 | Production SaaS (Docker, cloud, Postgres, password reset, billing): only after a real client or pilot | ~30 h | ~3 |

From Phase 1 on, keep ~2 h/week for small paid jobs or free pilots: they become the production experience that client proposals ask for. Product repo: `D:\demandpilot` (separate git repo).

## Current plan (2026-11-15)

1. `learn/` course, shortened: Lesson 0 metric (done) → 1 data + EDA (Step 3 in progress) → 2 benchmark ladder → 3 backtest and leakage → 12 quantile forecast → safety stock → order quantity. Lessons 4–11 (AMSI, Theta, LightGBM, Tweedie, dormancy, ensemble, pipeline, ablation) are paused and learned just-in-time during VN2.
2. Skills from VN1 lessons, built with the skeptical protocol above: `forecast-eval`, `demand-data-health`, then `agent-design` (SupChains Way) and `inventory-policy` (Inventory Optimization book + VN2 winners' webinar).
3. **VN2 blind attempt:** open only the competition data, rules and scoring code. Keep the VN2 winners' webinar, leaderboard and winners' files sealed until our attempt is pre-registered, frozen and committed. VN1 lessons are fair to use (they were public before VN2). Claude may carry VN2 knowledge from training; flag any idea that looks VN2-specific and disclose it in the report.
4. MVP after VN2, in a separate private repo: Streamlit UI over a UI-free `engine/` package (forecast, backtest, health, inventory) plus a Claude API explain layer; move to FastAPI + React only once there are paying clients.
