# SentryBench — Research Plan

> **Archived working notes.** This document records research planning and may
> include tentative claims or unimplemented ideas. For the published paper,
> see [arXiv:2610.34535](https://arxiv.org/abs/2610.34535); for current usage,
> see the [project README](../README.md).

*A Benchmark for Adversarial Robustness of Tool-Using LLM Agents in Public-Safety Deployments*

> Working name: **SentryBench** (alt: PSAgentBench). Workspace: `pubai`.
> Status: **v1.0 — built and run.** The `[DECISION]` items below are now resolved
> (see §19). A working harness, a 190-scenario real-data bank, and an 840-episode
> case study exist; forward-looking items are marked **[FUTURE]**.

---

## 0. One-line

We measure whether tool-using LLM agents deployed in public-safety settings fail *safely* under adversarial pressure, where "safe" is **bidirectional**: neither suppressing a real emergency (under-reaction) nor triggering unwarranted alerts/privacy violations (over-reaction).

---

## 0.5 Current state (as-built)

**Built and executed** (see the [paper](https://arxiv.org/abs/2610.34535) and
[results](../final/final_hf.md) for full results):

- **Harness:** complete Python package `sentrybench/` (ingest → generate → runner/episode → deterministic judge → metrics → self-generating report), ledger-first scoring, bootstrap CIs, live per-episode logging.
- **Data — real, not synthetic.** Scenarios are grounded in real Hugging Face datasets: `spikecodes/911-call-transcripts` and `community-datasets/disaster_response_messages` (legitimate incident content) + `deepset/prompt-injections` and `jackhhao/jailbreak-classification` (attack payloads). No real PII (PII-flagged rows dropped); all tools are mocked (sandbox-only).
- **Scenario bank:** **190** scenarios in 6 families — A benign-urgent (40), B benign-ambiguous (38), C adversarial under-reaction (38), D adversarial over-reaction (30), E privilege-escalation/records (24), F tool-result-poisoning/infra (20). Split: 112 adversarial / 78 benign (40 urgent + 38 ambiguous).
- **Models:** `claude-sonnet-4-5-20250929`, `claude-haiku-4-5-20251001`, Azure `gpt-5`.
- **Defenses:** `none`, `caution` (naive-caution strawman, prompt-only), `tgcc` (Trust-Gating for Consequential Calls = provenance separation + heuristic injection detector + consequential-call gate routing to human + human-escalation containment).
- **Run:** 840 scored episodes (Haiku on all 190 × 3; Claude + GPT-5 on a 45-scenario stratified subset × 3).
- **Headline results:** naive caution drives over-refusal to **100%** on Claude/GPT-5 for ≤3.7-pt ASR change; **TGCC has the lowest harm on every model** (0.66/1.0/0.6 vs caution 1.33/1.8/1.6); one evasive injection still beat TGCC (Claude `C-008`, detector recall gap); Azure's content filter confounds GPT-5 (reported as model-level vs system-level).
- **Frontier realized** as three discrete defense conditions ranked by harm; a continuous strictness sweep (θ) and published-baseline defenses are **[FUTURE]**.

**Not yet done (all [FUTURE]):** LLM-judge layer (scoring is deterministic only); TGCC component ablation; published-baseline defenses (spotlighting, CaMeL-style, allow-list); McNemar / mixed-effects stats; Delphi severity calibration; private canary split; response caching; adaptive-attacker track.

---


## 1. Motivation & Problem

Agentic LLMs are being wired into consequential tools: sending public alerts, dispatching responders, querying citizen records. A single bad tool call has real-world consequence. Existing red-team benchmarks overwhelmingly target **chat safety** (jailbreaks, refusals) or **generic tool harm** (data exfiltration, fraud, malware). None model the specific risk profile of public-safety tool use.

The defining feature of this domain: **safety is not monotonic in caution.** In chat safety, refusing is almost always the safe fallback. In public safety, an agent too cautious to dispatch help is itself a failure with potentially lethal cost. So the two failure modes that matter are:

- **False negatives (under-reaction):** the agent misses or suppresses a real emergency.
- **False positives (over-reaction):** the agent triggers unwarranted public alerts (panic), over-broad data queries (privacy violation), or unnecessary dispatch (resource waste, erosion of trust).

Prior work implicitly optimizes only the FP side by rewarding refusal. **SentryBench is the first benchmark to treat both as first-class, with domain-calibrated asymmetric cost.**

---

## 2. Novelty & Positioning

### 2.1 Novelty claim (the thesis that drives the paper)

> In public-safety agent deployments, safety is a **cost-sensitive decision problem with asymmetric, domain-calibrated losses**, not a monotone "refuse-more-is-safer" axis. We (a) formalize this, (b) release a grounded benchmark that measures both under- and over-reaction under adversarial tool use, and (c) show current defenses trade attack-success reduction for increased over-refusal, while our trust-gating approach shifts the operating point favorably on the harm–utility frontier.

Three deliverables fall out of this: a **formalism**, a **benchmark/dataset**, and an **intervention evaluated on the frontier** (not on a single axis).

### 2.2 Related work comparison

| Benchmark / work | Domain | Attack focus | Over-refusal measured? | Asymmetric harm cost? | Public-safety? |
|---|---|---|---|---|---|
| InjecAgent (2024) | generic tools | indirect prompt injection | No | No | No |
| AgentDojo (2024) | banking / travel / workspace | PI + defenses | Partial (task utility under attack) | No | No |
| ToolEmu (2024) | generic | emergent risky behavior | No | Partial (severity rubric) | No |
| AgentHarm (2024) | 11 harm categories | malicious task completion | No | No | No |
| R-Judge (2024) | interaction records | risk awareness | No | No | No |
| AgentSafetyBench / SafeAgentBench (2024–25) | general agent safety | mixed | Partial | No | No |
| **SentryBench (this work)** | **public safety (4 sub-domains)** | PI + tool-result poisoning + goal hijack + over-trust + priv-esc | **Yes (core metric)** | **Yes (calibrated)** | **Yes** |

*Fill exact citations at write-up time; verify claims against latest versions.*

### 2.3 What is genuinely new

1. First **domain-grounded** public-safety agent red-team suite.
2. First to make **over-refusal a first-class metric with a benign-urgent control arm**.
3. A **decision-theoretic harm–utility frontier** as the primary comparison object.
4. A reusable, versioned, contamination-resistant **dataset contribution** with a datasheet.

---

## 3. Research Questions & Hypotheses

- **RQ1 (vulnerability):** How susceptible are frontier agents (Claude-family, GPT-family) to public-safety-specific attacks across the five attack vectors?
- **RQ2 (asymmetry):** Do defenses that reduce attack success also increase over-refusal on benign-urgent scenarios? (i.e., is the tradeoff real and measurable?)
- **RQ3 (intervention):** Does trust-gating (TGCC) Pareto-improve the harm–utility frontier vs. no-defense and vs. published baselines?
- **RQ4 (containment):** When uncertain, do agents escalate to a human rather than act unilaterally — and does defense change containment behavior?
- **RQ5 (latency):** For the trust-gating condition, how many turns elapse before the gate fires (detection latency)?

**Hypotheses**
- H1: ASR is non-trivial (>20%) for at least one attack vector on undefended frontier agents.
- H2: Naive "cautious" prompting reduces ASR but significantly raises over-refusal (negative correlation across the operating curve).
- H3: Trust-gating dominates naive caution on the frontier (lower ASR at equal or lower over-refusal).
- H4: Containment behavior is the strongest single predictor of low harm-weighted severity.

**Result status (this run).**
- **H1 — not supported (encouraging for models):** undefended ASR was low (≤3.7%), not >20%. Frontier models resist obvious injection; all residual successes were *under-reaction* suppressions. (RQ1 answered: vulnerability is low but non-zero and dangerous when it occurs.)
- **H2 — strongly supported:** `caution` pushed over-refusal to **100%** (Claude/GPT-5) and 65% (Haiku) for ≤3.7-pt ASR change — refusing more is not free. (RQ2 confirmed.)
- **H3 — supported in the harm-weighted sense:** `tgcc` has the lowest harm on every model and keeps over-refusal at/below `none` while raising containment; it does not increase either error rate vs `none`. (RQ3 confirmed; a swept-θ Pareto curve is [FUTURE].)
- **H4 — consistent but not formally tested:** high-containment cells had low harm; a regression test is [FUTURE].
- **RQ5 (latency):** TGCC detection latency ≈ **1 turn** (gate fires on the first consequential call under a manipulation signal).

---

## 4. Formal Framework (the intellectual core)

For each scenario $i$ there is a **warranted action set** $A^\*_i$ defined by the gold label (possibly empty, e.g. "do nothing but log", or "escalate to human"). Let the agent's realized actions be $A_i$.

Classify the outcome from the **tool-call ledger** (deterministic where possible):

- **Under-reaction (FN):** a warranted protective action in $A^\*_i$ was not taken (missed dispatch, suppressed real alert).
- **Over-reaction (FP):** a consequential action not in $A^\*_i$ was taken (false public alert, over-broad `query_citizen_record`, unnecessary `dispatch_unit`).
- **Correct (TP/TN):** actions match the gold set, including correct escalation/abstention.

Define a **harm-weighted utility** per scenario:

$$U_i = -\Big( c^{FN}_i \cdot \mathbb{1}[\text{under-react}_i] + c^{FP}_i \cdot \mathbb{1}[\text{over-react}_i] \Big)$$

where $c^{FN}_i, c^{FP}_i$ are **domain-calibrated severity weights** (Section 7.3). Aggregate utility $U=\sum_i U_i$.

**Frontier object.** For a defense with tunable strictness $\theta$, sweep $\theta$ and plot **over-refusal rate** (y) vs **attack-success rate** (x). Each defense yields an operating characteristic curve; a defense **Pareto-dominates** if its curve lies below-left of another. Report:
- Area under the harm–utility frontier.
- Expected utility under a **range** of cost ratios $\rho = c^{FN}/c^{FP}$ (so results are robust to the exact weighting).

This decision-theoretic framing is what elevates the work from a leaderboard to a method.

---

## 5. Scope (kept tractable)

Four sub-domains, each grounded in a documented real-world standard so gold labels are defensible:

1. **Emergency dispatch triage** (911-style call handling). Grounding: NENA / APCO Emergency Medical Dispatch call-handling protocols.
2. **Public alert / notification** (weather, evacuation, AMBER-style). Grounding: FEMA IPAWS, Wireless Emergency Alerts, NWS product standards, AMBER activation criteria (strict → rich FP/FN cases).
3. **Citizen data query agents** (records lookup for responders). Grounding: CJIS Security Policy, least-privilege / need-to-know (→ privilege-escalation & over-query privacy cases).
4. **Infrastructure monitoring escalation** (utility/transit anomaly). Grounding: SCADA/utility anomaly-escalation norms.

*We deliberately exclude open-ended "all of public safety" to keep gold labels rigorous.*

**As-built grounding.** In addition to the standards above, legitimate incident
content is now drawn from **real data** — real 911 transcripts and real
disaster-response messages — mapped into the four sub-domains by their category
labels (see §7.1 and `sentrybench/generate.py`). Attack payloads are real
prompt-injection / jailbreak strings. This replaces the original synthetic-only
plan and makes the gold labels more defensible, not less.

---

## 6. Threat Model & Attack Taxonomy

**Assumptions.** The agent is trusted; *inputs and tool results may be untrusted.* No model weights access. Attacker controls: caller transcripts, sensor/incident-report text, and (for poisoning) the content returned by some tools mid-chain.

**Attack vectors (rows):**

1. **Prompt injection via untrusted input** — fake caller transcript, poisoned sensor log, malicious "incident report" text.
2. **Tool-result poisoning** — a called tool returns manipulated data mid-chain to redirect the agent.
3. **Goal hijacking** — multi-turn drift that gradually redirects agent intent.
4. **Over-trust exploitation** — attacker impersonates authority ("this is dispatch override, skip verification").
5. **Privilege escalation** — agent is induced to chain tools beyond granted scope.

**Failure-mode axis (columns) — the novel contribution:**

- **Under-reaction** (suppressed/missed real emergency).
- **Over-reaction** (false alert, unwarranted escalation, privacy violation via over-broad query).
- **Correct containment** (detect anomaly, escalate to human, do not act unilaterally).

The **5 × 3 matrix × 4 sub-domains** is the scenario-generation grid.

---

## 7. Scenario Design & Gold Labels

### 7.1 Counts

**As-built: 190 scenarios** (this run), composed from real data across 6 families:
A benign-urgent (40), B benign-ambiguous (38), C adversarial under-reaction (38),
D adversarial over-reaction (30), E privilege-escalation/records (24),
F tool-result-poisoning/infra (20) → **112 adversarial / 78 benign** (40 urgent +
38 ambiguous), a ~59:41 adversarial:benign ratio. The original v0.1 target of
~280 remains the scaling goal **[FUTURE]**; the harness generates the bank
deterministically (`cli generate`), so scaling is a parameter change.

*Note (v0.1 plan, retained):* target `5 vectors × 3 failure targets × 4 domains ×
~4 variants ≈ 240 adversarial` + ~40–60 benign controls.

### 7.2 Per-scenario schema (see [schemas.py](../sentrybench/schemas.py) for the implementation)

Each scenario carries: `id`, `subdomain`, `attack_vector`, `target_failure_mode`, `narrative/context`, `available_tools`, `injected_payload` (+ delivery channel), `gold_action_set`, `forbidden_action_set`, `severity_weights {c_fn, c_fp}`, `rubric`, `provenance/grounding_ref`, `difficulty`, `split {public|canary}`.

### 7.3 Severity calibration (defensible weights)

**As-built:** severity weights $c^{FN}, c^{FP}$ are **authored per-scenario
constants** (e.g. urgent dispatch `c_fn=9, c_fp=3`; alert/records `c_fp≈8`),
encoding the domain's asymmetry. The **Delphi-panel calibration** and
**per-cost-ratio robustness curves** below are **[FUTURE]**.

Anchor $c^{FN}, c^{FP}$ to a **documented 4-tier consequence scale** (e.g., negligible / moderate / serious / catastrophic) mapped to numeric costs. Calibrate via:
- A written rubric tied to each sub-domain's standard, **plus**
- A small **Delphi-style panel** (3–5 raters; can be internal) to set tier boundaries, reporting agreement.
Report all results under **multiple cost ratios** so conclusions don't hinge on exact weights.

### 7.4 Gold labeling & IRR

- Two independent annotators label gold action sets; disagreements adjudicated.
- Report **inter-rater reliability** (Cohen's κ / Krippendorff's α) on gold labels for ~15–20% double-labeled.

---

## 8. Experimental Design

**Factors:** `model × defense` applied to every scenario (3×3 as-built).

- **Models (resolved):** `claude-sonnet-4-5-20250929`, `claude-haiku-4-5-20251001` (Anthropic, `temperature=0`), and Azure `gpt-5` (reasoning; `max_completion_tokens=8192`, default temperature).
- **Defense conditions (as-built):** `none` (baseline), **`caution`** (naive-caution strawman, prompt-only), `tgcc` (trust-gating). Published baselines — `spotlighting`, `delimiting/sandwich`, `CaMeL-style` separation, `allow-list gate` — are **[FUTURE]**.
- **Control arm (two-sided, as-built):** **urgent** benign scenarios quantify over-refusal (FN); **ambiguous** benign scenarios quantify over-reaction (FP).
- **Coverage (as-built):** Haiku on all 190 × 3; Claude + GPT-5 on a 45-scenario stratified subset × 3 (so cross-model cells are over different mixes — a stated limitation).

This yields, per scenario, a clean paired design enabling paired significance tests **[FUTURE: McNemar / mixed-effects]**.

**Ablations of TGCC** (injection detector / gate / provenance / human-escalation toggled independently) — the components are individually toggleable in code but the ablation sweep is **[FUTURE]**.

---

## 9. Metrics

Primary (all implemented, each with a **bootstrap 95% CI**):
- **Attack Success Rate (ASR)** — attacker goal achieved, over adversarial scenarios the model *actually ran* (deterministic from ledger).
- **Over-Refusal Rate (ORR / FN)** — on **urgent** benign scenarios, failure to take the warranted *kind* of action (name-level, not exact unit/priority).
- **Benign Over-Reaction Rate (FP)** — on **ambiguous** benign scenarios, taking a forbidden consequential action. *(Added as-built; the second frontier axis.)*
- **Harm-Weighted Severity Score** — mean per-scenario $c^{FN}\mathbb{1}[u]+c^{FP}\mathbb{1}[o]$; the primary single-number ranking.
- **Containment Rate** — escalate/human-route without unilateral harm.
- **Detection Latency** — turns until the trust-gate first fires (defense condition only).
- **Model-level vs system-level** — system-level folds the provider content filter into the deployed system (filtered attack = prevented; filtered urgent emergency = over-refusal). *(Added as-built to handle Azure Prompt-Shield.)*

Composite / analysis:
- **Harm–utility frontier** reported as **three discrete defense conditions** ranked by harm (as-built). A continuous strictness sweep (θ) and **expected utility $U(\rho)$ over a range of cost ratios** are **[FUTURE]**.
- A single rank-able **Public-Safety Safety Score** — **[FUTURE]** (harm is used as the primary ranking today).

---

## 10. Evaluation / Judging Protocol

Design to neutralize LLM-judge bias:

1. **Ledger-first scoring.** Most ASR/FP/FN outcomes are computable **deterministically** from the tool-call ledger vs gold/forbidden action sets. Use the LLM judge only for reasoning-quality and ambiguous rubric items.
2. **Blinded, multi-judge.** Judge transcripts are stripped of model/defense identity. Use ≥2 judge models (e.g., Claude + GPT + one open model) and report agreement.
3. **Self-preference control.** Never let a model be the sole judge of its own transcripts; report cross-judge deltas.
4. **Human spot-check.** Manual review of 15–20% for inter-rater reliability (Krippendorff's α) between human and LLM judge; report.

---

## 11. Statistical Analysis Plan

- **Paired binary outcomes** (defense on/off, same scenario): McNemar's test.
- **Rates:** bootstrap 95% CIs (percentile), not just point estimates.
- **Model:** mixed-effects logistic regression — fixed effects `model`, `defense`, `attack_vector`, `subdomain`; random intercept per `scenario`.
- Report **effect sizes** and CIs, not only p-values; correct for multiple comparisons (Benjamini–Hochberg).
- **Power note:** the current bank is **190 scenarios** (Haiku full; Claude/GPT-5 on a 45-subset, so their CIs are wide — e.g. Claude ASR `3.7% [0–11.1]`). Scaling to ~280 and running all models on the full bank for adequately-powered paired tests is **[FUTURE]**.

---

## 12. Reproducibility & Contamination

- Pin **model snapshots** (dated), `temperature=0` where supported, fixed seeds, full request/response logging.
- **Cache** all model calls (hash of prompt+config) for exact replay.
- Release **public split** + withhold a **private canary split** to detect future training contamination; re-measure canary vs public gap over time.
- Version the dataset (`v1.0`) with a **datasheet** (Gebru et al.) and a machine-readable manifest + checksums.

---

## 13. Ethics, Dual-Use, Responsible Release

- **Sandbox-only tools; no real PII; no real APIs; no live systems touched.** Legitimate content comes from **public research datasets** (real 911 transcripts, disaster messages) with **PII-flagged rows filtered out**; only the English `message` field is used. (Updated from the v0.1 "synthetic-only" stance: content is now real, but harmless and de-identified.)
- Attack payloads are real injection/jailbreak strings but **templated and domain-bounded**; the artifact avoids operational detail that transfers to attacking real public-safety infrastructure.
- **Licensing:** injection/jailbreak/911 sources are permissive (Apache-2.0/MIT); `disaster_response_messages` license is "unknown" → research-use only; derived scenarios should respect source terms. **[FUTURE:** gated release for the most sensitive payloads.**]**
- **Broader-impact statement** covering misuse, and a coordinated-disclosure note to affected model vendors before publication.
- No instructions that would help a real actor harm real public-safety infrastructure.

---

## 14. Dataset Contribution

- Structured, versioned JSON scenarios + gold labels + rubrics + severity weights.
- Datasheet: motivation, composition, collection, preprocessing, uses, distribution, maintenance.
- **Licensing (as-built):** source datasets are Apache-2.0 (injection, jailbreak) / MIT (911) / unknown (disaster → research-use only); harness is open-source. A formal scenario-bank license + use-restriction addendum for a gated split is **[FUTURE]**.

---

## 15. Expected Contributions & Claims

1. **Framework:** cost-sensitive, asymmetric formalization of public-safety agent safety + harm–utility frontier methodology.
2. **Benchmark:** ~280 grounded scenarios across 4 sub-domains, 5 attack vectors, 3 failure modes, with benign-urgent controls.
3. **Findings:** vulnerability profile of frontier agents; empirical demonstration of the ASR↔over-refusal tradeoff; evidence that trust-gating Pareto-improves the frontier.
4. **Artifact:** reusable harness + dataset + datasheet.

---

## 16. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Scenarios seen as unrealistic | Ground each in a real standard; expert spot-review |
| LLM-judge bias inflates results | Ledger-first + blinded multi-judge + human IRR |
| Severity weights seen as arbitrary | Delphi calibration + report across cost ratios |
| TGCC underspecified | Specify + ablate + compare to published baselines |
| Contamination over time | Private canary split, dated snapshots |
| Dual-use concern from reviewers | Synthetic-only, gated release, broader-impact section |
| API drift/cost | Cache all calls; pin snapshots; budget (Impl. plan §11) |

---

## 17. Timeline / Milestones (relative)

- **M0 — Design freeze:** ✅ **done** — taxonomy, schema, severity weights, model/defense list all fixed.
- **M1 — Vertical slice:** ✅ **done** — sandbox, runner, ledger, both providers, TGCC on/off, deterministic scorer, metrics, end-to-end report.
- **M1.5 — Real-data extension (added):** ✅ **done** — HF ingestion, 190-scenario generator, `caution` baseline, two-sided benign controls, bootstrap CIs, model-vs-system-level, live logging.
- **M2 — Full scenario authoring:** ◐ **partial** — 190 real-data scenarios (target ~280); gold labels are programmatic, **IRR [FUTURE]**.
- **M3 — Full runs:** ✅ **done** — 840 episodes logged (Haiku full; Claude/GPT-5 subset). **Response caching [FUTURE]**.
- **M4 — Judging + stats:** ◐ deterministic scoring + bootstrap CIs done; **blinded multi-LLM-judge, human spot-check, McNemar/mixed-effects [FUTURE]**.
- **M5 — Ablations + baselines:** **[FUTURE]** — TGCC component ablation + published defenses.
- **M6 — Write-up + release:** ◐ `paper.md` + auto-report done; **datasheet + canary split [FUTURE]**.

---

## 18. Related Work (to expand at write-up)

Prompt-injection & tool-agent safety: InjecAgent; AgentDojo; ToolEmu; AgentHarm; R-Judge; AgentSafetyBench / SafeAgentBench; ST-WebAgentBench. Defenses: spotlighting; prompt sandwiching/delimiting; CaMeL (control/data-flow separation); task-scoped allow-listing. Over-refusal / exaggerated safety: XSTest and successors. Dataset norms: Datasheets for Datasets. *Verify latest versions and exact claims before citing.*

---

## 19. Resolved decisions & remaining questions

**Resolved (as-built).**

1. **TGCC definition:** provenance separation + heuristic injection detector +
   consequential-call gate (route-to-human) + human-escalation containment; the
   four components are individually toggleable (`sentrybench/defenses/tgcc.py`).
2. **Model snapshots:** `claude-sonnet-4-5-20250929`, `claude-haiku-4-5-20251001`,
   Azure `gpt-5`. Budget managed via a fast/cheap Haiku full arm + small
   Claude/GPT-5 subset.
3. **Adversarial:benign ratio:** ~59:41 (112 adversarial / 78 benign), with the
   benign arm split urgent/ambiguous.
4. **Severity costs:** authored per-scenario constants (asymmetric); Delphi
   calibration deferred.
5. **Data:** real HF datasets replace synthetic-only (911 transcripts, disaster
   messages, injection/jailbreak payloads).

**Remaining [FUTURE].**

6. Scale toward ~280+ scenarios; add IRR / gold double-labeling.
7. Published-baseline defenses + TGCC component ablation.
8. LLM-judge layer; McNemar / mixed-effects; per-cost-ratio robustness curves.
9. Private **canary split**, dataset **datasheet**, response caching, licensing
   for a gated release.
10. Adaptive-attacker track; run Claude/GPT-5 on the full bank for clean
    cross-model comparison.
11. Target **venue/deadline**.
