# Infrastructure and Signal Detection Theory — Source Notes

## ortools2024cpsat

**Citation key:** ortools2024cpsat
**Title:** CP-SAT Solver (Google OR-Tools documentation)
**Author/Publisher:** Google LLC
**Year:** accessed 2026 (documentation, no fixed publication date)
**URL:** https://developers.google.com/optimization/cp/cp_solver

**Verification status:** VERIFIED — page live and content confirmed

**What it supports for this paper:**
Provides the authoritative reference for CP-SAT as an integer constraint-programming solver. Defines five solver status categories: OPTIMAL, FEASIBLE, INFEASIBLE, MODEL_INVALID, and UNKNOWN. The INFEASIBLE status (proven infeasibility) is the oracle ground truth used to evaluate whether agents correctly claim a task is infeasible vs. attempt a solution.

---

## qwen2024qwen25

**Citation key:** qwen2024qwen25
**Title:** Qwen2.5: A Party of Foundation Models!
**Authors/Team:** Qwen Team (Alibaba)
**Date:** September 19, 2024
**URL:** https://qwenlm.github.io/blog/qwen2.5/

**Verification status:** VERIFIED — blog post live and content confirmed

**What it supports for this paper:**
Announces the Qwen2.5 open-weight model family with sizes 0.5B–72B, pretrained on 18 trillion tokens, supporting 128K context. Primary reference establishing the Qwen2.5 family as the candidate frozen-reference-agent substrate.

---

## qwen2024qwen25hf

**Citation key:** qwen2024qwen25hf
**Title:** Qwen/Qwen2.5-7B-Instruct (model repository)
**Authors/Team:** Qwen Team (Alibaba)
**Platform:** Hugging Face Model Hub
**URL:** https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
**License:** Apache 2.0

**Verification status:** VERIFIED — model repository live, publicly accessible

**What it supports for this paper:**
Confirms public availability of Qwen2.5-7B-Instruct as a concrete, reproducible model artifact (7.61B parameters, instruction-tuned, Apache 2.0). Supports reproducibility claims: any researcher can download and rerun calibration against the same checkpoint.

---

## green1966sdt

**Citation key:** green1966sdt
**Full title:** Signal Detection Theory and Psychophysics
**Authors:** David M. Green and John A. Swets
**Year:** 1966
**Publisher:** John Wiley & Sons, New York
**ISBN:** 978-0-471-32420-1 (Wiley original); reprint editions also available

**Verification status:** VERIFIED via bibliographic sources — book exists and publication details confirmed (not web-fetchable)

**What it supports for this paper:**
Foundational text establishing signal detection theory terminology: hit rate, false-alarm rate, discrimination measure d′, response bias criterion, and the ROC framework. Load-bearing for analyzing infeasibility-claim behavior as a detection task.

---

## macmillan2005detection

**Citation key:** macmillan2005detection
**Full title:** Detection Theory: A User's Guide (2nd edition)
**Authors:** Neil A. Macmillan and C. Douglas Creelman
**Year:** 2005
**Publisher:** Lawrence Erlbaum Associates (Psychology Press / Taylor & Francis)
**ISBN:** 978-0-8058-4231-9

**Verification status:** VERIFIED via bibliographic sources — book exists and publication details confirmed (not web-fetchable)

**What it supports for this paper:**
Standard reference for applied SDT methodology: computation of d′, criterion c, ROC analysis, and small-sample considerations (e.g., corrections for zero/one rates).

---

## maida2026gate

**Citation key:** maida2026gate
**Title:** The PR Gate for AI Agents
**Author/Organization:** Maida.AI
**URL:** https://www.maida.ai/
**Accessed:** 2026-06-10

**Verification status:** VERIFIED — website live, content confirmed

**What it supports for this paper:**
Describes Maida as a pre-merge behavioral regression gate monitoring structural behavioral signals: step count, tool call count, unexpected tool path, loop risk, guardrail events, latency envelope, cost envelope, stop condition. Per STYLE.md: cite only if Maida is named explicitly in the paper.

---

## maida2026repo

**Citation key:** maida2026repo
**Title:** maida-ai/maida (GitHub repository)
**Author/Organization:** Maida.AI
**URL:** https://github.com/maida-ai/maida
**Accessed:** 2026-06-10

**Verification status:** VERIFIED — repository live, content confirmed

**What it supports for this paper:**
Confirms the implementation substrate is local-first and CI-first: trace data stored locally as JSONL; baseline/policy workflow. Supports reproducibility claims about trace capture infrastructure. Like maida2026gate, cite only if Maida is named.
