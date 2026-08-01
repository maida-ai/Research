# Agent Failure Modes — Source Notes

## cuadron2025overthinking

**Citation key:** cuadron2025overthinking
**Full title:** The Danger of Overthinking: Examining the Reasoning-Action Dilemma in Agentic Tasks
**Authors:** Alejandro Cuadron, Dacheng Li, Wenjie Ma, Xingyao Wang, Yichuan Wang, Siyuan Zhuang, Shu Liu, Luis Gaspar Schroeder, Tian Xia, Huanzhi Mao, Nicholas Thumiger, Aditya Desai, Ion Stoica, Ana Klimovic, Graham Neubig, Joseph E. Gonzalez
**Year:** 2025
**arXiv ID:** 2502.08235
**Submitted:** February 12, 2025
**Venue status:** PREPRINT — arXiv only (cs.AI); no peer-reviewed venue acceptance found as of verification date (2026-06-10)
**URL:** https://arxiv.org/abs/2502.08235

**Verification status:** VERIFIED

**What it supports for this paper:**
Introduces three named failure-mode patterns — Analysis Paralysis, Rogue Actions, and Premature Disengagement — from trajectory-level analysis of 4,018 agent runs on software engineering tasks. Provides empirical grounding for the claim that agentic LLMs exhibit qualitatively distinct behavioral failure modes beyond simple task failure. The framing of "overthinking" as prioritizing internal reasoning over environmental interaction is directly relevant to the constraint-induced destabilization framing.

---

## cemri2025multiagent

**Citation key:** cemri2025multiagent
**Full title:** Why Do Multi-Agent LLM Systems Fail?
**Authors:** Mert Cemri, Melissa Z. Pan, Shuyi Yang, Lakshya A. Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, Matei Zaharia, Joseph E. Gonzalez, Ion Stoica
**Year:** 2025
**arXiv ID:** 2503.13657
**Submitted:** March 17, 2025; revised to v3 October 26, 2025
**Venue status:** PREPRINT — arXiv only (cs.AI); no peer-reviewed venue acceptance found as of verification date (2026-06-10)
**URL:** https://arxiv.org/abs/2503.13657

**Verification status:** VERIFIED

**What it supports for this paper:**
Introduces MAST (Multi-Agent System Failure Taxonomy) with 14 failure modes in three categories: system design issues, inter-agent misalignment, and task verification. Built on 1,600+ annotated traces across 7 MAS frameworks. Directly supports the claim that trace-annotated failure taxonomies are tractable and that multi-agent systems fail in systematic, classifiable ways. Provides methodological precedent for the paper's trace-annotation approach.

---

## chen2026elephant

**Citation key:** chen2026elephant
**Full title:** Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems
**Authors:** Mengzhuo Chen, Junjie Wang, Fangwen Mu, Yawen Wang, Zhe Liu, Huanxiang Feng, Qing Wang
**Year:** 2026
**arXiv ID:** 2604.22708
**Submitted:** April 24, 2026
**Venue status:** ACCEPTED — arXiv record explicitly states "Accepted by ACL 2026"
**URL:** https://arxiv.org/abs/2604.22708

**Verification status:** VERIFIED (ACL 2026 acceptance confirmed on arXiv record)

**What it supports for this paper:**
Introduces TraceElephant, a benchmark demonstrating that complete execution traces (vs. partial output-only observations) improve failure attribution accuracy by up to 76%. Directly supports the paper's methodological premise that full observability of execution traces is necessary for meaningful failure analysis. The "whole elephant" framing — that examining only outputs obscures many failure causes — is load-bearing for the argument that oracle-checked complete traces are the correct unit of analysis.
