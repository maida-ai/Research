# Planning and Constrained Agent Evaluation — Source Notes

## zhang2026deepplanning

**Citation key:** zhang2026deepplanning
**Full title:** DeepPlanning: Benchmarking Long-Horizon Agentic Planning with Verifiable Constraints
**Authors:** Yinger Zhang, Shutong Jiang, Renhao Li, Jianhong Tu, Yang Su, Lianghao Deng, Xudong Guo, Chenxu Lv, Junyang Lin
**Year:** 2026
**arXiv ID:** 2601.18137
**Submitted:** January 26, 2026 (v1)
**Venue status:** PREPRINT — arXiv only (cs.AI, cs.CL); no peer-reviewed venue acceptance found as of verification date (2026-06-10)
**URL:** https://arxiv.org/abs/2601.18137

**Verification status:** VERIFIED

**What it supports for this paper:**
Introduces a benchmark explicitly targeting "global constrained optimization" rather than step-level reasoning, using travel and shopping tasks requiring proactive information gathering under hard constraints. Directly supports the claim that existing benchmarks fail to test constraint-satisfaction pressure, and that frontier LLMs struggle with this class of task. The paper's framing of verifiable constraints as the distinguishing feature of genuinely hard planning tasks is load-bearing for the CIBD methodology's use of solver-verified infeasibility claims.

---

## sun2026apb

**Citation key:** sun2026apb
**Full title:** Agent Planning Benchmark: A Diagnostic Framework for Planning Capabilities in LLM Agents
**Authors:** Haoyu Sun, Wenxuan Wang, Mingyang Song, Jujie He, Weinan Zhang, Yang Liu, Yang Yang, Yu Cheng
**Year:** 2026
**arXiv ID:** 2606.04874
**Submitted:** June 3, 2026; revised June 5, 2026 (v2)
**Venue status:** PREPRINT — arXiv only (cs.CL); no peer-reviewed venue acceptance found as of verification date (2026-06-10). NOTE: submitted only ~1 week before verification; minimal community review time.
**URL:** https://arxiv.org/abs/2606.04874

**Verification status:** VERIFIED

**What it supports for this paper:**
Introduces a diagnostic benchmark of 4,200+ multimodal test cases across 22 domains, explicitly including unsolvable tasks, irrelevant/broken tools, and feasibility determination as test dimensions. Directly supports the CIBD design choices of including infeasible problem instances and extraneous tools as pressure levers. The finding that models fail at "appropriate task refusal" validates infeasibility-claim detection as a meaningful behavioral signal.
