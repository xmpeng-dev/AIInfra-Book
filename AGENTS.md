# AGENTS.md — shared engineering context

> Loaded at the start of every agent session.
> This file is owned by the `slab` knowledge repo and symlinked into
> sibling projects (MMOE, MonolithEP, Primus, Primus-dev, RocMoE, …)
> by `scripts/bootstrap-project-from-slab.sh`.
>
> Keep it stable, short, and high-signal. Project-specific context
> belongs in that project's own `README.md` or in a
> `.cursor/rules/90-<project>-local.mdc` file.

---

## 1. Whose environment this is

Xiaoming Peng — training systems & AI infrastructure for AMD GPUs.
The agent's work happens across a small set of sibling repositories
that all share the same hardware, container, SLURM cluster, and
knowledge base. Slab is the shared **knowledge layer**; sibling repos
are application code that imports slab's rules and skills.

## 2. Hardware & environment context

Day-job hardware (almost all notes are written against this):

- **AMD MI300X** (CDNA3, gfx942)
- **AMD MI325X** (CDNA3 successor, gfx942)
- **AMD MI355X** (CDNA4, gfx950) — current default for new experiments
- Comparison reference: **NVIDIA H100 / H200 / B200**

Training stack:

- ROCm + PyTorch, RCCL, Megatron-LM / Megatron-Core, TorchTitan, **Primus**
- SLURM-managed multi-node cluster
- Dev environment: Podman container `xiaoming-dev`
  (see `scripts/start-xiaoming-dev-fix-container.sh` in slab);
  image `docker.io/rocm/primus:v26.2`
- Profiling: PyTorch Profiler / Kineto traces; rocprof for HIP kernels

## 3. Where things live

There are two layers — the **shared knowledge repo (slab)** and the
**current project**.

### Slab knowledge repo (`~/workspace/slab/`)

```
slab/
├── AGENTS.md                # this file (symlinked into every project)
├── .cursor/
│   ├── rules/               # auto-attached engineering rules (by glob)
│   └── skills/              # task-driven SOPs the agent can invoke
├── knowledge/               # stable domain knowledge
│   ├── hardware/            # GPU specs (MI300X / MI355X / B200 etc.)
│   ├── systems/             # backend (TorchTitan / Megatron / Primus) know-how
│   ├── moe/                 # MoE dataflow, parallelism, research directions
│   ├── kernels/             # GEMM / FP8 / comm-compute overlap patterns
│   ├── libraries/           # 3rd-party kernel libraries (CK, AITER, …)
│   └── pilot/               # autopilot / tuning recipes
├── papers/                  # paper reading notes (one slug per paper)
├── notes/                   # project work logs (one dir per project)
├── weekly/                  # weekly reports, foldered by year
├── artifacts/               # generated outputs (canvases/, html/)
└── scripts/                 # utility shell scripts
```

### Current project (anything under `~/workspace/<proj>/`)

- The project's own `README.md` is the authoritative project intro.
- `.cursor/rules/` and `.cursor/skills/` are **symlinks back into slab** —
  do not edit them in-place; edit them in slab so every project benefits.
- A `.cursor/rules/90-<project>-local.mdc` may exist with
  project-specific invariants.
- Project work logs live in `~/workspace/slab/notes/<project>/`, not in
  the project repo itself.

### Three-tier mental model

| Tier | Location | Lifetime | Who reads | Typical action |
|---|---|---|---|---|
| **Engineering context** | `AGENTS.md`, `.cursor/rules/`, `.cursor/skills/` | Stable | Agent (every session / by glob / on demand) | Read → follow |
| **Domain knowledge** | `slab/knowledge/`, `slab/papers/` | Evolves slowly | Agent on demand, human reference | Grep → cite |
| **Project work** | `slab/notes/<project>/`, `slab/weekly/` | Append-only log | Human primarily | Write progress notes |

## 4. When agent should consult what

| If the user is doing... | Read first |
|---|---|
| Editing a HIP/CUDA kernel | `.cursor/rules/10-gpu-kernels.mdc` (auto), `slab/knowledge/kernels/`, `slab/knowledge/hardware/` |
| Working on MoE algorithms or training | `.cursor/rules/20-moe.mdc` (auto), `slab/knowledge/moe/`, recent `slab/papers/` on MoE |
| Touching SLURM scripts or cluster jobs | `.cursor/rules/30-slurm.mdc` (auto), `.cursor/skills/slurm-*` |
| Writing notes / weekly reports / paper notes | `.cursor/rules/40-notes-style.mdc` (auto), the matching skill |
| Optimizing/tuning a training run | `.cursor/skills/pilot-*` (when added), `slab/knowledge/pilot/` |
| Analyzing a Kineto trace | `.cursor/skills/gpu-trace-analysis` or `.cursor/skills/trace-vram-canvas` |
| Comparing backends (TorchTitan / Megatron / Primus) | `.cursor/skills/backend-gap-report`, `slab/knowledge/systems/` |
| Picking / wrapping a 3rd-party kernel library (CK, AITER, hipBLASLt, primus-turbo, …) | `slab/knowledge/libraries/README.md` first, then `slab/knowledge/libraries/<lib>.md`; `_patterns.md` for cross-library trade-offs |
| Distilling a new external library into knowledge | `.cursor/skills/distill-operator-repo/SKILL.md` → outputs `slab/knowledge/libraries/<slug>.md` |
| Wiring a brand-new knowledge doc into discovery paths | `.cursor/skills/wire-knowledge-into-system/SKILL.md` |

(Paths starting with `slab/` resolve to `~/workspace/slab/` regardless
of which sibling project the agent is currently inside.)

## 5. Hard conventions (do not violate)

Apply to **knowledge / notes / papers / docs** written into slab.
Project application code follows the project's own style.

- **Filenames**: kebab-case, lowercase, ASCII. No spaces. No `README_<x>.md`
  siblings — each directory has exactly one `README.md`.
- **Paper notes** live only in `slab/papers/<slug>.md` (flat) or
  `slab/papers/<slug>/` (when supporting material exists). Slug omits
  years unless disambiguation requires it.
- **Project notes** live only in `slab/notes/<project>/`. Each project
  has a `README.md` (status + timeline + next steps).
- **Progress notes**: `YYYY-MM-DD_HHMM_<topic>.md` (minute precision).
- **Archive notes**: `YYYY-MM-DD_<topic>.md` (day precision).
- **Weekly reports**: `slab/weekly/<year>/YYYY-MM-DD_weekly_report_<range>[_EN].md`.
- **Knowledge docs**: `slab/knowledge/<topic>/<slug>.md` — short, focused,
  one concept per file.
- **No emojis** in any file unless explicitly requested.
- **No new top-level directories** in slab without updating this file.

## 6. Commit & PR etiquette

- Conventional, low-noise commit messages: imperative subject ≤ 72 chars,
  followed by a brief body when the diff is non-obvious.
- Agent **never commits** without an explicit user instruction.
- One logical concern per commit; do not bundle unrelated edits.
- Large directory restructures must touch one logical group per commit.
- Repo-specific details (default branch, remote URL, CI requirements)
  live in that repo's `.cursor/rules/90-<project>-local.mdc`.

## 7. Things the agent should NOT do

- Do not invent paths, paper titles, model names, or hardware specs. If
  unsure, grep `slab/knowledge/` and `slab/papers/` first; if still
  unsure, ask.
- Do not create new `summary/`, `docs/`, or sibling `README_<x>.md`
  files in slab — they were deliberately retired in the May 2026
  restructure.
- Do not touch `slab/notes/<project>/raw/` (raw experiment logs are
  human-curated dumps; treat as read-only).
- Do not write into `slab/.cursor/skills/` to record findings — that's
  what `slab/notes/` and `slab/knowledge/` are for. Skills hold reusable
  procedures only.
- Do not edit the symlinked `.cursor/rules/` or `.cursor/skills/` files
  from inside a sibling project — they resolve into slab and any edit
  affects every project. Edit them in slab itself.
