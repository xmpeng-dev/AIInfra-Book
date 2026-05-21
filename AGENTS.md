# AGENTS.md — Always-loaded context for this knowledge repo

> This file is read at the start of every agent session. Keep it stable,
> short, and high-signal. Volatile content belongs in
> `notes/<project>/README.md` or the relevant `knowledge/<topic>/README.md`.

---

## 1. Who owns this repo

Xiaoming Peng — training systems & AI infrastructure. The repo is a
**personal knowledge base**, not application code. There is no build,
no test suite, no deployable artifact. The deliverables are Markdown
notes, design docs, Cursor skills/rules, and an occasional canvas / HTML
export.

## 2. Hardware & environment context

Day-job hardware (almost all notes are written against this):

- **AMD MI300X** (CDNA3, gfx942)
- **AMD MI325X** (CDNA3 successor, gfx942)
- **AMD MI355X** (CDNA4, gfx950) — current default for new experiments
- Comparison reference: **NVIDIA H100 / H200 / B200**

Training stack:

- ROCm + PyTorch, RCCL, Megatron-LM / Megatron-Core, TorchTitan, **Primus**
- SLURM-managed multi-node cluster
- Dev environment: Podman container `xiaoming-dev` (see
  `scripts/start-xiaoming-dev-fix-container.sh`); image
  `docker.io/rocm/primus:v26.2`
- Profiling: PyTorch Profiler / Kineto traces; rocprof for HIP kernels

## 3. Repo layout (top-level only)

```
slab/
├── AGENTS.md                # this file
├── README.md                # personal/repo intro
├── .cursor/
│   ├── rules/               # auto-attached engineering rules (by glob)
│   └── skills/              # task-driven SOPs the agent can invoke
├── knowledge/               # stable domain knowledge (hardware/systems/moe/kernels/pilot)
├── papers/                  # paper reading notes (one file per paper, flat)
├── notes/                   # project work logs (one directory per project)
├── weekly/                  # weekly reports, foldered by year
├── artifacts/               # generated outputs (canvases/, html/)
└── scripts/                 # utility shell scripts
```

Three-tier mental model of what lives where:

| Tier | Location | Lifetime | Who reads | Typical action |
|---|---|---|---|---|
| **Engineering context** | `AGENTS.md`, `.cursor/rules/`, `.cursor/skills/` | Stable | Agent (every session / by glob / on demand) | Read → follow |
| **Domain knowledge** | `knowledge/`, `papers/` | Evolves slowly | Agent on demand, human reference | Grep → cite |
| **Project work** | `notes/<project>/`, `weekly/` | Append-only log | Human primarily | Write progress notes |

## 4. When agent should consult what

| If the user is doing... | Read first |
|---|---|
| Editing a HIP/CUDA kernel | `.cursor/rules/10-gpu-kernels.mdc` (auto), `knowledge/kernels/`, `knowledge/hardware/` |
| Working on MoE algorithms or training | `.cursor/rules/20-moe.mdc` (auto), `knowledge/moe/`, recent `papers/` on MoE |
| Touching SLURM scripts or cluster jobs | `.cursor/rules/30-slurm.mdc` (auto), `.cursor/skills/slurm-*` |
| Writing notes / weekly reports / paper notes | `.cursor/rules/40-notes-style.mdc` (auto), the matching skill |
| Optimizing/tuning a training run | `.cursor/skills/pilot-*` (when added), `knowledge/pilot/` |
| Analyzing a Kineto trace | `.cursor/skills/gpu-trace-analysis` or `.cursor/skills/trace-vram-canvas` |
| Comparing backends (TorchTitan/Megatron/Primus) | `.cursor/skills/backend-gap-report`, `knowledge/systems/` |
| Picking / wrapping a 3rd-party kernel library (CK, AITER, hipBLASLt, primus-turbo, …) | `knowledge/libraries/README.md` first, then `knowledge/libraries/<lib>.md`; `_patterns.md` for cross-library trade-offs |
| Distilling a new external library into knowledge | `.cursor/skills/distill-operator-repo/SKILL.md` → outputs `knowledge/libraries/<slug>.md` |

## 5. Hard conventions (do not violate)

- **Filenames**: kebab-case, lowercase, ASCII. No spaces. No `README_<x>.md`
  siblings — each directory has exactly one `README.md`.
- **Paper notes** live only in `papers/<slug>.md` (flat) or
  `papers/<slug>/` (when supporting material exists). Slug omits years
  unless disambiguation requires it.
- **Project notes** live only in `notes/<project>/`. Each project has a
  `README.md` (status + timeline + next steps).
- **Progress notes**: `YYYY-MM-DD_HHMM_<topic>.md` (minute precision).
- **Archive notes**: `YYYY-MM-DD_<topic>.md` (day precision).
- **Weekly reports**: `weekly/<year>/YYYY-MM-DD_weekly_report_<range>[_EN].md`.
- **Knowledge docs**: `knowledge/<topic>/<slug>.md` — short, focused,
  one concept per file.
- **No emojis** in any file unless explicitly requested.
- **No new top-level directories** without updating this file.

## 6. Commit & PR etiquette

- Repo is `xmpeng-dev/system-lab` on GitHub, default branch `main`.
- Conventional, low-noise commit messages: imperative subject ≤ 72 chars,
  followed by a brief body when the diff is non-obvious.
- Agent **never commits** without an explicit user instruction.
- Large directory restructures must touch one logical group per commit.

## 7. Things the agent should NOT do

- Do not invent paths, paper titles, model names, or hardware specs. If
  unsure, grep `knowledge/` and `papers/` first; if still unsure, ask.
- Do not create new `summary/`, `docs/`, or sibling `README_<x>.md`
  files — they were deliberately retired in the May 2026 restructure.
- Do not touch `notes/<project>/raw/` (raw experiment logs are
  human-curated dumps; treat as read-only).
- Do not write into `.cursor/skills/` to record findings — that's what
  `notes/` and `knowledge/` are for. Skills hold reusable procedures only.
