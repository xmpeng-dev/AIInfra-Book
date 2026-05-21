---
name: wire-knowledge-into-system
description: >-
  After a new doc is added to knowledge/<topic>/, wire it into every
  discovery path so the agent actually reaches for it on relevant tasks
  — local sub-directory README index, top-level knowledge/README.md,
  AGENTS.md §4 routing table, the matching .cursor/rules/*.mdc (extend
  globs and/or add a reference line), and at least one cross-reference
  from a related .cursor/skills/<name>/SKILL.md. Use when the user adds
  a knowledge file and asks to "接进来", "wire it up", "link this into
  the system", "让 agent 用上这个", or when finishing a `distill-*` /
  `read-paper` flow that produced a new knowledge doc. Do NOT use for
  notes/<project>/ files (those don't need system-wide wiring — only
  their own project README) or for papers/<slug>.md (those auto-flow via
  papers/README.md only).
---

# wire-knowledge-into-system

Bridge the gap between **"the knowledge file exists"** and **"the agent
actually reads it on relevant tasks"**. New `knowledge/<topic>/<slug>.md`
files are dark matter until five discovery paths point at them:

| # | Path | Lever | Effort |
|---|---|---|---|
| 1 | Sub-dir `README.md` index table | row that names the file and what it answers | 1 line |
| 2 | Top-level `knowledge/README.md` | only if a brand-new sub-topic was created | 1 line |
| 3 | `AGENTS.md` §4 routing table | row mapping a task-shape to the file | 1 line |
| 4 | A `.cursor/rules/*.mdc` | extend `globs:` and/or add a body reference | a few lines |
| 5 | One or more `.cursor/skills/<name>/SKILL.md` | cross-reference in the skill body | 1–3 lines per skill |

Paths 1–2 make it **searchable**. Path 3 makes the agent **reflexively
aware**. Paths 4–5 make it **auto-loaded** by the right triggers.
Without 3–5, new knowledge tends to never get read.

## When to apply

- A new file was just added under `knowledge/<topic>/`.
- A new sub-directory was created under `knowledge/`.
- A `distill-*` / `read-paper` / similar skill just finished and produced
  a knowledge artifact.
- The user says "接进来", "接路由", "wire it up", "link this in", "让
  agent 看到这个文档".

## When NOT to apply

- File is under `notes/<project>/` — those only update their own project
  README; they're not system-wide knowledge.
- File is under `papers/<slug>.md` — single-paper notes flow through
  `papers/README.md` and the `40-notes-style.mdc` rule already; do not
  add per-paper AGENTS.md rows.
- File is a workspace-local config (`.cursor/`, `scripts/`, `weekly/`).

## Workflow

### 1. Identify what was added

Take inventory:

```bash
git status --short knowledge/ AGENTS.md .cursor/rules/ .cursor/skills/
```

For each new / changed file under `knowledge/`:

- Is it a new file in an existing sub-dir?       → paths 1, 3, 4, 5
- Is it a new sub-dir (`knowledge/<new>/`)?     → all 5 paths
- Is it a substantive expansion of an existing  → review paths 3–5 to
  doc that changes its scope?                     ensure they still apply

### 2. Sub-dir README index (path 1)

Open `knowledge/<topic>/README.md`. The file index is a table:

```markdown
| 文件 | 内容 |
|---|---|
| [`<slug>.md`](./<slug>.md) | <one-line description of what it answers> |
```

Add a row for the new file. Group with similar entries if the table has
sections. If the README is missing an index table, this skill's first
job is to create one (do not skip).

### 3. Top-level `knowledge/README.md` (path 2, only if new sub-dir)

Add a row to the "子领域" table:

```markdown
| [`<new>/`](./<new>/README.md) | <one-line scope> | <when to consult> |
```

Order does not matter strictly; keep related sub-topics adjacent
(hardware/systems/kernels/libraries cluster; moe/pilot cluster).

### 4. `AGENTS.md` §4 routing table (path 3)

Open `AGENTS.md` and find the table headed "When agent should consult
what". Add one row that maps a **task shape** to the new doc, not a
keyword:

| If the user is doing... | Read first |
|---|---|
| <task description in user-facing terms> | `<absolute or relative path>` |

**Good row:** `Picking / wrapping a 3rd-party kernel library (CK, AITER,
hipBLASLt, primus-turbo, …) | knowledge/libraries/README.md first, then
knowledge/libraries/<lib>.md`

**Bad row:** `MoE | knowledge/moe/` — too vague, no task shape.

If the new doc is purely additive within an existing routing row, edit
that row in place instead of adding a duplicate.

### 5. Extend a `.cursor/rules/*.mdc` (path 4)

Find the rule whose `globs:` matches the files an agent is editing when
this knowledge becomes relevant. Common matches:

| Knowledge topic | Likely rule |
|---|---|
| `hardware/`, `kernels/`, `libraries/` | `10-gpu-kernels.mdc` |
| `moe/` | `20-moe.mdc` |
| `systems/`, `pilot/` | possibly a new rule; otherwise none |
| `papers/` | `40-notes-style.mdc` (existing, no change needed) |

Two ways to wire:

- **Extend `globs:`** so the rule auto-loads when relevant files are
  touched (e.g. add `knowledge/libraries/**` and `3rd/**`).
- **Add a body section** named after the new sub-topic with a one-paragraph
  "before doing X, consult `<path>`" instruction.

If no existing rule fits, create a new `.cursor/rules/<NN>-<topic>.mdc`
following the existing numbering convention (`00–49` for rules from
slab; project-local rules use `90+`).

### 6. Cross-reference from related skills (path 5)

Identify 1–3 skills whose workflows should consume the new knowledge:

```bash
ls .cursor/skills/
```

Add a short header callout near the top of each, not at the bottom —
placement at the top guarantees the agent sees it before doing the
work:

```markdown
> **Before doing X**, consult `knowledge/<topic>/<slug>.md` for <why>.
```

If no existing skill is relevant, this is fine — wiring 1, 3, 4 is
enough for retrieval. Do not invent a skill just to host a cross-ref.

### 7. Verify and report

```bash
git diff --stat
```

Tell the user:

- Paths touched (sub-dir README, knowledge/README.md if changed,
  AGENTS.md, the rule file(s), each cross-referenced skill)
- A draft commit message (e.g. `wire knowledge/<topic>/<slug> into
  AGENTS.md, <rule>, and <skills>`)
- Anything you intentionally skipped (e.g. "no existing skill fit the
  topic, so path 5 was left empty")

## Patterns

### Pattern: AGENTS.md row wording

The left column is **what the user is doing**, not **what the file is
about**. Past tense / imperative / noun phrases all work:

| User-doing | Read first |
|---|---|
| Optimizing a Triton kernel on MI355X | `.cursor/rules/10-gpu-kernels.mdc` (auto), `knowledge/hardware/...`, `.cursor/skills/mi355_hardware_aware/` |
| Picking a 3rd-party kernel library | `knowledge/libraries/README.md` then `<lib>.md` |

### Pattern: rule body addition

Keep new sections short (≤ 10 lines) and instructive, not narrative:

```markdown
## <Concern>

Before doing <X>, consult `<path>`. <Why this is non-obvious.>
Do not <anti-pattern> without citing which document informed the choice.
```

### Pattern: skill cross-reference placement

Top of file, right after the H1, before the "When to apply" section. A
blockquote works because it's visually distinct from the body:

```markdown
# <Skill Title>

> Before <task>, consult `knowledge/<topic>/<slug>.md` — <one-line why>.

## When to apply
...
```

## Anti-patterns

- ❌ Adding the knowledge file but skipping paths 3–5. The file then
  exists only for users who already know the path. Agent never finds it.
- ❌ AGENTS.md row that says only the topic name ("MoE", "hardware") —
  the routing table is keyed on task shape, not topic.
- ❌ Extending `globs:` of every rule "just in case" — rules consume
  tokens on every matching edit. Only extend where the rule body will
  actually need to reference the new knowledge.
- ❌ Adding cross-references to skills that do not actually consume the
  knowledge. Bloats unrelated skills and dilutes signal.
- ❌ Writing a brand-new rule when an existing one fits — proliferation
  of single-concern rules makes the auto-load behavior unpredictable.
- ❌ Wiring `papers/<slug>.md` individually into AGENTS.md — papers are
  many; route to the directory, never to single files.

## Example (canonical)

The May 2026 wiring of `knowledge/libraries/` is the reference example:

| Path | Change | Commit |
|---|---|---|
| 1 sub-dir README | created `knowledge/libraries/README.md` with index table for 4 libs + `_patterns.md` | `knowledge/libraries: distillations of CK / AITER / ...` |
| 2 top-level README | added `libraries/` row to `knowledge/README.md` | same |
| 3 AGENTS.md §4 | added 2 rows: "Picking a 3rd-party kernel library" + "Distilling a new external library" | `wire knowledge/libraries/ into AGENTS.md, ...` |
| 4 rule | extended `10-gpu-kernels.mdc` globs to include `knowledge/libraries/**` + `3rd/**`; added "Library / backend selection" section | same |
| 5 skills | added header cross-refs to `amd-gemm-optimization`, `cco-pipeline-overlap`, `cuda_gemm_optimization` | same |

`git log -1 --grep="wire knowledge/libraries"` shows the exact diff to
mimic.

## Related

- Slab routing table: `AGENTS.md` §4
- Knowledge layer overview: `knowledge/README.md`
- Sibling skills: `.cursor/skills/distill-operator-repo/`,
  `.cursor/skills/read-paper/`, `.cursor/skills/paper-deep-analysis/`,
  `.cursor/skills/create-slab-skill/`
