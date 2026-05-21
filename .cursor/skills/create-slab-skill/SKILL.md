---
name: create-slab-skill
description: >-
  Author a new Cursor Agent Skill and install it inside the slab knowledge
  repo at <slab-root>/.cursor/skills/<name>/SKILL.md so it is shared across
  every sibling project that has been bootstrapped against slab. Use when
  the user asks to add / write / create a skill, says "新建一个 skill",
  "写一个 slab skill", "把这个流程沉淀成 skill", or asks where a new skill
  should live. Do NOT use to create personal one-off skills outside slab —
  for that, defer to the upstream `create-skill` skill.
---

# create-slab-skill

Author a new skill **inside the slab knowledge repo**, not in any
sibling project, not in `~/.cursor/skills/`, and **never** in
`~/.cursor/skills-cursor/` (that path is Cursor-managed and overwritten
on sync).

## Why this exists (and how it differs from upstream `create-skill`)

The upstream skill at `~/.cursor/skills-cursor/create-skill/SKILL.md`
covers general SKILL.md authoring (frontmatter, descriptions,
progressive disclosure, anti-patterns). **Read it first** for the
craft. This skill adds the slab-specific constraints:

| Concern | Upstream `create-skill` | This skill |
|---|---|---|
| Storage location | asks the user (personal vs project) | **always** `<slab>/.cursor/skills/<name>/` |
| Distribution to other projects | n/a | inherited automatically via slab bootstrap symlinks |
| House style | generic | slab's tone (concise, table-driven, English/中文 mixed where appropriate) |
| Cross-references | self-contained | encourages linking to `knowledge/`, `papers/`, sibling skills |

## When to apply

- User wants a new skill that should be available across **every** project
  bootstrapped from slab (MMOE / MonolithEP / Primus / RocMoE / etc.).
- A workflow / SOP emerged in conversation and the user says "记一下这个
  流程" / "把它做成 skill".

## When NOT to apply

- The user explicitly wants a one-off skill scoped to a single sibling
  project — that goes under `<project>/.cursor/skills/<name>/` directly,
  not in slab. Use the upstream `create-skill` skill for those.
- The skill needs to be private to one user across many machines — that's
  the upstream personal `~/.cursor/skills/` use case.
- The user just wants a Cursor **rule** (engineering convention applied
  by glob), not a workflow. Skills are for SOPs; rules go in
  `.cursor/rules/`.

## Workflow

### 1. Resolve slab root

Slab is the workspace containing both `AGENTS.md` and `.cursor/skills/`.
Default path on this machine: `~/workspace/slab`. If unsure, run:

```bash
test -f ~/workspace/slab/AGENTS.md && test -d ~/workspace/slab/.cursor/skills \
  && echo "slab ok" || echo "slab not found at default path"
```

If the current working dir is itself a slab clone elsewhere, prefer that.

### 2. Gather requirements

Use `AskQuestion` when available, otherwise ask conversationally:

| Field | Constraint |
|---|---|
| `name` | kebab-case, ≤ 64 chars, lowercase letters/digits/hyphens, no `_` |
| `description` (frontmatter) | third person, includes WHAT + WHEN + trigger phrases in both EN and 中文, ≤ 1024 chars |
| Scope | confirm "shared across all slab-bootstrapped projects" — if no, this skill does not apply |
| Helpers needed | `reference.md` / `examples.md` / `scripts/` / `templates/` ? |

### 3. Check for collisions

```bash
ls <slab>/.cursor/skills/ | grep -i <name-or-prefix>
```

If a near-name exists, either extend that skill or pick a more specific
slug. Avoid generic names: `helper`, `utils`, `tools`.

### 4. Read references (one level deep, only what's needed)

- `~/.cursor/skills-cursor/create-skill/SKILL.md` — generic craft (frontmatter,
  description quality, anti-patterns). Skim once if it's not already in context.
- 1–2 existing slab skills with similar shape:
  - **SOP / workflow skill** → `.cursor/skills/progress-note/SKILL.md`
  - **Domain-knowledge skill** → `.cursor/skills/cco-pipeline-overlap/SKILL.md`
  - **Tool-usage skill** → `.cursor/skills/slurm-xiaoming-dev-container/SKILL.md`
  - **Skill with templates** → `.cursor/skills/trace-vram-canvas/SKILL.md`

### 5. Write `<slab>/.cursor/skills/<name>/SKILL.md`

Use the template below. Then add `reference.md`, `examples.md`,
`scripts/`, `templates/` only when the SKILL.md genuinely benefits from
progressive disclosure (rule of thumb: split when SKILL.md crosses
~300 lines or when a helper is a multi-file artifact).

### 6. Verify

- File length: `wc -l SKILL.md` → should be ≤ 500, ideally ≤ 250
- Frontmatter parses (no tabs, correct YAML)
- No emojis in the body (slab convention) unless the skill is about
  generating user-facing output that legitimately uses them
- All cross-references resolve: `knowledge/...`, `papers/...`, sibling
  skills exist
- `ReadLints` on the new file is clean

### 7. Tell the user the result

State exactly:

- Absolute path of the new SKILL.md
- Which sibling projects already have it (those bootstrapped with
  `scripts/bootstrap-project-from-slab.sh` — the symlink is automatic
  on next bootstrap; existing bootstraps may need a re-run to pick up
  newly added skill directories)
- Whether `AGENTS.md` or any rule should be updated to point at the new
  skill (only if it's load-bearing for an existing workflow)

## SKILL.md template (slab style)

```markdown
---
name: <kebab-case-name>
description: >-
  <WHAT this skill does in third person, one or two sentences>
  <WHEN to apply: triggers, file types, user phrases — EN and 中文>
  <Optional: when NOT to apply, to disambiguate from neighbors>
---

# <Skill Title>

> One-paragraph context: where this skill fits in slab's workflow and
> which neighboring skills it complements or replaces.

## When to apply

- Bullet trigger 1 (concrete situation)
- Bullet trigger 2
- ...

## When NOT to apply

- Boundary case 1
- Boundary case 2

## Workflow

### 1. <First step>
<concrete command or decision>

### 2. <Second step>
...

## Templates / outputs

<Inline a template if small; otherwise link to `templates/<name>.md`.>

## Examples

<1–2 concrete examples from real slab projects (gpt-oss / monolith-moe /
mlperf-llama / pilot). Cite paths.>

## Anti-patterns

- ❌ ...
- ❌ ...

## Related

- Slab knowledge: `knowledge/<topic>/...`
- Other skills: `.cursor/skills/<sibling-name>/`
- Upstream reference: `~/.cursor/skills-cursor/<name>/` (if applicable)
```

## Slab house style for skill descriptions

- **Third person**, present tense.
- Lead with WHAT in one clause, then WHEN with concrete trigger phrases.
- Include both English and 中文 triggers when the workflow is bilingual
  (most of slab's are, because conversations switch language).
- End with **NOT** triggers if there is a sibling skill it might be
  confused with.

Good example (from `progress-note`):

> Write a progress note into notes/<project>/ with minute-precision
> filename and a detailed timestamp+environment header, covering when /
> what problem / what was done / achieved effect / next directions, then
> update the project README. Use when the user asks to write a note,
> log progress, record current work, or says "写一个 note", "记一下进展",
> "把这次实验写下来", "log 一下".

## Anti-patterns specific to slab

- ❌ Writing to `~/.cursor/skills-cursor/<name>/` — auto-synced, will be
  clobbered. Hard rule.
- ❌ Writing to a sibling project's `<project>/.cursor/skills/<name>/` —
  defeats the "single source of truth" the bootstrap setup provides.
- ❌ Duplicating upstream `create-skill` guidance — link to it instead.
- ❌ Putting domain knowledge inside the SKILL.md when it belongs under
  `knowledge/<topic>/` — SKILL.md should be the *procedure*, not the
  *facts*. Procedures cite facts.
- ❌ Adding "history" / "changelog" sections inside SKILL.md — use git
  log for that.

## Propagation reminder

New skill dirs created under `<slab>/.cursor/skills/` propagate to
sibling projects only on the next run of
`scripts/bootstrap-project-from-slab.sh`. For already-bootstrapped
projects, re-run it (idempotent; only adds the new symlink):

```bash
~/workspace/slab/scripts/bootstrap-project-from-slab.sh ~/workspace/<project>
```

## Related

- Upstream: `~/.cursor/skills-cursor/create-skill/SKILL.md` (general craft)
- Distribution: `scripts/bootstrap-project-from-slab.sh`
- Layout reference: `AGENTS.md` §3 (Repo layout)
