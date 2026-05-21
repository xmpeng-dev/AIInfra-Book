---
name: distill-operator-repo
description: >-
  Distill the design philosophy and reusable design patterns of a single
  operator / kernel-library repository (Composable Kernel, AITER,
  hipBLASLt, primus-turbo, CUTLASS, FlashAttention, Triton, etc.) into
  one short note under `knowledge/libraries/<slug>.md`. The output is
  about IDEAS, not implementation details — no file walkthroughs, no
  dtype matrices, no code snippets. Use when the user asks to extract
  design 思路 / 设计理念 / 可借鉴的模式 from a library at `3rd/<repo>`
  or any external kernel repo, or says "蒸馏算子库", "总结设计思想",
  "提炼可复用模式", "把 xxx 的设计写成领域知识". Do NOT use for paper
  notes (`read-paper`), upstream-vs-fork comparisons
  (`backend-gap-report`), single kernel pattern docs (`knowledge/kernels/`
  directly), or per-file code archaeology — this skill explicitly
  refuses code-level detail.
---

# distill-operator-repo

Produce one short, idea-dense note per kernel library so future
operator development can grep `knowledge/libraries/` and learn each
library's design philosophy and the patterns worth borrowing — in
5 minutes, without re-reading the source tree.

The output is **conceptual**. The skill explicitly avoids:

- File walkthroughs (X calls Y in `foo.hpp` line 123)
- Dtype × arch support matrices
- Build/install instructions
- Code snippets longer than 3 lines
- Exhaustive directory listings

Those belong in upstream docs, our own kernel notes
(`knowledge/kernels/<pattern>.md`), or hardware notes
(`knowledge/hardware/<arch>.md`). This skill answers two questions only:

1. **What design ideas is this library built around?**
2. **Which of those ideas should we steal for our own kernel work?**

## When to apply

- User points at a repo (often under `3rd/<name>/`, but can be any
  checkout) and asks to characterize / summarize / distill its **design
  思路** or **可借鉴的模式**.
- User says "把这个库的设计思路总结成 knowledge", "梳理 X 库的核心设计
  理念", "X 库有哪些模式我们能借鉴", "把 3rd 下的库写成 knowledge".
- A new dependency landed under `3rd/` and we want a short stable
  reference before integrating it.

## When NOT to apply

- The artifact is a paper, not a repo — use `read-paper` or
  `paper-deep-analysis`.
- The goal is a delta-vs-upstream report on a Primus backend — use
  `backend-gap-report`.
- The goal is a focused single-kernel pattern doc (e.g. "FP8 expert
  GEMM on gfx950") — write directly under `knowledge/kernels/`.
- The user wants implementation-level walkthroughs, line-by-line code
  reading, or a build/integration guide — this skill explicitly refuses
  that scope. Suggest reading the upstream README or writing a project
  note under `notes/<project>/` instead.
- Multi-repo cross-cutting synthesis — first distill each repo with
  this skill, then write `knowledge/libraries/_patterns.md` directly.

## Scope: one repo per invocation, ideas only

If the user names multiple repos in a single ask, distill them
sequentially — one note per repo — and only at the end consider writing
a cross-cutting synthesis. Each note is **conceptual** (≤ ~300 lines
of body, more typically 150–250).

## Output location

| Artifact | Path |
|---|---|
| Per-library note | `knowledge/libraries/<repo-slug>.md` |
| Index | `knowledge/libraries/README.md` |
| Top-level routing | row in `knowledge/README.md` "子领域" table |

`<repo-slug>` is kebab-case, lowercase, matches the repo's canonical
name with vendor prefixes stripped (`composable-kernel`, `aiter`,
`hipblas`, `hipblaslt`, `primus-turbo`, `cutlass`, `flash-attention`).

If `knowledge/libraries/` does not yet exist:

1. Create `knowledge/libraries/README.md` using the index template
   below.
2. Add one row to `knowledge/README.md`'s 子领域 table pointing at it.

## Workflow

### 1. Resolve inputs

| Input | Default |
|---|---|
| Repo path | `3rd/<name>/` (or user-provided absolute path) |
| Output slug | derived from repo's canonical name |
| Language | EN headers + 中文 body, matching `knowledge/kernels/` |

If the repo's purpose is not obvious from its name, ask before
proceeding.

### 2. Anchor with provenance

Capture once at the top of the note:

```bash
cd <repo>
git remote -v | head -1
git log -1 --format='%h %ci %s'
git rev-parse --abbrev-ref HEAD
du -sh .
```

Five lines into the note header — future readers know which snapshot
the analysis reflects.

### 3. Read the public-facing layer only

The whole point of this skill is to harvest **stated intent**, not
reverse-engineer it from code. Read, in order, until you can answer
both top-level questions:

1. `README.md` (always)
2. `docs/` index, `ARCHITECTURE.md`, `TERMINOLOGY.md`, `DESIGN.md`
3. The top of one or two key public headers (header doxygen comments
   often state the abstraction's purpose)
4. Top of `setup.py` / `pyproject.toml` / top-level `CMakeLists.txt`
   only if the build story directly reveals a design choice

**Stop here.** Do NOT trace call chains, do NOT open kernel
implementations, do NOT enumerate intrinsics. If you find yourself
opening a third `.cu` / `.hip` / `.cpp` file, you're past the skill's
scope.

Prefer **verbatim quotes** from the README / docs when stating design
philosophy. Verbatim earns its token cost; paraphrase usually doesn't.

### 4. Sketch a conceptual layout

One small table mapping the 4–8 top-level directories to their **role
in the design**, not their file contents. Skip plumbing dirs
(`.github/`, `cmake/`, `script/`). The reader should learn the
library's shape, not its file census.

### 5. Identify 3–7 core design ideas

For each: **name**, one paragraph covering (a) what the idea is,
(b) what problem it solves, (c) why the library chose it over the
alternative. **No code. No file paths beyond a single "see X" pointer
per idea.**

Examples of the kind of idea to surface:

- "Tile-based programming model with tensor coordinate transformation"
  (CK) — separates layout from algorithm.
- "JIT compile on first call with on-disk cache" (AITER) — trades
  startup latency for kernel specialization breadth.
- "Marshalling layer over multiple backends" (hipBLAS) — same API
  whether ROCm or CUDA is below.
- "Op registry with multi-backend dispatch inside a single op"
  (AITER: Triton vs CK vs ASM) — lets one Python entry pick the best
  backend per shape.
- "Single-header HIP utility template library" (AITER's Opus) —
  optimizes build time at the cost of recompilation surface.

Stop at 7. If the library has more, pick the ideas that **matter for
our future kernel work**, not all of them.

### 6. Extract the reusable patterns (THE CENTERPIECE)

3–8 patterns we can borrow. Each row in the table answers four
questions:

| Pattern | What it solves | Where it applies in our work | Caveats |
|---|---|---|---|

"Our work" = Primus / Primus-Turbo kernels, MoE training, MonolithMoE,
new HIP/Triton kernels for MI300X/MI355X. Be specific. Vague entries
("good abstraction") fail the bar.

This section is the *reason* the skill exists. If sections 4–5 are
strong but this one is weak, the note has failed.

### 7. Sketch ecosystem position

One-line dependency arrow plus one paragraph of context. Examples:

```
AITER         → CK / ck_tile (GEMM) + Triton (norms) + ASM (attention)
primus-turbo  → CK + hipBLASLt + AITER
hipBLAS       → rocBLAS (AMD) | cuBLAS (NVIDIA)
```

If neighbor libraries have already been distilled under
`knowledge/libraries/`, link to them.

### 8. Wire up indices

1. Add or update a row in `knowledge/libraries/README.md` (create the
   file using the index template if it does not yet exist).
2. If you just created `knowledge/libraries/`, add a row to
   `knowledge/README.md`'s 子领域 table.
3. Do not commit. Tell the user the resulting paths.

## Output template — `knowledge/libraries/<slug>.md`

```markdown
# <Repo display name>

> **Repo:** `<owner>/<repo>` &nbsp; **Local path:** `3rd/<name>/`
> **Snapshot:** `<short-sha>` `<commit-date>` on branch `<branch>`
> **Size:** ~`<MB>` MB &nbsp; **License:** `<license>`
> **Distilled:** `YYYY-MM-DD`

## TL;DR

2–3 sentences: what this library is conceptually, and the single most
important design idea worth borrowing.

## 1. 库定位 (positioning)

- 一句话定位
- 它**是什么** (2–4 bullets, conceptual)
- 它**不是什么 / 不做什么** (1–3 bullets — explicit non-goals matter
  more than feature lists)
- 谁在用它 (downstream consumers; one line each)

## 2. 顶层架构 (conceptual layout)

| Directory | Role in the design | Notes |
|---|---|---|
| ... | ... | ... |

(4–8 rows, max. Skip plumbing dirs.)

## 3. 核心设计理念 (core design ideas)

### 3.1 <Idea 1>
One paragraph. What the idea is, what it solves, why this library
chose it. Quote the README verbatim when possible. **No code.**

### 3.2 <Idea 2>
...

(3–7 ideas total.)

## 4. 可借鉴的设计模式 (patterns to borrow) ★

| Pattern | What it solves | Where it applies to us | Caveats |
|---|---|---|---|
| ... | ... | Primus / MoE / kernel dev | ... |
| ... | ... | ... | ... |

(3–8 rows. **This section is the point of the note.** "Where it
applies to us" must be specific — name a Primus module, kernel name,
or workflow.)

## 5. 与生态的关系 (ecosystem position)

One-line dependency arrow.

One paragraph: where this library sits in the AMD / NVIDIA kernel-lib
stack, and how it interacts with the others under
`knowledge/libraries/`.

## 6. 进一步阅读 / TODO

- Open questions left after this pass
- ≤ 5 entry-point files worth reading next, with one-line "why"
- Patterns we are uncertain about and want to validate
```

## Output template — `knowledge/libraries/README.md`

```markdown
# knowledge/libraries/ — 第三方算子库设计理念

每个文件蒸馏 **一个** 第三方算子 / kernel 库的**设计理念**和**可借鉴的
模式**,作为后续算子开发和库选型时的快速参考。不做代码级别的说明 —
那是 upstream README 的工作,或者写到 `knowledge/kernels/<pattern>.md`
里去。生成流程见 `.cursor/skills/distill-operator-repo/SKILL.md`。

## 索引

| 库 | 位置 | 一句话定位 | 状态 |
|---|---|---|---|
| [Composable Kernel](composable-kernel.md) | `3rd/composable_kernel` | AMD 算子模板库 (tile-based, tensor-coord transform) | active |
| [AITER](aiter.md) | `3rd/aiter` | ROCm 推理/训练算子集合 (CK + Triton + ASM) | active |
| [hipBLAS](hipblas.md) | `3rd/hipBLAS` | 经典 BLAS marshalling 层 | deprecated -> rocm-libraries |
| [Primus-Turbo](primus-turbo.md) | `3rd/turbo` | AMD 训练侧 fused-op 库 | active |

## 跨库综述

- [`_patterns.md`](_patterns.md) — 当 ≥ 3 个库被蒸馏后,在此沉淀重复
  出现的设计模式 (templated dispatch / JIT cache / op registry / tile
  policy / marshalling 等)。
```

## Anti-patterns

- ❌ Tracing one operator end-to-end with file paths and code at each
  layer. That's a project note, not a library distillation. Move it to
  `notes/<project>/`.
- ❌ Listing every dtype × every arch × every tile shape the library
  supports. Hardware tables belong in `knowledge/hardware/`; a
  conceptual library note doesn't need them.
- ❌ Copy-pasting README sections wholesale. Quote selectively; the
  note's job is to compress.
- ❌ Code snippets longer than 3 lines. If you need code to make a
  point, the point is probably too low-level for this skill.
- ❌ A weak "patterns to borrow" section (Section 4). That section is
  the entire point. Empty or generic entries ("good abstraction layer",
  "well-designed templates") mean the note has failed.
- ❌ "Where it applies to us" entries that don't name a concrete Primus
  module / kernel / workflow.
- ❌ A "key file index" of 20+ paths. ≤ 5 entry points in Section 6,
  each with a "why".
- ❌ Writing the file under `knowledge/kernels/` because it's about
  kernels. `knowledge/kernels/` is for portable one-pattern docs;
  `knowledge/libraries/` is for per-library conceptual notes.

## Examples

**1. Foundational library:**

User: "把 `3rd/composable_kernel` 蒸馏成领域知识"

→ Write `knowledge/libraries/composable-kernel.md`, ~200 lines.
→ Sections 3 and 4 emphasize three pillars: tile-based programming
  model, tensor coordinate transformation, and per-(dtype,arch)
  instance enumeration. Section 4 lists how each could apply to our
  Primus kernel work.
→ Bootstrap `knowledge/libraries/README.md` (first library distilled).
→ Add a row to `knowledge/README.md`'s 子领域 table.

**2. Deprecated wrapper:**

User: "也写一份 hipBLAS 的"

→ Write `knowledge/libraries/hipblas.md`, ~100 lines.
→ Single design idea (marshalling pattern) gets the spotlight.
→ Section 4 has one strong row: "How to design a portable wrapper
  across two backends, applied to our own framework-portability
  layer."
→ Append a row to `knowledge/libraries/README.md`.

**3. After 3+ libraries:**

User: "现在 4 个库都有了,做个综述"

→ This skill **stops here**. Write `knowledge/libraries/_patterns.md`
  directly, referencing each per-library file, without re-invoking
  this skill.

## Related

- Layout reference: `AGENTS.md` §3, `.cursor/rules/00-always.mdc`
- Style: `.cursor/rules/40-notes-style.mdc`
- GPU conventions used inside the notes:
  `.cursor/rules/10-gpu-kernels.mdc` (gfx942 vs gfx950, wavefront=64)
- Sibling skills:
  - `read-paper` / `paper-deep-analysis` — for papers, not repos
  - `backend-gap-report` — for upstream-vs-fork comparisons
  - `progress-note` / `archive-notes` — for project work, not stable
    library knowledge
- Cross-cutting kernel patterns live in `knowledge/kernels/` and
  `knowledge/hardware/`; this skill's output complements them at a
  higher level of abstraction.
