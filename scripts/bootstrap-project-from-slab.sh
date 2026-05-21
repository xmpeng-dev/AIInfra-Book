#!/usr/bin/env bash
# Bootstrap a project to use the slab knowledge repo as its shared
# engineering context (rules + skills + AGENTS.md pointer).
#
# Idempotent. Safe to re-run. Uses symlinks; nothing is copied.
# Per-machine (the slab path is absolute) — do NOT commit the
# symlinks; add the suggested .gitignore entries below.
#
# Usage:
#   bootstrap-project-from-slab.sh            # bootstrap current dir
#   bootstrap-project-from-slab.sh /path/proj # bootstrap a specific path
#   bootstrap-project-from-slab.sh --unlink   # remove all slab symlinks
#   bootstrap-project-from-slab.sh --check    # dry-run: show what would change

set -euo pipefail

SLAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-$(pwd)}"
MODE="install"

for arg in "$@"; do
  case "$arg" in
    --unlink) MODE="unlink" ;;
    --check)  MODE="check" ;;
  esac
done

if [[ "${1:-}" == --* ]]; then
  TARGET="$(pwd)"
fi

if [[ ! -d "$TARGET" ]]; then
  echo "error: target dir does not exist: $TARGET" >&2
  exit 1
fi

if [[ "$TARGET" == "$SLAB" ]]; then
  echo "error: cannot bootstrap slab into itself" >&2
  exit 1
fi

log() { printf "%s\n" "$*"; }

# ---------- discovery ----------
RULES=()
while IFS= read -r f; do RULES+=("$f"); done < <(ls "$SLAB/.cursor/rules"/*.mdc 2>/dev/null)

SKILLS=()
while IFS= read -r d; do SKILLS+=("$d"); done < <(ls -d "$SLAB/.cursor/skills"/*/ 2>/dev/null)

log "slab:    $SLAB"
log "target:  $TARGET"
log "rules:   ${#RULES[@]} files"
log "skills:  ${#SKILLS[@]} dirs"
log "mode:    $MODE"
log ""

do_link() {
  local src="$1" dst="$2"
  if [[ -L "$dst" ]]; then
    local cur; cur="$(readlink "$dst")"
    if [[ "$cur" == "$src" ]]; then
      [[ "$MODE" == "check" ]] && log "  [ok]   $dst"
      return
    fi
    if [[ "$MODE" == "check" ]]; then log "  [retarget] $dst -> $src (was $cur)"; return; fi
    rm "$dst"
  elif [[ -e "$dst" ]]; then
    log "  [skip] $dst (exists, not a symlink)"
    return
  fi
  if [[ "$MODE" == "check" ]]; then
    log "  [link] $dst -> $src"
  else
    ln -s "$src" "$dst"
    log "  linked $dst"
  fi
}

do_unlink() {
  local dst="$1"
  if [[ -L "$dst" ]]; then
    local cur; cur="$(readlink "$dst")"
    if [[ "$cur" == "$SLAB"/* ]]; then
      if [[ "$MODE" == "check" ]]; then log "  [unlink] $dst"; else rm "$dst" && log "  unlinked $dst"; fi
    fi
  fi
}

# ---------- install / check ----------
if [[ "$MODE" == "install" || "$MODE" == "check" ]]; then
  mkdir -p "$TARGET/.cursor/rules" "$TARGET/.cursor/skills"

  log "== rules =="
  for f in "${RULES[@]}"; do
    do_link "$f" "$TARGET/.cursor/rules/$(basename "$f")"
  done

  log "== skills =="
  for d in "${SKILLS[@]}"; do
    name="$(basename "$d")"
    do_link "${d%/}" "$TARGET/.cursor/skills/$name"
  done

  log "== AGENTS.md =="
  agents="$TARGET/AGENTS.md"
  if [[ -e "$agents" ]]; then
    log "  [skip] AGENTS.md exists — please ensure it references slab manually."
  elif [[ "$MODE" == "check" ]]; then
    log "  [write] AGENTS.md (starter template)"
  else
    project_name="$(basename "$TARGET")"
    cat > "$agents" <<EOF
# AGENTS.md — $project_name

> Project-specific context for $project_name.
> Shared engineering context lives in the slab knowledge repo:
> see \`$SLAB/AGENTS.md\` and the symlinked rules / skills under
> \`.cursor/rules/\` and \`.cursor/skills/\`.

## What this project is

<one paragraph: what $project_name is, who uses it, its entry points>

## Where things live (project-local)

<directory tour: code/, configs/, etc.>

## Shared knowledge to consult

When working in this project, the agent should consult:

- \`$SLAB/AGENTS.md\` — global hardware / SLURM / commit conventions
- \`$SLAB/knowledge/hardware/\` — GPU specs (MI300X / MI355X / B200 etc.)
- \`$SLAB/knowledge/systems/\` — backend (TorchTitan / Megatron / Primus) know-how
- \`$SLAB/knowledge/moe/\` — MoE dataflow, parallelism, research directions
- \`$SLAB/knowledge/kernels/\` — GEMM / FP8 / comm-compute overlap patterns
- \`$SLAB/papers/\` — paper notes (read the README first for the index)
- \`$SLAB/notes/\` — sibling project work logs (gpt-oss / monolith-moe / etc.)

The 5 slab rules under \`.cursor/rules/0[0-4]-*.mdc\` and all skills under
\`.cursor/skills/\` are symlinked into this project — they auto-apply.

## Project-local rules / skills

Add files with prefix \`90-\` and above under \`.cursor/rules/\` or
new directories under \`.cursor/skills/\` for things that are
specific to $project_name. Do NOT edit symlinked content here —
edit it in slab so all projects benefit.
EOF
    log "  wrote $agents"
  fi

  log ""
  log "== .gitignore reminder =="
  log "  Add these lines to $TARGET/.gitignore (symlinks are per-machine):"
  log ""
  log "    # slab symlinks (per-machine bootstrap, not portable)"
  log "    .cursor/rules/0*.mdc"
  log "    /.cursor/skills/amd-gemm-optimization"
  log "    /.cursor/skills/archive-notes"
  log "    /.cursor/skills/backend-gap-report"
  log "    /.cursor/skills/canvas-to-html"
  log "    /.cursor/skills/cco-pipeline-overlap"
  log "    /.cursor/skills/cuda_*"
  log "    /.cursor/skills/gpu-trace-analysis"
  log "    /.cursor/skills/mi355_hardware_aware"
  log "    /.cursor/skills/paper-deep-analysis"
  log "    /.cursor/skills/progress-note"
  log "    /.cursor/skills/read-paper"
  log "    /.cursor/skills/slurm-*"
  log "    /.cursor/skills/ssh-node-xiaoming-dev-container"
  log "    /.cursor/skills/trace-vram-canvas"
fi

# ---------- unlink ----------
if [[ "$MODE" == "unlink" ]]; then
  log "== removing slab symlinks =="
  for f in "$TARGET/.cursor/rules"/*.mdc; do [[ -e "$f" || -L "$f" ]] && do_unlink "$f"; done
  for d in "$TARGET/.cursor/skills"/*; do [[ -e "$d" || -L "$d" ]] && do_unlink "$d"; done
  log ""
  log "Project-local rules/skills (non-symlinks) left untouched."
  log "AGENTS.md left untouched."
fi

log ""
log "done."
