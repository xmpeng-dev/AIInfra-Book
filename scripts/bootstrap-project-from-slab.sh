#!/usr/bin/env bash
# Bootstrap a project to use the slab knowledge repo as its shared
# engineering context.
#
# Installs three symlinks into the target project:
#   - AGENTS.md                  -> slab/AGENTS.md
#   - .cursor/rules/[00-89]*.mdc -> slab/.cursor/rules/*  (excludes 9*-local)
#   - .cursor/skills/<each>      -> slab/.cursor/skills/<each>
#
# Idempotent. Safe to re-run. Uses symlinks; nothing is copied.
# Per-machine (the slab path is absolute) — do NOT commit the
# symlinks; add the suggested .gitignore entries below.
#
# Safety: if a target path already exists as a plain file (not a
# symlink), it is left untouched and reported as [skip]. To re-bootstrap
# such a project, delete the plain file first.
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
# Cross-project rules only (00..89). Files named 90-*-local.mdc stay
# slab-local and are NOT symlinked into sibling projects.
RULES=()
while IFS= read -r f; do
  base="$(basename "$f")"
  case "$base" in
    9[0-9]-*) continue ;;  # project-local; never symlink across projects
  esac
  RULES+=("$f")
done < <(ls "$SLAB/.cursor/rules"/*.mdc 2>/dev/null)

SKILLS=()
while IFS= read -r d; do SKILLS+=("$d"); done < <(ls -d "$SLAB/.cursor/skills"/*/ 2>/dev/null)

log "slab:     $SLAB"
log "target:   $TARGET"
log "AGENTS:   $SLAB/AGENTS.md"
log "rules:    ${#RULES[@]} cross-project files"
log "skills:   ${#SKILLS[@]} dirs"
log "mode:     $MODE"
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
  # AGENTS.md is symlinked back into slab so every project gets the
  # same shared engineering context. Project-specific context belongs
  # in .cursor/rules/90-<project>-local.mdc, not in this file.
  do_link "$SLAB/AGENTS.md" "$TARGET/AGENTS.md"

  log ""
  log "== .gitignore reminder =="
  log "  Add these lines to $TARGET/.gitignore (symlinks are per-machine):"
  log ""
  log "    # slab symlinks (per-machine bootstrap, not portable)"
  log "    /AGENTS.md"
  log "    /.cursor/rules/[0-8]*.mdc"
  log "    /.cursor/skills/amd-gemm-optimization"
  log "    /.cursor/skills/archive-notes"
  log "    /.cursor/skills/backend-gap-report"
  log "    /.cursor/skills/canvas-to-html"
  log "    /.cursor/skills/cco-pipeline-overlap"
  log "    /.cursor/skills/cuda_*"
  log "    /.cursor/skills/distill-operator-repo"
  log "    /.cursor/skills/gpu-trace-analysis"
  log "    /.cursor/skills/mi355_hardware_aware"
  log "    /.cursor/skills/paper-deep-analysis"
  log "    /.cursor/skills/progress-note"
  log "    /.cursor/skills/read-paper"
  log "    /.cursor/skills/slurm-*"
  log "    /.cursor/skills/ssh-node-xiaoming-dev-container"
  log "    /.cursor/skills/trace-vram-canvas"
  log "    /.cursor/skills/wire-knowledge-into-system"
  log "    /.cursor/skills/create-slab-skill"
fi

# ---------- unlink ----------
if [[ "$MODE" == "unlink" ]]; then
  log "== removing slab symlinks =="
  do_unlink "$TARGET/AGENTS.md"
  for f in "$TARGET/.cursor/rules"/*.mdc; do [[ -e "$f" || -L "$f" ]] && do_unlink "$f"; done
  for d in "$TARGET/.cursor/skills"/*; do [[ -e "$d" || -L "$d" ]] && do_unlink "$d"; done
  log ""
  log "Project-local rules/skills (non-symlinks) left untouched."
  log "A plain (non-symlink) AGENTS.md, if present, is left untouched."
fi

log ""
log "done."
