# Agent Instructions

## Issue Tracking

We use **bd (beads)** for issue tracking instead of Markdown TODOs or external tools.
Hooks auto-inject `bd prime` at session start. Run `bd prime` manually after compaction or context clear.

### Quick Reference

```bash
bd ready --json                    # Find ready work (no blockers)
bd create "Title" -t task -p 2 -d "Description" --json  # Create issue
bd update <id> --status in_progress --json              # Claim work
bd close <id> --reason "Done" --json                    # Complete work
bd show <id> --json                # Get issue details
bd dep add <child> <parent> --type discovered-from      # Link discovered work
bd dep tree <id>                   # Show dependency tree
```

### Workflow

1. **Check for ready work**: `bd ready` to see what's unblocked
2. **Claim your task**: `bd update <id> --status in_progress`
3. **Work on it**: Implement, test, document
4. **Discover new work**: If you find bugs or TODOs, create issues and link them:
   - `bd create "Found bug in auth" -t bug -p 1 --json`
   - `bd dep add <new-id> <current-id> --type discovered-from`
5. **Complete**: `bd close <id> --reason "Implemented"`

### Issue Types

| Type | Use for |
|------|---------|
| `bug` | Something broken that needs fixing |
| `feature` | New functionality |
| `task` | Work item (tests, docs, refactoring) |
| `epic` | Large feature composed of multiple issues |
| `chore` | Maintenance work (dependencies, tooling) |

### Priorities

| Priority | Meaning |
|----------|---------|
| `0` | Critical (security, data loss, broken builds) |
| `1` | High (major features, important bugs) |
| `2` | Medium (default - nice-to-have features, minor bugs) |
| `3` | Low (polish, optimization) |
| `4` | Backlog (future ideas) |

### Dependency Types

| Type | Effect |
|------|--------|
| `blocks` | Hard dependency - affects `bd ready` queue |
| `related` | Soft relationship (issues are connected) |
| `parent-child` | Epic/subtask relationship |
| `discovered-from` | Track issues discovered during work |

Only `blocks` dependencies affect the ready work queue.

### Idea Tracking

Ideas live in the **Ideation epic** (`w26-ryu`) with **priority 4** until promoted.

| User says | Action |
|-----------|--------|
| "idea about X" | Create under Ideation epic, P4, `related` to X if X exists |
| "track this for later" | P4 under relevant epic (not necessarily Ideation) |
| "this is now real work" | Update priority + move to appropriate parent epic |
| "never mind this idea" | Close with reason |

**Convention**: Ideas don't need an `[idea]` prefix—being P4 under Ideation is sufficient signal.

**Promotion workflow**:
```bash
bd update <id> --priority 2                           # Raise priority
bd dep remove <id> w26-ryu                            # Remove from Ideation
bd dep add <id> <target-epic> --type parent-child     # Move to real epic
```

---

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

