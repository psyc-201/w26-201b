# CLAUDE.md - W26 Course Project Context

## Project Overview

**PSYC 201B: Statistical Intuitions for Social Scientists** (Winter 2026) — graduate statistics course at UC San Diego. Quarto-based course website.

**Current Status**: Week 1 complete. Week 2 scaffolded.

---

## Issue Tracking

Uses `bd` (beads) for persistent issue tracking. Hooks auto-inject workflow context.

```bash
bd ready          # What can I work on?
bd create --title="..." --type=task --priority=2
bd close <id>     # Mark complete
```

---

## Skills

### `/summarize-source` - Extract summaries from readings

Extracts structured summaries from PDFs/web without loading full files into context. Uses `rga` for PDF search. See `.claude/skills/summarize-source/SKILL.md` for details.

**Output**: `assets/summaries/{source-name}.md`

---

## GitHub Classroom

**Key insight**: Template repos are boilerplate. Develop content in the **assignment repo** (created in Classroom UI). Students fork from that.

- **Org**: psyc-201
- **Classroom**: https://classroom.github.com/classrooms/232475786-201b-w26
- **Local workspace**: `classroom/` (gitignored)
- **Templates**: `201b-ghct-01-basic` (Week 1), `201b-ghct-02-pydata` (Week 2, cumulative)

---

## Key Directories

```
w26/
├── weeks/              # Weekly content (01, 02, final; more added as needed)
├── guides/             # Student-facing guides
├── slides/             # Source slide PDFs (copied to weeks/XX/slides/)
├── assets/
│   ├── pdfs/           # Reading PDFs
│   └── summaries/      # Generated reading summaries
├── dev/                # Development scratch (not rendered)
├── planning/           # Pedagogical plans (gitignored)
└── classroom/          # GitHub Classroom work (gitignored)
    └── templates/      # 201b-ghct-01-basic, 201b-ghct-02-pydata
```

---

## Essential Commands

```bash
uv run poe quarto       # Preview site
quarto render           # Full build
```

---

## GitHub Pages Troubleshooting

If GitHub Actions deployments get stuck in "queued" status and the UI won't cancel/restart them:

```bash
# Check Pages status
gh api repos/psyc-201/w26-201b/pages --jq '{status}'

# If status is "errored", trigger a fresh build to reset:
gh api -X POST repos/psyc-201/w26-201b/pages/builds

# Monitor progress
gh run list --limit 5
```

**Root cause**: A failed deployment can leave Pages in an "errored" state, blocking subsequent runs. The manual build trigger resets this state.

---

## Source Materials (W25)

- **Labs/HWs**: `/Users/esh/Dropbox/docs/teaching/201b/w25/classroom/`
- **Lectures**: `/Users/esh/Dropbox/docs/teaching/201b/w25/201b-site/lectures/`

---

## Planning Philosophy

- **Class structure**: 3 days/week (M/T/W), 1hr 50min each
- **Core principle**: "Coding = language learning" — daily practice, scaffolded complexity
- **Early coding**: Get students hands-on ASAP

**Reference docs**: `planning/WEEKLY_REFERENCE.md`, `planning/CONTENT_MAPPING.md`, `schedule.qmd`
