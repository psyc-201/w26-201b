# CLAUDE.md - W26 Course Project Context

## Project Overview

**PSYC 201B: Statistical Intuitions for Social Scientists** (Winter 2026) — graduate statistics course at UC San Diego. Quarto-based course website.

**Pedagogical Reference**: See [`PEDAGOGY.md`](PEDAGOGY.md) for teaching philosophy, content sequencing, scaffolding patterns, and assessment design.

## Essential Commands

```bash
uv run poe quarto       # Preview site (localhost:9999)
quarto render           # Full build to docs/
```

---

## GitHub Classroom

- **Org**: psyc-201
- **Classroom**: https://classroom.github.com/classrooms/232475786-201b-w26
- **Local workspace**: `classroom/` (gitignored from main repo)

> **Important**: Each directory under `classroom/templates/` and `classroom/assignments/` is a **separate git repository** (not a submodule). This is intentional — they have their own remotes in the psyc-201 org. When working in these directories, git commands operate on that repo, not the main course repo.

## Grading

- **Directory**: `grading/` (gitignored from main repo)
- **Roster**: `grading/classroom_roster.csv` — canonical student list with `identifier`, `github_username`, `github_id`, `name`
- Use `classroom_roster.csv` as the source of truth for scripting against student repos

### Collecting Student Submissions

`grading/fetch_proposals.sh` fetches proposal PDFs for all students:
1. Reads handles from `classroom_roster.csv`
2. Tries `pdfs/proposal.pdf` then `proposal.pdf` at repo root
3. Validates downloaded files are actual PDFs
4. Reports students needing local builds from `.qmd`

```bash
bash grading/fetch_proposals.sh    # Downloads to grading/proposals/
```

Output naming convention: `first-last.pdf` (lowercase, hyphenated).

Student repos follow the pattern `final-project-<github_handle>` in the `psyc-201` org.

---

## Key Directories

```
w26/
├── weeks/              # Weekly content (01-06, final)
│   └── XX/slides/      # Lecture PDFs per week
├── guides/             # Student-facing guides (published)
├── slides/             # Source slide PDFs (master copies)
├── assets/
│   ├── pdfs/           # Reading PDFs
│   └── summaries/      # Generated reading summaries
├── grading/            # Grading workspace (gitignored)
│   ├── classroom_roster.csv  # Student roster (identifier, github_username, github_id, name)
│   ├── grade.py             # CLI grading toolkit (uv run python grade.py --help)
│   ├── fetch_proposals.sh   # Download proposal PDFs from student repos
│   └── proposals/           # Collected student proposal PDFs
├── dev/                # Development scratch (not rendered)
├── planning/           # Legacy planning docs (gitignored)
└── classroom/          # GitHub Classroom work (gitignored)
    ├── templates/      # Assignment templates
    └── assignments/    # Student submissions (cloned repos)
```

---

## Source Materials (W25)

Prior year materials for reference:
- **Labs/HWs**: `/Users/esh/Dropbox/docs/teaching/201b/w25/classroom/`
- **Lectures**: `/Users/esh/Dropbox/docs/teaching/201b/w25/201b-site/lectures/`

---

## Core Libraries

| Library | Role | Docs |
|---------|------|------|
| **bossanova** | Statistical modeling (replaces statsmodels `ols`/R's `lm`/`glm`/`lmer`) | https://sciminds.ucsd.edu/bossanova/ |
| **polars** | DataFrames (primary, replaces pandas) | https://docs.pola.rs/ |
| **seaborn** | Visualization | https://seaborn.pydata.org/ |
| **marimo** | Reactive notebooks (replaces Jupyter) | https://docs.marimo.io/ |

> Always refer to the [bossanova docs](https://sciminds.ucsd.edu/bossanova/cheatsheet/) for latest API

---

## Quick Reference

- **Class structure**: 3 days/week (M/T/W), 1hr 50min each
- **Core principle**: "Coding = language learning" — daily practice, scaffolded complexity
- **Model comparison**: Prefer "worth it?" framing over cookbook approach
- **Marginal effects**: "Mixing board" metaphor — sliders (continuous) and switches (categorical)
- **Schedule**: See `schedule.qmd` for weekly topics and deadlines

---

## Lab Notebook Conventions

Lab notebooks are **marimo** `.py` files (not Jupyter) in classroom template repos.

- **Voice**: Conversational, second-person ("We'll", "Let's"), informal but precise
- **Structure**: Explain → Show → Interpret rhythm; `hide_code=True` on markdown cells
- **Formatting**: Blockquote callouts for deeper explanations, tables for comparisons, LaTeX for formulas
- **Challenges**: Interleaved "Your Turn" / challenge cells with empty code cells
- **Resources**: Each template repo has a `resources/` directory (gitignored from students) with reference notebooks
- **Data**: Datasets live in `data/` within each template repo

---
