# CLAUDE.md - W26 Course Project Context

## Project Overview

**PSYC 201B: Statistical Intuitions for Social Scientists** (Winter 2026) — graduate statistics course at UC San Diego. Quarto-based course website.

**Pedagogical Reference**: See [`stat-intuitions.exe.xyz/docs/PEDAGOGY.md`](stat-intuitions.exe.xyz/docs/PEDAGOGY.md) for teaching philosophy, content sequencing, scaffolding patterns, and assessment design (removed from repo root post-course).

## Essential Commands

All site commands run from `site/` (the Quarto project root):

```bash
cd site
uv run poe quarto       # Preview site (localhost:9999)
quarto render           # Full build to ../docs/ (repo root)
```

> **Layout rationale**: The Quarto project lives in `site/` with `output-dir: ../docs` because (1) GitHub Pages deploys from the branch and can only serve `/` or `/docs`, and (2) Quarto walks the *entire* project directory on every render/preview — keeping `classroom/` (~560k files) and `grades/` outside `site/` cut project inspection from 66s to ~2s. Never move large non-site directories into `site/`.
>
> **Caveat**: Because `docs/` is outside the project dir, Quarto refuses to auto-clean stale output files. If cruft accumulates: `rm -r docs && (cd site && quarto render)`.

---

## GitHub Classroom

- **Org**: psyc-201
- **Classroom**: https://classroom.github.com/classrooms/232475786-201b-w26
- **Local workspace**: `classroom/` (gitignored from main repo)

> **Important**: Each directory under `classroom/templates/` and `classroom/assignments/` is a **separate git repository** (not a submodule). This is intentional — they have their own remotes in the psyc-201 org. When working in these directories, git commands operate on that repo, not the main course repo.

## Grading (archived — course complete)

- **Directory**: `grades/` (gitignored from main repo)
- Contents: `final-project-grades.csv`/`.md`, per-student final project work under `grades/gh-classroom/<last-first>/`, teaching evaluation PDF
- Student repos follow the pattern `final-project-<github_handle>` in the `psyc-201` org

---

## Key Directories

```
w26/
├── site/               # Quarto project root (run all quarto/uv commands here)
│   ├── _quarto.yml     # output-dir: ../docs
│   ├── pyproject.toml  # Site build env (uv); jupyter kernel: w26-201b
│   ├── weeks/          # Weekly content (01-10, final)
│   │   ├── XX/slides/  # Lecture PDFs per week (published copies)
│   │   └── XX/lab/     # Lab notebooks (wks 1-2: .qmd/jupyter; wks 4/6: marimo .py)
│   ├── guides/         # Student-facing guides (published)
│   └── assets/
│       ├── pdfs/       # Reading PDFs (gitignored)
│       └── summaries/  # Generated reading summaries (gitignored)
├── docs/               # Rendered site (GitHub Pages serves this; stat-intuitions.com)
├── slides/             # Source slide PDFs (master copies, untracked)
├── grades/             # Grading archive (gitignored)
├── dev/                # Development scratch (gitignored)
├── sidecar/            # Side pipeline (lectures/papers notes, untracked)
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

## Marimo Labs on the Site (weeks 4 & 6)

The wk4/wk6 lab notebooks are marimo `.py` files (source of truth, shared with
classroom templates). The site renders them via the **quarto-marimo engine
extension** (v0.4.5, requires marimo >=0.23.1) from *derived* `.qmd` files:

```bash
cd site
uv run python marimo2qmd.py weeks/04/lab/01-sampling.py   # .py -> .qmd (see script docstring)
```

Re-run after editing a lab `.py`; commit both files. Notes:

- Each generated `.qmd` carries `external-env: true` in its frontmatter (the
  engine ignores the project-level key) so cells execute in `site/.venv`.
- `_extensions/marimo-team/marimo/extract.py` has a **local patch** (marked
  `LOCAL PATCH`) that chdirs to the document dir so `helpers.py` imports and
  `./data/*.csv` reads resolve. `_extensions/` is gitignored — re-apply the
  patch if `quarto update marimo-team/quarto-marimo` overwrites it.
- Rendered pages are static-with-islands. Full in-browser reactivity (WASM)
  would additionally need `helpers.py` folded into the notebooks and data
  loading over HTTP — not currently wired up.

## Lab Notebook Conventions

Lab notebooks are **marimo** `.py` files (not Jupyter) in classroom template repos.

- **Voice**: Conversational, second-person ("We'll", "Let's"), informal but precise
- **Structure**: Explain → Show → Interpret rhythm; `hide_code=True` on markdown cells
- **Formatting**: Blockquote callouts for deeper explanations, tables for comparisons, LaTeX for formulas
- **Challenges**: Interleaved "Your Turn" / challenge cells with empty code cells
- **Resources**: Each template repo has a `resources/` directory (gitignored from students) with reference notebooks
- **Data**: Datasets live in `data/` within each template repo

---
