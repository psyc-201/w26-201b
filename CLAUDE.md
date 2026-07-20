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
│   │   └── XX/lab/     # Lab notebooks (.qmd, jupyter-executed; wk4 keeps helpers.py)
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

## Week 4 & 6 Labs on the Site

All lab pages (including wks 4/6) are now plain `.qmd` files executed by the
jupyter engine (`w26-201b` kernel → `site/.venv`), same as wks 1-2. The wk4/wk6
`.qmd`s were converted one-time from the classroom marimo `.py` notebooks
(`marimo export ipynb` → `quarto convert` → post-processing) and are now the
site's source of truth; the marimo `.py` copies were removed from `site/`
(classroom template repos still carry them).

- The 8 wk4 interactive anywidget explorables render as **static snapshots**:
  hidden cells instantiate each explorer from `weeks/04/lab/helpers.py` and
  display its default-settings `chart_base64`/`stats_html`. Keep `helpers.py`
  — the wk4 pages execute against it.
- The full marimo/WASM pipeline (quarto-marimo engine, wheels, patched
  extension, in-browser pyodide reactivity) is preserved on the
  **`marimo-wasm-labs` branch** — see its two commits if reviving it.

## Lab Notebook Conventions

Lab notebooks are **marimo** `.py` files (not Jupyter) in classroom template repos.

- **Voice**: Conversational, second-person ("We'll", "Let's"), informal but precise
- **Structure**: Explain → Show → Interpret rhythm; `hide_code=True` on markdown cells
- **Formatting**: Blockquote callouts for deeper explanations, tables for comparisons, LaTeX for formulas
- **Challenges**: Interleaved "Your Turn" / challenge cells with empty code cells
- **Resources**: Each template repo has a `resources/` directory (gitignored from students) with reference notebooks
- **Data**: Datasets live in `data/` within each template repo

---
