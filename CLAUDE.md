# CLAUDE.md - W26 Course Project Context

## Project Overview

**PSYC 201B: Statistical Intuitions for Social Scientists** (Winter 2026) — graduate statistics course at UC San Diego. Quarto-based course website.

**Pedagogical Reference**: See [`PEDAGOGY.md`](PEDAGOGY.md) for teaching philosophy, content sequencing, scaffolding patterns, and assessment design.

---

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

### Templates (cumulative)
| Template | Content |
|----------|---------|
| `201b-ghct-01-basic` | Python fundamentals |
| `201b-ghct-02-pydata` | + Polars, Seaborn |
| `201b-ghct-04-pydata-resampling` | + Resampling/bootstrap |
| `201b-ghct-05-models` | + bossanova, GLM, model comparison |
| `w26-hw-01` | HW1 template |
| `w26-hw-02` | HW2 template |

### Active Assignments (GitHub Classroom repos)
| Assignment | Template | Week |
|------------|----------|------|
| `201b-w26-wk01-lab-*` | `201b-ghct-01-basic` | 1 |
| `201b-w26-wk04-lab-*` | `201b-ghct-04-pydata-resampling` | 4 |
| `201b-w26-wk06-lab-*` | `201b-ghct-05-models` | 6 |
| `201b-w26-hw-01-*` | `w26-hw-01` | 2-3 |
| `201b-w26-hw-02-*` | `w26-hw-02` | 4-5 |

### Updating a Live Assignment

When updating a template repo that students have already forked:
- **NEVER edit existing files** — causes merge conflicts when students `git pull`
- **Only add new files** (notebooks, data, etc.)
- Safe to edit: `pyproject.toml`, `README.md` (students don't modify these)
- Always verify with `git diff --cached` before pushing

---

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

> **Always consult bossanova docs first** for API questions — resource notebooks in `classroom/` may use older statsmodels patterns.

### Key bossanova API patterns
- `model("y ~ x", data).fit()` — fit a model
- `.set_contrasts(var='treatment'|'sum'|'poly')` — coding schemes
- `.explore("y ~ x", **kwargs)` → `.effects` — marginal effects (the "mixing board")
- `.jointtest()` / `.infer("joint")` — ANOVA-style omnibus tests
- `compare(m1, m2)` — nested model comparison (F-test, PRE)
- `.plot_resid()`, `.plot_design()`, `.plot_mee()`, `.plot_vif()` — diagnostics

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

## GitHub Pages Troubleshooting

If deployments get stuck in "queued" status:

```bash
gh api repos/psyc-201/w26-201b/pages --jq '{status}'  # Check status
gh api -X POST repos/psyc-201/w26-201b/pages/builds   # Reset if "errored"
gh run list --limit 5                                  # Monitor progress
```

---

## Linear (Issue Tracking)

- **Team**: Ejolly
- **Project**: W26-201b

### Statuses

| Name | Type | Use |
|------|------|-----|
| Backlog | unstarted | Default for new issues not yet prioritized |
| Todo | unstarted | Prioritized, ready to work |
| In Progress | started | Actively being worked on |
| Done | completed | Finished |
| Canceled | canceled | Won't do |

### Labels (required on every issue)

| Label | When to use |
|-------|-------------|
| Lab | Lab exercises and coding practice |
| Lecture | Lecture content and slides |
| HW | Homework assignments |
| Bug | Broken behavior, regressions |
| Improvement | Refactoring, performance, code quality |

### Milestones

Assign based on scope — no default:
- **Week 6** through **Week 10**
- **Finals**

### Priorities

0=None, 1=Urgent, 2=High, 3=Normal, 4=Low

### Key MCP Tools

| Action | Tool |
|--------|------|
| Find work | `mcp__linear-server__list_issues` (filter by state, project) |
| View details | `mcp__linear-server__get_issue` |
| Create issue | `mcp__linear-server__create_issue` (team: "Ejolly", project: "W26-201b") |
| Update issue | `mcp__linear-server__update_issue` |
| Add comment | `mcp__linear-server__create_comment` |
| List labels | `mcp__linear-server__list_issue_labels` |
| List milestones | `mcp__linear-server__list_milestones` (project: "W26-201b") |

### Session Start

- Check for **In Progress** issues (stale claims from previous sessions)
- Check **Backlog** and **Todo** for available work

### Session Close Protocol

Before saying "done" or "complete":
1. Create issues for remaining/discovered work
2. Update issue states (close completed, note in-progress)
3. Commit code changes
