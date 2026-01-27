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
| `w26-hw-01` | HW1 template |

---

## Key Directories

```
w26/
├── weeks/              # Weekly content (01-04, final)
│   └── XX/slides/      # Lecture PDFs per week
├── guides/             # Student-facing guides (published)
├── slides/             # Source slide PDFs (master copies)
├── assets/
│   ├── pdfs/           # Reading PDFs
│   └── summaries/      # Generated reading summaries
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

## Quick Reference

- **Class structure**: 3 days/week (M/T/W), 1hr 50min each
- **Core principle**: "Coding = language learning" — daily practice, scaffolded complexity
- **Model comparison**: Prefer "worth it?" framing over cookbook approach
- **Schedule**: See `schedule.qmd` for weekly topics and deadlines

---

## GitHub Pages Troubleshooting

If deployments get stuck in "queued" status:

```bash
gh api repos/psyc-201/w26-201b/pages --jq '{status}'  # Check status
gh api -X POST repos/psyc-201/w26-201b/pages/builds   # Reset if "errored"
gh run list --limit 5                                  # Monitor progress
```
