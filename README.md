# PSYC 201B - W26

Source code for course materials and website for the Winter 2026 edition of *Statistical Intuitions for Social Scientists*

Python and all its dependencies are managed by `uv` in an isolated environment.  All pages and computational documents are rendered & executed with `quarto` using the environment. 

## Quick guide

- Always `uv run quarto render` **before** `git commit` and `git push` to build the site to the `docs/` folder
- The website is automatically updated from this folder each push!

## Initial setup

1. Install [homebrew](https://brew.sh/) a macOS package manager to make the `brew` command available in your terminal
2. Install [Quarto](https://quarto.org/): `brew install --cask quarto`
3. Install [uv](https://docs.astral.sh/uv/) for Python management: `brew install uv`
4. Clone the repository: `git clone copyURLfromButton`
5. Install dependencies: `uv sync`
6. Register the Jupyter kernel (one-time): `uv run poe setup`

> **Note**: Step 6 registers a Jupyter kernel that Quarto uses to run Python code. This works across VSCode, RStudio, Positron, and the command line. Re-run it if you delete and recreate the `.venv` directory.

## Using Quarto or Marimo

We've setup [Poe the Poet](https://poethepoet.natn.io/index.html) which lets us define custom commands in `pyproject.toml` to reduce typing:

## VSCode / Codium

The `.vscode/` folder includes workspace settings that:

- Use Python, Ruff, and ty from the project's `.venv` (no global installs needed)
- Recommend essential extensions (Quarto, Jupyter, Ruff, ty)
- Provide a clean, distraction-free UI with telemetry disabled

When you open the project, VSCode will prompt to install recommended extensions. Accept to get the full experience.

> **Other IDEs**: RStudio and Positron also work — run `uv run poe setup` first to register the Jupyter kernel

## Adding or Removing Python Packages

`uv add mypackage`: installs `mypackage` and adds it to `pyproject.toml`  

`uv remove mypackage`: removes `mypackage` and the corresponding entry in `pyproject.toml`  

## Collaborating with Claude Code

This project uses [Claude Code](https://docs.anthropic.com/en/docs/claude-code) for AI-assisted development. Project context is in `CLAUDE.md`.

### Getting Started

```bash
# Install Claude Code (requires Node.js 18+)
npm install -g @anthropic-ai/claude-code

# Install beads for issue tracking
brew tap ejfox/tap && brew install beads

# Run from project root
claude
```

### Issue Tracking with Beads

We use `bd` (beads) for persistent issue tracking across sessions:

```bash
bd ready              # What's available to work on?
bd list --status=open # All open issues
bd show <id>          # Issue details
bd update <id> --status=in_progress  # Claim work
bd close <id>         # Mark complete
```

