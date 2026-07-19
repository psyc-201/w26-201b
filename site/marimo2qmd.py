"""Convert marimo .py lab notebooks to Quarto .qmd (marimo engine format).

The marimo .py notebooks remain the source of truth (they're what students get
via GitHub Classroom); the .qmd files are derived build inputs for the site.

Usage (from site/):
    uv run python marimo2qmd.py weeks/04/lab/01-sampling.py [more .py files]

Transforms applied on top of `marimo export md --flavor qmd`:
1. Fences: ```{marimo .python ...} -> ```{python .marimo ...}
   (marimo 0.23.x emits the former; quarto-marimo 0.4.5's engine only
   matches python/sql-first variants)
2. Frontmatter: hoist the title/author/date block that marimo embeds at the
   top of the body (from the notebook's first mo.md cell) into the YAML
   header, and add `sidebar: weeks`
3. Insert a blank line before `<!---->` cell markers so a marker directly
   after a raw HTML block (e.g. <img>) doesn't glue the next markdown block
   into it (pandoc type-6 HTML blocks only end at a blank line)
"""

import re
import subprocess
import sys
from pathlib import Path

FENCE = re.compile(r"^```\{marimo \.python", flags=re.M)
BODY_META = re.compile(r"\A---\n(.*?)\n---\n(?:<!---->\n)?", flags=re.S)


def convert(py_path: Path) -> Path:
    out = py_path.with_suffix(".qmd")
    md = subprocess.run(
        ["marimo", "export", "md", str(py_path), "--flavor", "qmd"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    # Split off marimo's generated frontmatter (generic title + marimo-version)
    m = BODY_META.match(md)
    assert m, f"no frontmatter in export of {py_path}"
    gen_meta, body = m.group(1), md[m.end() :]
    version = next(
        (l for l in gen_meta.splitlines() if l.startswith("marimo-version:")), ""
    )

    # Hoist the real metadata from the notebook's first mo.md cell: either an
    # embedded ---...--- yaml block (wk4 labs) or a leading H1 (wk6 labs)
    body = body.lstrip("\n")
    if m := BODY_META.match(body):
        meta, body = m.group(1), body[m.end() :]
    elif m := re.match(r"# (.+)\n+", body):
        meta, body = f'title: "{m.group(1)}"\nauthor: "Eshin Jolly"', body[m.end() :]
    else:
        raise ValueError(f"no title block or leading H1 in {py_path}")

    body = FENCE.sub("```{python .marimo", body)
    body = body.replace("\n<!---->", "\n\n<!---->")

    # external-env must be in document metadata: the engine reads it from
    # target.metadata, which the project-level _quarto.yml key doesn't reach
    out.write_text(
        f"---\n{meta}\nsidebar: weeks\nexternal-env: true\n{version}\n---\n\n{body}"
    )
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(convert(Path(arg)))
