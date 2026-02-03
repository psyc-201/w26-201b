# PSYC 201B Marp Slides

Marp-based slide system for PSYC 201B lectures.

## Files

| File | Description |
|------|-------------|
| `template.md` | Comprehensive feature showcase and reference |
| `psyc201b.css` | Custom theme matching Keynote design |

## Quick Start

```bash
# Preview in browser (recommended for development)
marp --server .

# Preview specific file with live reload
marp --preview template.md

# Export to PDF
marp --pdf template.md

# Export to PowerPoint
marp --pptx template.md

# Export to HTML
marp template.md -o output.html
```

## Creating New Slides

1. Copy `template.md` and rename (e.g., `week05-models.md`)
2. Delete example slides you don't need
3. Add your content
4. Preview with `marp --server .`

## Slide Classes

Use `<!-- _class: classname -->` before a slide:

| Class | Use For |
|-------|---------|
| `title` | Title slide with course branding |
| `section` | Section divider (centered) |
| `invert` | Dark background for emphasis |
| `focus` | Key takeaway (dark, centered) |
| `discussion` | Discussion prompt (coral background) |
| `poll` | Live poll (cyan background) |
| `small` | Dense content (smaller text) |
| `summary` | Summary slide |
| `closing` | Final slide with next steps |

## Two-Column Layout

```markdown
<div class="columns">
<div>

Left column content

</div>
<div>

Right column content

</div>
</div>
```

## Comparison Labels

```markdown
<div class="label-explanation">Explanation</div>
<div class="label-prediction">Prediction</div>
```

## Callout Boxes

```markdown
<div class="callout-info">Info callout</div>
<div class="callout-warning">Warning callout</div>
<div class="discussion">Discussion prompt</div>
<div class="math-box">$$equation$$</div>
```

## Images

```markdown
![w:400](image.png)           # Width 400px
![h:300](image.png)           # Height 300px
![center](image.png)          # Centered
![bg](image.png)              # Background
![bg left:40%](image.png)     # Split left
![bg right:40%](image.png)    # Split right
```

## Pagination Control

```markdown
<!-- _paginate: skip -->      # Don't count this slide
<!-- _paginate: false -->     # Hide page number
<!-- _paginate: hold -->      # Keep same number
```

## Presenter Notes

```markdown
Content visible to audience

<!--
Notes only visible in presenter view.
Export with: marp --notes slides.md
-->
```

## Troubleshooting

**Browser timeout on PDF export:**
If `marp --pdf` times out, try:
```bash
marp --browser chrome --browser-timeout 60 --pdf slides.md
```

Or use the server mode and print to PDF from browser:
```bash
marp --server .
# Open http://localhost:8080, then Cmd+P to print
```

## Reference

- [Marp Documentation](https://marp.app/)
- [Marpit Framework](https://marpit.marp.app/)
- [Marp CLI](https://github.com/marp-team/marp-cli)
