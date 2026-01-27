# PSYC 201B Pedagogical Reference

> **Purpose**: Reference document for Claude sessions working on course materials. Captures the instructor's teaching philosophy, content sequencing, and assessment approach.

---

## Course Identity

**PSYC 201B: Statistical Intuitions for Social Scientists**
Graduate statistics course for first-year Psychology PhD students at UC San Diego (Winter 2026).

### Three Overarching Goals
1. **Developing statistical intuitions** - Statistical thinking as essential literacy, not just mechanical procedures
2. **Bridging cultural differences** - Connecting the "two cultures" of statistics (prediction vs. explanation)
3. **Programming as theory-building** - "A program is a hypothesis made executable" (code forces precision)

### Core Philosophy
- **"Coding = language learning"** - Daily practice, scaffolded complexity, get hands-on ASAP
- **"Slow is smooth, smooth is fast"** - Thorough onboarding before content; extensive reference material built in
- **Model comparison over cookbook** - Explicit alternative to "look up the right test" approach
- **Statistical thinking over inference** - The process of becoming less wrong, not finding "the answer"

---

## Four Fundamental Intuitions Framework

Introduced in Week 2, referenced throughout the course:

| Intuition | Definition | Common Misconception |
|-----------|------------|---------------------|
| **Aggregation** | Compression - "throwing away information on purpose" to create a model | "More aggregation = better data" |
| **Sampling** | Generalization - connects data to world-at-large | "Random sampling is always required" |
| **Uncertainty** | Limits of inference - reflects what we don't know | "p < .05 removes uncertainty" |
| **Learning** | Modeling & updating - principled mind-changing | "Learning = finding the 'right' model" |

---

## Content Sequencing (Weeks 1-4)

### Week 1: Foundations and Setup
**Focus**: Environment setup + Python/Polars fundamentals

| Day | Topic |
|-----|-------|
| Mon | Course intro, philosophy, logistics (mastery-based grading) |
| Tue-Wed | Pre-flight lab: Terminal, Git, GitHub Classroom workflow |

**Lab Content**:
- Python fundamentals (for R users): variables, types, functions, lists, slicing, loops, comprehensions
- Polars crash course: contexts (select, with_columns, filter, group_by) and expressions (col, operations)
- Emphasis on 0-based indexing, method chaining, `help()` function

**Key Bridging Concepts** (R → Python):
- `%>%` → method chaining
- `function()` → `def`
- 1-based → 0-based indexing
- tidyverse → Polars (similar mental model, different syntax)

---

### Week 2: Statistical Thinking and Visualization
**Focus**: Conceptual foundations + Seaborn/EDA

| Day | Topic |
|-----|-------|
| Mon | What is Statistics? Four Fundamental Intuitions |
| Tue | What is a Model? Data = Model + Error |
| Wed | Lab: Seaborn crash course + EDA workflows |

**Lecture Content**:
- Definition: "The science of principled abstraction from data with explicit acknowledgment of uncertainty"
- Three Big Theorems: LLN, CLT, No Free Lunch
- Data = Model + Error formalism
- Mean as a model, residuals, SSE vs SAE
- Galton board/quincunx analogy for CLT

**Lab Content**:
- Seaborn: relplot, displot, catplot, lmplot
- Figure-level vs axis-level functions
- EDA workflow: 6-phase visual detective work
- "Show all data when you can" principle
- Simpson's Paradox awareness

**Key Readings**:
- Breiman (2001) - Two Cultures
- Yarkoni & Westfall (2017) - Generalizability crisis
- Gelman & Vehtari (2021) - What are the most important statistical ideas?
- Jolly & Chang (2019) - The Flatland Fallacy

---

### Week 3: Bias, Variance, and Model Comparison
**Focus**: Mathematical formalism + consolidation

| Day | Topic |
|-----|-------|
| Mon | No class (holiday) |
| Tue | Bias-variance tradeoff, Math ↔ Python translation |
| Wed | Hypothesis testing as model comparison (PRE, F-statistic) |

**Lecture Content**:
- Sigma notation → for-loops (explicit translation)
- Bias: systematic mis-prediction; Variance: data sensitivity
- Under-fitting (high bias) vs Over-fitting (low bias)
- Compact Model (H0) vs Augmented Model (H1)
- PRE = (ERROR(C) - ERROR(A)) / ERROR(C)
- "Worth it?" as the unifying question

**Visual Resources**:
- Bull's-eye/target diagrams for bias-variance
- mlu-explain.github.io/bias-variance/

---

### Week 4: Sampling Distributions and Inference
**Focus**: Computational bridge to resampling

| Day | Topic |
|-----|-------|
| Mon | Three distributions (Population, Sampling, Sample) |
| Tue | Two approaches: Asymptotic vs Resampling |
| Wed | Lab: Bootstrap/permutation (computational approach) |

**Key Framing**:
- Feynman: "What I cannot create, I do not understand"
- F-distribution = sampling distribution of PRE
- "Math is hard and formulas don't build intuitions, so instead..." → simulation
- Standard error and CIs as the "uncertainty bridge"

**Transition**: From conceptual to computational; video-based learning component

---

## Technical Stack Choices

| Tool | Rationale |
|------|-----------|
| **Python** (not R) | Industry standard, transferable skill |
| **Polars** (not Pandas) | Simpler mental model (contexts + expressions), more consistent API |
| **Seaborn** (not matplotlib) | Higher-level, statistical focus, integrates well with DataFrames |
| **Marimo** (reactive notebooks) | Prevents out-of-order execution bugs, automatic dependency tracking |
| **Quarto** | Reproducible documents, supports both R and Python |
| **uv** | Fast, reliable Python environment management |
| **GitHub Classroom** | Real-world workflow, persistent access post-course |

---

## Scaffolding Patterns

### Callout Box Conventions
```
:::{.callout-note}      # Conceptual explanations, "why" behind patterns
:::{.callout-tip}       # Pro tips, shortcuts, "Your Turn" exercises
:::{.callout-important} # Common errors (SyntaxError, TypeError, etc.)
:::{.callout-warning}   # Gotchas, painpoints (e.g., seaborn prefers pandas)
:::{.callout-caution}   # Assignment links, deadlines
```

### Exercise Structure
1. "Your Turn" blocks with `...` placeholders
2. Hints in comments or callout boxes
3. Collapsible solution blocks (`collapse="true"`)
4. Progressive difficulty within tutorials
5. Expected output shown before exercise

### R-to-Python Bridging
- Explicit side-by-side code comparisons
- ggplot2 ↔ seaborn mapping tables
- Notes on syntax differences (indentation, quotes, etc.)
- Leverage existing DataFrame intuitions

### Error-Positive Approach
- Intentionally show error-producing cells with explanations
- Cover: SyntaxError, TypeError, IndexError, IndentationError, DuplicateError
- Frame errors as learning moments, not failures

---

## Assessment Philosophy

### Grading Breakdown
- 30% Labs/Engagement (completion-based)
- 40% Homework (with revision opportunities)
- 30% Final Project

### Rubric Dimensions (5 competencies × 3 levels)
| Competency | Focus |
|------------|-------|
| **Computation** | Correctness of code, absence of extraneous code |
| **Analysis** | Appropriateness of approach for data/context |
| **Synthesis** | Interpretation of results in context |
| **Visual presentation** | Plot quality, labeling, clarity |
| **Written communication** | Clear, precise, convincing explanations |

### Key Principles
- **Process over correctness**: Document thought process; partial credit for good-faith attempts
- **Iterative improvement**: Post-deadline revision allowed after class review
- **GenAI as tool, not crutch**: Allowed with attribution + full transcript; copy/paste forbidden
- **Show your work**: Grading emphasizes reasoning, not just output

---

## Communication Style

### Tone
- Approachable and practical; conversational without being casual
- Action-oriented; focus on what students will *do*
- Anticipate struggles; provide "escape hatches" for common errors
- Frame effort as investment in future self

### Analogies Used
- Git as "social time-machine"; commits as "snapshots"
- Galton board/quincunx for CLT
- Bull's-eye diagrams for bias-variance
- "Worth it?" question for model comparison

### Forward-Looking Motivation
- Emphasize long-term value: "You'll *always* have access to..."
- "The more you engage, the higher quality resources for your own future reference"
- Statistical thinking as "essential literacy for citizenship"

---

## Running Examples and Datasets

| Example | Used In | Concepts |
|---------|---------|----------|
| Happiness ~ Chocolate | Weeks 2-3 | Data = Model + Error, PRE, model comparison |
| Palmer Penguins | Labs 1-2 | DataFrames, visualization, EDA |
| Star Wars characters | Lab 2 | Missing data, derived variables |
| Cat name recognition | HW-01 | Full EDA workflow, interpretation |

---

## Key Sources and Citations

| Author(s) | Year | Topic |
|-----------|------|-------|
| Wilks | 1951 | Statistical literacy quote |
| Breiman | 2001 | Two Cultures |
| Judd, McClelland & Ryan | 2011 | Model Comparison textbook |
| Bzdok & Ioannidis | 2019 | Explanation vs Prediction |
| Yarkoni & Westfall | 2017 | Generalizability crisis |
| Jolly & Chang | 2019 | Flatland Fallacy |
| Gelman & Vehtari | 2021 | Important statistical ideas |
| Tong | 2019 | Statistical thinking enables good science |

---

## Implementation Notes for Claude

### When Creating Course Materials

1. **Match the scaffolding level** to where students are in the sequence:
   - Week 1: High scaffolding, explicit syntax, R comparisons
   - Week 2-3: Medium scaffolding, hints but not solutions
   - Week 4+: Lower scaffolding, documentation-first approach

2. **Use consistent callout conventions** (see Scaffolding Patterns above)

3. **Include "Your Turn" exercises** with:
   - `...` placeholders
   - Hints in comments
   - Collapsible solutions
   - Expected output shown

4. **Prefer Polars over Pandas** - use contexts + expressions mental model

5. **Prefer Seaborn over matplotlib** - use figure-level functions when possible

6. **Connect to the Four Intuitions** when introducing statistical concepts

7. **Use the "Worth it?" framing** for hypothesis testing / model comparison

### When Writing Assessments

1. **Progressive difficulty** within tasks (simple → combined → synthesis)

2. **Include interpretation questions** - not just code, but "what does this tell us?"

3. **Provide scaffolding that fades** - more hints early, fewer later

4. **Real research context** - use realistic datasets with meaningful questions

5. **Rubric alignment** - consider all 5 competencies, not just computation

### Communication Principles

1. **Anticipate R-user struggles** - 0-based indexing, method chaining, string quotes

2. **Error messages are learning opportunities** - explain what went wrong and why

3. **Documentation as skill** - encourage `help()`, API reference, official docs

4. **Frame challenges positively** - "frustration is normal, hours will become minutes"
