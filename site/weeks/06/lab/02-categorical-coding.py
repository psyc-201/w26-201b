import marimo

__generated_with = "0.19.10"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Categorical Predictors & Coding Schemes

    In the previous notebook we added a **2-level** categorical predictor (`Student`: Yes / No) using **treatment coding**. That was straightforward — one dummy variable, one reference group, done.

    But what happens when a predictor has **3 or more levels**? And does it matter *how* we code those levels into numbers?

    Spoiler: the individual parameter estimates **do** change with different coding schemes, but there's a test that gives the **same answer no matter what**. Understanding when and why is the key lesson of this notebook.

    By the end you'll know how to:

    - Encode a **3+ level** categorical predictor as multiple dummy variables
    - Apply three coding schemes: **treatment**, **sum**, and **polynomial**
    - Inspect the **design matrix** to see exactly what each scheme does
    - Use `.jointtest()` to ask **"does this predictor matter at all?"** — and understand why the answer is **invariant** to coding
    - Distinguish between the **joint test** (omnibus) and **individual parameter** questions
    - Combine multiple categorical predictors (additive & interaction)
    """)
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from polars import col
    import seaborn as sns
    from bossanova import model

    return col, mo, model, pl, sns


@app.cell
def _(pl):
    credit = pl.read_csv("./data/credit.csv")
    credit.head()
    return (credit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Quick reminder of the variables we'll focus on today:

    | Variable   | Description                          | Type        |
    |------------|--------------------------------------|-------------|
    | Balance    | Average credit card debt ($)         | Continuous  |
    | Ethnicity  | African American / Asian / Caucasian | Categorical (3 levels) |
    | Gender     | Male / Female                        | Categorical (2 levels) |
    | Student    | Yes / No                             | Categorical (2 levels) |
    | Married    | Yes / No                             | Categorical (2 levels) |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Three-Level Categorical: The Default

    In the previous notebook we modeled `Student` (2 levels) — that required **1** dummy column ($k - 1 = 2 - 1 = 1$).

    Now let's look at `Ethnicity`, which has **3 levels**: African American, Asian, Caucasian. That means we need $k - 1 = 2$ dummy columns.

    By default the *reference* level is the first *alphabetical* level of the factor. In this case *African American*:

    $$
    Balance_i = \beta_0 + \beta_1 \cdot Ethnicity[Asian]_i + \beta_2 \cdot Ethnicity[Caucasian]_i
    $$

    Let's visualize first:
    """)
    return


@app.cell
def _(credit, sns):
    sns.catplot(
        data=credit.to_pandas(),
        x="Ethnicity",
        y="Balance",
        kind="violin",
        inner="stick",
        height=5,
        aspect=1.3,
        color="steelblue",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    Now let's take a look at the design matrix:
    """)
    return


@app.cell
def _(credit, model):
    m_eth = model("Balance ~ Ethnicity", credit)
    m_eth.plot_design()
    return (m_eth,)


@app.cell
def _(m_eth):
    m_eth.designmat.head()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Let's try to understand what the default encoding is doing here by just grabbing the unique rows that correspond to each unique factor level of Ethnicity:
    """)
    return


@app.cell
def _(m_eth):
    # Sort just to make it easier to discuss
    m_eth.designmat.unique().sort(by='Ethnicity[Caucasian]')
    return


@app.cell
def _(mo):
    mo.md(r"""
    Here's how we can read this:
    - Each **row** corresponds to one unique possible factor level. From top to bottom:
      - `[1 0 0]`: African American
      - `[1 1 0]`: Asian
      - `[1 0 1]`: Caucasian
    - Each **column** corresponds to what that model parameter represents. From left to right:
      - `Intercept`: mean of African American
      - `Ethnicity[Asian]`: difference between Asian - African American
      - `Ethnicity[Caucasian]`: difference between Caucasian - African American
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And in GLM terms:

    - The **reference level** is chosen alphabetically → **African American**
    - $\beta_0$ (Intercept) = mean Balance for African Americans
    - $\beta_1$ (`Ethnicity[Asian]`) = Asian mean $-$ African American mean
    - $\beta_2$ (`Ethnicity[Caucasian]`) = Caucasian mean $-$ African American mean

    With 3 levels we get $k - 1 = 2$ dummy columns. The reference group *doesn't need its own column* — it's captured by the **Intercept.**
    """)
    return


@app.cell
def _(col, credit):
    # Group Means:
    group_means = (
        credit
        .group_by("Ethnicity")
        .agg(col("Balance").mean().alias("mean_balance"))
        .sort("Ethnicity")
    )
    group_means
    return (group_means,)


@app.cell
def _(m_eth):
    # Parameter estimates
    m_eth.fit().params
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Three Coding Schemes

    Above we met the *default* way the GLM handles categorical predictors (switches) and we learned that this called **treatment coding** in class. But this is just *one* way to convert categories into numbers.

    Different coding schemes change what the Intercept and slopes *mean*, but — as we'll see — they don't change the model's overall fit or predictions.

    Let's compare three schemes side-by-side.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 1. Treatment Coding (default)

    We've already seen this one:

    | Property | Value |
    |----------|-------|
    | **Intercept** | Mean of the reference group |
    | **Each slope** | Difference from reference group |
    | **Design matrix** | 0/1 pattern (reference = all zeros) |
    | **Best when** | There's a natural reference group (e.g., control condition) |
    """)
    return


@app.cell
def _(m_eth):
    # Recap: treatment coding params (already fit above)
    m_eth.params
    return


@app.cell
def _(mo):
    mo.md(r"""
    But we can control which group is the reference group directly *in the model formula* using `treatment()` and fitting again:
    """)
    return


@app.cell
def _(credit, model):
    # bossanova methods can be chained like polars!
    (
        model("Balance ~ treatment(Ethnicity, ref='Caucasian')", credit) # contrasts in-formula
        .fit() # fit model
        .params # inspect params
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    Notice how the *Intercept* now reflect the mean of the Caucasian group.
    Like other software `bossanova` will automatically calculate the remaining codes for the other factor levels, but loses their names (because you can specify very complicated codes!).
    """)
    return


@app.cell
def _(group_means):
    # Reverse alphabetical sort using the negative list slicing trick!
    group_means[::-1]
    return


@app.cell
def _(mo):
    mo.md(r"""
    > Your Turn

    Can you figure out what the other parameters represent?
    Make a figure in seaborn to help you visualize the data and check

    *hint: we've simply moved things around so the "frame-of-reference" is the Caucasian group*
    """)
    return


@app.cell
def _():
    # Your code here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Sum (Deviation) Coding

    In class we discussed how we can **mean center** continuous (slider) variables in our model to help with interpretability (and multi-collinearity). You can think of **sum (deviation) coding** as approximately equivalent for categorical (switch) variables.

    Sum coding changes what the Intercept and slopes represent:

    | Property | Value |
    |----------|-------|
    | **Intercept** | Grand mean (unweighted mean of group means) |
    | **Each slope** | Deviation of that group from the grand mean |
    | **Design matrix** | -1 / 0 / 1 pattern (last group = all -1s) |
    | **Best when** | No natural reference; you want each group compared to the overall average |

    The key property: each column in the design matrix **sums to zero** across levels — hence "sum" coding.
    """)
    return


@app.cell
def _(credit, model):
    m_sum = model("Balance ~ sum(Ethnicity)", credit)
    m_sum.plot_design()
    return (m_sum,)


@app.cell
def _(m_sum):
    m_sum.designmat.unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the **-1 / 0 / 1** pattern in the design matrix. The last level alphabetically (Caucasian) gets coded as -1 on both columns — it's the "omitted" group whose deviation can be recovered as the negative sum of the other deviations.

    You can interpret this encoding scheme as having the same effect that *mean centering* has on continuous (slider) predictors. It removes collinearity, but in the case of factor levels makes things *harder* to interpet, with the benefit being that it also "balances out" any differences in group *size* that you may have when you make comparisons.

    > **Why columns sum to zero**
    >
    > In sum coding, the contrast columns are designed so that the Intercept estimates the **grand mean** — the unweighted average across all group means. Each slope then measures how far a group deviates from that average.
    >
    > This is the coding scheme most naturally connected to ANOVA-style thinking, where we ask "how do groups differ from the overall average?" while accounting for imbalances in group-sizes.
    """)
    return


@app.cell
def _(col, group_means):
    # Verify: Intercept should be the grand mean (unweighted mean of group means)
    grand_mean = group_means.select(col("mean_balance").mean())
    grand_mean
    return


@app.cell
def _(m_sum):
    m_sum.fit().params
    return


@app.cell
def _(mo):
    mo.md(r"""
    > Your Turn

    Like before, see if you can figure out what the other two parameters represent in terms of group-means. It's tricky! And you'll need to "mix" groups together
    """)
    return


@app.cell
def _():
    # Your code here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3. Polynomial Coding

    Polynomial coding is extremely useful whenever your factor levels (or hypotheses about them) are **ordered**. For example: "is there a linear *trend* across groups?"

    | Property | Value |
    |----------|-------|
    | **Intercept** | Grand mean |
    | **Slopes** | Linear trend, quadratic trend, etc. |
    | **Design matrix** | Orthogonal polynomial contrasts |
    | **Best when** | Levels have a natural ordering (e.g., Low / Medium / High) |

    Let's see what happens when we apply it to Ethnicity — even though Ethnicity has **no natural ordering**.

    > For `Ethnicity`, the order is just alphabetical — there's no reason "African American → Asian → Caucasian" should be treated as a progression, so the "linear" and "quadratic" trends are not particularly informative in this example, but you can imagine a study design or dataset where they might be (e.g. pain severity, ses, etc).
    """)
    return


@app.cell
def _(credit, model):
    m_poly = model("Balance ~ poly(Ethnicity)", credit)
    m_poly.plot_design()
    return (m_poly,)


@app.cell
def _(m_poly):
    m_poly.designmat.unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The design matrix shows **orthogonal** contrast weights. The first parameter captures a "linear" trend across levels, and the second captures a "quadratic" (U-shaped) trend. Orthogonal means "statistical independent" (in a linear sense), so we've de-correlated these predictors as much as possible.

    For example check out the correlation between polynomial encoding:
    """)
    return


@app.cell
def _(m_poly):
    m_poly.plot_vif()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Vs sum coding:
    """)
    return


@app.cell
def _(m_sum):
    m_sum.plot_vif()
    return


@app.cell
def _(mo):
    mo.md(r"""
    But **critically** notice that the model's *predictions* do not change!

    We're just mixing things around but not fundamentally changing anything about the GLM's behavior!
    """)
    return


@app.cell
def _(m_eth, m_poly, m_sum, pl, sns):
    # Each model's predictions (.fit() first so this cell doesn't rely on
    # fits that happened in other cells, which marimo's graph can't order on)
    all_fits = pl.DataFrame({"Treatment": m_eth.fit().data['fitted'], "Sum": m_sum.fit().data['fitted'], "Poly": m_poly.fit().data["fitted"]})

    # Make a correlation matrix
    ax = sns.heatmap(all_fits.corr().to_pandas(), vmin=-1, vmax=1, cmap='coolwarm', linewidths=1, annot=True, yticklabels=all_fits.columns)
    ax.set_title("Correlation Matrix of Model Predictions")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Coding Schemes Summary

    | | Treatment | Sum | Polynomial |
    |---|---|---|---|
    | **Intercept** | Reference group mean | Grand mean | Grand mean |
    | **Slopes** | Differences from reference | Deviations from grand mean | Linear, quadratic, … trends |
    | **Design matrix** | 0 / 1 | -1 / 0 / 1 | Orthogonal weights |
    | **Use when** | Natural reference group | No natural reference, ANOVA-style | Ordered levels |
    | **Predictions** | Same | Same | Same |
    | **$R^2$** | Same | Same | Same |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Joint (Omnibus/ANOVA tests) & Model Comparison

    Here's the big idea to connect a few pieces together:
    - We have a factor with 3 levels, but the exact estimates we calculate depend on our coding scheme
    - We're just interested in whether Ethnicity matters *overall* for predicting credit-card balance
    - So how can we do this? **Model comparison**
    """)
    return


@app.cell
def _():
    from bossanova import compare

    return (compare,)


@app.cell
def _(compare, credit, m_eth, model):
    # Compact model: no Ethnicity (intercept only)
    m_compact = model("Balance ~ 1", credit).fit()

    # compare() should give the same F and p-value as jointtest()
    compare(m_compact, m_eth)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Now run the next code cell and check-out the output:
    """)
    return


@app.cell
def _(m_eth):
    m_eth.jointtest()
    return


@app.cell
def _(mo):
    mo.md(r"""
    This is the most important idea in this notebook: **ANOVA = Nested Model Comparison**

    The `.jointtest()` method is how `bossanova` computes a what we call traditional Analysis of Variance (ANOVA). However, we've named it more informatively based on what it actually *does*, just like the same function in R's `emmeans` package.

    An omnibus/F/jointtest, tests all parameters that reflect a factor variable simulataneously (hence the name). But what does "test simultaneously mean?"

    It's just **nested model comparison** between a compact model *without* the term, and an augmented model *including* the term.
    The F-ratio tells us the uncertainty we should have in our PRE when this term is included in the model - *how much better does it predict credit card balance?*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Critically, our coding scheme *does not effect* the joint-test!

    We'll explain more in class, but what `jointtest` is doing behind-the-scenes is what Psychologists (almost exclusively) call a "Type-III Sum-of-Squares" test.

    This is consistent with modern tools in R like the tidyverse that automatically handle this for you.

    However, if you're manually doing this with other tools in Python or in R (e.g. the `aov()` or `anova()` functions) you **first must use either sum or polynomial encoding** to get valid traditional ANOVA/jointtest results!

    This is just a (pointless) statistical relic that `bossanova` just does the "right thing" for you when you ask for a `jointtest`.
    """)
    return


@app.cell
def _(m_sum):
    # Sum coding
    m_sum.jointtest()
    return


@app.cell
def _(m_poly):
    # Polynomial coding
    m_poly.jointtest()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The F-statistic and p-value from `compare()` match `.jointtest()` exactly. They're asking the same question: **"Is Ethnicity worth including in the model?"**

    > **Joint test vs individual parameters — when to use which**
    >
    > | Question | Tool | Coding-dependent? |
    > |----------|------|-------------------|
    > | "Does Ethnicity matter *at all*?" | `.jointtest()` or `compare()` | **No** — always the same answer |
    > | "How does Asian differ from African American?" | `.params` with treatment coding | **Yes** — depends on reference level |
    > | "How does each group deviate from the grand mean?" | `.params` with sum coding | **Yes** — depends on coding scheme |
    >
    > Think of it this way: the **joint test** asks whether the predictor is *worth including*, while the **individual parameters** tell you the *specific pattern* — and the pattern's description depends on how you coded the levels.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Two Factor Models

    Let's increase the complexity a bit and include Student (which we've previously seen) as another variable.
    Remember it has 2 levels (Yes/No).

    We'll start with the default (treatment coding):
    """)
    return


@app.cell
def _(credit, model):
    m_add = model("Balance ~ Ethnicity + Student", credit)
    m_add.plot_design()
    return (m_add,)


@app.cell
def _(m_add):
    m_add.designmat.unique()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Notice now how there are 6 unique combinations required to capture all of our factor level combinations:
    - Ethnicity: 3 levels, k = 2
    - Student: 2 levels, k = 1
    - 2 + 1 = **3 columns to represent all possible differences**
    - 3 + 2 (+ 1) = **6 combinations to represent each unique difference**
    """)
    return


@app.cell
def _(m_add):
    m_add.fit().params
    return


@app.cell
def _(mo):
    mo.md(r"""
    > Your turn

    See if you can figure out what these parameter estimates represent?
    """)
    return


@app.cell
def _():
    # Your code here
    return


@app.cell
def _(mo):
    mo.md(r"""
    Let's take it a step further and include an interaction:
    """)
    return


@app.cell
def _(credit, model):
    m_int = model("Balance ~ Ethnicity * Student", credit)
    m_int.plot_design()
    return (m_int,)


@app.cell
def _(m_int):
    m_int.designmat.unique()
    return


@app.cell
def _(mo):
    mo.md(r"""
    > Your turn

    Call `.fit()` and then inspect `.params` and try to understand what each parameter reflects in terms of factor level differences.

    It'll be helpful to use `polars` to aggregate the groups first and verify
    """)
    return


@app.cell
def _():
    # You code here
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's run a jointtest (traditional ANOVA):
    """)
    return


@app.cell
def _(m_int):
    m_int.jointtest()
    return


@app.cell
def _(mo):
    mo.md(r"""
    - Row 1 represents the "main effect" of Ethnicity
    - Row 2 represents the "main effect" of Student
    - Row 3 represents their interaction

    Let's try to understand what's happening by seeing this as the equivalent set of model comparisons starting with the interaction.

    To test if the interaction is **worth it** we need create a *compact* model thats the same in everyway *except* the interaction:

    That's just our `Balance ~ Ethnicity + Student` model from earlier!
    """)
    return


@app.cell
def _(compare, m_add, m_int):
    compare(m_add, m_int)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Notice how the F-ration and p-values are identitical to the last row of the `jointtest`

    Now to test "main effects" we need to remove 1 variable at a time, while keeping in mind they contribute to the interaction. It's helpful to keep in mind that this formula:

    `Balance ~ Ethnicity * Student` is really:

    `Balance ~ Ethnicity + Student + Ethnicity:Student`
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    So we need to create 2 more nested models:
    """)
    return


@app.cell
def _(credit, model):
    # Drop Ethnicity (keep interaction!)
    drop_eth = model("Balance ~ Student + Ethnicity:Student", credit).fit()

    # Drop Student (keep interaction!)
    drop_student = model("Balance ~ Ethnicity + Ethnicity:Student", credit).fit()
    return drop_eth, drop_student


@app.cell
def _(compare, drop_eth, m_int):
    # Main effect of Ethnicity
    compare(drop_eth, m_int)
    return


@app.cell
def _(compare, drop_student, m_int):
    # Main effect of Student
    compare(drop_student, m_int)
    return


@app.cell
def _(m_int):
    # Just to compare
    m_int.jointtest()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Your Turn

    Use the cells below to practice. Some ideas:

    1. Fit `Balance ~ Gender` and `Balance ~ Married` — check the joint tests and params
    2. Try changing the reference level for treatment coding
    3. Verify that the joint test is still the same with a different reference level
    4. Fit a large model with all categorical predictors and inspect its `.jointtest()`: `Balance ~ Ethnicity * Student * Gender`
    5. Use `compare()` and any new nested mdoels you need to create to match the output of each main-effect and interaction

    *Tip: If you're confused make some plots to help you out!*
    """)
    return


@app.cell
def _(credit):
    credit.head()
    return


@app.cell
def _():
    # Your code here
    return


@app.cell
def _():
    # Your code here
    return


@app.cell
def _():
    # Your code here
    return


if __name__ == "__main__":
    app.run()
