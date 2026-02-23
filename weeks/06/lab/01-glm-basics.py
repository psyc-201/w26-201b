import marimo

__generated_with = "0.19.9"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Regression Basics

    We're *finally* going to fit some statistical models the easy way! This tutorial introduces [`bossanova`](https://sciminds.ucsd.edu/bossanova), a new Python library maintained by Eshin & team at the [SciMinds Research Studio](https://sciminds.studio). It's been purposely designed to borrow the best patterns from R and improve upon them based upon the approaches we've been learning about in class. It's also a fully functional statistical library that replaces `lm`, `glm`, `lmer` and `glmer` in R that you can use for any of your projects. We'll continue using `bossanova` for the rest of the course as we get to these additional topics.

    By the end of this tutorial you'll know how to:

    - Create and fit regression models using `bossanova`
    - Inspect the **design matrix** — what the model actually "sees"
    - Evaluate model fit with residual diagnostics
    - Improve coefficient interpretation with **centering**
    - Test hypotheses using **model comparison**
    - Include **multiple continuous predictors** and their **interactions**
    - Diagnose **multicollinearity** with variance inflation factors (VIF)
    - Add **categorical predictors** with treatment coding
    - Fit **parallel lines** and **different slopes** models

    ## The Credit Dataset

    We'll use the credit-card dataset mention in class which contains records from 400 people:

    | Variable   | Description                        |
    |------------|------------------------------------|
    | Income     | Annual income (thousands of $)     |
    | Limit      | Credit limit                       |
    | Rating     | Credit rating                      |
    | Cards      | Number of credit cards             |
    | Age        | Age in years                       |
    | Education  | Years of education                 |
    | Gender     | Male / Female                      |
    | Student    | Yes / No                           |
    | Married    | Yes / No                           |
    | Ethnicity  | African American / Asian / Caucasian |
    | Balance    | Average credit card debt ($)       |
    """)
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from polars import col
    import seaborn as sns

    return col, mo, pl, sns


@app.cell
def _(pl):
    credit = pl.read_csv("./data/credit.csv")
    credit.head()
    return (credit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Exploration

    Before building any model, **always plot your data first**. Our primary question:

    **Is there a relationship between *Income* (how much a person makes) and their *Balance* (average credit card debt)?**

    $$
    Balance_i = \beta_0 + \beta_1 \cdot Income_i
    $$

    Let's start by visualizing the relationship:
    """)
    return


@app.cell
def _(credit, sns):
    sns.relplot(
        data=credit.to_pandas(),
        x="Income",
        y="Balance",
        kind="scatter",
        height=4,
        aspect=1.3,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There seems to be a positive relationship — as Income increases, so does Balance. But notice the cluster of observations at the low end of Income. Let's zoom in on people with Income below $50K:
    """)
    return


@app.cell
def _(col, credit, sns):
    sns.relplot(
        data=credit.filter(col("Income") < 50).to_pandas(),
        x="Income",
        y="Balance",
        kind="scatter",
        height=4,
        aspect=1.3,
    ).set(title="Income < $50K")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A large number of people with incomes below this threshold carry a Balance of **$0**! There's very little *variance* in Balance for this region of the data. Keep this in mind — our model may struggle here.

    Let's ask `seaborn` to add a regression line to the full data. This gives us a preview before we fit our own model:
    """)
    return


@app.cell
def _(credit, sns):
    sns.lmplot(
        data=credit.to_pandas(),
        x="Income",
        y="Balance",
        ci=None,
        scatter_kws={"alpha": 0.25},
        height=4,
        aspect=1.3,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This confirms our visual intuition about a positive linear relationship. Let's model it ourselves with `bossanova`!
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Univariate Regression

    ### Creating a model

    `bossanova`'s `model()` function takes two arguments:

    - **formula**: a string specifying the model using [Wilkinson notation](https://www.econometrics.blog/post/the-r-formula-cheatsheet/) (same as R's `lm()`)
    - **data**: a polars DataFrame

    Here's a quick reference for formula syntax:

    | Formula | Meaning |
    |---------|---------|
    | `y ~ x` | Intercept + slope for x |
    | `y ~ 1 + x` | Same as above (intercept is implicit) |
    | `y ~ 0 + x` | Slope only, no intercept |
    | `y ~ x1 + x2` | Additive: two main effects |
    | `y ~ x1 * x2` | Main effects + interaction |
    | `y ~ x1:x2` | Interaction only (no main effects) |
    """)
    return


@app.cell
def _():
    from bossanova import model

    return (model,)


@app.cell
def _(credit, model):
    m1 = model("Balance ~ Income", credit)
    m1
    return (m1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Remember that formula syntax automatically adds an intercept. These two formulas are equivalent:

    - `Balance ~ Income`
    - `Balance ~ 1 + Income`

    ### The Design Matrix

    Before fitting, we can inspect the **design matrix** — the actual numbers the model will work with. This is the $X$ matrix from the GLM equation $\hat{y} = X\hat{\beta}$:
    """)
    return


@app.cell
def _(m1):
    m1.designmat.head()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Let's visualize it which is helpful as we get to more complex models
    """)
    return


@app.cell
def _(m1):
    m1.plot_design()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Reading the design matrix**
    >
    > The design matrix has one row per observation and one column per model term:
    >
    > - **Intercept** — a column of 1s (the "constant" that captures the baseline level of Balance)
    > - **Income** — the raw income values for each person
    >
    > The heatmap visualization lets you see the structure at a glance. This becomes more useful with multiple predictors.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Fitting the model

    To estimate the model parameters ($\hat{\beta}$), we can call `.fit()`

    This won't return anything, but stores the **parameter estimates** as a dataframe in `.params`
    """)
    return


@app.cell
def _(m1):
    m1.fit()
    m1.params
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpreting the coefficients

    Each coefficient tells you the expected change in `Balance` for a **1-unit increase** in that predictor, **when all other predictors are fixed at 0**:

    - **Intercept** ($\hat{\beta}_0 \approx 247$) — the predicted Balance when Income = 0
    - **Income** ($\hat{\beta}_1 \approx 6.05$) — for each additional $1,000 of income, Balance increases by ~$6

    In plain language: *"For each additional thousand dollars of income, a person's average credit card debt is expected to increase by about $6."*

    But wait — is the Intercept interpretation meaningful? We'll come back to this...

    ### Model Evaluation

    Before drawing conclusions, we should check that we're not violating the core assumptions of the GLM: **independent and identically distributed (iid) errors**.

    The `.diagnostics` property gives us key fit statistics:
    """)
    return


@app.cell
def _(m1):
    m1.diagnostics
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Coefficient of Determination ($R^2$)**
    >
    > $R^2$ measures the proportion of variance in Balance that our model explains:
    >
    > $$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$
    >
    > - $R^2 \approx 1$ → model explains most of the variance
    > - $R^2 \approx 0$ → model explains very little
    > - $R^2 < 0$ → model is systematically worse than just predicting the mean
    >
    > Our $R^2 \approx 0.21$ means Income alone explains about 21% of the variance in Balance. Not bad for a single predictor, but there's clearly more going on.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    We can also check the `.metadata` which summarizes useful info about the data the model use for fitting. This is helpful because we have to drop drows where data is missing and this lets you inspect if/when that's happening!
    """)
    return


@app.cell
def _(m1):
    m1.metadata
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Augmented Data**

    Similar to libraries like `broom` in R, after calling `.fit()`, the `.data` property of a model stores the original data plus several computed columns. We'll explore these in more detail soon:

    | Column | Description |
    |--------|-------------|
    | `fitted` | Predicted values ($\hat{y}_i$) |
    | `resid` | Residuals ($y_i - \hat{y}_i$) |
    | `hat` | Leverage values |
    | `std_resid` | Standardized residuals |
    | `cooksd` | Cook's distance (influence) |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Most helpful is to inspect the model residuals (errors) to see if any patterns help us diagnose violations of assumptions:
    """)
    return


@app.cell
def _(m1):
    m1.plot_resid()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Reading residual plots**
    >
    > The 4-panel display helps check our GLM assumptions:
    >
    > 1. **Residuals vs Fitted** — look for random scatter around 0 (no patterns)
    > 2. **Q-Q Plot** — check if residuals are approximately normal (points on the diagonal)
    > 3. **Scale-Location** — check for equal variance across fitted values (should be flat)
    > 4. **Residuals vs Leverage** — identify influential observations (watch for high Cook's distance)
    >
    > Notice the cluster of large negative residuals at low fitted values — these are the people with low incomes who carry $0 Balance. Our model over-predicts for them because there's no variance in Balance to learn from in that region.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    We can see and access these columns directly if needed:
    """)
    return


@app.cell
def _(m1):
    m1.data.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's plot predicted vs observed Balance to see how well our model does:
    """)
    return


@app.cell
def _(m1, sns):
    sns.relplot(
        data=m1.data.to_pandas(),
        x="fitted",
        y="Balance",
        kind="scatter",
        height=4,
        aspect=1.2,
    ).set_axis_labels("Predicted Balance", "Observed Balance")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpreting Parameter Estimates

    You might be thinking: *"I thought the Intercept was always the mean of $y$ — what gives?"*

    The key lesson when interpreting **any** GLM parameter:

    > A parameter estimate is the expected change in $y$ for a 1-unit change in its predictor, **when all other model parameters are fixed at 0**.

    So our Intercept of ~$247 is the predicted Balance **when Income = 0**. But is that even in the range of our data?
    """)
    return


@app.cell
def _(credit, sns):
    sns.displot(
        data=credit.to_pandas(),
        x="Income",
        height=5,
        aspect=1.3,
        bins=50,
    ).set_axis_labels("Income ($K)").set(xlim=(0, 200))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Nobody in our sample has an Income near $0! The Intercept is **extrapolating** to a region of the data we never observed. Models will happily make predictions outside the range of your data, but those predictions aren't always meaningful.

    ## Centering for Interpretability

    What we really want is for the Intercept to represent Balance at some *reasonable, observed* value of Income. We can achieve this by **mean-centering** — subtracting the mean from each value:

    $$Income\_c_i = Income_i - \overline{Income}$$

    This shifts Income so that **0 = the sample mean**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    We can do this manually in polars by creating a new column like this:
    """)
    return


@app.cell
def _(col, credit):
    credit.with_columns(Income_c=(col("Income") - col("Income").mean())).head()
    return


@app.cell
def _(mo):
    mo.md(r"""
    *But* `bossanova` makes this easy for us by allowing us to specify this directly in the formula when we create a `model`:
    """)
    return


@app.cell
def _(credit, model):
    # Just wrap a column in center() to transform it!
    m1c = model("Balance ~ center(Income)", credit).fit()
    return (m1c,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Original**
    """)
    return


@app.cell
def _(m1):
    m1.params
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Centered**
    """)
    return


@app.cell
def _(m1c):
    m1c.params
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice that the Intercept changed, what's going on?

    > - **Intercept changed** — it's now the predicted Balance when Income is at its *mean*. For a univariate model, this equals the mean of Balance itself!
    > - **Slope is identical** — centering never changes the slope of the variable being centered
    > - **Predictions are identical** — centering changes *interpretation*, not model fit

    Let's verify that the centered Intercept really is the mean of Balance:
    """)
    return


@app.cell
def _(col, credit):
    credit.select(col("Balance").mean())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And let's verify that the fitted values are identical between both models:
    """)
    return


@app.cell
def _(m1, m1c, pl, sns):
    # Make a new df
    both_fitted = pl.DataFrame({
        'Original Fitted': m1.data['fitted'],
        'Centered Fitted': m1c.data['fitted'],
    })

    sns.relplot(
        both_fitted.to_pandas(),
        x='Original Fitted',
        y='Centered Fitted',
        height=4,
        aspect=1.3
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Centering takeaways

    - Centering fixes a parameter at a meaningful value (e.g., the mean) to aid interpretation of *other* parameters
    - It **never changes** the slope of the centered variable or the model's predictions
    - You should *almost always* center continuous predictors unless a value of 0 is of theoretical interest
    - Centering becomes **essential** with interactions (as we'll see shortly)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Model Comparison

    We've estimated a relationship between Income and Balance, but how do we know it's **meaningful**? We can formulate hypotheses as **model comparisons** — comparing a *compact* model to an *augmented* model and asking whether the trade-off between added complexity and reduction in error is **worth it**.

    <div align="center">
    <img src="./figs/model_comparison.png" width="90%" alt="Model comparison diagram">
    </div>

    ### Proportional Reduction in Error (PRE)

    The key idea: compare two nested models and ask what **proportion** of the compact model's error the augmented model eliminates:

    - **Compact model**: $Balance_i = \beta_0$ (intercept only — just predict the mean)
    - **Augmented model**: $Balance_i = \beta_0 + \beta_1 \cdot Income_i$ (add Income)

    $$
    PRE = \frac{SSE_{compact} - SSE_{augmented}}{SSE_{compact}}
    $$

    For single-predictor models, PRE is identical to $R^2$! But the PRE framework generalizes to comparing *any* two nested models.

    ### From PRE to the F-statistic

    PRE alone doesn't tell us whether the improvement is *worth* the added complexity. We convert PRE to an **F-statistic** that accounts for degrees of freedom:

    <div align="center">
    <img src="./figs/fstat.png" width="50%" alt="PRE to F conversion">
    </div>

    The F-statistic weighs the error reduction against how many parameters we added. We can then look up the probability of observing an F this large under the null hypothesis (compact model is correct) to get a **p-value**.

    > **Degrees of freedom**
    >
    > Degrees of freedom (df) capture how many values are "free to vary" after estimating parameters.
    >
    > Consider the values $x = [3, 5, 7, 9, 11]$ with mean $\bar{x} = 7$. If we know the mean and all values except one, we can recover the missing value — it's "fixed." Estimating the mean *consumed* one degree of freedom, which is why the sample variance divides by $n - 1$:
    >
    > $$\sigma^2_x = \frac{\sum_{i=1}^n (x_i - \bar{x})^2}{n-1}$$
    >
    > In the GLM:
    >
    > - **Model df** = number of parameters ($p$). Compact: $p = 1$ (intercept). Augmented: $p = 2$ (intercept + slope).
    > - **Error df** = $n - p$. With 400 observations: compact has $399$, augmented has $398$.
    > - The F-test uses the *difference* in model df (numerator) and the error df of the augmented model (denominator).

    ### Using `compare()`

    `bossanova` provides `compare()` to automate this entire process. Pass the models from simplest to most complex:
    """)
    return


@app.cell
def _():
    from bossanova import compare

    return (compare,)


@app.cell
def _(credit, model):
    m0 = model("Balance ~ 1", credit).fit()
    return (m0,)


@app.cell
def _(compare, m0, m1):
    compare(m0, m1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading the `compare()` output

    | Column | Meaning |
    |--------|---------|
    | `model` | The formula for each model |
    | `df_resid` | Residual degrees of freedom ($n - p$) |
    | `rss` | Residual sum of squares (total prediction error) |
    | `df` | Degrees of freedom for this comparison |
    | `ss` | Sum of squares explained by the added predictor(s) |
    | `F` | F-statistic |
    | `p_value` | Probability of observing this F under the null |
    | `PRE` | Proportional reduction in error |

    The first row is the baseline (compact model) — it has no comparison values. The second row compares the augmented model *to* the compact.

    With a large F-statistic and $p < .001$, adding Income significantly improves our predictions. And notice the PRE matches the $R^2$ from `.diagnostics` — for single-predictor models they're the same!
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    > Note for R users

    `compare()` is similar to `anova()` in R (nested model comparison) but a little smarter and adapts the type of comparison statistic and test based on the *type* of model(s) you're comparing. It's also a bit confusingly named based on how you might familiar with talking about ANOVA.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Univariate regression recap

    We've learned a standard modeling workflow:

    1. **Create** a model: `m = model("y ~ x", data)`
    2. **Inspect** the design matrix: `m.designmat`, `m.plot_design()`
    3. **Fit** the model: `m.fit()`
    4. **Examine** parameters: `m.params`
    5. **Evaluate** fit: `m.diagnostics`, `m.plot_resid()`
    6. **Explore** augmented data: `m.data` (fitted values, residuals, leverage, Cook's distance)
    7. **Center** predictors for interpretability: `center()`
    8. **Compare** models to test hypotheses: `compare(compact, augmented)`

    Now let's see what happens when we add more predictors.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multiple Regression

    We'll now build **multiple regression** models that combine several predictors at once. We'll work with two continuous predictors:

    - **Income** — annual income in thousands of dollars
    - **Rating** — credit rating score

    Let's start by visualizing these relationships:
    """)
    return


@app.cell
def _(credit, sns):
    sns.pairplot(
        data=credit.select("Balance", "Income", "Rating").to_pandas(),
        kind="scatter",
        diag_kind="kde",
        plot_kws={"alpha": 0.5},
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Additive model: `Balance ~ Income + Rating`

    Our first multiple regression includes both predictors **additively** — each has its own slope, but their effects are independent:

    $$
    \hat{Balance}_i = \beta_0 + \beta_1 \cdot Income_i + \beta_2 \cdot Rating_i
    $$
    """)
    return


@app.cell
def _(credit, model):
    m2 = model("Balance ~ Income + Rating", credit)
    return (m2,)


@app.cell
def _(m2):
    m2.designmat
    return


@app.cell
def _(m2):
    m2.plot_design()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The design matrix now has three columns: **Intercept**, **Income**, and **Rating**. Each row is still one observation. The heatmap helps you see how the columns differ in scale.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpreting the coefficients

    Remember: each coefficient tells you the expected change in `Balance` for a **1-unit increase** in that predictor, **holding all other predictors at 0**.

    - **Intercept** — predicted Balance when Income = 0 *and* Rating = 0
    - **Income** — change in Balance per additional $1,000 of income, when Rating = 0
    - **Rating** — change in Balance per additional rating point, when Income = 0
    """)
    return


@app.cell
def _(m2):
    m2.fit()
    m2.params
    return


@app.cell
def _(mo):
    mo.md(r"""
    Let's inspect diagnostics:
    """)
    return


@app.cell
def _(m2):
    m2.plot_resid()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Look at how they've changed from before. If we plot against predictions we can see the same pattern:
    """)
    return


@app.cell
def _(m2, sns):
    sns.relplot(
        data=m2.data.to_pandas(),
        x="fitted",
        y="Balance",
        kind="scatter",
        alpha=0.5,
        height=4,
        aspect=1.3,
    ).set_axis_labels("Predicted Balance", "Observed Balance")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Interaction model: `Balance ~ Income * Rating`

    Does the effect of Income on Balance *depend on* a person's Rating? An **interaction** lets us test this:

    $$
    \hat{Balance}_i = \beta_0 + \beta_1 \cdot Income_i + \beta_2 \cdot Rating_i + \beta_3 \cdot Income_i \times Rating_i
    $$

    The `*` shorthand in the formula automatically includes both main effects and their interaction:
    """)
    return


@app.cell
def _(credit, model):
    m3 = model("Balance ~ Income * Rating", credit)
    m3.plot_design()
    return (m3,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Multi-collinearity

    Remember we learned that creating new predictors that products of others (interaction) will always introduce a correlation. We can see this in the mode before fiting:
    """)
    return


@app.cell
def _(m3):
    m3.plot_vif()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Variance Inflation Factors (VIF)**
    >
    > VIF measures how much the variance of each coefficient is *inflated* due to correlation with other predictors (**multicollinearity**).
    >
    > - **VIF = 1** → no correlation with other predictors
    > - **VIF > 5** → concerning multicollinearity
    > - **VIF > 10** → serious multicollinearity
    >
    > Income and Rating are highly correlated (people with higher incomes tend to have higher credit ratings), so expect elevated VIFs here.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Like before we can solve this *and* the interpretation problem by **centering**:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Centering with multiple predictors

    **Mean-centering** subtracts the mean from each predictor before fitting. This changes *what the coefficients refer to* without changing the model's predictions:

    | Without centering | With centering |
    |---|---|
    | Intercept = Balance at Income=0, Rating=0 | Intercept = Balance at **mean** Income, **mean** Rating |
    | Income slope = effect when Rating=0 | Income slope = effect at **mean** Rating |
    | Interaction = same either way | Interaction = same either way |
    """)
    return


@app.cell
def _(credit, model):
    m3c = model("Balance ~ center(Income) * center(Rating)", credit).fit()
    m3c.vif()

    return (m3c,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Uncentered**
    """)
    return


@app.cell
def _(m3):
    m3.fit()
    m3.params.select("term", "estimate")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Centered**
    """)
    return


@app.cell
def _(m3c):
    m3c.params.select("term", "estimate")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **What changed and what didn't?**
    >
    > - **Intercept changed** — now it's the predicted Balance at the *mean* of both predictors (a meaningful value!)
    > - **Main effect slopes changed** — now interpreted at the *mean* of the other predictor
    > - **Interaction is identical** — centering never affects interactions
    > - **Predictions are identical** — centering changes interpretation, not fit
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Additive vs interaction

    Let's `compare` both models:
    """)
    return


@app.cell
def _(compare, m2, m3c):
    compare(m2,m3c)
    return


@app.cell
def _(mo):
    mo.md(r"""
    This shows us that the addition of the *interaction* is worth-it, tho it only reduces error by about 2%
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Multiple regression takeaways

    1. **Multiple regression** lets you include several predictors — each slope is the effect *holding others constant*
    2. **Interactions** (`*`) let one predictor's effect depend on another — the interaction term is a product column in the design matrix
    3. **Centering** changes what the coefficients *mean* but not what the model *predicts* — use it whenever 0 is not a meaningful value for your predictors
    4. **VIF** helps you diagnose multicollinearity — centering dramatically reduces the artificial inflation caused by interaction terms
    5. **Always inspect the design matrix** (`m.designmat`) to understand what your model actually sees

    Now let's see what happens when we add a *categorical* predictor.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Categorical Predictors (2-levels)

    We'll now add a **categorical** predictor: `Student` (Yes / No).

    The key questions:

    1. Do students carry different balances than non-students, *after accounting for* Income?
    2. Does the *Income → Balance* relationship differ for students vs non-students?

    Let's start by visualizing the data:
    """)
    return


@app.cell
def _(credit, sns):
    sns.catplot(
        data=credit.to_pandas(),
        x="Student",
        y="Balance",
        kind="violin",
        inner="stick",
        height=5,
        aspect=1,
        color="steelblue",
    )
    return


@app.cell
def _(credit, sns):
    sns.lmplot(
        data=credit.to_pandas(),
        x="Income",
        y="Balance",
        hue="Student",
        height=5,
        aspect=1.3,
        scatter_kws={"alpha": 0.3},
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Parallel lines: `Balance ~ Income + Student`

    This is a **parallel lines** model — students and non-students get different intercepts but the *same* slope for Income:

    $$
    \hat{Balance}_i = \beta_0 + \beta_1 \cdot Income_i + \beta_2 \cdot Student[Yes]_i
    $$
    """)
    return


@app.cell
def _(credit, model):
    m4 = model("Balance ~ Income + Student", credit)
    return (m4,)


@app.cell
def _(m4):
    m4.designmat
    return


@app.cell
def _(m4):
    m4.plot_design()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Treatment (dummy) coding**
    >
    > Notice the design matrix has a column called **Student[Yes]** (not "Student"). This is **treatment coding** — the default for categorical variables:
    >
    > - The *reference level* is chosen alphabetically → **No** (because N < Y)
    > - `Student[Yes] = 1` for students, `Student[Yes] = 0` for non-students
    > - The intercept represents the reference group (Student = No)
    > - The `Student[Yes]` coefficient is the *difference* between Yes and No
    >
    > This is the same coding scheme used in R's `lm()` by default.
    """)
    return


@app.cell
def _(m4):
    m4.fit()
    m4.params
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interpreting the coefficients

    - **Intercept** — predicted Balance for a **non-student** (Student = No) when Income = 0
    - **Income** — change in Balance per $1,000 of income (same for both groups)
    - **Student[Yes]** — how much more (or less) Balance students carry than non-students, **holding Income constant**

    The intercept interpretation is awkward — Income = 0 isn't meaningful. We'll fix that with centering shortly.
    """)
    return


@app.cell
def _(m4):
    m4.diagnostics
    return


@app.cell
def _(m4):
    m4.plot_resid()
    return


@app.cell
def _(m4):
    m4.data
    return


@app.cell
def _(m4, sns):
    # Visualize parallel lines from augmented data
    _g = sns.relplot(
        data=m4.data.to_pandas(),
        x="Income",
        y="Balance",
        hue="Student",
        kind="scatter",
        alpha=0.3,
        height=5,
        aspect=1.3,
    )
    # Overlay fitted lines (parallel — same slope, different intercept)
    _g.map_dataframe(sns.lineplot, x="Income", y="fitted", hue="Student", sort=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the lines are **parallel** — adding a categorical predictor with `+` shifts the intercept but keeps the slope the same for both groups.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Different slopes: `Balance ~ Income * Student`

    Does the **Income → Balance** relationship differ for students? An interaction lets us test this — now each group gets its own slope:

    $$
    \hat{Balance}_i = \beta_0 + \beta_1 \cdot Income_i + \beta_2 \cdot Student[Yes]_i + \beta_3 \cdot Income_i \times Student[Yes]_i
    $$
    """)
    return


@app.cell
def _(credit, model):
    m5 = model("Balance ~ Income * Student", credit)
    return (m5,)


@app.cell
def _(m5):
    m5.designmat
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The new **Income:Student[Yes]** column is the product of the dummy variable x Income — it's non-zero *only* for students. This is what allows the slope to differ between groups.
    """)
    return


@app.cell
def _(m5):
    m5.fit()
    m5.params
    return


@app.cell
def _(m5):
    m5.diagnostics
    return


@app.cell
def _(m5):
    m5.plot_resid()
    return


@app.cell
def _(m5, sns):
    # Visualize different slopes per Student group
    _g = sns.relplot(
        data=m5.data.to_pandas(),
        x="Income",
        y="Balance",
        hue="Student",
        kind="scatter",
        alpha=0.3,
        height=5,
        aspect=1.3,
    )
    # Overlay fitted lines (different slopes per group)
    _g.map_dataframe(sns.lineplot, x="Income", y="fitted", hue="Student", sort=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now the lines have **different slopes** — the interaction lets the Income → Balance relationship vary by Student status.

    ### Centering with categorical predictors

    The raw coefficients are interpreted "when Income = 0" — which isn't meaningful. Let's center Income so coefficients are interpreted at the **mean** of Income instead.

    Here we'll create the centered column ourselves with `center()` as a polars expression and use it in the formula:
    """)
    return


@app.cell
def _(credit, model):
    m5c = model("Balance ~ center(Income) * Student", credit).fit()
    return (m5c,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Uncentered**
    """)
    return


@app.cell
def _(m5):
    m5.params.select("term", "estimate")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Centered**
    """)
    return


@app.cell
def _(m5c):
    m5c.params.select("term", "estimate")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Interpreting centered parameters**
    >
    > With centering, each coefficient has a concrete meaning:
    >
    > | Parameter | Interpretation |
    > |-----------|---------------|
    > | **Intercept** | Predicted Balance for **Student = No** at **mean Income** |
    > | **Income_c** | Income slope for **non-students** |
    > | **Student[Yes]** | Difference in Balance between Students and Non-students **at mean Income** |
    > | **Income_c:Student[Yes]** | How much *steeper* (or shallower) the Income slope is for students vs non-students |
    >
    > The interaction term is unchanged by centering — it's always the *difference in slopes*.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Parallel lines vs different slopes

    Is this new interaction term worth it?
    """)
    return


@app.cell
def _(compare, m4, m5c):
    compare(m4, m5c)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    We've built five regression models today, each building on the last:

    | Model | Formula | Key concept |
    |-------|---------|-------------|
    | m1 | `Balance ~ Income` | Univariate regression, baseline workflow |
    | m2 | `Balance ~ Income + Rating` | Multiple predictors, VIF |
    | m3 | `Balance ~ Income * Rating` | Continuous x continuous interaction |
    | m4 | `Balance ~ Income + Student` | Categorical predictor, parallel lines |
    | m5 | `Balance ~ Income * Student` | Continuous x categorical interaction, different slopes |

    ### Key ideas

    1. **Categorical predictors** are automatically converted to dummy (0/1) columns — check `m.designmat` to see how
    2. **Treatment coding** uses the first level alphabetically as the reference — the coefficient is the *difference from that reference*
    3. **Additive models** (`+`) give parallel lines — same slope, different intercepts per group
    4. **Interactions** (`*`) allow different slopes per group — the interaction coefficient is the *difference in slopes*
    5. **Centering** continuous predictors makes all other coefficients interpretable at the *mean* of the centered predictor (not at 0)
    6. **VIF** diagnoses multicollinearity — centering dramatically reduces the artificial inflation from interaction terms
    7. **Model comparison** (`compare()`) tests whether added complexity is worth the reduction in error
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Your Turn

    `bossanova` includes the `penguins` dataset you've seen before. Try using the cells below to play around and make sure you understand these core concepts
    """)
    return


@app.cell
def _():
    from bossanova import load_dataset

    penguins = load_dataset('penguins')
    penguins.head()
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


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
