# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.4.1",
#     "wigglystuff==0.2.16",
#     "altair==6.0.0",
#     "pandas==3.0.0",
# ]
# ///

import marimo

__generated_with = "0.19.5"
app = marimo.App(width="medium")


# ============================================================
# IMPORTS
# ============================================================
@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    from wigglystuff import Slider2D

    alt.data_transformers.disable_max_rows()
    return Slider2D, alt, mo, np, pd


# ============================================================
# TITLE & FRAME THE GOAL
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        title: "Linear Algebra for the GLM"
        author: "Eshin Jolly"
        date: "Feb 2026"
        ---
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Linear Algebra for the General Linear Model

        *The language your statistical model speaks*

        This notebook builds **geometric intuition** for the General Linear Model
        (GLM). The goal isn't to teach linear algebra as a math course — it's to
        give you the visual vocabulary to understand what your model actually does
        when you run a regression.

        > **The big idea:** Every modeling choice — adding a predictor, creating an
        > interaction, coding a categorical variable — is a choice about the
        > *geometry* of your design matrix. Interpretation follows from geometry.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **What you need from linear algebra for the GLM:**

            - The design matrix **X** defines what the model can "see"
            - Columns of **X** are directions in data space; betas scale them
            - Prediction $\hat{y} = X\beta$ must live in the column space (span) of **X**
            - Geometry $\to$ interpretation: multicollinearity, unique contributions, coding schemes
            """
        ),
        kind="info",
    )
    return


# ============================================================
# SECTION 1: VECTORS
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## 1. Vectors: Each Variable is a Direction

        In the GLM, every variable you measure is a **vector** — a list of values,
        one per observation.

        | Person | Height | Age |
        |--------|--------|-----|
        | 1      | 68     | 25  |
        | 2      | 72     | 30  |

        - `height = [68, 72]` $\to$ a vector in 2D "observation space"
        - `age = [25, 30]` $\to$ another vector in the same space
        - Your outcome `y` is also a vector: `y = [3.2, 4.1]`

        Each axis represents a **person**. Each vector represents a **variable**.

        **Try it:** Drag the point below to create different vectors.
        """
    )
    return


@app.cell(hide_code=True)
def _(Slider2D, mo):
    vector_slider = mo.ui.anywidget(
        Slider2D(
            x=0.7,
            y=0.5,
            width=300,
            height=300,
            x_bounds=(-2.0, 2.0),
            y_bounds=(-2.0, 2.0),
        )
    )
    return (vector_slider,)


@app.cell(hide_code=True)
def _(alt, mo, np, pd, vector_slider):
    _vx, _vy = vector_slider.x, vector_slider.y

    _gr = np.linspace(-2, 2, 9)
    _gx, _gy = np.meshgrid(_gr, _gr)
    _grid = (
        alt.Chart(pd.DataFrame({"x": _gx.flatten(), "y": _gy.flatten()}))
        .mark_circle(size=20, color="lightgray")
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-2.5, 2.5]), title="Person 1"),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[-2.5, 2.5]), title="Person 2"),
        )
    )

    _arrow = (
        alt.Chart(pd.DataFrame({"x": [0], "y": [0], "x2": [_vx], "y2": [_vy]}))
        .mark_line(strokeWidth=3, color="steelblue")
        .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
    )

    _endpoint = (
        alt.Chart(pd.DataFrame({"x": [_vx], "y": [_vy]}))
        .mark_circle(size=100, color="steelblue")
        .encode(x="x:Q", y="y:Q")
    )

    _origin = (
        alt.Chart(pd.DataFrame({"x": [0], "y": [0]}))
        .mark_circle(size=80, color="black")
        .encode(x="x:Q", y="y:Q")
    )

    _chart = (_grid + _arrow + _endpoint + _origin).properties(
        width=300, height=300, title="A Vector in Observation Space"
    )

    mo.hstack(
        [
            mo.vstack(
                [mo.md("**Drag to move the vector:**"), vector_slider],
                align="center",
            ),
            _chart,
            mo.vstack(
                [
                    mo.md("**This vector says:**"),
                    mo.md(f"Person 1 = {_vx:.2f}"),
                    mo.md(f"Person 2 = {_vy:.2f}"),
                    mo.md(f"**Length:** {np.sqrt(_vx**2 + _vy**2):.2f}"),
                ],
                align="start",
            ),
        ],
        justify="center",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **GLM connection:** Each predictor in your model is a vector in
            *n*-dimensional space (one dimension per observation). We use 2D here
            for visualization, but the same geometry works in 100D or 1000D.

            - Each **predictor** = a vector (column of X)
            - The **outcome** y = also a vector in the same space
            - Regression asks: *how do we combine the predictor vectors to get close to y?*
            """
        ),
        kind="info",
    )
    return


# ============================================================
# SECTION 2: LINEAR COMBINATIONS
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## 2. Linear Combinations: Betas are Volume Knobs

        A prediction in the GLM is a **linear combination** of predictor vectors:

        $$\hat{y} = \beta_1 \mathbf{x}_1 + \beta_2 \mathbf{x}_2$$

        Each $\beta$ is a **volume knob** — it controls how much each predictor
        contributes to the prediction. The prediction $\hat{y}$ is the weighted sum
        of these scaled vectors.

        Below, **x₁** (red) and **x₂** (blue) are two fixed predictor vectors.
        The gold star is the outcome **y** we're trying to predict.
        Adjust $\beta_1$ and $\beta_2$ to move $\hat{y}$ (purple) as close to **y** as possible.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    beta1_slider = mo.ui.slider(
        start=-2, stop=2, step=0.05, value=0.5, label="β₁ (weight on x₁)"
    )
    beta2_slider = mo.ui.slider(
        start=-2, stop=2, step=0.05, value=0.5, label="β₂ (weight on x₂)"
    )
    mo.hstack([beta1_slider, beta2_slider], justify="center", gap=2)
    return beta1_slider, beta2_slider


@app.cell(hide_code=True)
def _(alt, beta1_slider, beta2_slider, mo, np, pd):
    _b1 = beta1_slider.value
    _b2 = beta2_slider.value

    # Fixed predictor vectors and target
    _x1 = np.array([1.5, 0.5])
    _x2 = np.array([0.3, 1.4])
    _y = np.array([1.0, 1.2])

    # Prediction and residual
    _y_hat = _b1 * _x1 + _b2 * _x2
    _resid = _y - _y_hat
    _ss_resid = float(_resid @ _resid)

    # OLS solution for reference
    _X = np.column_stack([_x1, _x2])
    _beta_ols = np.linalg.lstsq(_X, _y, rcond=None)[0]

    # Grid
    _gr = np.linspace(-2, 2, 9)
    _gx, _gy = np.meshgrid(_gr, _gr)
    _grid = (
        alt.Chart(pd.DataFrame({"x": _gx.flatten(), "y": _gy.flatten()}))
        .mark_circle(size=15, color="lightgray")
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-2.5, 2.5]), title="Obs 1"),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[-2.5, 2.5]), title="Obs 2"),
        )
    )

    # Predictor direction arrows (faint dashed)
    _pred_df = pd.DataFrame(
        {
            "x": [0, 0],
            "y": [0, 0],
            "x2": [_x1[0], _x2[0]],
            "y2": [_x1[1], _x2[1]],
            "c": ["#e41a1c", "#377eb8"],
        }
    )
    _pred_arrows = (
        alt.Chart(_pred_df)
        .mark_line(strokeWidth=2, opacity=0.4, strokeDash=[4, 4])
        .encode(
            x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q",
            color=alt.Color("c:N", scale=None),
        )
    )

    # Scaled predictor components (solid)
    _comp_df = pd.DataFrame(
        {
            "x": [0, 0],
            "y": [0, 0],
            "x2": [_b1 * _x1[0], _b2 * _x2[0]],
            "y2": [_b1 * _x1[1], _b2 * _x2[1]],
            "c": ["#e41a1c", "#377eb8"],
        }
    )
    _comp_arrows = (
        alt.Chart(_comp_df)
        .mark_line(strokeWidth=3)
        .encode(
            x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q",
            color=alt.Color("c:N", scale=None),
        )
    )

    # Prediction vector (purple)
    _yhat_arrow = (
        alt.Chart(
            pd.DataFrame({"x": [0], "y": [0], "x2": [_y_hat[0]], "y2": [_y_hat[1]]})
        )
        .mark_line(strokeWidth=4, color="purple")
        .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
    )
    _yhat_pt = (
        alt.Chart(pd.DataFrame({"x": [_y_hat[0]], "y": [_y_hat[1]]}))
        .mark_circle(size=100, color="purple")
        .encode(x="x:Q", y="y:Q")
    )

    # Target y (gold star)
    _y_pt = (
        alt.Chart(pd.DataFrame({"x": [_y[0]], "y": [_y[1]]}))
        .mark_point(size=200, color="goldenrod", shape="cross", filled=True, strokeWidth=3)
        .encode(x="x:Q", y="y:Q")
    )

    # Residual (dashed line from ŷ to y)
    _resid_line = (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [_y_hat[0]],
                    "y": [_y_hat[1]],
                    "x2": [_y[0]],
                    "y2": [_y[1]],
                }
            )
        )
        .mark_line(strokeWidth=2, color="goldenrod", strokeDash=[3, 3])
        .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
    )

    # Origin
    _origin = (
        alt.Chart(pd.DataFrame({"x": [0], "y": [0]}))
        .mark_circle(size=60, color="black")
        .encode(x="x:Q", y="y:Q")
    )

    _chart = (
        _grid + _pred_arrows + _comp_arrows + _yhat_arrow + _yhat_pt
        + _y_pt + _resid_line + _origin
    ).properties(width=350, height=350, title="ŷ = β₁x₁ + β₂x₂")

    _quality = (
        "Perfect! ŷ = y"
        if _ss_resid < 0.01
        else ("Close!" if _ss_resid < 0.1 else "Keep adjusting...")
    )

    mo.hstack(
        [
            _chart,
            mo.vstack(
                [
                    mo.md(f"**β₁ = {_b1:.2f}**, **β₂ = {_b2:.2f}**"),
                    mo.md(f"ŷ = ({_y_hat[0]:.2f}, {_y_hat[1]:.2f})"),
                    mo.md(f"y = ({_y[0]:.2f}, {_y[1]:.2f})"),
                    mo.md(f"**Residual² = {_ss_resid:.3f}**"),
                    mo.md(f"*{_quality}*"),
                    mo.md(""),
                    mo.md("Red = β₁ · x₁  &nbsp;|&nbsp;  Blue = β₂ · x₂"),
                    mo.md("Purple = their sum (the prediction)"),
                    mo.md(""),
                    mo.md("---"),
                    mo.md(f"**OLS solution:** β₁ = {_beta_ols[0]:.2f}, β₂ = {_beta_ols[1]:.2f}"),
                ],
                align="start",
            ),
        ],
        justify="center",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Key insight:** OLS regression doesn't do anything mysterious — it
            finds the $\beta$ weights that make $\hat{y}$ as close to $y$ as possible
            (minimizing the squared length of the residual). You just did OLS by hand!
            """
        ),
        kind="success",
    )
    return


# ============================================================
# SECTION 3: SPAN & MODEL CAPACITY
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## 3. Span: What the Model Can Possibly Predict

        The **span** (or **column space**) of your predictors is the set of all
        possible predictions — every $\hat{y}$ that *could* be produced for some
        choice of $\beta$.

        - With **1 predictor**: span = a line ($\hat{y}$ can only slide along x₁'s direction)
        - With **2 independent predictors**: span = a plane (in 2D, that's everything)
        - **Adding a predictor expands** what the model can express

        **Misspecification** = the true $y$ falls *outside* the model's span. No
        choice of $\beta$ can fix this — you need different predictors.

        Below, the model has only **one predictor** (x₁). Slide $\beta_1$ and notice
        that $\hat{y}$ is stuck on a line — the **span** of x₁. The gold cross is $y$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    span_beta = mo.ui.slider(
        start=-2, stop=2, step=0.05, value=0.5, label="β₁ (slide ŷ along the span)"
    )
    mo.hstack([span_beta], justify="center")
    return (span_beta,)


@app.cell(hide_code=True)
def _(alt, mo, np, pd, span_beta):
    _b1 = span_beta.value
    _x1 = np.array([1.5, 0.5])
    _y = np.array([1.0, 1.2])

    # ŷ constrained to span of x₁
    _y_hat = _b1 * _x1

    # OLS: optimal β₁ = (x₁·y) / (x₁·x₁)
    _beta_ols = float(_x1 @ _y / (_x1 @ _x1))
    _y_hat_ols = _beta_ols * _x1
    _resid_ols = _y - _y_hat_ols
    _ss_ols = float(_resid_ols @ _resid_ols)

    # Current residual
    _resid = _y - _y_hat
    _ss = float(_resid @ _resid)

    # Span line (extend through origin along x1)
    _t = np.linspace(-2, 2, 50)
    _span_pts = np.outer(_t, _x1)
    _span_df = pd.DataFrame(
        {"x": _span_pts[:, 0], "y": _span_pts[:, 1], "order": range(50)}
    )

    # Grid
    _gr = np.linspace(-2, 2, 9)
    _gx, _gy = np.meshgrid(_gr, _gr)
    _grid = (
        alt.Chart(pd.DataFrame({"x": _gx.flatten(), "y": _gy.flatten()}))
        .mark_circle(size=15, color="lightgray")
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-2.5, 2.5]), title="Obs 1"),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[-2.5, 2.5]), title="Obs 2"),
        )
    )

    # Span line (thick faint purple)
    _span_line = (
        alt.Chart(_span_df)
        .mark_line(strokeWidth=10, opacity=0.12, color="purple")
        .encode(x="x:Q", y="y:Q", order="order:O")
    )

    # Span label
    _span_label_df = pd.DataFrame(
        {"x": [_x1[0] * 1.6], "y": [_x1[1] * 1.6], "text": ["span of x₁"]}
    )
    _span_label = (
        alt.Chart(_span_label_df)
        .mark_text(fontSize=11, color="purple", fontStyle="italic")
        .encode(x="x:Q", y="y:Q", text="text:N")
    )

    # x₁ direction (faint dashed)
    _x1_arrow = (
        alt.Chart(
            pd.DataFrame({"x": [0], "y": [0], "x2": [_x1[0]], "y2": [_x1[1]]})
        )
        .mark_line(strokeWidth=2, color="#e41a1c", strokeDash=[4, 4], opacity=0.5)
        .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
    )

    # ŷ
    _yhat_arrow = (
        alt.Chart(
            pd.DataFrame({"x": [0], "y": [0], "x2": [_y_hat[0]], "y2": [_y_hat[1]]})
        )
        .mark_line(strokeWidth=3, color="purple")
        .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
    )
    _yhat_pt = (
        alt.Chart(pd.DataFrame({"x": [_y_hat[0]], "y": [_y_hat[1]]}))
        .mark_circle(size=80, color="purple")
        .encode(x="x:Q", y="y:Q")
    )

    # Target y
    _y_pt = (
        alt.Chart(pd.DataFrame({"x": [_y[0]], "y": [_y[1]]}))
        .mark_point(
            size=200, color="goldenrod", shape="cross", filled=True, strokeWidth=3
        )
        .encode(x="x:Q", y="y:Q")
    )

    # Residual line
    _resid_line = (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [_y_hat[0]],
                    "y": [_y_hat[1]],
                    "x2": [_y[0]],
                    "y2": [_y[1]],
                }
            )
        )
        .mark_line(strokeWidth=2, color="goldenrod", strokeDash=[3, 3])
        .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
    )

    # Origin
    _origin = (
        alt.Chart(pd.DataFrame({"x": [0], "y": [0]}))
        .mark_circle(size=60, color="black")
        .encode(x="x:Q", y="y:Q")
    )

    _chart = (
        _grid + _span_line + _span_label + _x1_arrow + _yhat_arrow
        + _yhat_pt + _y_pt + _resid_line + _origin
    ).properties(width=350, height=350, title="ŷ Constrained to Span of x₁")

    _is_optimal = abs(_b1 - _beta_ols) < 0.06

    mo.hstack(
        [
            _chart,
            mo.vstack(
                [
                    mo.md("**With only 1 predictor:**"),
                    mo.md("ŷ can only move along the purple line (the **span**)"),
                    mo.md(""),
                    mo.md(f"β₁ = {_b1:.2f} → ŷ = ({_y_hat[0]:.2f}, {_y_hat[1]:.2f})"),
                    mo.md(f"Residual² = {_ss:.3f}"),
                    mo.md(""),
                    mo.md(
                        f"**OLS optimal:** β₁ = {_beta_ols:.2f} "
                        f"(min Residual² = {_ss_ols:.3f})"
                    ),
                    mo.md(
                        "**The residual is perpendicular to x₁!**"
                        if _is_optimal
                        else ""
                    ),
                    mo.md(""),
                    mo.md("*Even at the optimum, residual remains*"),
                    mo.md("*because y doesn't lie on x₁'s span.*"),
                ],
                align="start",
            ),
        ],
        justify="center",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Span expansion:** Adding a second non-collinear predictor x₂ would
            expand the span from a *line* to a *plane* — in 2D, that covers
            everything, so $\hat{y}$ could reach $y$ exactly.

            In real data ($n \gg 2$), each new predictor adds a direction to the
            span. But adding predictors always expands what the model can express,
            at the risk of fitting noise (overfitting).

            **Misspecification** = the true data-generating process creates a $y$
            that falls outside the span of $X$. No choice of $\beta$ can fix this —
            you need different predictors.
            """
        ),
        kind="warn",
    )
    return


# ============================================================
# SECTION 4: DESIGN MATRIX
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## 4. The Design Matrix: What the Model Can See

        The **design matrix** $X$ packages all your predictors as columns:

        $$X = \begin{bmatrix} | & | \\ \mathbf{x}_1 & \mathbf{x}_2 \\ | & | \end{bmatrix}$$

        - **Columns** = directions available to the model (predictor vectors)
        - **Rows** = observations (one row per person/trial)
        - The model's prediction: $\hat{y} = X\beta$ is a linear combination of columns

        Everything the model "knows" comes from these columns. If a pattern in $y$
        can't be expressed as a combination of the columns of $X$, the model simply
        cannot capture it.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Example:** Predicting exam score from hours studied and sleep:

        | Student | Hours Studied (x₁) | Hours Sleep (x₂) | Exam Score (y) |
        |---------|-------------------|------------------|---------------|
        | 1       | 3                 | 7                | 72            |
        | 2       | 5                 | 6                | 85            |
        | 3       | 2                 | 8                | 65            |
        | 4       | 6                 | 5                | 90            |

        $$X = \begin{bmatrix} 1 & 3 & 7 \\ 1 & 5 & 6 \\ 1 & 2 & 8 \\ 1 & 6 & 5 \end{bmatrix} \quad \beta = \begin{bmatrix} \beta_0 \\ \beta_1 \\ \beta_2 \end{bmatrix} \quad y = \begin{bmatrix} 72 \\ 85 \\ 65 \\ 90 \end{bmatrix}$$

        The first column of ones is the **intercept** — it gives the model a
        constant baseline. Columns 2 and 3 are the predictor variables. Together,
        these three columns define the three-dimensional subspace the model explores.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Key principle:** The design matrix is **not just bookkeeping**. Its
            columns define directions in observation space. $\hat{y} = X\beta$ means
            "find the best-fitting point in the subspace spanned by the columns of X."
            Every modeling decision — adding predictors, creating interactions, choosing
            coding schemes — changes the columns of $X$ and therefore changes the
            subspace the model searches within.
            """
        ),
        kind="info",
    )
    return


# ============================================================
# SECTION 5: MULTICOLLINEARITY
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## 5. Multicollinearity: When Predictors Point the Same Way

        **Multicollinearity** happens when two predictors point in nearly the same
        direction. Geometrically, their vectors are almost parallel.

        This creates a problem of **interpretation, not prediction:**

        - $\hat{y}$ is still fine (the span is barely affected)
        - But individual $\beta$ values become **unstable** — many different
          $(\beta_1, \beta_2)$ combinations produce virtually the same $\hat{y}$
        - Betas swing wildly because credit can't be uniquely assigned

        **Try it:** Adjust the angle between predictors. Watch what happens to the
        betas as the predictors become more similar.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    angle_slider = mo.ui.slider(
        start=5, stop=90, step=1, value=60, label="Angle between predictors (degrees)"
    )
    mo.hstack([angle_slider], justify="center")
    return (angle_slider,)


@app.cell(hide_code=True)
def _(alt, angle_slider, mo, np, pd):
    _angle = angle_slider.value
    _angle_rad = np.radians(_angle)

    # x₁ is fixed; x₂ rotates based on angle
    _x1 = np.array([1.0, 0.0])
    _x2 = np.array([np.cos(_angle_rad), np.sin(_angle_rad)])
    _y = np.array([0.8, 0.6])

    # OLS solution
    _X = np.column_stack([_x1, _x2])
    _beta_ols = np.linalg.lstsq(_X, _y, rcond=None)[0]
    _y_hat = _X @ _beta_ols
    _resid = _y - _y_hat
    _ss = float(_resid @ _resid)

    # Condition number (measure of collinearity)
    _cond = np.linalg.cond(_X)

    # Grid
    _gr = np.linspace(-2, 2, 9)
    _gx, _gy = np.meshgrid(_gr, _gr)
    _grid = (
        alt.Chart(pd.DataFrame({"x": _gx.flatten(), "y": _gy.flatten()}))
        .mark_circle(size=15, color="lightgray")
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-2.5, 2.5]), title="Obs 1"),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[-2.5, 2.5]), title="Obs 2"),
        )
    )

    # Predictor vectors
    _pred_df = pd.DataFrame(
        {
            "x": [0, 0],
            "y": [0, 0],
            "x2": [_x1[0], _x2[0]],
            "y2": [_x1[1], _x2[1]],
            "c": ["#e41a1c", "#377eb8"],
            "label": ["x₁", "x₂"],
        }
    )
    _pred_arrows = (
        alt.Chart(_pred_df)
        .mark_line(strokeWidth=3)
        .encode(
            x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q",
            color=alt.Color("c:N", scale=None),
        )
    )
    _pred_pts = (
        alt.Chart(_pred_df)
        .mark_circle(size=60)
        .encode(
            x="x2:Q", y="y2:Q",
            color=alt.Color("c:N", scale=None),
        )
    )
    _pred_labels = (
        alt.Chart(_pred_df)
        .mark_text(fontSize=14, fontWeight="bold", dx=12, dy=-8)
        .encode(x="x2:Q", y="y2:Q", text="label:N", color=alt.Color("c:N", scale=None))
    )

    # ŷ (always near y since 2 vectors span R² unless degenerate)
    _yhat_arrow = (
        alt.Chart(
            pd.DataFrame({"x": [0], "y": [0], "x2": [_y_hat[0]], "y2": [_y_hat[1]]})
        )
        .mark_line(strokeWidth=3, color="purple")
        .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
    )
    _yhat_pt = (
        alt.Chart(pd.DataFrame({"x": [_y_hat[0]], "y": [_y_hat[1]]}))
        .mark_circle(size=80, color="purple")
        .encode(x="x:Q", y="y:Q")
    )

    # Target y
    _y_pt = (
        alt.Chart(pd.DataFrame({"x": [_y[0]], "y": [_y[1]]}))
        .mark_point(
            size=200, color="goldenrod", shape="cross", filled=True, strokeWidth=3
        )
        .encode(x="x:Q", y="y:Q")
    )

    # Origin
    _origin = (
        alt.Chart(pd.DataFrame({"x": [0], "y": [0]}))
        .mark_circle(size=60, color="black")
        .encode(x="x:Q", y="y:Q")
    )

    _chart = (
        _grid + _pred_arrows + _pred_pts + _pred_labels
        + _yhat_arrow + _yhat_pt + _y_pt + _origin
    ).properties(width=330, height=330, title="Predictor Geometry")

    # Stability indicator
    _stability = (
        "Severely unstable"
        if _cond > 20
        else ("Unstable" if _cond > 5 else "Stable")
    )
    _stability_color = (
        "danger" if _cond > 20 else ("warn" if _cond > 5 else "success")
    )

    # Beta bar chart
    _beta_df = pd.DataFrame(
        {
            "predictor": ["β₁", "β₂"],
            "value": [_beta_ols[0], _beta_ols[1]],
            "c": ["#e41a1c", "#377eb8"],
        }
    )
    _beta_chart = (
        alt.Chart(_beta_df)
        .mark_bar(size=40)
        .encode(
            x=alt.X("predictor:N", title=""),
            y=alt.Y("value:Q", scale=alt.Scale(domain=[-10, 10]), title="Beta value"),
            color=alt.Color("c:N", scale=None),
        )
        .properties(width=120, height=200, title="OLS Betas")
    )

    mo.hstack(
        [
            _chart,
            _beta_chart,
            mo.vstack(
                [
                    mo.md(f"**Angle = {_angle}°**"),
                    mo.md(f"β₁ = {_beta_ols[0]:.2f}, β₂ = {_beta_ols[1]:.2f}"),
                    mo.md(f"ŷ = ({_y_hat[0]:.2f}, {_y_hat[1]:.2f})"),
                    mo.md(f"Residual² = {_ss:.4f}"),
                    mo.md(""),
                    mo.callout(
                        mo.md(f"**Betas: {_stability}**"),
                        kind=_stability_color,
                    ),
                    mo.md(""),
                    mo.md("*Notice: ŷ barely changes*"),
                    mo.md("*but β₁ and β₂ swing wildly*"),
                    mo.md("*as the angle shrinks!*"),
                ],
                align="start",
            ),
        ],
        justify="center",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Why this matters:** When predictors are collinear, many different
            $(\beta_1, \beta_2)$ combinations give essentially the same prediction.
            OLS picks one, but small perturbations in the data would give very
            different betas. This is a problem of **interpretation** (which predictor
            gets credit?) not **prediction** ($\hat{y}$ is fine either way).

            **Rule of thumb:** If betas flip sign or change drastically when you
            add/remove a correlated predictor, you have a collinearity problem.
            """
        ),
        kind="warn",
    )
    return


# ============================================================
# SECTION 6: UNIQUE CONTRIBUTION
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## 6. Unique Contribution: Why Regression ≠ Correlation

        When you have multiple predictors, OLS estimates the effect of each
        predictor **after removing the overlap** with the other predictors.

        This is why regression coefficients differ from simple correlations:

        - **Correlation** of $x_1$ with $y$: ignores $x_2$ entirely
        - **Regression coefficient** $\beta_1$: the effect of $x_1$ *after accounting for* $x_2$

        Geometrically, OLS **projects** each predictor onto the subspace
        orthogonal to the other predictors, then uses only the unique remainder
        to predict $y$. If two predictors share a lot of variance (overlap), each
        one's unique contribution shrinks.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **The Venn diagram analogy:**

            Imagine two overlapping circles (shared variance between $x_1$ and $x_2$):

            - **Total R²** uses all variance from both circles
            - **$\beta_1$** uses only the *non-overlapping* part of $x_1$
            - **$\beta_2$** uses only the *non-overlapping* part of $x_2$
            - The **overlap** can't be uniquely attributed to either predictor

            This is why adding a correlated predictor can change existing betas —
            it reshapes each predictor's unique territory. Predictors **compete**
            for variance in regression.
            """
        ),
        kind="info",
    )
    return


# ============================================================
# SECTION 7: INTERACTIONS
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## 7. Interactions: Adding New Directions

        An interaction term ($x_1 \times x_2$) is **not** just "multiplication." In
        design matrix terms, it's a **new column** — a new predictor vector that
        points in a direction neither $x_1$ nor $x_2$ can reach alone.

        $$X = \begin{bmatrix} | & | & | \\ \mathbf{x}_1 & \mathbf{x}_2 & \mathbf{x}_1 \odot \mathbf{x}_2 \\ | & | & | \end{bmatrix}$$

        where $\odot$ means element-wise multiplication.

        **What this does geometrically:**

        - **Expands the column space** — the model can now express patterns that
          require both predictors to act *together*
        - Enables **context-dependent effects** (moderation): the effect of $x_1$
          depends on the level of $x_2$
        - The interaction vector is genuinely new information — it's not redundant
          with $x_1$ or $x_2$ (unless one of them is constant)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Example:** Does caffeine help studying?

            - $x_1$ = hours studied, $x_2$ = cups of coffee, $x_1 \times x_2$ = interaction
            - Without interaction: caffeine has the same effect regardless of study hours
            - With interaction: caffeine might help *more* when you study longer (or vice versa)
            - Adding the interaction column lets the model capture this — it expands
              the span to include context-dependent patterns
            """
        ),
        kind="info",
    )
    return


# ============================================================
# SECTION 8: CATEGORICAL PREDICTORS
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## 8. Categorical Predictors: Groups as Directions

        A categorical variable with $k$ levels becomes $k - 1$ columns in the
        design matrix. Why $k - 1$ and not $k$?

        Because the **intercept** already consumes one dimension. If you have
        groups A, B, and C, you only need two indicator columns to distinguish
        all three groups — the third is determined by "neither B nor C → must be A."

        Using all $k$ columns would create **perfect collinearity** with the
        intercept (they'd sum to the intercept column), making betas
        unidentifiable — exactly the geometry problem we saw in the
        multicollinearity section.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Think of it this way:**

            - $k$ categories represent a conceptual space of group differences
            - The intercept fixes a "reference point" in that space
            - You need exactly $k - 1$ independent directions to navigate from
              the reference to any other group
            - Those $k - 1$ directions are your dummy/indicator columns
            """
        ),
        kind="info",
    )
    return


# ============================================================
# SECTION 9: CODING SCHEMES
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## 9. Coding Schemes: Same Span, Different Basis

        **Coding** is a choice of *basis* for the categorical subspace. Different
        coding schemes produce different columns in $X$ — but they span the
        **same subspace**. The model fits equally well; only the *interpretation*
        of the betas changes.

        Select a coding scheme below to see how the design matrix changes for
        three groups (A, B, C) with means 3, 5, and 7:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    coding_dropdown = mo.ui.dropdown(
        options=["Dummy (treatment) coding", "Effects (sum-to-zero) coding"],
        value="Dummy (treatment) coding",
        label="Coding scheme",
    )
    return (coding_dropdown,)


@app.cell(hide_code=True)
def _(alt, coding_dropdown, mo, pd):
    _scheme = coding_dropdown.value

    # Group means
    _means = {"A": 3, "B": 5, "C": 7}
    _grand_mean = 5.0

    if _scheme == "Dummy (treatment) coding":
        _design_df = pd.DataFrame(
            {
                "Intercept": [1, 1, 1],
                "D_B": [0, 1, 0],
                "D_C": [0, 0, 1],
            },
            index=["Group A", "Group B", "Group C"],
        )
        _beta_desc = pd.DataFrame(
            {
                "Parameter": ["β₀ (intercept)", "β₁ (D_B)", "β₂ (D_C)"],
                "Value": [3, 2, 4],
                "Meaning": [
                    "Mean of reference group (A)",
                    "B − A  (deviation from reference)",
                    "C − A  (deviation from reference)",
                ],
            }
        )
        _coding_name = "Dummy (Treatment) Coding"
    else:
        _design_df = pd.DataFrame(
            {
                "Intercept": [1, 1, 1],
                "E_1": [-1, 1, 0],
                "E_2": [-1, 0, 1],
            },
            index=["Group A", "Group B", "Group C"],
        )
        _beta_desc = pd.DataFrame(
            {
                "Parameter": ["β₀ (intercept)", "β₁ (E_1)", "β₂ (E_2)"],
                "Value": [5, 0, 2],
                "Meaning": [
                    "Grand mean (average of all groups)",
                    "B − grand mean  (deviation from average)",
                    "C − grand mean  (deviation from average)",
                ],
            }
        )
        _coding_name = "Effects (Sum-to-Zero) Coding"

    # Bar chart of group means with annotation
    _bar_df = pd.DataFrame(
        {
            "group": ["A", "B", "C"],
            "mean": [3, 5, 7],
        }
    )
    _bars = (
        alt.Chart(_bar_df)
        .mark_bar(opacity=0.7, color="steelblue", size=50)
        .encode(
            x=alt.X("group:N", title="Group"),
            y=alt.Y("mean:Q", scale=alt.Scale(domain=[0, 8]), title="Group Mean"),
        )
    )

    # Reference line
    _ref_val = 3.0 if _scheme == "Dummy (treatment) coding" else 5.0
    _ref_label = "β₀ = reference (A)" if _scheme == "Dummy (treatment) coding" else "β₀ = grand mean"
    _ref_line = (
        alt.Chart(pd.DataFrame({"y": [_ref_val]}))
        .mark_rule(strokeWidth=2, color="red", strokeDash=[4, 4])
        .encode(y="y:Q")
    )
    _ref_text = (
        alt.Chart(pd.DataFrame({"y": [_ref_val], "text": [_ref_label]}))
        .mark_text(
            align="left", dx=5, dy=-8, color="red", fontSize=11, fontWeight="bold"
        )
        .encode(y="y:Q", text="text:N")
    )

    _chart = (_bars + _ref_line + _ref_text).properties(
        width=200, height=250, title=_coding_name
    )

    mo.vstack(
        [
            coding_dropdown,
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("**Design Matrix (X):**"),
                            _design_df,
                            mo.md(""),
                            mo.md("**Beta Interpretation:**"),
                            _beta_desc,
                        ],
                        align="start",
                    ),
                    _chart,
                ],
                justify="center",
                gap=3,
            ),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **The punchline:** Different coding schemes produce different columns
            in $X$, but those columns **span the same subspace**. The model's
            predictions ($\hat{y}$) are identical — only what the betas *mean*
            changes.

            - **Dummy coding:** intercept = reference group mean; slopes = deviations from reference
            - **Effects coding:** intercept = grand mean; slopes = deviations from grand mean
            - Same $R^2$, same $\hat{y}$, different interpretation
            """
        ),
        kind="info",
    )
    return


# ============================================================
# SECTION 10: UNIFYING IDEA
# ============================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## 10. The Unifying Idea

        Every concept in this notebook is a variation on one theme:

        > **All modeling choices = choosing directions in the column space of X.**

        | Decision | Geometric consequence |
        |----------|----------------------|
        | Adding a continuous predictor | Adds a new direction to the span |
        | Adding an interaction | Adds a direction that captures joint effects |
        | Categorical with $k$ levels | $k - 1$ new directions |
        | Choosing a coding scheme | Different basis for the same subspace |
        | Multicollinearity | Redundant directions → ambiguous credit |
        | Misspecification | True pattern lies outside the span |
        | OLS estimation | Project $y$ onto the column space of $X$ |

        The design matrix $X$ is **not just bookkeeping** — it is the model. Its
        columns define what the model can see, what it can predict, and how its
        parameters should be interpreted.

        **When in doubt, ask:** *What directions am I giving my model?*
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Key takeaways:**

            1. Each predictor is a **vector**; betas are volume knobs on those vectors
            2. $\hat{y}$ must live in the **span** of the predictor vectors
            3. Adding predictors **expands the span** (model capacity)
            4. Collinear predictors → same span, but **ambiguous credit** assignment
            5. OLS finds the **closest point** in the span to $y$ (projection)
            6. Regression ≠ correlation because predictors **compete** for variance
            7. Interactions and categorical dummies are just **more columns** in $X$
            8. Coding schemes = same span, **different interpretation**
            """
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        :::{.callout-tip title="Your Turn"}
        **Practice problems** to check your understanding:

        1. If you add a predictor that is an exact copy of an existing predictor,
           what happens to the span? What happens to the betas?

        2. You run a regression and β₁ = 0.5. Then you add a new predictor x₃
           that correlates with x₁, and now β₁ = 0.1. Did x₁'s *actual* effect
           change? What happened geometrically?

        3. In dummy coding with 4 groups, how many dummy columns do you need?
           What does the intercept represent?

        4. A colleague says "my predictor was significant in the correlation but
           not in the regression." Explain why this can happen using the geometry
           of unique contributions.
        :::
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ---

        *Built with [marimo](https://marimo.io) and
        [wigglystuff](https://koaning.github.io/wigglystuff/) for*
        *PSYC 201B: Statistical Intuitions for Social Scientists*
        """
    ).style({"text-align": "center", "color": "#666", "font-size": "0.9em"})
    return


if __name__ == "__main__":
    app.run()
