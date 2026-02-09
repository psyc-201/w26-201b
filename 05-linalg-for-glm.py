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

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    from wigglystuff import Matrix, Slider2D

    alt.data_transformers.disable_max_rows()
    return Matrix, Slider2D, alt, mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    title: "Linear Algebra for the GLM"
    author: "Eshin Jolly"
    date: "Feb 2026"
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Linear Algebra for the General Linear Model

    *The language your statistical model speaks*

    This notebook builds **geometric intuition** for the General Linear Model
    (GLM).

    > **The big idea:** Every modeling choice — adding a predictor, creating an
    > interaction, coding a categorical variable — is a choice about the
    > *geometry* of your design matrix. Interpretation follows from geometry.
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Matrix Transformations: What Multiplication Actually Does

    You've seen that betas scale individual vectors. But what does the full
    matrix-vector product $A\mathbf{x}$ do?

    **The punchline:** a matrix transforms *all of space* — every point moves
    to a new location, simultaneously. The columns of $A$ tell you where the
    basis vectors land, and everything else follows by linearity.

    There are four transformations most relevant to the GLM:

    - **Scaling** — stretches or compresses along axes (what betas do to predictors)
    - **Rotation** — preserves distances and angles (change of basis, as in PCA)
    - **Reflection** — flips orientation (determinant goes negative)
    - **Projection** — collapses a dimension (this is what OLS does to $y$!)

    *(Matrices can also *shear* space, but we'll focus on the four most relevant to regression.)*

    **Try it:** Select a transformation and adjust the parameters.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    transform_type = mo.ui.dropdown(
        options=["Scaling", "Rotation", "Reflection", "Projection"],
        value="Rotation",
        label="Transformation",
    )
    transform_angle = mo.ui.slider(
        start=0, stop=360, step=1, value=45, label="Angle (degrees)"
    )
    transform_scale = mo.ui.slider(
        start=0.1, stop=3.0, step=0.05, value=1.5, label="Scale factor"
    )
    mo.hstack(
        [transform_type, transform_angle, transform_scale],
        justify="center",
        gap=2,
    )
    return transform_angle, transform_scale, transform_type


@app.cell(hide_code=True)
def _(alt, mo, np, pd, transform_angle, transform_scale, transform_type):
    _kind = transform_type.value
    _angle_deg = transform_angle.value
    _angle_rad = np.radians(_angle_deg)
    _s = transform_scale.value

    # Build 2×2 transformation matrix
    if _kind == "Scaling":
        _A = np.array([[_s, 0], [0, 1 / _s]])
    elif _kind == "Rotation":
        _c, _sn = np.cos(_angle_rad), np.sin(_angle_rad)
        _A = np.array([[_c, -_sn], [_sn, _c]])
    elif _kind == "Reflection":
        _half = _angle_rad / 2
        _c2, _s2 = np.cos(2 * _half), np.sin(2 * _half)
        _A = np.array([[_c2, _s2], [_s2, -_c2]])
    else:  # Projection
        _d = np.array([np.cos(_angle_rad), np.sin(_angle_rad)])
        _A = np.outer(_d, _d)

    # Unit circle + transform
    _t = np.linspace(0, 2 * np.pi, 80)
    _circle = np.column_stack([np.cos(_t), np.sin(_t)])
    _transformed = _circle @ _A.T

    _circ_df = pd.DataFrame(
        {"x": _circle[:, 0], "y": _circle[:, 1], "order": range(80)}
    )
    _trans_df = pd.DataFrame(
        {"x": _transformed[:, 0], "y": _transformed[:, 1], "order": range(80)}
    )

    # Basis vectors + transform
    _e1, _e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    _Ae1, _Ae2 = _A @ _e1, _A @ _e2

    # Grid
    _gr = np.linspace(-2, 2, 9)
    _gx, _gy = np.meshgrid(_gr, _gr)
    _grid = (
        alt.Chart(pd.DataFrame({"x": _gx.flatten(), "y": _gy.flatten()}))
        .mark_circle(size=15, color="lightgray")
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-2.5, 2.5]), title=""),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[-2.5, 2.5]), title=""),
        )
    )

    # Original unit circle
    _orig_circle = (
        alt.Chart(_circ_df)
        .mark_line(strokeWidth=1.5, opacity=0.3, color="gray")
        .encode(x="x:Q", y="y:Q", order="order:O")
    )

    # Transformed shape
    _trans_shape = (
        alt.Chart(_trans_df)
        .mark_line(strokeWidth=2.5, color="purple")
        .encode(x="x:Q", y="y:Q", order="order:O")
    )

    # Original basis vectors (dashed)
    _orig_basis = (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [0, 0], "y": [0, 0],
                    "x2": [_e1[0], _e2[0]], "y2": [_e1[1], _e2[1]],
                    "c": ["#e41a1c", "#377eb8"],
                }
            )
        )
        .mark_line(strokeWidth=2, strokeDash=[4, 4], opacity=0.5)
        .encode(
            x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q",
            color=alt.Color("c:N", scale=None),
        )
    )

    # Transformed basis vectors (solid, thick)
    _trans_basis = (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [0, 0], "y": [0, 0],
                    "x2": [_Ae1[0], _Ae2[0]], "y2": [_Ae1[1], _Ae2[1]],
                    "c": ["#e41a1c", "#377eb8"],
                }
            )
        )
        .mark_line(strokeWidth=3.5)
        .encode(
            x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q",
            color=alt.Color("c:N", scale=None),
        )
    )
    _trans_pts = (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [_Ae1[0], _Ae2[0]],
                    "y": [_Ae1[1], _Ae2[1]],
                    "c": ["#e41a1c", "#377eb8"],
                }
            )
        )
        .mark_circle(size=50)
        .encode(x="x:Q", y="y:Q", color=alt.Color("c:N", scale=None))
    )

    # Origin
    _origin = (
        alt.Chart(pd.DataFrame({"x": [0], "y": [0]}))
        .mark_circle(size=60, color="black")
        .encode(x="x:Q", y="y:Q")
    )

    _chart = (
        _grid + _orig_circle + _trans_shape
        + _orig_basis + _trans_basis + _trans_pts + _origin
    ).properties(width=350, height=350, title=f"{_kind} Transformation")

    # Determinant and rank
    _det = float(np.linalg.det(_A))
    _rank = int(np.linalg.matrix_rank(_A))

    # Which slider matters
    _slider_note = {
        "Scaling": "**Active slider:** Scale factor",
        "Rotation": "**Active slider:** Angle",
        "Reflection": "**Active slider:** Angle (axis of reflection = angle / 2)",
        "Projection": "**Active slider:** Angle (projection onto this direction)",
    }[_kind]

    # Per-type observation
    _observation = {
        "Scaling": (
            "Circle → **ellipse**. Distances change along axes, "
            "but axis directions are preserved. "
            "This is what different betas do — scale each predictor direction by a different amount."
        ),
        "Rotation": (
            "Circle → **circle**. Distances and angles are preserved "
            "(det = 1). This is a rigid change of basis — "
            "PCA rotates to a new coordinate system without distortion."
        ),
        "Reflection": (
            "Circle → **circle**, but orientation flips (det = -1). "
            "Notice e₁ and e₂ swap their 'handedness.' "
            "Contrast coding flips the sign convention for group comparisons."
        ),
        "Projection": (
            "Circle → **line segment**. A dimension collapses — rank drops to 1. "
            "This is exactly what OLS does: the hat matrix $H$ projects $y$ "
            "onto the column space of $X$."
        ),
    }[_kind]

    _sidebar = mo.vstack(
        [
            mo.md(f"**Matrix A** ({_kind}):"),
            mo.md(
                f"$$A = \\begin{{bmatrix}} {_A[0,0]:.2f} & {_A[0,1]:.2f} \\\\"
                f" {_A[1,0]:.2f} & {_A[1,1]:.2f} \\end{{bmatrix}}$$"
            ),
            mo.md(f"det(A) = {_det:.2f} &nbsp;|&nbsp; rank(A) = {_rank}"),
            mo.md(""),
            mo.md(_slider_note),
            mo.md(""),
            mo.md(f"*{_observation}*"),
            mo.md(""),
            mo.md(
                "Dashed = original basis (e₁, e₂)  \n"
                "Solid = where A sends them  \n"
                "Gray circle = unit circle  \n"
                "Purple = transformed shape"
            ),
        ],
        align="start",
    )

    mo.hstack([_chart, _sidebar], justify="center", gap=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **GLM connections:**

            - **Projection = OLS.** The hat matrix $H = X(X^\top X)^{-1}X^\top$
              projects $y$ onto the column space of $X$ — exactly the projection
              operation above, just in higher dimensions.
            - **Scaling = what betas do.** Different $\beta$ values scale different
              predictor directions by different amounts.
            - **Columns of $A$ = where basis vectors land.** The columns of $X$
              show where "one unit" of each predictor maps to in observation space.
            - *Rotation will return when we discuss PCA — it's a change of basis
              that preserves distances.*
            """
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Span: What the Model Can Possibly Predict

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
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Now let's add a second predictor** and see the span expand from a
    line to a plane. Adjust both $\beta$ sliders to move $\hat{y}$ around
    the full 2D space:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    span2_b1 = mo.ui.slider(
        start=-2, stop=2, step=0.05, value=0.3, label="β₁"
    )
    span2_b2 = mo.ui.slider(
        start=-2, stop=2, step=0.05, value=0.5, label="β₂"
    )
    mo.hstack([span2_b1, span2_b2], justify="center", gap=2)
    return span2_b1, span2_b2


@app.cell(hide_code=True)
def _(alt, mo, np, pd, span2_b1, span2_b2):
    _b1, _b2 = span2_b1.value, span2_b2.value
    _x1 = np.array([1.5, 0.5])
    _x2 = np.array([0.3, 1.4])
    _y = np.array([1.0, 1.2])

    _y_hat = _b1 * _x1 + _b2 * _x2
    _resid = _y - _y_hat
    _ss = float(_resid @ _resid)

    _X = np.column_stack([_x1, _x2])
    _beta_ols = np.linalg.lstsq(_X, _y, rcond=None)[0]
    _y_hat_ols = _X @ _beta_ols
    _ss_ols = float((_y - _y_hat_ols) @ (_y - _y_hat_ols))

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

    _span_bg = (
        alt.Chart(pd.DataFrame({"x": [-2.5], "y": [-2.5], "x2": [2.5], "y2": [2.5]}))
        .mark_rect(opacity=0.04, color="purple")
        .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
    )

    _pred_df = pd.DataFrame(
        {
            "x": [0, 0], "y": [0, 0],
            "x2": [_x1[0], _x2[0]], "y2": [_x1[1], _x2[1]],
            "c": ["#e41a1c", "#377eb8"], "label": ["x₁", "x₂"],
        }
    )
    _pred_arrows = (
        alt.Chart(_pred_df)
        .mark_line(strokeWidth=2, opacity=0.5, strokeDash=[4, 4])
        .encode(
            x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q",
            color=alt.Color("c:N", scale=None),
        )
    )
    _pred_labels = (
        alt.Chart(_pred_df)
        .mark_text(fontSize=14, fontWeight="bold", dx=10, dy=-8)
        .encode(
            x="x2:Q", y="y2:Q", text="label:N",
            color=alt.Color("c:N", scale=None),
        )
    )

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

    _y_pt = (
        alt.Chart(pd.DataFrame({"x": [_y[0]], "y": [_y[1]]}))
        .mark_point(
            size=200, color="goldenrod", shape="cross", filled=True, strokeWidth=3
        )
        .encode(x="x:Q", y="y:Q")
    )

    _resid_line = (
        alt.Chart(
            pd.DataFrame(
                {"x": [_y_hat[0]], "y": [_y_hat[1]], "x2": [_y[0]], "y2": [_y[1]]}
            )
        )
        .mark_line(strokeWidth=2, color="goldenrod", strokeDash=[3, 3])
        .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
    )

    _origin = (
        alt.Chart(pd.DataFrame({"x": [0], "y": [0]}))
        .mark_circle(size=60, color="black")
        .encode(x="x:Q", y="y:Q")
    )

    _chart = (
        _grid + _span_bg + _pred_arrows + _pred_labels + _yhat_arrow
        + _yhat_pt + _y_pt + _resid_line + _origin
    ).properties(width=350, height=350, title="ŷ = β₁x₁ + β₂x₂ — Span = Full Plane")

    _quality = (
        "Perfect fit! Residual = 0"
        if _ss < 0.01
        else ("Close!" if _ss < 0.1 else "Keep adjusting...")
    )

    mo.hstack(
        [
            _chart,
            mo.vstack(
                [
                    mo.md("**With 2 non-collinear predictors:**"),
                    mo.md("Span covers the **entire plane** (faint purple)"),
                    mo.md(""),
                    mo.md(f"β₁ = {_b1:.2f}, β₂ = {_b2:.2f}"),
                    mo.md(f"ŷ = ({_y_hat[0]:.2f}, {_y_hat[1]:.2f})"),
                    mo.md(f"Residual² = {_ss:.3f}"),
                    mo.md(f"*{_quality}*"),
                    mo.md(""),
                    mo.md(f"**OLS:** β₁ = {_beta_ols[0]:.2f}, β₂ = {_beta_ols[1]:.2f}"),
                    mo.md(f"OLS Residual² = {_ss_ols:.4f}"),
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
            **Compare:** With 1 predictor, $\hat{y}$ was stuck on a *line* —
            residual was unavoidable. With 2 non-collinear predictors, $\hat{y}$
            can reach *anywhere* in the plane, so residual goes to zero.

            In higher dimensions the same principle applies: each new independent
            predictor adds a direction, expanding the model's reach. But in real
            data ($n \gg 2$), even many predictors rarely span all of
            $\mathbb{R}^n$.
            """
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. The Design Matrix: What the Model Can See

    The **design matrix** $X$ packages all your predictors as columns:

    $$X = \begin{bmatrix} | & | \\ \mathbf{x}_1 & \mathbf{x}_2 \\ | & | \end{bmatrix}$$

    - **Columns** = directions available to the model (predictor vectors)
    - **Rows** = observations (one row per person/trial)
    - The model's prediction: $\hat{y} = X\beta$ is a linear combination of columns

    Everything the model "knows" comes from these columns. If a pattern in $y$
    can't be expressed as a combination of the columns of $X$, the model simply
    cannot capture it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 6. Multicollinearity: When Predictors Point the Same Way

    **Multicollinearity** happens when two predictors point in nearly the same
    direction. Geometrically, their vectors are almost parallel.

    This creates a problem of **interpretation, not prediction:**

    - $\hat{y}$ is still fine (the span is barely affected)
    - But individual $\beta$ values become **unstable** — many different
      $(\beta_1, \beta_2)$ combinations produce virtually the same $\hat{y}$
    - Betas swing wildly because credit can't be uniquely assigned

    **Try it:** Adjust the angle between predictors. Watch what happens to the
    betas as the predictors become more similar.
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exploring with a Design Matrix

    Now let's see multicollinearity with actual "data." Below is a small
    design matrix — **drag the cell values** to change the predictor columns.
    The outcome y is fixed. Watch how the OLS betas respond as you make
    columns more or less similar:
    """)
    return


@app.cell(hide_code=True)
def _(Matrix, mo):
    design_widget = mo.ui.anywidget(
        Matrix(
            matrix=[
                [1.0, 0.2],
                [0.5, 0.8],
                [0.8, 0.5],
                [0.3, 1.0],
                [0.9, 0.3],
            ],
            min_value=-2,
            max_value=2,
            step=0.1,
        )
    )
    return (design_widget,)


@app.cell(hide_code=True)
def _(alt, design_widget, mo, np, pd):
    _X_raw = np.array(design_widget.matrix)
    _y_dw = np.array([0.8, 1.2, 0.9, 1.5, 0.7])

    _n_dw = _X_raw.shape[0]
    _X_full = np.column_stack([np.ones(_n_dw), _X_raw])

    _beta_dw = np.linalg.lstsq(_X_full, _y_dw, rcond=None)[0]
    _y_hat_dw = _X_full @ _beta_dw
    _resid_dw = _y_dw - _y_hat_dw
    _ss_dw = float(_resid_dw @ _resid_dw)

    _r_dw = float(np.corrcoef(_X_raw[:, 0], _X_raw[:, 1])[0, 1])
    _cond_dw = float(np.linalg.cond(_X_raw))

    _stability_dw = (
        "Severely unstable"
        if _cond_dw > 15
        else ("Unstable" if _cond_dw > 5 else "Stable")
    )
    _stab_kind_dw = (
        "danger" if _cond_dw > 15 else ("warn" if _cond_dw > 5 else "success")
    )

    _beta_bar_df = pd.DataFrame(
        {
            "param": ["β₀", "β₁", "β₂"],
            "value": list(_beta_dw),
            "c": ["gray", "#e41a1c", "#377eb8"],
        }
    )
    _beta_bar = (
        alt.Chart(_beta_bar_df)
        .mark_bar(size=35)
        .encode(
            x=alt.X("param:N", title="", sort=["β₀", "β₁", "β₂"]),
            y=alt.Y("value:Q", scale=alt.Scale(domain=[-8, 8]), title="Value"),
            color=alt.Color("c:N", scale=None),
        )
        .properties(width=140, height=200, title="OLS Betas")
    )

    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md("**Drag to edit the design matrix (x₁, x₂):**"),
                    design_widget,
                    mo.md(
                        f"y = [{', '.join(f'{_v:.1f}' for _v in _y_dw)}]"
                    ),
                ],
                align="center",
            ),
            _beta_bar,
            mo.vstack(
                [
                    mo.md(f"**Predictor correlation:** r = {_r_dw:.2f}"),
                    mo.callout(
                        mo.md(f"**Betas: {_stability_dw}**"),
                        kind=_stab_kind_dw,
                    ),
                    mo.md(f"β₀ = {_beta_dw[0]:.2f}"),
                    mo.md(f"β₁ = {_beta_dw[1]:.2f}"),
                    mo.md(f"β₂ = {_beta_dw[2]:.2f}"),
                    mo.md(f"Residual² = {_ss_dw:.3f}"),
                    mo.md(""),
                    mo.md("*Try making columns identical*"),
                    mo.md("*and watch the betas explode!*"),
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
    mo.md(r"""
    ---

    ## 7. Unique Contribution: Why Regression ≠ Correlation

    When you have multiple predictors, OLS estimates the effect of each
    predictor **after removing the overlap** with the other predictors.

    - **Correlation** of x₁ with y: ignores x₂ entirely
    - **Regression coefficient** β₁: the effect of x₁ *after accounting for* x₂

    Geometrically, OLS **projects** each predictor onto the subspace
    orthogonal to the other predictors, then uses only the unique remainder
    to predict y.

    The visualization below demonstrates this with the
    **Frisch-Waugh-Lovell theorem**: you can recover β₁ by first removing
    x₂ from *both* x₁ and y, then running a simple regression on the
    residuals.

    **Try it:** Increase the predictor correlation and watch the partial
    relationship change:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    pred_corr_slider = mo.ui.slider(
        start=0.0, stop=0.95, step=0.05, value=0.3,
        label="Correlation between x₁ and x₂",
    )
    mo.hstack([pred_corr_slider], justify="center")
    return (pred_corr_slider,)


@app.cell(hide_code=True)
def _(alt, mo, np, pd, pred_corr_slider):
    _r_pc = pred_corr_slider.value
    np.random.seed(42)
    _n_pc = 80

    _x1_pc = np.random.randn(_n_pc)
    _x2_pc = _r_pc * _x1_pc + np.sqrt(max(1 - _r_pc**2, 0.01)) * np.random.randn(_n_pc)

    _y_pc = 0.6 * _x1_pc + 0.4 * _x2_pc + 0.3 * np.random.randn(_n_pc)

    # Simple regression slope
    _slope_simple = float(np.cov(_x1_pc, _y_pc)[0, 1] / np.var(_x1_pc))

    # Multiple regression (partial)
    _X_pc = np.column_stack([np.ones(_n_pc), _x1_pc, _x2_pc])
    _betas_pc = np.linalg.lstsq(_X_pc, _y_pc, rcond=None)[0]
    _slope_partial = float(_betas_pc[1])

    # Frisch-Waugh: residualise x₂ out of x₁ and y
    _Z = np.column_stack([np.ones(_n_pc), _x2_pc])
    _x1_resid = _x1_pc - _Z @ np.linalg.lstsq(_Z, _x1_pc, rcond=None)[0]
    _y_resid = _y_pc - _Z @ np.linalg.lstsq(_Z, _y_pc, rcond=None)[0]

    _simple_df = pd.DataFrame({"x": _x1_pc, "y": _y_pc})
    _partial_df = pd.DataFrame({"x": _x1_resid, "y": _y_resid})

    _simple_scatter = (
        alt.Chart(_simple_df)
        .mark_circle(size=25, opacity=0.5, color="steelblue")
        .encode(
            x=alt.X("x:Q", title="x₁", scale=alt.Scale(domain=[-3.5, 3.5])),
            y=alt.Y("y:Q", title="y", scale=alt.Scale(domain=[-3, 3])),
        )
    )
    _simple_reg = (
        alt.Chart(_simple_df).transform_regression("x", "y")
        .mark_line(color="#e41a1c", strokeWidth=3)
        .encode(x="x:Q", y="y:Q")
    )
    _simple_chart = (_simple_scatter + _simple_reg).properties(
        width=260, height=260, title=f"Simple: slope = {_slope_simple:.2f}"
    )

    _partial_scatter = (
        alt.Chart(_partial_df)
        .mark_circle(size=25, opacity=0.5, color="steelblue")
        .encode(
            x=alt.X(
                "x:Q", title="x₁ (x₂ removed)",
                scale=alt.Scale(domain=[-3.5, 3.5]),
            ),
            y=alt.Y(
                "y:Q", title="y (x₂ removed)",
                scale=alt.Scale(domain=[-3, 3]),
            ),
        )
    )
    _partial_reg = (
        alt.Chart(_partial_df).transform_regression("x", "y")
        .mark_line(color="purple", strokeWidth=3)
        .encode(x="x:Q", y="y:Q")
    )
    _partial_chart = (_partial_scatter + _partial_reg).properties(
        width=260, height=260, title=f"Partial: slope = {_slope_partial:.2f}"
    )

    mo.vstack(
        [
            mo.hstack(
                [_simple_chart, _partial_chart], justify="center", gap=2
            ),
            mo.hstack(
                [
                    mo.md(f"**Predictor correlation:** r(x₁, x₂) = {_r_pc:.2f}"),
                    mo.md(f"**Simple slope:** {_slope_simple:.2f}"),
                    mo.md(f"**Partial slope (β₁):** {_slope_partial:.2f}"),
                ],
                justify="center",
                gap=3,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **What's happening:**

            - **Left plot:** Simple relationship between x₁ and y (ignoring x₂)
            - **Right plot:** Relationship *after removing x₂* from both variables
              (a **partial regression plot** / added-variable plot)

            The partial slope IS the regression coefficient β₁. As the correlation
            between predictors increases:

            - The simple slope gets inflated (x₁ takes credit for x₂'s effect too)
            - The partial slope isolates x₁'s **unique** contribution
            - The right-side scatter gets noisier (less unique signal left in x₁)

            This is why regression ≠ correlation: **predictors compete for variance**.
            """
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 8. Interactions: Context-Dependent Effects

    An interaction term ($x_1 \times x_2$) is a **new column** in the design
    matrix — a new predictor vector that captures how the effect of one
    variable *depends on the level of another*.

    $$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 (x_1 \cdot x_2)$$

    Without the interaction, $x_1$ has the same effect regardless of $x_2$
    (parallel lines). With the interaction, the slope of $x_1$ **changes**
    depending on $x_2$ (non-parallel lines).

    **Try it:** Adjust the interaction strength to see the lines fan out:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    inter_strength = mo.ui.slider(
        start=-1.0, stop=1.0, step=0.05, value=0.0,
        label="Interaction strength (β₃)",
    )
    mo.hstack([inter_strength], justify="center")
    return (inter_strength,)


@app.cell(hide_code=True)
def _(alt, inter_strength, mo, np, pd):
    _b3_is = inter_strength.value
    np.random.seed(42)
    _n_is = 120

    _b0_is, _b1_is, _b2_is = 2.0, 0.8, 0.5

    _x1_is = np.random.randn(_n_is)
    _x2_is = np.random.randn(_n_is)
    _y_is = (
        _b0_is + _b1_is * _x1_is + _b2_is * _x2_is
        + _b3_is * _x1_is * _x2_is + 0.4 * np.random.randn(_n_is)
    )

    _x2_q = np.percentile(_x2_is, [33, 67])
    _x2_grp = np.where(
        _x2_is < _x2_q[0], "Low x₂",
        np.where(_x2_is > _x2_q[1], "High x₂", "Mid x₂"),
    )

    _sc_df = pd.DataFrame({"x1": _x1_is, "y": _y_is, "x2_group": _x2_grp})

    _x1_r = np.linspace(-2.5, 2.5, 50)
    _ld = []
    for _x2v, _lab in [(-1, "x₂ = -1"), (0, "x₂ = 0"), (1, "x₂ = +1")]:
        _yl = _b0_is + _b1_is * _x1_r + _b2_is * _x2v + _b3_is * _x1_r * _x2v
        for _k in range(len(_x1_r)):
            _ld.append({"x1": _x1_r[_k], "y": _yl[_k], "x2_level": _lab})
    _line_df = pd.DataFrame(_ld)

    _scatter = (
        alt.Chart(_sc_df)
        .mark_circle(size=20, opacity=0.3)
        .encode(
            x=alt.X("x1:Q", title="x₁", scale=alt.Scale(domain=[-3, 3])),
            y=alt.Y("y:Q", title="y", scale=alt.Scale(domain=[-3, 7])),
            color=alt.Color(
                "x2_group:N", title="x₂ tercile",
                scale=alt.Scale(
                    domain=["Low x₂", "Mid x₂", "High x₂"],
                    range=["#377eb8", "gray", "#e41a1c"],
                ),
            ),
        )
    )

    _lines = (
        alt.Chart(_line_df)
        .mark_line(strokeWidth=3)
        .encode(
            x="x1:Q", y="y:Q",
            color=alt.Color(
                "x2_level:N", title="Conditional on",
                scale=alt.Scale(
                    domain=["x₂ = -1", "x₂ = 0", "x₂ = +1"],
                    range=["#377eb8", "gray", "#e41a1c"],
                ),
            ),
        )
    )

    _chart = (_scatter + _lines).properties(
        width=400, height=300,
        title=f"Interaction: slope of x₁ varies with x₂  (β₃ = {_b3_is:.2f})",
    )

    mo.vstack(
        [
            _chart,
            mo.hstack(
                [
                    mo.md(
                        f"**Slope of x₁ when x₂ = -1:** "
                        f"{_b1_is + _b3_is * (-1):.2f}"
                    ),
                    mo.md(f"**Slope of x₁ when x₂ = 0:** {_b1_is:.2f}"),
                    mo.md(
                        f"**Slope of x₁ when x₂ = +1:** "
                        f"{_b1_is + _b3_is:.2f}"
                    ),
                ],
                justify="center",
                gap=3,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Geometry of interactions:**

            - Without the interaction ($\beta_3 = 0$): all three lines are
              **parallel** — the effect of $x_1$ is the same at every level of $x_2$
            - With a positive interaction: the lines **fan out** — higher $x_2$
              means steeper slope for $x_1$
            - The interaction term $x_1 \cdot x_2$ is a genuinely new column in
              $X$ — it **expands the column space**, enabling patterns that $x_1$
              and $x_2$ alone cannot express
            """
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why centering matters for interactions

    When predictors are **uncentered**, the product $x_1 \cdot x_2$ tends to
    be correlated with $x_1$ and $x_2$ individually — creating
    multicollinearity.

    **Centering** (subtracting the mean) fixes this: the product of centered
    variables is much less correlated with the main effects.

    More importantly, centering changes **what the betas mean:**

    - **Uncentered:** $\beta_1$ = effect of $x_1$ when $x_2 = 0$ (possibly meaningless!)
    - **Centered:** $\beta_1$ = effect of $x_1$ when $x_2 = \overline{x}_2$ (meaningful)

    Toggle centering below and watch the correlations change:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    center_toggle = mo.ui.radio(
        options=["Uncentered", "Centered (x - mean)"],
        value="Uncentered",
        label="",
    )
    mo.hstack([center_toggle], justify="center")
    return (center_toggle,)


@app.cell(hide_code=True)
def _(alt, center_toggle, mo, np, pd):
    _do_center = center_toggle.value == "Centered (x - mean)"
    np.random.seed(42)
    _n_ct = 150

    _x1_raw_ct = np.random.uniform(2, 8, _n_ct)
    _x2_raw_ct = np.random.uniform(4, 10, _n_ct)

    _x1_ct = _x1_raw_ct - _x1_raw_ct.mean() if _do_center else _x1_raw_ct
    _x2_ct = _x2_raw_ct - _x2_raw_ct.mean() if _do_center else _x2_raw_ct

    _x1x2_ct = _x1_ct * _x2_ct

    _r_x1_int = float(np.corrcoef(_x1_ct, _x1x2_ct)[0, 1])
    _r_x2_int = float(np.corrcoef(_x2_ct, _x1x2_ct)[0, 1])
    _r_x1_x2 = float(np.corrcoef(_x1_ct, _x2_ct)[0, 1])

    _corr_mat = np.array(
        [
            [1.0, _r_x1_x2, _r_x1_int],
            [_r_x1_x2, 1.0, _r_x2_int],
            [_r_x1_int, _r_x2_int, 1.0],
        ]
    )
    _labels = ["x₁", "x₂", "x₁·x₂"]
    _hd = []
    for _i in range(3):
        for _j in range(3):
            _hd.append(
                {"row": _labels[_i], "col": _labels[_j], "r": _corr_mat[_i, _j]}
            )
    _hm_df = pd.DataFrame(_hd)

    _heatmap = (
        alt.Chart(_hm_df)
        .mark_rect()
        .encode(
            x=alt.X("col:N", title="", sort=_labels),
            y=alt.Y("row:N", title="", sort=_labels),
            color=alt.Color(
                "r:Q", title="Correlation",
                scale=alt.Scale(domain=[-1, 1], scheme="redblue"),
            ),
        )
    )
    _hm_text = (
        alt.Chart(_hm_df)
        .mark_text(fontSize=14, fontWeight="bold")
        .encode(
            x=alt.X("col:N", sort=_labels),
            y=alt.Y("row:N", sort=_labels),
            text=alt.Text("r:Q", format=".2f"),
            color=alt.condition(
                alt.datum.r > 0.5, alt.value("white"), alt.value("black")
            ),
        )
    )

    _hm_chart = (_heatmap + _hm_text).properties(
        width=250, height=250,
        title="Centered" if _do_center else "Uncentered",
    )

    _x1_lab = "x₁ - mean" if _do_center else "x₁"
    _x2_lab = "x₂ - mean" if _do_center else "x₂"

    mo.hstack(
        [
            _hm_chart,
            mo.vstack(
                [
                    mo.md("**Correlations between predictors:**"),
                    mo.md(f"r({_x1_lab}, {_x2_lab}) = {_r_x1_x2:.2f}"),
                    mo.md(f"r({_x1_lab}, interaction) = **{_r_x1_int:.2f}**"),
                    mo.md(f"r({_x2_lab}, interaction) = **{_r_x2_int:.2f}**"),
                    mo.md(""),
                    mo.callout(
                        mo.md(
                            "Interaction is nearly **orthogonal** to main effects!"
                            if _do_center
                            else "Interaction is **highly correlated** with main effects!"
                        ),
                        kind="success" if _do_center else "danger",
                    ),
                    mo.md(""),
                    mo.md(
                        "β₁ = effect of x₁ at **mean(x₂)**"
                        if _do_center
                        else "β₁ = effect of x₁ when **x₂ = 0** (extrapolation!)"
                    ),
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
            **Why this matters for the GLM:**

            - Without centering, the interaction column ($x_1 \cdot x_2$) points
              in a similar direction to $x_1$ and $x_2$ → **multicollinearity**
            - Centering makes the interaction column more **orthogonal** to the
              main effects → betas are more stable and interpretable
            - Centering also changes the meaning of $\beta_1$ from "effect when
              $x_2 = 0$" to "effect at the mean of $x_2$" — usually a more
              meaningful baseline
            - **The model fits equally well either way** (same $R^2$) — centering
              only changes what the parameters *mean*
            """
        ),
        kind="warn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 9. Putting It Together: Group × Continuous Interactions

    In psychology, one of the most common models includes a **categorical
    predictor** (treatment vs. control), a **continuous predictor** (e.g.
    age), and their **interaction** (does the treatment effect depend on
    age?).

    $$\hat{y} = \beta_0 + \beta_1 \cdot \text{Group} + \beta_2 \cdot x + \beta_3 \cdot \text{Group} \times x$$

    With Group coded as 0 / 1:

    - **Group 0 line:** $\hat{y} = \beta_0 + \beta_2 x$ &nbsp; (intercept $\beta_0$, slope $\beta_2$)
    - **Group 1 line:** $\hat{y} = (\beta_0 + \beta_1) + (\beta_2 + \beta_3) x$

    Adjust the parameters to explore the geometry:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    cat_intercept = mo.ui.slider(
        start=-2, stop=4, step=0.1, value=2.0,
        label="β₀ (Group 0 intercept)",
    )
    cat_group_eff = mo.ui.slider(
        start=-3, stop=3, step=0.1, value=1.0,
        label="β₁ (Group difference)",
    )
    cat_slope = mo.ui.slider(
        start=-1, stop=2, step=0.05, value=0.5,
        label="β₂ (Slope for Group 0)",
    )
    cat_inter = mo.ui.slider(
        start=-1, stop=1, step=0.05, value=0.0,
        label="β₃ (Interaction: slope difference)",
    )
    mo.vstack(
        [
            mo.hstack([cat_intercept, cat_group_eff], justify="center", gap=2),
            mo.hstack([cat_slope, cat_inter], justify="center", gap=2),
        ]
    )
    return cat_group_eff, cat_inter, cat_intercept, cat_slope


@app.cell(hide_code=True)
def _(alt, cat_group_eff, cat_inter, cat_intercept, cat_slope, mo, np, pd):
    _b0_c = cat_intercept.value
    _b1_c = cat_group_eff.value
    _b2_c = cat_slope.value
    _b3_c = cat_inter.value

    np.random.seed(42)
    _n_per = 40

    _x_g0 = np.random.uniform(1, 9, _n_per)
    _x_g1 = np.random.uniform(1, 9, _n_per)
    _y_g0 = _b0_c + _b2_c * _x_g0 + 0.5 * np.random.randn(_n_per)
    _y_g1 = (
        (_b0_c + _b1_c) + (_b2_c + _b3_c) * _x_g1
        + 0.5 * np.random.randn(_n_per)
    )

    _scat_df = pd.DataFrame(
        {
            "x": np.concatenate([_x_g0, _x_g1]),
            "y": np.concatenate([_y_g0, _y_g1]),
            "Group": ["Group 0"] * _n_per + ["Group 1"] * _n_per,
        }
    )

    _xr_c = np.linspace(0, 10, 50)
    _ld_c = []
    for _xi in _xr_c:
        _ld_c.append({"x": _xi, "y": _b0_c + _b2_c * _xi, "Group": "Group 0"})
        _ld_c.append(
            {
                "x": _xi,
                "y": (_b0_c + _b1_c) + (_b2_c + _b3_c) * _xi,
                "Group": "Group 1",
            }
        )
    _line_df_c = pd.DataFrame(_ld_c)

    _scatter_c = (
        alt.Chart(_scat_df)
        .mark_circle(size=30, opacity=0.4)
        .encode(
            x=alt.X(
                "x:Q", title="Continuous predictor (x)",
                scale=alt.Scale(domain=[0, 10]),
            ),
            y=alt.Y(
                "y:Q", title="Outcome (y)",
                scale=alt.Scale(domain=[-3, 12]),
            ),
            color=alt.Color(
                "Group:N",
                scale=alt.Scale(
                    domain=["Group 0", "Group 1"],
                    range=["#377eb8", "#e41a1c"],
                ),
            ),
        )
    )

    _lines_c = (
        alt.Chart(_line_df_c)
        .mark_line(strokeWidth=3)
        .encode(
            x="x:Q", y="y:Q",
            color=alt.Color(
                "Group:N",
                scale=alt.Scale(
                    domain=["Group 0", "Group 1"],
                    range=["#377eb8", "#e41a1c"],
                ),
            ),
        )
    )

    _int_pts_c = pd.DataFrame(
        {
            "x": [0, 0],
            "y": [_b0_c, _b0_c + _b1_c],
            "Group": ["Group 0", "Group 1"],
        }
    )
    _int_marks_c = (
        alt.Chart(_int_pts_c)
        .mark_point(size=100, filled=True, shape="diamond")
        .encode(
            x="x:Q", y="y:Q",
            color=alt.Color(
                "Group:N",
                scale=alt.Scale(
                    domain=["Group 0", "Group 1"],
                    range=["#377eb8", "#e41a1c"],
                ),
            ),
        )
    )

    _chart_c = (_scatter_c + _lines_c + _int_marks_c).properties(
        width=420, height=300,
        title="Uncentered: β₁ = group difference at x = 0",
    )

    mo.vstack(
        [
            _chart_c,
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("**Group 0 (blue):**"),
                            mo.md(f"Intercept = β₀ = {_b0_c:.1f}"),
                            mo.md(f"Slope = β₂ = {_b2_c:.2f}"),
                        ],
                        align="start",
                    ),
                    mo.vstack(
                        [
                            mo.md("**Group 1 (red):**"),
                            mo.md(f"Intercept = β₀+β₁ = {_b0_c + _b1_c:.1f}"),
                            mo.md(f"Slope = β₂+β₃ = {_b2_c + _b3_c:.2f}"),
                        ],
                        align="start",
                    ),
                    mo.vstack(
                        [
                            mo.md("**β₁ = group diff at x = 0**"),
                            mo.md(f"= {_b1_c:.1f}"),
                            mo.md("**β₃ = difference in slopes**"),
                            mo.md(f"= {_b3_c:.2f}"),
                        ],
                        align="start",
                    ),
                ],
                justify="center",
                gap=3,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Notice:** $\beta_1$ represents the group difference **at x = 0**.
            But if $x$ is something like "age" or "hours studied," $x = 0$ might
            be meaningless or impossible! The group difference $\beta_1$ is
            *numerically* correct but potentially **uninterpretable** because it
            refers to an impossible baseline.
            """
        ),
        kind="warn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The centering fix

    If we **center** the continuous predictor (subtract its mean), $\beta_1$
    becomes the group difference **at the mean of x** — a much more
    meaningful comparison. The model fit is identical; only the
    interpretation changes.

    Compare the two parameterizations below:
    """)
    return


@app.cell(hide_code=True)
def _(alt, mo, np, pd):
    np.random.seed(42)
    _n_per2 = 50
    _b0_f, _b1_f, _b2_f, _b3_f = 2.0, 1.5, 0.5, 0.3

    _x_raw_f = np.random.uniform(2, 8, _n_per2 * 2)
    _group_f = np.array([0] * _n_per2 + [1] * _n_per2)
    _y_f = (
        _b0_f + _b1_f * _group_f + _b2_f * _x_raw_f
        + _b3_f * _group_f * _x_raw_f
        + 0.5 * np.random.randn(_n_per2 * 2)
    )
    _x_mean_f = float(_x_raw_f.mean())
    _x_cen_f = _x_raw_f - _x_mean_f

    _X_raw_f = np.column_stack(
        [np.ones(_n_per2 * 2), _group_f, _x_raw_f, _group_f * _x_raw_f]
    )
    _X_cen_f = np.column_stack(
        [np.ones(_n_per2 * 2), _group_f, _x_cen_f, _group_f * _x_cen_f]
    )
    _betas_raw_f = np.linalg.lstsq(_X_raw_f, _y_f, rcond=None)[0]
    _betas_cen_f = np.linalg.lstsq(_X_cen_f, _y_f, rcond=None)[0]

    _scat_df2 = pd.DataFrame(
        {
            "x": _x_raw_f, "y": _y_f,
            "Group": [
                "Group 0" if _g == 0 else "Group 1" for _g in _group_f
            ],
        }
    )

    _xr_f = np.linspace(1, 9, 50)
    _raw_ld = []
    for _xi in _xr_f:
        _raw_ld.append(
            {
                "x": _xi,
                "y": _betas_raw_f[0] + _betas_raw_f[2] * _xi,
                "Group": "Group 0",
            }
        )
        _raw_ld.append(
            {
                "x": _xi,
                "y": (
                    _betas_raw_f[0] + _betas_raw_f[1]
                    + (_betas_raw_f[2] + _betas_raw_f[3]) * _xi
                ),
                "Group": "Group 1",
            }
        )
    _raw_line_df = pd.DataFrame(_raw_ld)

    _y0_g0 = float(_betas_raw_f[0])
    _y0_g1 = float(_betas_raw_f[0] + _betas_raw_f[1])
    _ym_g0 = float(_betas_raw_f[0] + _betas_raw_f[2] * _x_mean_f)
    _ym_g1 = float(
        _betas_raw_f[0] + _betas_raw_f[1]
        + (_betas_raw_f[2] + _betas_raw_f[3]) * _x_mean_f
    )

    _gc_scale = alt.Scale(
        domain=["Group 0", "Group 1"], range=["#377eb8", "#e41a1c"]
    )

    _scatter_f = (
        alt.Chart(_scat_df2)
        .mark_circle(size=25, opacity=0.35)
        .encode(
            x=alt.X(
                "x:Q", title="x (continuous)",
                scale=alt.Scale(domain=[0, 10]),
            ),
            y=alt.Y("y:Q", title="y", scale=alt.Scale(domain=[-1, 10])),
            color=alt.Color("Group:N", scale=_gc_scale),
        )
    )

    _line_f = (
        alt.Chart(_raw_line_df)
        .mark_line(strokeWidth=3)
        .encode(x="x:Q", y="y:Q", color=alt.Color("Group:N", scale=_gc_scale))
    )

    _vline_0 = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(strokeWidth=2, strokeDash=[4, 4], color="gray")
        .encode(x="x:Q")
    )
    _vline_m = (
        alt.Chart(pd.DataFrame({"x": [_x_mean_f]}))
        .mark_rule(strokeWidth=2, strokeDash=[4, 4], color="green")
        .encode(x="x:Q")
    )

    _lbl_0 = (
        alt.Chart(pd.DataFrame({"x": [0.3], "y": [9.5], "t": ["x = 0"]}))
        .mark_text(color="gray", fontSize=11, fontWeight="bold", align="left")
        .encode(x="x:Q", y="y:Q", text="t:N")
    )
    _lbl_m = (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [_x_mean_f + 0.2], "y": [9.5],
                    "t": [f"x = mean ({_x_mean_f:.1f})"],
                }
            )
        )
        .mark_text(color="green", fontSize=11, fontWeight="bold", align="left")
        .encode(x="x:Q", y="y:Q", text="t:N")
    )

    _diff0 = (
        alt.Chart(
            pd.DataFrame({"x": [0.0], "y": [_y0_g0], "x2": [0.0], "y2": [_y0_g1]})
        )
        .mark_line(strokeWidth=3, color="gray")
        .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
    )
    _diffm = (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": [_x_mean_f], "y": [_ym_g0],
                    "x2": [_x_mean_f], "y2": [_ym_g1],
                }
            )
        )
        .mark_line(strokeWidth=3, color="green")
        .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
    )

    _chart_f = (
        _scatter_f + _line_f + _vline_0 + _vline_m
        + _lbl_0 + _lbl_m + _diff0 + _diffm
    ).properties(
        width=450, height=320, title="Same model, different parameterizations"
    )

    mo.vstack(
        [
            _chart_f,
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("**Uncentered model:**"),
                            mo.md(f"β₁ = {_betas_raw_f[1]:.2f}"),
                            mo.md("= group diff at x = 0 (gray)"),
                            mo.md(f"β₂ = {_betas_raw_f[2]:.2f}"),
                            mo.md("= slope for Group 0"),
                        ],
                        align="start",
                    ),
                    mo.vstack(
                        [
                            mo.md("**Centered model:**"),
                            mo.md(f"β₁ = {_betas_cen_f[1]:.2f}"),
                            mo.md(
                                f"= group diff at mean(x) (green)"
                            ),
                            mo.md(f"β₂ = {_betas_cen_f[2]:.2f}"),
                            mo.md("= slope for Group 0"),
                        ],
                        align="start",
                    ),
                ],
                justify="center",
                gap=3,
            ),
            mo.md(
                "*Both models produce identical predictions and R². "
                "Same lines, same fit — just a different reference point for β₁.*"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **The unifying insight for categorical models:**

            - A categorical variable with $k$ levels → $k - 1$ dummy columns in $X$
            - The **intercept absorbs one level** (the reference group)
            - With interactions, **centering the continuous predictor** makes the
              main effect of group ($\beta_1$) refer to a sensible baseline
            - **Different coding / centering → same span, same fit, different interpretation**
            """
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 10. The Unifying Idea

    Every concept in this notebook is a variation on one theme:

    > **All modeling choices = choosing directions in the column space of X.**

    | Decision | Geometric consequence |
    |----------|----------------------|
    | Adding a continuous predictor | Adds a new direction to the span |
    | Matrix transformation | Defines how coefficient space maps to observation space |
    | Adding an interaction | Adds a direction that captures joint effects |
    | Categorical with $k$ levels | $k - 1$ new directions |
    | Centering a predictor | Same span, different beta interpretation |
    | Multicollinearity | Redundant directions → ambiguous credit |
    | Misspecification | True pattern lies outside the span |
    | OLS estimation | Project $y$ onto the column space of $X$ |

    The design matrix $X$ is **not just bookkeeping** — it is the model. Its
    columns define what the model can see, what it can predict, and how its
    parameters should be interpreted.

    **When in doubt, ask:** *What directions am I giving my model?*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Key takeaways:**

            1. Each predictor is a **vector**; betas are volume knobs on those vectors
            2. A matrix-vector product is a **transformation** of space — OLS is a projection
            3. $\hat{y}$ must live in the **span** of the predictor vectors
            4. Adding predictors **expands the span** (model capacity)
            5. Collinear predictors → same span, but **ambiguous credit** assignment
            6. OLS finds the **closest point** in the span to $y$ (projection)
            7. Regression ≠ correlation because predictors **compete** for variance
            8. Interactions add new columns — **centering** keeps them interpretable
            9. Categorical dummies are just more columns; coding = choice of basis
            """
        ),
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
