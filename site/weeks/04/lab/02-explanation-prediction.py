import marimo

__generated_with = "0.19.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    title: "Permutation & Cross-Validation"
    author: "Eshin Jolly"
    date: "Jan 27, 2026"
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the previous notebook we learned two resampling approaches from **Statistics for Hackers**:

    1. **Monte Carlo Simulation** — generate synthetic data from assumed distributions
    2. **Bootstrap** — resample from actual data with replacement

    Both help us understand the **sampling distribution** — how our estimates bounce around.

    In this notebook, we'll complete the toolkit with two more approaches:

    1. **Permutation** — shuffle data to test whether an effect is real
    2. **Cross-Validation** — split data to test whether an estimator generalizes

    :::{.callout-tip title="Explorable Interactives"}
    This notebook includes **interactive widgets** you can play with to build intuitions *before* writing any code. Drag sliders, change parameters, and watch what happens!
    :::
    """)
    return


@app.cell
def _():
    # Imports
    import marimo as mo
    import polars as pl
    from polars import col
    import seaborn as sns
    import numpy as np
    return col, mo, np, pl, sns


@app.cell
def _(pl, sns):
    # Load the dataset
    penguins = pl.DataFrame(sns.load_dataset("penguins")).drop_nulls()
    penguins.head()
    return (penguins,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (1) Permutation: Resampling *without* replacement

    We've seen that the bootstrap can tell us how **uncertain** our estimate is (e.g., "the mean flipper length is 200 ± 3 mm").

    But what about a different question: **"Could this pattern arise purely by chance?"**

    For example, Gentoo and Adelie penguins seem to have different flipper lengths. But maybe that difference is just random noise — maybe species has *nothing to do with* flipper length and we just got unlucky with our sample.

    **Permutation testing** (also called *shuffling* or *randomization*) answers this by simulating a world where **there is no effect**. We do this by **shuffling the group labels** — breaking the connection between species and flipper length — and seeing how big of a difference we'd get by chance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="text-align:center">
      <img src="./figs/permutation.jpg" width="60%" alt="Permutation illustration">
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The logic is:

    1. **Calculate** the observed statistic (e.g., difference in means between groups)
    2. **Shuffle** the group labels (break the relationship)
    3. **Recalculate** the statistic on the shuffled data
    4. **Repeat** many times to build a **null distribution** — what differences look like when there's no real effect
    5. **Compare** the observed statistic to the null distribution

    If our observed difference is in the **tails** of the null distribution, it's unlikely to have arisen by chance.

    Try it yourself — adjust the effect size and click **Shuffle** to build a null distribution:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    from helpers import permutation_explorer

    _perm_widget = mo.ui.anywidget(permutation_explorer())
    _perm_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :::{.callout-note title="What just happened?"}
    Each shuffle **randomly reassigns** the group labels, then computes the difference in means. This simulates what differences would look like if group membership didn't matter.

    Key things to notice:
    - The null distribution is **centered at zero** — when labels are random, there's no systematic difference
    - The **observed difference** (red line) is far in the tail when the effect is large
    - The **p-value** is the fraction of shuffled differences as extreme as (or more extreme than) what we observed
    - Try setting **effect size to 0** — now the observed difference *blends into* the null distribution
    :::
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Now let's build this in code with real penguins data

    We'll test whether **Gentoo** and **Adelie** penguins have different flipper lengths. First, let's see the data:
    """)
    return


@app.cell
def _(col, penguins):
    agg_means = penguins.filter(
        col('species').is_in(['Adelie', 'Gentoo'])
    ).group_by('species', maintain_order=True).agg(
        col('flipper_length_mm').mean().alias('mean_flength')
    )

    agg_means
    return (agg_means,)


@app.cell
def _(agg_means, col):
    # .diff() computes the difference between the current row and the previous, and .last() gets the last row since there is no first row
    # in this case: Gentoo - Adelie

    observed_diff = agg_means.select(
        col('mean_flength').diff().last().alias('mean_difference')
        ).item()

    observed_diff
    return (observed_diff,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That's a big difference! But let's build a distribution of differences we might expect to see if *species* wasn't related to *flipper length*, aka a **null distribution.**

    First, let's see what **one** shuffle looks like. We can use the same approach as bootstrapping with `polars` `.sample()` method.
    But this time we can use it within an *expression*, i.e. chaining it lik `col('column').sample()`
    """)
    return


@app.cell
def _(col, penguins):
    # Filter down two the 2 species
    one_shuffle = penguins.filter(
        col('species').is_in(['Adelie', 'Gentoo'])
    ).select(
        col('species'),
        # .sample() just this column without replacement and with shuffling, then rename it
        col('flipper_length_mm').sample(fraction=1, shuffle=True).alias('flipper_length_shuffled')
    )

    one_shuffle.head()
    return (one_shuffle,)


@app.cell
def _(col, one_shuffle):
    # Get the means after shuffling
    agg_means_one_shuffle = one_shuffle.group_by('species',maintain_order=True).agg(
        col('flipper_length_shuffled').mean().alias('mean_flength_shuffled')
        )

    agg_means_one_shuffle
    return (agg_means_one_shuffle,)


@app.cell
def _(agg_means_one_shuffle, col):
    # And their difference
    agg_means_one_shuffle.select(
        col('mean_flength_shuffled').diff().last()
        ).item()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice how the shuffled difference isn't the same (and typically much smaller) than the observed difference — because the shuffled groups are **random** mixtures of both species.

    The key intuition is that by shuffling, we've **broken the relationship** between `species` and `flipper_length_mm` in the data. Repeating, this *deliberate* shuffling is what allows us to create the the **null distribution**:
    """)
    return


@app.cell
def _(col, penguins, pl):
    # Number of permutations
    nperm = 1000

    # Store results
    perm_diffs = []

    # If you can loop you can do stats!
    for _ in range(nperm):

        # Created shuffled dataframe
        this_shuffle = penguins.filter(col("species").is_in(["Adelie", "Gentoo"])).select(
            col("species"),
            col("flipper_length_mm").sample(fraction=1, shuffle=True).alias("flipper_length_shuffled"),
        )

        # Calculate mean diff
        shuffled_diff = (
            this_shuffle.group_by("species", maintain_order=True)
            .agg(col("flipper_length_shuffled").mean())
            .select(col("flipper_length_shuffled").diff().last()).item()
        )

        # Save it
        perm_diffs.append(shuffled_diff)

    # Convert to DataFrame
    perm_df = pl.DataFrame(
        {
            "permutation": range(nperm),
            "diff": perm_diffs,
        }
    )

    perm_df.head()
    return nperm, perm_df


@app.cell
def _(mo):
    mo.md(r"""
    Now we can calculate a "p-value." Let's think through the logic here:
    - We're interested in how likely we are to see our *estimated* mean difference if *species* didn't matter (randomized)
    - We can test this by seeing how often our *shuffled* mean differences match or exceed our *estimated* mean difference

    If we don't care about a *directional* difference (i.e. which particlar species' mean is greater) we can take the **absolute value.**

    This is commonly called a "two-tailed" test/p-value:
    """)
    return


@app.cell
def _(col, np, nperm, observed_diff, perm_df):
    # Filter dataframe for rows where the permuted mean difference is >= the observed
    # Then calculate the proportion relative to the number of permutations we ran
    # This *is* our p-value!

    perm_df.filter(
        col('diff').abs() >= np.abs(observed_diff)
    ).height / nperm
    return


@app.cell
def _(mo):
    mo.md(r"""
    :::{.callout-tip title="Note: Important! Adjusted formula"}
    In practice we use a slightly adjusted formula for calculating a permuted p-value called the **Phipson-Smyth correction** based on [this paper](https://pubmed.ncbi.nlm.nih.gov/21044043/).

    We add 1 to the numerator and denominator of the proportion formula for 2 reasons:

    1. To avoid getting p-values that = 0 just like above
    2. To make the proportion we calculate *sensitive* to the number of permutations we run
    :::
    """)
    return


@app.cell
def _(col, np, nperm, observed_diff, perm_df):
    # Note: I've just broken up the single line in the previous cell to make things clearer

    # Number of permuted means >= observed + 1
    numerator = perm_df.filter(
        col('diff').abs() >= np.abs(observed_diff)
    ).height + 1

    # Total number of permutation + 1
    denominator = nperm + 1

    proportion = numerator / denominator
    proportion
    return (numerator,)


@app.cell
def _(mo):
    mo.md(r"""
    Notice that had we run more permutations the p-value would change even if the permuted distribution isn't different:
    """)
    return


@app.cell
def _(numerator):
    numerator / (10000 + 1)
    return


@app.cell
def _(mo):
    mo.md(r"""
    In this way this the **number of resamples** of a permutation test is directly tied to the **precision** of the p-value
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's visualize the null distribution and see where our observed difference falls:
    """)
    return


@app.cell
def _(col, np, nperm, observed_diff, perm_df, sns):
    # Calculate p-value: fraction of permuted differences as extreme as observed
    perm_mean = perm_df.select('diff').mean().item()
    perm_df.filter(
        col('diff').abs() > np.abs(observed_diff)
    )

    # Plot the null distribution
    _grid = sns.displot(
        data=perm_df.to_pandas(),
        x="diff",
        kind="hist",
        bins=30,
        aspect=2,
    )
    _grid.refline(x=observed_diff, color="red", ls="--", lw=2)
    _grid.set(
        title=f'Permuted Null Distribution (n={nperm})\nMean Diff (estimated): {observed_diff:.3f}\nMean Diff (perm): {perm_mean:.3f}',
        xlabel="Difference in Means (mm)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :::{.callout-note title="Key Insight: What's a p-value?"}
    The p-value answers: *"If there were truly no difference between these groups, how often would I see a difference at least this extreme?"*

    In this case **none** of our permuted differences were even close to the observed difference.

    Remember a p-value, whether calculated via permutation like we just did or via analytic formula (e.g. R/Python default) **does not** tell you any thing about the properties of your **estimator!**

    It just shows you how the **same estimator** would behave under randomization. How large or small the **magnitude** of your estimator is (i.e. the mean difference ) is independent of this randomization. A "tiny effect" can be extremely robust to randomization, i.e. have a "small p-value" and visa-versa.

    We'll discuss this more when we get to power and sensitivity.
    :::
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :::{.callout-tip title="Your Turn"}
    Now test whether **Adelie** and **Chinstrap** penguins have different flipper lengths.

    1. Extract Chinstrap flipper lengths (just like we did for Gentoo)
    2. Calculate the observed difference
    3. Build a null distribution by shuffling
    4. Compute the p-value

    *Hint: You can copy and adapt the code from above (or challenge yourself to write a function!) — just change which species you filter, and remember to use new variable names*
    :::
    """)
    return


@app.cell
def _():
    # Your code here: 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :::{.callout-tip title="Your Turn: One-tailed p-value"}
    Above we mentioned you can take the absolute value of your observed and permuted estimators to calculate a two-tailed test/p-value.

    Can you adapt this to test a **directional** hypothesis instead?
    Try adjusting the calculation, visualizing and seeing what happens
    :::
    """)
    return


@app.cell
def _():
    # Your code here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (2) Cross-Validation: How well does our estimator generalize?

    We've now seen three resampling methods:
    - **Simulation** → "What would we expect under these assumptions?"
    - **Bootstrap** → "How uncertain is our estimate?"
    - **Permutation** → "Could this pattern arise by chance?"

    But there's one more question we haven't addressed: **"How well does our estimator work on *new, unseen* data?"**

    Remember from the previous notebook: the **mean** minimizes SSE (sum of squared errors) and the **median** minimizes SAE. Both are good estimators on the data we *observed*.

    But how well do they **generalize** — how good are they at predicting observations we *haven't seen yet*?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="text-align:center">
      <img src="./figs/crossval.png" width="60%" alt="Cross-validation illustration">
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Cross-validation** simulates this generalization using following steps:

    1. **Split** the data into **training** and a **testing** **folds**
    2. **Fit** the estimator on the training fold (e.g., calculate the mean)
    3. **Evaluate** the estimator on the testing fold — how wrong are the predictions?
    4. **Repeat** with different random folds to get a stable estimate of generalization error

    In the image above we're seeing one type of approach for step 4 called $k$ fold CV.
    This just means we split the data into $k$ folds instead of just 2. A common $k$ is 5 or 10, which means we fit the estimator on 4/5ths of our data and test it on 1/5th, then we rotate (we'll explore this more below).

    Cross-validation is what allows us to **decompose** our prediction error into **bias** and **variance**. Remember the classic equation:

    **Error = Bias² + Variance + Irreducible Error**

    Play around with the widget below to see how model complexity affects this decomposition:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    from helpers import cv_explorer

    _cv_widget = mo.ui.anywidget(cv_explorer())
    _cv_widget
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :::{.callout-note title="What just happened?"}
    This is the classic **bias-variance tradeoff**. We fit polynomials of increasing degree to simulated data (y = sin(2πx) + noise) and decompose the test error.

    Key things to notice:
    - **Left panel**: The U-shaped curve shows total MSE (green). Too simple (low degree) = high bias (underfitting). Too complex (high degree) = high variance (overfitting).
    - **Bias²** (blue) decreases as complexity increases — more flexible models can capture the true pattern
    - **Variance** (orange) increases as complexity increases — more flexible models are more sensitive to the specific training data
    - The **optimal model** (red star) balances bias and variance — it's not the simplest or most complex
    - **Right panel**: Shows example fits at low (d=1), optimal, and high complexity — notice how overfitting wiggles through the noise
    - Try increasing **noise** — the optimal degree shifts left (simpler models win when signal is weak)
    - Try increasing **sample size** — variance shrinks and you can afford more complexity
    :::
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Now let's build this in code with real penguins data

    We'll compare how well the **mean** and **median** of flipper length predict held-out penguins. First, let's see what **one** train/test split looks like:
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    First, depending on what *type* of outcome variable you have (e.g. continuous or categorical) there are different ways we can evaluate an estimator. A common approach for continuous data is **RMSE** (Root Mean Squared Error):

    $$\text{RMSE} = \sqrt{\frac{1}{n_{\text{test}}} \sum_{i=1}^{n_{\text{test}}} (x_i - \hat\theta)^2}$$

    where $x_i$ are the test observations and $\hat\theta$ is the estimate from the training data.

    Let's start simple and just do a single CV split (skip step 4 above).
    We'll use 80% of the data to fit our estimator, and then evaluate it on the left-out 20%
    """)
    return


@app.cell
def _(np, penguins):
    # Choose the train/test ratio
    train_pct = 80  # 80% train, 20% test
    n_total = penguins.height
    n_train = int(train_pct / 100 * n_total)
    n_test = n_total - n_train

    print(f"Total penguins: {n_total}")
    print(f"Training set: {n_train} ({train_pct}%)")
    print(f"Test set: {n_test} ({100 - train_pct}%)")

    # Shuffle and split
    # Note: shuffling here is just shuffling the order of ALL rows
    # it's not breaking the relationships between variables like permutation!
    # It just ensure's we don't always use the first 80% of rows to train data
    # and pick a different 80% each time
    shuffled_penguins = penguins.sample(fraction=1.0, shuffle=True)

    # Quickly get train/test splits
    train_data = shuffled_penguins.head(n_train)
    test_data = shuffled_penguins.tail(n_test)

    # Fit estimators on training data
    train_mean = train_data.select("flipper_length_mm").mean().item()
    train_values = train_data.select("flipper_length_mm").to_series().to_numpy()
    train_rmse = np.sqrt(np.mean((train_values - train_mean) ** 2))

    test_values = test_data.select("flipper_length_mm").to_series().to_numpy()
    test_rmse = np.sqrt(np.mean((test_values - train_mean) ** 2))
    # train_median = train_data.select("flipper_length_mm").median().item()

    print(f"\nTraining mean:   {train_mean:.3f} mm")
    print(f"Train RMSE:   {train_rmse:.3f} mm")
    print(f"Test RMSE:   {test_rmse:.3f} mm")
    return n_test, n_train, train_data, train_mean


@app.cell
def _(col, train_data, train_mean):
    train_data.select(
        mse = (col('flipper_length_mm') - train_mean).pow(2).mean()
    ).select(col('mse').sqrt()).item()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Try rerunning the cell above a few times and notice how the mean, train RMSE, and test RMSE keep changing based on what random 80% we use for training
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That's one split. But the result depends on *which* penguins ended up in train vs test! Let's repeat this many times to get a stable comparison:
    """)
    return


@app.cell
def _(col, n_test, n_train, penguins, pl):
    # Number of random splits
    nsplits = 500

    # Store results
    cv_results = []

    # If you can loop you can do stats!
    for i in range(nsplits):

        # Shuffle and split
        _shuffled = penguins.sample(fraction=1.0, shuffle=True)
        _train = _shuffled.head(n_train)
        _test = _shuffled.tail(n_test)

        # Get test values as numpy array
        _train_estimator = _train.select("flipper_length_mm").mean().item()

        _train_rmse = _train.select(
            mse = (col('flipper_length_mm') - _train_estimator).pow(2).mean()
        ).select(col('mse').sqrt()).item()

        _test_rmse = _test.select(
            mse = (col('flipper_length_mm') - _train_estimator).pow(2).mean()
        ).select(col('mse').sqrt()).item()

        cv_results.append({
            "split": i,
            "train_rmse": _train_rmse,
            "test_rmse": _test_rmse,
        })

    # Convert to DataFrame
    cv_df = pl.DataFrame(cv_results)
    cv_df.head()
    return (cv_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's visualize the comparison. First we need a **tidy dataframe**.
    """)
    return


@app.cell
def _(cv_df):
    # Extra helpers
    import polars.selectors as cs

    cv_df_tidy = cv_df.unpivot(
        on=cs.ends_with("rmse"),
        index="split",
        value_name="rmse",
        variable_name="fold"
    )

    cv_df_tidy.head()
    return (cv_df_tidy,)


@app.cell
def _(mo):
    mo.md(r"""
    Let's take a quick look at the average error and the variance:
    """)
    return


@app.cell
def _(col, cv_df_tidy):
    cv_df_tidy.select('fold','rmse').group_by('fold').agg(col('rmse').mean())
    return


@app.cell
def _(col, cv_df_tidy):
    cv_df_tidy.select('fold','rmse').group_by('fold').agg(col('rmse').std())
    return


@app.cell
def _(mo):
    mo.md(r"""
    Let's visualize this using a histogram colored by training and testing splits:
    """)
    return


@app.cell
def _(cv_df_tidy, sns):
    sns.displot(
        data=cv_df_tidy.to_pandas(),
        x='rmse',
        hue='fold',
        kind='hist',
        aspect=2
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :::{.callout-note title="Interpreting the results"}

    Notice that our **training** errors are lower and more consistent (lower mean and smaller standard-deviation), while our **testing** errors are higher and more variable.

    This is the bias-variance trade-off in action:
    - a model can have different patterns of errors in *training* and *testing*
    - the trade-off between them tells you how much a model is *overfitting* or *underfitting* your data

    While we're using a simple model here (the mean), this general approach can be use with **any type of model** (e.g. regression, neural network etc)
    :::
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :::{.callout-tip title="Your Turn"}
    Try modifying the cross-validation code to use the **median** as an estimator instead. How does its RMSE compare to the mean for training and testing?
    :::
    """)
    return


@app.cell
def _():
    # Your code here
    return


@app.cell
def _(mo):
    mo.md(r"""
    :::{.callout-tip title="Your Turn"}
    Why might RMSE not be the right metric to evaluate the **median**?
    *Hint: think about the loss-function*

    Can you evaluate the median differently? Try it out and see what happens
    :::
    """)
    return


@app.cell
def _():
    # Your code here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Splitting strategies with scikit-learn

    We've been writing our own train/test splits by hand, which is great for understanding the mechanics. But in practice, the [`scikit-learn`](https://scikit-learn.org/stable/modules/cross_validation.html) library provides well-tested tools for this.

    We won't dive deep into `scikit-learn` here — it's a massive library for machine learning - for now, let's learn just three useful functions for splitting data for cross-validation:

    | Function | What it does |
    |----------|-------------|
    | [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split) | One random split into train + test |
    | [`KFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold) | Split into *k* non-overlapping folds, rotating which fold is the test set |
    | [`LeaveOneOut`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html#sklearn.model_selection.LeaveOneOut) | Extreme case: each observation takes a turn as the lone test point |

    There are many more [splitters](https://scikit-learn.org/stable/api/sklearn.model_selection.html) available especially when working with more complicated data (e.g. repeated observations, multiple levels, etc). We encourage you to check them out to see what's most appropriate for the kind of data you typically deal with.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **`train_test_split`** — the one-liner version of what we did by hand:
    """)
    return


@app.cell
def _():
    # Import the function
    from sklearn.model_selection import train_test_split
    return (train_test_split,)


@app.cell
def _(mo):
    mo.md(r"""
    If you examine the function help or online docs, you'll notice it needs an *array*. This is a data-type we haven't really worked with, but don't worry `polars` has our back!

    Just use the `.to_numpy()` method to convert any column to an array that the `sklearn` library understands
    """)
    return


@app.cell
def _(penguins):
    # Get flipper lengths as 1-dimensional arrayfj
    flippers = penguins.select("flipper_length_mm").to_numpy().flatten()
    flippers
    return (flippers,)


@app.cell
def _(flippers, np, train_test_split):
    # One random 80/20 split (same as our hand-written version!)
    _train, _test = train_test_split(flippers, test_size=0.2)

    _train_mean = np.mean(_train)
    _train_rmse = np.sqrt(np.mean((_train - _train_mean) ** 2))
    _test_rmse = np.sqrt(np.mean((_test - _train_mean) ** 2))

    print(f"Train: {len(_train)} observations")
    print(f"Test:  {len(_test)} observations")
    print(f"Train RMSE: {_train_rmse:.3f} mm")
    print(f"Test RMSE:  {_test_rmse:.3f} mm")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **`KFold`** — instead of one random split, divide the data into *k* non-overlapping chunks ("folds"). Each fold takes a turn as the test set while the rest is used for training. This is more efficient because **every observation gets tested exactly once**:
    """)
    return


@app.cell
def _(flippers, np):
    from sklearn.model_selection import KFold

    # KFold allows us to create a custom function that splits the data for us
    my_kfold_splitter = KFold(n_splits=5, shuffle=True)

    # Make splits
    splits = my_kfold_splitter.split(flippers)

    fold_rmses = []

    # enumerate() gives us the current number of the loop we're on 
    # which I'm calling fold_num
    for fold_num, (train_indices, test_indices) in enumerate(splits):
        _train = flippers[train_indices]
        _test = flippers[test_indices]

        _pred = np.mean(_train)
        _rmse = np.sqrt(np.mean((_test - _pred) ** 2))
        fold_rmses.append(_rmse)

        print(f"Fold {fold_num}: train={len(train_indices)}, test={len(test_indices)}, RMSE={_rmse:.3f}")

    print(f"\nAverage RMSE across 5 folds: {np.mean(fold_rmses):.2f} (± {np.std(fold_rmses):.3f})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **`LeaveOneOut`** — the extreme case where *k* = *n*. Each observation gets held out one at a time, and the estimator is trained on the remaining *n − 1* observations. This gives the most thorough evaluation, but is slow for large datasets:
    """)
    return


@app.cell
def _(flippers, np):
    from sklearn.model_selection import LeaveOneOut

    # Same as before
    loo = LeaveOneOut()

    loo_errors = []
    for all_train, left_out_observation in loo.split(flippers):

        _train = flippers[all_train]
        _test = flippers[left_out_observation]

        _pred = np.mean(_train)
        _error = (_test[0] - _pred) ** 2  # single test point
        loo_errors.append(_error)

    loo_rmse = np.sqrt(np.mean(loo_errors))
    print(f"Leave-One-Out: {len(loo_errors)} splits (one per observation)")
    print(f"LOO RMSE: {loo_rmse:.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :::{.callout-note title="Comparing the approaches"}
    Notice the progression from less splitting to more splitting:

    | Method | # of splits | Train size | Test size | Speed |
    |--------|------------|------------|-----------|-------|
    | `train_test_split` | 1 | ~80% | ~20% | Instant |
    | `KFold(k=5)` | 5 | ~80% each | ~20% each | Fast |
    | `KFold(k=10)` | 10 | ~90% each | ~10% each | Fast |
    | `LeaveOneOut` | *n* | *n* − 1 | 1 | Slow for big data |

    More splits → more stable RMSE estimate, but slower. **5-fold or 10-fold CV** is the most common choice in practice.
    :::
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    :::{.callout-tip title="Your Turn"}

    Try playing with another splitter from `sklearn` (linked above) or adjusting the values of any of the ones we explored here to really make sure you understand what's going on.
    :::
    """)
    return


@app.cell
def _():
    # Your code here
    return


@app.cell
def _(mo):
    mo.md(r"""
    :::{.callout-tip title="Your Turn"}
    Earlier you were asked to compare a different metric for evaluating the median instead of RMSE.
    `sklearn` offers many such "metrics" and "scorers" to do that for you.

    See if you can use one of [the following](https://scikit-learn.org/stable/api/sklearn.metrics.html#regression-metrics) to evaluate the mean/median differently.

    Then change up the *splitter* and see how they interact
    :::
    """)
    return


@app.cell
def _():
    # Your code here
    return


@app.cell
def _():
    # Your code here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary: The Complete Resampling Toolkit

    Over the last two notebooks, we've learned four computational approaches to statistics — all based on the same simple idea: **use a for-loop to repeat a random process and see what happens**.

    | Approach | Question | Method | From |
    |----------|----------|--------|------|
    | **Monte Carlo** | "What if the world worked like this?" | Simulate from assumed distribution | Notebook 01 |
    | **Bootstrap** | "How uncertain is my estimate?" | Resample WITH replacement | Notebook 01 |
    | **Permutation** | "Could this arise by chance?" | Shuffle WITHOUT replacement | This notebook |
    | **Cross-Validation** | "Does my estimator generalize?" | Train/test splits | This notebook |

    Each method answers a **different question** about your data. The beauty is that they all follow the same computational pattern:

    ```
    results = []
    for i in range(n_simulations):
        resampled_data = resample_somehow(data)   # the method differs
        result = compute_statistic(resampled_data)
        results.append(result)
    ```

    This is what Jake VanderPlas means when he says:

    <img src="./figs/jake2.png" width="40%" alt="Jake VanderPlas">
    """)
    return


if __name__ == "__main__":
    app.run()
