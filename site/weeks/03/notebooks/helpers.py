import numpy as np
import polars as pl
from polars import col
import seaborn as sns
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, fixed, IntSlider
import warnings
warnings.filterwarnings('ignore',category=UserWarning)

def generate_student_data(n_samples=100):
    np.random.seed(0)

    # Dataset 1: Study time and test performance
    study_time = np.random.normal(loc=15, scale=5, size=n_samples)
    performance = 60 + 2 * study_time + np.random.normal(0, 10, n_samples)
    performance = np.clip(performance, 0, 100)

    # Student labels
    student_labels = _generate_labels(n_samples)

    students = pl.DataFrame({
        'student': student_labels,
        'study_time': study_time,
        'test_score': performance,
    })
    return students

def plot_dot(sample_size=100,test_scale=1.0, time_scale=1.0):

    np.random.seed(0)
    df = generate_student_data(sample_size)

    to_plot = df.with_columns(
        col('study_time') * time_scale,
        col('test_score') * test_scale
    )
    dot_product = np.dot(
        to_plot['study_time'].to_numpy(),
        to_plot['test_score'].to_numpy()
    )

    f, ax = plt.subplots()
    ax = sns.scatterplot(
        data=to_plot,
        x="study_time",
        y="test_score",
        ax=ax
    )

    ax.set(
        xlabel = 'Study Time (hours)',
        ylabel = 'Test Score (out of 100)',
        title = f"Dot Product: {dot_product:.3f}"
    )
    plt.show()

def dot_widget():
    interact(
        plot_dot,
        time_scale=FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='Time multiplier'),
        test_scale=FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='Score multiplier'),
        sample_size=IntSlider(value=100, min=50, max=500, step=10, description='Sample size')

    )

def plot_dot_avg(sample_size=100, test_scale=1.0, time_scale=1.0):

    np.random.seed(0)
    df = generate_student_data(sample_size)

    to_plot = df.with_columns(
        col('study_time') * time_scale,
        col('test_score') * test_scale
    )
    dot_product = np.dot(
        to_plot['study_time'].to_numpy(),
        to_plot['test_score'].to_numpy()
    ) / to_plot.height

    f, ax = plt.subplots()
    ax = sns.scatterplot(
        data=to_plot,
        x="study_time",
        y="test_score",
        ax=ax
    )

    ax.set(
        xlabel = 'Study Time (hours)',
        ylabel = 'Test Score (out of 100)',
        title = f"Average Dot Product: {dot_product:.3f}"
    )
    plt.show()

def dot_avg_widget():
    interact(
        plot_dot_avg,
        time_scale=FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='Time multiplier'),
        test_scale=FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='Score multiplier'),
        sample_size=IntSlider(value=100, min=50, max=500, step=10, description='Sample size')
    )

def plot_cov(sample_size=100, test_scale=1.0, time_scale=1.0):

    np.random.seed(0)
    df = generate_student_data(sample_size)

    to_plot = df.with_columns(
        col('study_time') * time_scale,
        col('test_score') * test_scale
    )

    study_mean = to_plot['study_time'].mean()
    test_mean = to_plot['test_score'].mean()
    study_var = to_plot['study_time'].var()
    test_var = to_plot['test_score'].var()


    to_plot = to_plot.with_columns(
        col('study_time') - col('study_time').mean(),
        col('test_score') - col('test_score').mean()
    )

    cov = np.mean(
        to_plot['study_time'].to_numpy() *
        to_plot['test_score'].to_numpy()
    )

    f, ax = plt.subplots()
    ax = sns.scatterplot(
        data=to_plot,
        x="study_time",
        y="test_score",
        ax=ax
    )
    ax.axvline(x=0, ls='--', color='k')
    ax.axhline(y=0, ls='--', color='k')

    ax.set(
        xlabel = 'Study Time (hours)',
        ylabel = 'Test Score (out of 100)',
        title = f"Co-variance: {cov:.3f}\nX_mean: {study_mean:.2f}, X_var: {study_var:.2f}\nY_mean: {test_mean:.2f}, Y_var: {test_var:.2f}"
    )
    plt.show()

def cov_widget():
    interact(
        plot_cov,
        time_scale=FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='Time multiplier'),
        test_scale=FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='Score multiplier'),
        sample_size=IntSlider(value=100, min=50, max=500, step=10, description='Sample size')
    )


def plot_corr(sample_size=100, test_scale=1.0, time_scale=1.0):

    np.random.seed(0)
    df = generate_student_data(sample_size)

    to_plot = df.with_columns(
        col('study_time') * time_scale,
        col('test_score') * test_scale
    )

    study_time = to_plot['study_time'].to_numpy()
    test_score = to_plot['test_score'].to_numpy()
    study_norm = np.sqrt(np.sum([student**2 for student in study_time]))   
    test_norm = np.sqrt(np.sum([student**2 for student in test_score]))
    cos = np.dot(study_time, test_score) / (study_norm * test_norm)

    scale = lambda c: (col(c) - col(c).mean()) / col(c).std()

    to_plot = to_plot.with_columns(
        study_time_z = scale('study_time'),
        test_score_z = scale('test_score')
    )

    study_time = to_plot['study_time_z'].to_numpy()
    test_score = to_plot['test_score_z'].to_numpy()
    corr = np.mean(study_time * test_score)

    f, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0] = sns.scatterplot(
        data=to_plot,
        x="study_time",
        y="test_score",
        ax=ax[0]
    )

    ax[0].set(
        xlabel = 'Study Time (hours)',
        ylabel = 'Test Score (out of 100)',
        title = f"Cosine Similarity: {cos:.3f}"
    )
    ax[1] = sns.scatterplot(
        data=to_plot,
        x="study_time_z",
        y="test_score_z",
        ax=ax[1]
    )

    ax[1].axvline(x=0, ls='--', color='k')
    ax[1].axhline(y=0, ls='--', color='k')

    ax[1].set(
        xlabel = 'Study Time (z-score)',
        ylabel = 'Test Score (z-score)',
        title = f"Correlation: {corr:.3f}"
    )
    plt.show()

def corr_widget():
    interact(
        plot_corr,
        time_scale=FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='Time multiplier'),
        test_scale=FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='Score multiplier'),
        sample_size=IntSlider(value=100, min=50, max=500, step=10, description='Sample size')
    )


def plot_cos(sample_size=100, test_scale=1.0, time_scale=1.0):

    np.random.seed(0)
    df = generate_student_data(sample_size)

    to_plot = df.with_columns(
        col('study_time') * time_scale,
        col('test_score') * test_scale
    )

    study_time = to_plot['study_time'].to_numpy()
    test_score = to_plot['test_score'].to_numpy()

    study_norm = np.sqrt(np.sum([student**2 for student in study_time]))   
    test_norm = np.sqrt(np.sum([student**2 for student in test_score]))

    cos = np.dot(study_time, test_score) / (study_norm * test_norm)

    f, ax = plt.subplots()
    ax = sns.scatterplot(
        data=to_plot,
        x="study_time",
        y="test_score",
        ax=ax
    )

    ax.set(
        xlabel = 'Study Time (hours)',
        ylabel = 'Test Score (out of 100)',
        title = f"Cosine Similarity: {cos:.3f}\nX_norm: {study_norm:.2f}, Y_norm: {test_norm:.2f}"
    )
    plt.show()

def cos_widget():
    interact(
        plot_cos,
        time_scale=FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='Time multiplier'),
        test_scale=FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description='Score multiplier'),
        sample_size=IntSlider(value=100, min=50, max=500, step=10, description='Sample size')
    )

def generate_ice_cream_data():
    np.random.seed(0)
    n_samples = 100

    # Dataset 2: Ice cream and swimming (both influenced by temperature)
    temperature = np.random.normal(loc=25, scale=5, size=n_samples)  # Hidden common cause
    ice_cream = 100 + 5 * temperature + np.random.normal(0, 20, n_samples)
    swimming = 50 + 3 * temperature + np.random.normal(0, 15, n_samples)

    city_labels = _generate_labels(n_samples)

    students = pl.DataFrame({
        'cit': city_labels,
        'ice_cream_concsumption': ice_cream,
        'swimming_hours': swimming,
    })
    return students

def _generate_labels(n):
    from string import ascii_letters

    letters = ascii_letters[26:]
    labels = []
    multiplier = 1
    while len(labels) < n:
        labels += [''.join(letter * multiplier) for letter in letters]
        multiplier += 1
    labels = labels[:n]
    assert len(set(labels)) == n
    return labels