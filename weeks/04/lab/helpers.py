"""
Interactive statistical explorables for PSYC 201B Week 4.

These self-contained widgets help build intuitions about:
- Loss functions (SSE vs SAE)
- Law of Large Numbers (LLN)
- Central Limit Theorem (CLT)

Usage:
    from helpers import loss_explorer, lln_explorer, clt_explorer

    # In a marimo cell:
    loss_explorer()  # Returns interactive widget
"""

import anywidget
import traitlets
import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # noqa: F401 (used in BootstrapExplorer._render)
import io
import base64

# =============================================================================
# Data Loading
# =============================================================================

def load_penguins() -> pl.DataFrame:
    """Load penguins dataset as Polars DataFrame."""
    return pl.DataFrame(sns.load_dataset("penguins")).drop_nulls()


def load_tips() -> pl.DataFrame:
    """Load tips dataset as Polars DataFrame."""
    return pl.DataFrame(sns.load_dataset("tips")).drop_nulls()


# =============================================================================
# Utility Functions
# =============================================================================

def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='white')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return encoded


# =============================================================================
# Loss Function Explorer
# =============================================================================

class LossExplorer(anywidget.AnyWidget):
    """
    Interactive explorer for SSE vs SAE loss functions.

    Drag the estimate line to see how different loss functions respond.
    The mean minimizes SSE; the median minimizes SAE.
    """

    _esm = """
    function render({ model, el }) {
        const container = document.createElement('div');
        container.style.fontFamily = 'system-ui, -apple-system, sans-serif';
        container.style.maxWidth = '800px';

        // Title
        const title = document.createElement('div');
        title.innerHTML = '<strong>Loss Function Explorer</strong>: Drag the slider to move your estimate';
        title.style.marginBottom = '12px';
        title.style.fontSize = '14px';
        container.appendChild(title);

        // Slider row
        const sliderRow = document.createElement('div');
        sliderRow.style.display = 'flex';
        sliderRow.style.alignItems = 'center';
        sliderRow.style.gap = '12px';
        sliderRow.style.marginBottom = '8px';

        const label = document.createElement('span');
        label.textContent = 'Estimate:';
        label.style.fontSize = '14px';

        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = model.get('est_min');
        slider.max = model.get('est_max');
        slider.step = '0.1';
        slider.value = model.get('estimate');
        slider.style.width = '300px';
        slider.style.cursor = 'pointer';

        const valueDisplay = document.createElement('span');
        valueDisplay.textContent = model.get('estimate').toFixed(1);
        valueDisplay.style.fontFamily = 'monospace';
        valueDisplay.style.fontSize = '14px';
        valueDisplay.style.minWidth = '50px';

        slider.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            model.set('estimate', val);
            model.save_changes();
            valueDisplay.textContent = val.toFixed(1);
        });

        sliderRow.appendChild(label);
        sliderRow.appendChild(slider);
        sliderRow.appendChild(valueDisplay);
        container.appendChild(sliderRow);

        // Stats display
        const stats = document.createElement('div');
        stats.style.fontFamily = 'monospace';
        stats.style.fontSize = '13px';
        stats.style.marginBottom = '12px';
        stats.style.padding = '8px';
        stats.style.backgroundColor = '#f5f5f5';
        stats.style.borderRadius = '4px';
        stats.innerHTML = model.get('stats_html');
        container.appendChild(stats);

        model.on('change:stats_html', () => {
            stats.innerHTML = model.get('stats_html');
        });

        // Chart image
        const img = document.createElement('img');
        img.src = 'data:image/png;base64,' + model.get('chart_base64');
        img.style.maxWidth = '100%';

        model.on('change:chart_base64', () => {
            img.src = 'data:image/png;base64,' + model.get('chart_base64');
        });

        container.appendChild(img);
        el.appendChild(container);
    }
    export default { render };
    """

    # Synced traits
    estimate = traitlets.Float(200.0).tag(sync=True)
    est_min = traitlets.Float(180.0).tag(sync=True)
    est_max = traitlets.Float(220.0).tag(sync=True)
    chart_base64 = traitlets.Unicode("").tag(sync=True)
    stats_html = traitlets.Unicode("").tag(sync=True)

    def __init__(self, data: pl.DataFrame | None = None, column: str = "flipper_length_mm", **kwargs):
        super().__init__(**kwargs)

        # Load data
        if data is None:
            data = load_penguins()

        self._data = data.select(column).to_series().to_list()
        self._column = column

        # Calculate estimators
        self._mean = np.mean(self._data)
        self._median = np.median(self._data)

        # Set slider bounds around the data
        data_range = max(self._data) - min(self._data)
        self.est_min = self._mean - data_range * 0.15
        self.est_max = self._mean + data_range * 0.15
        self.estimate = self._mean  # Start at mean

        # Precompute loss curves
        self._estimate_range = np.linspace(self.est_min, self.est_max, 200)
        sse_raw = np.array([sum((x - est) ** 2 for x in self._data) for est in self._estimate_range])
        sae_raw = np.array([sum(abs(x - est) for x in self._data) for est in self._estimate_range])

        # Normalize: shift so minimum = 0, then scale so max = 1
        self._sse_min = float(sse_raw.min())
        self._sae_min = float(sae_raw.min())
        sse_shifted = sse_raw - self._sse_min
        sae_shifted = sae_raw - self._sae_min
        self._sse_range = float(sse_shifted.max()) if sse_shifted.max() > 0 else 1.0
        self._sae_range = float(sae_shifted.max()) if sae_shifted.max() > 0 else 1.0
        self._sse_curve = sse_shifted / self._sse_range
        self._sae_curve = sae_shifted / self._sae_range

        # Create figure
        self._fig, self._ax = plt.subplots(figsize=(8, 4))

        # Initial render
        self._render()
        self.observe(self._on_estimate_change, names=['estimate'])

    def _compute_errors(self, estimate: float) -> tuple:
        """Compute SSE and SAE for current estimate."""
        sse = sum((x - estimate) ** 2 for x in self._data)
        sae = sum(abs(x - estimate) for x in self._data)
        return sse, sae

    def _normalize_error(self, sse: float, sae: float) -> tuple:
        """Normalize errors using same shift+scale as curves."""
        return (sse - self._sse_min) / self._sse_range, (sae - self._sae_min) / self._sae_range

    def _render(self):
        """Render the chart and update stats."""
        ax = self._ax
        ax.clear()

        # Plot normalized loss curves (minimum at 0)
        ax.plot(self._estimate_range, self._sse_curve,
                'r-', linewidth=2, label='SSE (squared)', alpha=0.8)
        ax.plot(self._estimate_range, self._sae_curve,
                'b-', linewidth=2, label='SAE (absolute)', alpha=0.8)

        # Mark mean and median
        ax.axvline(self._mean, color='red', linestyle='--', alpha=0.6,
                   label=f'Mean = {self._mean:.1f}')
        ax.axvline(self._median, color='blue', linestyle='--', alpha=0.6,
                   label=f'Median = {self._median:.1f}')

        # Mark current estimate
        ax.axvline(self.estimate, color='green', linewidth=2,
                   label=f'Your estimate = {self.estimate:.1f}')

        # Current error points on the curves
        sse, sae = self._compute_errors(self.estimate)
        sse_norm, sae_norm = self._normalize_error(sse, sae)
        ax.scatter([self.estimate], [sse_norm], color='red', s=100, zorder=5)
        ax.scatter([self.estimate], [sae_norm], color='blue', s=100, zorder=5)

        ax.set_xlabel('Estimate', fontsize=11)
        ax.set_ylabel('Error above minimum (scaled)', fontsize=11)
        ax.set_title('Which estimate minimizes error?', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_ylim(-0.05, 1.1)

        self._fig.tight_layout()
        self.chart_base64 = fig_to_base64(self._fig)

        # Update stats
        sse_at_mean, sae_at_mean = self._compute_errors(self._mean)
        sse_at_median, sae_at_median = self._compute_errors(self._median)

        self.stats_html = f"""
        <table style="width:100%; border-collapse: collapse;">
        <tr style="border-bottom: 1px solid #ddd;">
            <td></td>
            <td><b>SSE</b></td>
            <td><b>SAE</b></td>
        </tr>
        <tr>
            <td>Your estimate ({self.estimate:.1f})</td>
            <td style="color: {'green' if abs(sse - sse_at_mean) < 1 else 'black'}">{sse:,.0f}</td>
            <td style="color: {'green' if abs(sae - sae_at_median) < 1 else 'black'}">{sae:,.0f}</td>
        </tr>
        <tr style="color: #666;">
            <td>At mean ({self._mean:.1f})</td>
            <td>{sse_at_mean:,.0f} ✓</td>
            <td>{sae_at_mean:,.0f}</td>
        </tr>
        <tr style="color: #666;">
            <td>At median ({self._median:.1f})</td>
            <td>{sse_at_median:,.0f}</td>
            <td>{sae_at_median:,.0f} ✓</td>
        </tr>
        </table>
        """

    def _on_estimate_change(self, change):
        """Re-render when estimate changes."""
        self._render()

    def __del__(self):
        """Clean up matplotlib figure."""
        plt.close(self._fig)


def loss_explorer(data: pl.DataFrame | None = None, column: str = "flipper_length_mm") -> LossExplorer:
    """
    Create an interactive loss function explorer.

    Drag the slider to see how SSE and SAE respond to different estimates.
    Notice: mean minimizes SSE, median minimizes SAE.

    Parameters
    ----------
    data : pl.DataFrame, optional
        Data to use. Defaults to penguins dataset.
    column : str
        Numeric column to analyze. Default: "flipper_length_mm"

    Returns
    -------
    LossExplorer widget

    Example
    -------
    >>> from helpers import loss_explorer
    >>> loss_explorer()  # Interactive widget appears
    """
    return LossExplorer(data=data, column=column)


# =============================================================================
# Law of Large Numbers Explorer
# =============================================================================

class LLNExplorer(anywidget.AnyWidget):
    """
    Interactive explorer for the Law of Large Numbers.

    Uses synthetic heavy-tailed data (t-distribution) to show how the
    sample mean converges to the population mean as n increases.
    Each observation is plotted as a dot connected by a line.
    """

    _esm = """
    function render({ model, el }) {
        const container = document.createElement('div');
        container.style.fontFamily = 'system-ui, -apple-system, sans-serif';
        container.style.maxWidth = '800px';

        // Title
        const title = document.createElement('div');
        title.innerHTML = '<strong>Law of Large Numbers</strong>: Watch the running mean converge';
        title.style.marginBottom = '12px';
        title.style.fontSize = '14px';
        container.appendChild(title);

        // Slider row
        const sliderRow = document.createElement('div');
        sliderRow.style.display = 'flex';
        sliderRow.style.alignItems = 'center';
        sliderRow.style.gap = '12px';
        sliderRow.style.marginBottom = '8px';

        const label = document.createElement('span');
        label.textContent = 'Number of observations (n):';
        label.style.fontSize = '14px';

        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = model.get('n_min');
        slider.max = model.get('n_max');
        slider.step = '1';
        slider.value = model.get('n');
        slider.style.width = '300px';
        slider.style.cursor = 'pointer';

        const valueDisplay = document.createElement('span');
        valueDisplay.textContent = model.get('n');
        valueDisplay.style.fontFamily = 'monospace';
        valueDisplay.style.fontSize = '14px';
        valueDisplay.style.minWidth = '40px';

        slider.addEventListener('input', (e) => {
            const val = parseInt(e.target.value);
            model.set('n', val);
            model.save_changes();
            valueDisplay.textContent = val;
        });

        sliderRow.appendChild(label);
        sliderRow.appendChild(slider);
        sliderRow.appendChild(valueDisplay);
        container.appendChild(sliderRow);

        // Stats display
        const stats = document.createElement('div');
        stats.style.fontFamily = 'monospace';
        stats.style.fontSize = '13px';
        stats.style.marginBottom = '12px';
        stats.style.padding = '8px';
        stats.style.backgroundColor = '#f5f5f5';
        stats.style.borderRadius = '4px';
        stats.innerHTML = model.get('stats_html');
        container.appendChild(stats);

        model.on('change:stats_html', () => {
            stats.innerHTML = model.get('stats_html');
        });

        // Chart image
        const img = document.createElement('img');
        img.src = 'data:image/png;base64,' + model.get('chart_base64');
        img.style.maxWidth = '100%';

        model.on('change:chart_base64', () => {
            img.src = 'data:image/png;base64,' + model.get('chart_base64');
        });

        container.appendChild(img);
        el.appendChild(container);
    }
    export default { render };
    """

    # Synced traits
    n = traitlets.Int(20).tag(sync=True)
    n_min = traitlets.Int(2).tag(sync=True)
    n_max = traitlets.Int(500).tag(sync=True)
    chart_base64 = traitlets.Unicode("").tag(sync=True)
    stats_html = traitlets.Unicode("").tag(sync=True)

    def __init__(self, pop_mean: float = 100.0, pop_std: float = 15.0,
                 df: int = 3, n_max: int = 500, seed: int = 42, **kwargs):
        super().__init__(**kwargs)

        self._pop_mean = pop_mean
        self._pop_std = pop_std
        self.n_max = n_max

        # Generate heavy-tailed synthetic data (t-distribution)
        # t-dist with low df has heavy tails → occasional extreme values
        rng = np.random.default_rng(seed)
        self._samples = rng.standard_t(df=df, size=n_max) * pop_std + pop_mean

        # Create figure
        self._fig, self._ax = plt.subplots(figsize=(8, 4))

        # Initial render
        self._render()
        self.observe(self._on_n_change, names=['n'])

    def _render(self):
        """Render the convergence chart."""
        ax = self._ax
        ax.clear()

        observations = self._samples[:self.n]
        x_vals = np.arange(1, self.n + 1)

        # Compute running mean at each step
        running_means = np.cumsum(observations) / x_vals

        # Plot line connecting running means
        ax.plot(x_vals, running_means, '-', color='steelblue', linewidth=1.5,
                alpha=0.7, zorder=2)

        # Plot dots for each running mean estimate
        # Use smaller dots if many points, larger if few
        dot_size = max(8, min(40, 600 / self.n))
        ax.scatter(x_vals, running_means, s=dot_size, color='steelblue',
                   edgecolors='white', linewidths=0.5, zorder=3,
                   label='Running mean')

        # Population mean reference line
        ax.axhline(self._pop_mean, color='red', linestyle='--', linewidth=2,
                   label=f'True mean = {self._pop_mean:.1f}', zorder=4)

        # Theoretical SE envelope (±1.96 SE)
        se_band = 1.96 * self._pop_std / np.sqrt(x_vals)
        ax.fill_between(x_vals,
                        self._pop_mean - se_band,
                        self._pop_mean + se_band,
                        alpha=0.15, color='red', label='95% expected range')

        ax.set_xlabel('Number of observations (n)', fontsize=11)
        ax.set_ylabel('Running mean', fontsize=11)
        ax.set_title('Law of Large Numbers: More data → better estimate', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)

        # Fix y-limits to full range so you can see the wild early bouncing
        ax.set_ylim(self._pop_mean - self._pop_std * 3, self._pop_mean + self._pop_std * 3)
        ax.set_xlim(0, self.n + max(5, self.n * 0.05))

        self._fig.tight_layout()
        self.chart_base64 = fig_to_base64(self._fig)

        # Update stats
        current_mean = running_means[-1]
        error = abs(current_mean - self._pop_mean)
        se = self._pop_std / np.sqrt(self.n)

        self.stats_html = (
            f"True mean: <b>{self._pop_mean:.1f}</b> | "
            f"Running mean (n={self.n}): <b>{current_mean:.2f}</b> | "
            f"Error: <b>{error:.2f}</b> | "
            f"SE (σ/√n): <b>{se:.2f}</b>"
        )

    def _on_n_change(self, change):
        """Re-render when n changes."""
        self._render()

    def __del__(self):
        plt.close(self._fig)


def lln_explorer(pop_mean: float = 100.0, pop_std: float = 15.0,
                 df: int = 3, n_max: int = 500, seed: int = 42) -> LLNExplorer:
    """
    Create an interactive Law of Large Numbers explorer.

    Uses synthetic heavy-tailed data (t-distribution with df=3) so
    convergence is visibly bumpy. Each running mean is plotted as a
    dot connected by a line.

    Parameters
    ----------
    pop_mean : float
        True population mean. Default: 100.0
    pop_std : float
        Population standard deviation (scale). Default: 15.0
    df : int
        Degrees of freedom for t-distribution (lower = heavier tails). Default: 3
    n_max : int
        Maximum number of observations. Default: 500
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    LLNExplorer widget
    """
    return LLNExplorer(pop_mean=pop_mean, pop_std=pop_std, df=df, n_max=n_max, seed=seed)


# =============================================================================
# Central Limit Theorem Explorer
# =============================================================================

# Distribution generators: each returns (samples, label, true_mean, true_std)
def _generate_population(name: str, size: int, rng: np.random.Generator) -> tuple:
    """Generate synthetic population data for a named distribution."""
    if name == "Normal":
        data = rng.normal(loc=50, scale=15, size=size)
        return data, "Normal(50, 15)", 50.0, 15.0
    elif name == "Uniform":
        data = rng.uniform(low=0, high=100, size=size)
        return data, "Uniform(0, 100)", 50.0, 100 / np.sqrt(12)
    elif name == "Exponential":
        data = rng.exponential(scale=20, size=size)
        return data, "Exponential(λ=1/20)", 20.0, 20.0
    elif name == "Bimodal":
        half = size // 2
        data = np.concatenate([
            rng.normal(loc=30, scale=8, size=half),
            rng.normal(loc=70, scale=8, size=size - half),
        ])
        return data, "Bimodal(30 & 70)", 50.0, float(np.std(data))
    else:
        raise ValueError(f"Unknown distribution: {name}")


def _compute_statistic(data: np.ndarray, name: str) -> float:
    """Compute a named statistic on data."""
    if name == "Mean":
        return float(np.mean(data))
    elif name == "Median":
        return float(np.median(data))
    elif name == "Std Dev":
        return float(np.std(data, ddof=1))
    elif name == "Trimmed Mean (10%)":
        sorted_data = np.sort(data)
        trim = max(1, int(len(data) * 0.1))
        return float(np.mean(sorted_data[trim:-trim]))
    else:
        raise ValueError(f"Unknown statistic: {name}")


class CLTExplorer(anywidget.AnyWidget):
    """
    Interactive explorer for the Central Limit Theorem.

    Choose a population distribution and estimator, then click "Run Batch"
    to progressively build up the sampling distribution. Watch it converge
    to a normal shape regardless of the source distribution.
    """

    _esm = """
    function render({ model, el }) {
        const container = document.createElement('div');
        container.style.fontFamily = 'system-ui, -apple-system, sans-serif';
        container.style.maxWidth = '900px';

        // Title
        const title = document.createElement('div');
        title.innerHTML = '<strong>Central Limit Theorem</strong>: Build a sampling distribution';
        title.style.marginBottom = '12px';
        title.style.fontSize = '14px';
        container.appendChild(title);

        // --- Row 1: Dropdowns ---
        const row1 = document.createElement('div');
        row1.style.display = 'flex';
        row1.style.alignItems = 'center';
        row1.style.gap = '16px';
        row1.style.marginBottom = '8px';
        row1.style.flexWrap = 'wrap';

        // Distribution dropdown
        const distGroup = document.createElement('div');
        distGroup.style.display = 'flex';
        distGroup.style.alignItems = 'center';
        distGroup.style.gap = '6px';
        const distLabel = document.createElement('span');
        distLabel.textContent = 'Distribution:';
        distLabel.style.fontSize = '13px';
        const distSelect = document.createElement('select');
        distSelect.style.fontSize = '13px';
        distSelect.style.padding = '2px 6px';
        const dists = JSON.parse(model.get('distribution_options'));
        dists.forEach(d => {
            const opt = document.createElement('option');
            opt.value = d;
            opt.textContent = d;
            if (d === model.get('distribution')) opt.selected = true;
            distSelect.appendChild(opt);
        });
        distSelect.addEventListener('change', (e) => {
            model.set('distribution', e.target.value);
            model.save_changes();
        });
        distGroup.appendChild(distLabel);
        distGroup.appendChild(distSelect);
        row1.appendChild(distGroup);

        // Estimator dropdown
        const estGroup = document.createElement('div');
        estGroup.style.display = 'flex';
        estGroup.style.alignItems = 'center';
        estGroup.style.gap = '6px';
        const estLabel = document.createElement('span');
        estLabel.textContent = 'Estimator:';
        estLabel.style.fontSize = '13px';
        const estSelect = document.createElement('select');
        estSelect.style.fontSize = '13px';
        estSelect.style.padding = '2px 6px';
        const ests = JSON.parse(model.get('estimator_options'));
        ests.forEach(e => {
            const opt = document.createElement('option');
            opt.value = e;
            opt.textContent = e;
            if (e === model.get('estimator')) opt.selected = true;
            estSelect.appendChild(opt);
        });
        estSelect.addEventListener('change', (e) => {
            model.set('estimator', e.target.value);
            model.save_changes();
        });
        estGroup.appendChild(estLabel);
        estGroup.appendChild(estSelect);
        row1.appendChild(estGroup);
        container.appendChild(row1);

        // --- Row 2: Sliders + Button ---
        const row2 = document.createElement('div');
        row2.style.display = 'flex';
        row2.style.alignItems = 'center';
        row2.style.gap = '16px';
        row2.style.marginBottom = '8px';
        row2.style.flexWrap = 'wrap';

        // Sample size slider
        const nGroup = document.createElement('div');
        nGroup.style.display = 'flex';
        nGroup.style.alignItems = 'center';
        nGroup.style.gap = '6px';
        const nLabel = document.createElement('span');
        nLabel.textContent = 'Sample size (n):';
        nLabel.style.fontSize = '13px';
        const nSlider = document.createElement('input');
        nSlider.type = 'range';
        nSlider.min = '5';
        nSlider.max = '200';
        nSlider.step = '5';
        nSlider.value = model.get('n');
        nSlider.style.width = '120px';
        nSlider.style.cursor = 'pointer';
        const nValue = document.createElement('span');
        nValue.textContent = model.get('n');
        nValue.style.fontFamily = 'monospace';
        nValue.style.minWidth = '30px';
        nSlider.addEventListener('input', (e) => {
            const val = parseInt(e.target.value);
            model.set('n', val);
            model.save_changes();
            nValue.textContent = val;
        });
        nGroup.appendChild(nLabel);
        nGroup.appendChild(nSlider);
        nGroup.appendChild(nValue);
        row2.appendChild(nGroup);

        // Batch size slider
        const batchGroup = document.createElement('div');
        batchGroup.style.display = 'flex';
        batchGroup.style.alignItems = 'center';
        batchGroup.style.gap = '6px';
        const batchLabel = document.createElement('span');
        batchLabel.textContent = 'Per click:';
        batchLabel.style.fontSize = '13px';
        const batchSlider = document.createElement('input');
        batchSlider.type = 'range';
        batchSlider.min = '10';
        batchSlider.max = '200';
        batchSlider.step = '10';
        batchSlider.value = model.get('batch_size');
        batchSlider.style.width = '80px';
        batchSlider.style.cursor = 'pointer';
        const batchValue = document.createElement('span');
        batchValue.textContent = model.get('batch_size');
        batchValue.style.fontFamily = 'monospace';
        batchValue.style.minWidth = '30px';
        batchSlider.addEventListener('input', (e) => {
            const val = parseInt(e.target.value);
            model.set('batch_size', val);
            model.save_changes();
            batchValue.textContent = val;
        });
        batchGroup.appendChild(batchLabel);
        batchGroup.appendChild(batchSlider);
        batchGroup.appendChild(batchValue);
        row2.appendChild(batchGroup);

        // Run button
        const runBtn = document.createElement('button');
        runBtn.textContent = '▶ Draw Samples';
        runBtn.style.fontSize = '13px';
        runBtn.style.padding = '4px 14px';
        runBtn.style.cursor = 'pointer';
        runBtn.style.backgroundColor = '#4a90d9';
        runBtn.style.color = 'white';
        runBtn.style.border = 'none';
        runBtn.style.borderRadius = '4px';
        runBtn.addEventListener('click', () => {
            model.set('run_trigger', model.get('run_trigger') + 1);
            model.save_changes();
        });
        row2.appendChild(runBtn);

        // Reset button
        const resetBtn = document.createElement('button');
        resetBtn.textContent = '↺ Reset';
        resetBtn.style.fontSize = '13px';
        resetBtn.style.padding = '4px 10px';
        resetBtn.style.cursor = 'pointer';
        resetBtn.style.border = '1px solid #ccc';
        resetBtn.style.borderRadius = '4px';
        resetBtn.style.backgroundColor = '#f5f5f5';
        resetBtn.addEventListener('click', () => {
            model.set('reset_trigger', model.get('reset_trigger') + 1);
            model.save_changes();
        });
        row2.appendChild(resetBtn);

        container.appendChild(row2);

        // Stats display
        const stats = document.createElement('div');
        stats.style.fontFamily = 'monospace';
        stats.style.fontSize = '13px';
        stats.style.marginBottom = '12px';
        stats.style.padding = '8px';
        stats.style.backgroundColor = '#f5f5f5';
        stats.style.borderRadius = '4px';
        stats.innerHTML = model.get('stats_html');
        container.appendChild(stats);

        model.on('change:stats_html', () => {
            stats.innerHTML = model.get('stats_html');
        });

        // Chart image
        const img = document.createElement('img');
        img.src = 'data:image/png;base64,' + model.get('chart_base64');
        img.style.maxWidth = '100%';

        model.on('change:chart_base64', () => {
            img.src = 'data:image/png;base64,' + model.get('chart_base64');
        });

        container.appendChild(img);
        el.appendChild(container);
    }
    export default { render };
    """

    # Synced traits
    distribution = traitlets.Unicode("Normal").tag(sync=True)
    estimator = traitlets.Unicode("Mean").tag(sync=True)
    n = traitlets.Int(30).tag(sync=True)
    batch_size = traitlets.Int(50).tag(sync=True)
    run_trigger = traitlets.Int(0).tag(sync=True)
    reset_trigger = traitlets.Int(0).tag(sync=True)
    chart_base64 = traitlets.Unicode("").tag(sync=True)
    stats_html = traitlets.Unicode("").tag(sync=True)
    distribution_options = traitlets.Unicode("").tag(sync=True)
    estimator_options = traitlets.Unicode("").tag(sync=True)

    _DISTRIBUTIONS = ["Normal", "Uniform", "Exponential", "Bimodal"]
    _ESTIMATORS = ["Mean", "Median", "Std Dev", "Trimmed Mean (10%)"]

    def __init__(self, pop_size: int = 10_000, seed: int = 42, **kwargs):
        super().__init__(**kwargs)

        import json
        self.distribution_options = json.dumps(self._DISTRIBUTIONS)
        self.estimator_options = json.dumps(self._ESTIMATORS)

        self._pop_size = pop_size
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._accumulated: list[float] = []

        # Generate initial population
        self._regenerate_population()

        # Create figure with 2 subplots
        self._fig, self._axes = plt.subplots(1, 2, figsize=(10, 4))

        # Initial render
        self._render()

        # Observe changes
        self.observe(self._on_run, names=['run_trigger'])
        self.observe(self._on_reset, names=['reset_trigger'])
        self.observe(self._on_settings_change, names=['distribution', 'estimator', 'n'])

    def _regenerate_population(self):
        """Generate population data for current distribution."""
        self._pop_data, self._pop_label, self._pop_mean, self._pop_std = \
            _generate_population(self.distribution, self._pop_size, self._rng)
        self._accumulated = []

    def _on_run(self, change):
        """Add a batch of samples when button is clicked."""
        for _ in range(self.batch_size):
            sample = self._rng.choice(self._pop_data, size=self.n, replace=True)
            stat = _compute_statistic(sample, self.estimator)
            self._accumulated.append(stat)
        self._render()

    def _on_reset(self, change):
        """Clear accumulated samples and re-render."""
        self._rng = np.random.default_rng(self._seed)
        self._regenerate_population()
        self._render()

    def _on_settings_change(self, change):
        """Reset when distribution, estimator, or n changes."""
        self._rng = np.random.default_rng(self._seed)
        self._regenerate_population()
        self._render()

    def _render(self):
        """Render the CLT visualization."""
        ax_pop, ax_samp = self._axes
        ax_pop.clear()
        ax_samp.clear()

        # Left panel: Population distribution
        ax_pop.hist(self._pop_data, bins=40, density=True, alpha=0.7,
                    color='steelblue', edgecolor='white')
        ax_pop.axvline(np.mean(self._pop_data), color='red', linestyle='--', linewidth=2)
        ax_pop.set_xlabel('Value', fontsize=10)
        ax_pop.set_ylabel('Density', fontsize=10)
        ax_pop.set_title(f'Population: {self._pop_label}', fontsize=11)

        # Right panel: Sampling distribution (accumulated)
        total = len(self._accumulated)
        if total == 0:
            ax_samp.text(0.5, 0.5, 'Click "Draw Samples"\nto start building\nthe distribution',
                         transform=ax_samp.transAxes, ha='center', va='center',
                         fontsize=14, color='#888')
            ax_samp.set_title(f'Sampling Distribution of {self.estimator}', fontsize=11)
        else:
            stats_array = np.array(self._accumulated)
            ax_samp.hist(stats_array, bins=min(40, max(10, total // 5)),
                         density=True, alpha=0.7, color='coral', edgecolor='white')

            # Overlay theoretical normal
            sim_mean = np.mean(stats_array)
            sim_std = np.std(stats_array)
            if sim_std > 0:
                x_norm = np.linspace(sim_mean - 4*sim_std, sim_mean + 4*sim_std, 100)
                y_norm = (1 / (sim_std * np.sqrt(2*np.pi))) * \
                         np.exp(-0.5 * ((x_norm - sim_mean) / sim_std) ** 2)
                ax_samp.plot(x_norm, y_norm, 'k-', linewidth=2, alpha=0.6,
                             label='Normal fit')

            ax_samp.axvline(sim_mean, color='red', linestyle='--', linewidth=2,
                            label=f'Mean = {sim_mean:.2f}')
            ax_samp.set_title(f'Sampling Dist. of {self.estimator} '
                              f'({total} samples, n={self.n})', fontsize=11)
            ax_samp.legend(loc='upper right', fontsize=9)

        ax_samp.set_xlabel(f'Sample {self.estimator}', fontsize=10)
        ax_samp.set_ylabel('Density', fontsize=10)

        self._fig.tight_layout()
        self.chart_base64 = fig_to_base64(self._fig)

        # Update stats
        if total == 0:
            self.stats_html = (
                f"<b>Distribution</b>: {self._pop_label} | "
                f"<b>Estimator</b>: {self.estimator} | "
                f"<b>n</b> = {self.n} | "
                "Samples drawn: <b>0</b> — click <b>Draw Samples</b> to begin"
            )
        else:
            sim_se = np.std(self._accumulated)
            theoretical_se = self._pop_std / np.sqrt(self.n)
            self.stats_html = (
                f"<b>Distribution</b>: {self._pop_label} | "
                f"<b>Estimator</b>: {self.estimator} | "
                f"<b>n</b> = {self.n} | "
                f"Samples: <b>{total}</b> | "
                f"Simulated SE: <b>{sim_se:.3f}</b> | "
                f"Theoretical SE (σ/√n): <b>{theoretical_se:.3f}</b>"
            )

    def __del__(self):
        plt.close(self._fig)


def clt_explorer(pop_size: int = 10_000, seed: int = 42) -> CLTExplorer:
    """
    Create an interactive Central Limit Theorem explorer.

    Choose a population distribution and estimator, then click "Draw Samples"
    to progressively build up the sampling distribution. Watch it converge
    to a normal shape regardless of the source distribution.

    Parameters
    ----------
    pop_size : int
        Size of the synthetic population. Default: 10,000
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    CLTExplorer widget
    """
    return CLTExplorer(pop_size=pop_size, seed=seed)


# =============================================================================
# No Free Lunch Explorer
# =============================================================================

class NFLExplorer(anywidget.AnyWidget):
    """
    Interactive explorer for the No Free Lunch principle.

    Choose a distribution and sample size to see which estimator (mean vs
    median) has lower RMSE. The winner changes depending on the data shape —
    no single estimator is universally best.
    """

    _esm = """
    function render({ model, el }) {
        const c = document.createElement('div');
        c.style.fontFamily = 'system-ui, -apple-system, sans-serif';
        c.style.maxWidth = '900px';

        const title = document.createElement('div');
        title.innerHTML = '<strong>No Free Lunch</strong>: Which estimator wins depends on the data';
        title.style.marginBottom = '12px';
        title.style.fontSize = '14px';
        c.appendChild(title);

        // Controls row
        const row = document.createElement('div');
        row.style.display = 'flex';
        row.style.alignItems = 'center';
        row.style.gap = '16px';
        row.style.marginBottom = '8px';
        row.style.flexWrap = 'wrap';

        // Distribution dropdown
        const distGroup = document.createElement('div');
        distGroup.style.display = 'flex';
        distGroup.style.alignItems = 'center';
        distGroup.style.gap = '6px';
        const distLabel = document.createElement('span');
        distLabel.textContent = 'Distribution:';
        distLabel.style.fontSize = '13px';
        const distSelect = document.createElement('select');
        distSelect.style.fontSize = '13px';
        distSelect.style.padding = '2px 6px';
        const dists = JSON.parse(model.get('distribution_options'));
        dists.forEach(d => {
            const opt = document.createElement('option');
            opt.value = d;
            opt.textContent = d;
            if (d === model.get('distribution')) opt.selected = true;
            distSelect.appendChild(opt);
        });
        distSelect.addEventListener('change', (e) => {
            model.set('distribution', e.target.value);
            model.save_changes();
        });
        distGroup.appendChild(distLabel);
        distGroup.appendChild(distSelect);
        row.appendChild(distGroup);

        // Sample size slider
        const nGroup = document.createElement('div');
        nGroup.style.display = 'flex';
        nGroup.style.alignItems = 'center';
        nGroup.style.gap = '6px';
        const nLabel = document.createElement('span');
        nLabel.textContent = 'Sample size (n):';
        nLabel.style.fontSize = '13px';
        const nSlider = document.createElement('input');
        nSlider.type = 'range';
        nSlider.min = '10';
        nSlider.max = '100';
        nSlider.step = '5';
        nSlider.value = model.get('sample_size');
        nSlider.style.width = '150px';
        nSlider.style.cursor = 'pointer';
        const nValue = document.createElement('span');
        nValue.textContent = model.get('sample_size');
        nValue.style.fontFamily = 'monospace';
        nValue.style.fontSize = '13px';
        nValue.style.minWidth = '30px';
        nSlider.addEventListener('input', (e) => {
            const val = parseInt(e.target.value);
            model.set('sample_size', val);
            model.save_changes();
            nValue.textContent = val;
        });
        nGroup.appendChild(nLabel);
        nGroup.appendChild(nSlider);
        nGroup.appendChild(nValue);
        row.appendChild(nGroup);

        c.appendChild(row);

        // Stats display
        const stats = document.createElement('div');
        stats.style.fontFamily = 'monospace';
        stats.style.fontSize = '13px';
        stats.style.marginBottom = '12px';
        stats.style.padding = '8px';
        stats.style.backgroundColor = '#f5f5f5';
        stats.style.borderRadius = '4px';
        stats.innerHTML = model.get('stats_html');
        c.appendChild(stats);
        model.on('change:stats_html', () => {
            stats.innerHTML = model.get('stats_html');
        });

        // Chart image
        const img = document.createElement('img');
        img.src = 'data:image/png;base64,' + model.get('chart_base64');
        img.style.maxWidth = '100%';
        model.on('change:chart_base64', () => {
            img.src = 'data:image/png;base64,' + model.get('chart_base64');
        });
        c.appendChild(img);

        el.appendChild(c);
    }
    export default { render };
    """

    # Synced traits
    distribution = traitlets.Unicode("Normal").tag(sync=True)
    sample_size = traitlets.Int(30).tag(sync=True)
    chart_base64 = traitlets.Unicode("").tag(sync=True)
    stats_html = traitlets.Unicode("").tag(sync=True)
    distribution_options = traitlets.Unicode("").tag(sync=True)

    _DISTRIBUTIONS = ["Normal", "Heavy-tailed (t, df=3)", "Right-skewed (Exponential)", "Bimodal"]

    def __init__(self, n_sims: int = 1000, pop_size: int = 10_000, seed: int = 42, **kwargs):
        super().__init__(**kwargs)

        import json
        self.distribution_options = json.dumps(self._DISTRIBUTIONS)

        self._n_sims = n_sims
        self._pop_size = pop_size
        self._seed = seed

        self._fig, self._axes = plt.subplots(1, 2, figsize=(10, 4))

        self._render()
        self.observe(self._on_change, names=['distribution', 'sample_size'])

    def _generate_pop(self, name: str) -> tuple[np.ndarray, str]:
        """Generate population data for the given distribution name."""
        rng = np.random.default_rng(self._seed)
        size = self._pop_size
        if name == "Normal":
            data = rng.normal(loc=50, scale=15, size=size)
            return data, "Normal(50, 15)"
        elif name == "Heavy-tailed (t, df=3)":
            data = rng.standard_t(df=3, size=size) * 15 + 50
            return data, "t(df=3), centered at 50"
        elif name == "Right-skewed (Exponential)":
            data = rng.exponential(scale=20, size=size)
            return data, "Exponential(λ=1/20)"
        elif name == "Bimodal":
            half = size // 2
            data = np.concatenate([
                rng.normal(loc=30, scale=8, size=half),
                rng.normal(loc=70, scale=8, size=size - half),
            ])
            return data, "Bimodal(30 & 70)"
        else:
            raise ValueError(f"Unknown distribution: {name}")

    def _on_change(self, change):
        """Re-render when controls change."""
        self._render()

    def _render(self):
        """Run simulation and render comparison chart."""
        ax_pop, ax_bar = self._axes
        ax_pop.clear()
        ax_bar.clear()

        # Generate population
        pop_data, pop_label = self._generate_pop(self.distribution)
        true_center = float(np.mean(pop_data))

        # Vectorized simulation
        rng = np.random.default_rng(self._seed + 1)
        samples = rng.choice(pop_data, size=(self._n_sims, self.sample_size), replace=True)
        mean_estimates = samples.mean(axis=1)
        median_estimates = np.median(samples, axis=1)

        mean_rmse = float(np.sqrt(np.mean((mean_estimates - true_center) ** 2)))
        median_rmse = float(np.sqrt(np.mean((median_estimates - true_center) ** 2)))

        # Left: Population histogram
        ax_pop.hist(pop_data, bins=40, density=True, alpha=0.7,
                    color='steelblue', edgecolor='white')
        ax_pop.axvline(true_center, color='red', linestyle='--', linewidth=2,
                       label=f'Mean = {true_center:.1f}')
        ax_pop.axvline(float(np.median(pop_data)), color='orange', linestyle='--',
                       linewidth=2, label=f'Median = {np.median(pop_data):.1f}')
        ax_pop.set_xlabel('Value', fontsize=10)
        ax_pop.set_ylabel('Density', fontsize=10)
        ax_pop.set_title(f'Population: {pop_label}', fontsize=11)
        ax_pop.legend(fontsize=9)

        # Right: RMSE comparison bars
        colors = ['#1f77b4', '#ff7f0e']
        winner_idx = 0 if mean_rmse < median_rmse else 1
        edge_colors = ['gold' if i == winner_idx else 'white' for i in range(2)]
        edge_widths = [3 if i == winner_idx else 1 for i in range(2)]

        bars = ax_bar.bar(
            ['Mean', 'Median'], [mean_rmse, median_rmse],
            color=colors, alpha=0.8,
            edgecolor=edge_colors, linewidth=edge_widths,
        )

        for bar, val in zip(bars, [mean_rmse, median_rmse]):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold',
            )

        winner = "Mean" if mean_rmse < median_rmse else "Median"
        ax_bar.set_ylabel('RMSE (lower = better)', fontsize=10)
        ax_bar.set_title(f'Winner: {winner} ★', fontsize=11)

        self._fig.tight_layout()
        self.chart_base64 = fig_to_base64(self._fig)

        # Stats
        efficiency = min(mean_rmse, median_rmse) / max(mean_rmse, median_rmse)
        self.stats_html = (
            f"<b>Distribution</b>: {pop_label} | "
            f"<b>n</b> = {self.sample_size} | "
            f"<b>1000 simulations</b> | "
            f"Mean RMSE: <b>{mean_rmse:.3f}</b> | "
            f"Median RMSE: <b>{median_rmse:.3f}</b> | "
            f"Winner: <b>{winner}</b> (efficiency: {efficiency:.1%})"
        )

    def __del__(self):
        plt.close(self._fig)


def nfl_explorer(
    n_sims: int = 1000, pop_size: int = 10_000, seed: int = 42,
) -> NFLExplorer:
    """
    Create an interactive No Free Lunch explorer.

    Switch between distributions and sample sizes to see which estimator
    (mean vs median) has lower RMSE. The winner depends on the data shape.

    Parameters
    ----------
    n_sims : int
        Number of simulations per comparison. Default: 1000
    pop_size : int
        Size of the synthetic population. Default: 10,000
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    NFLExplorer widget
    """
    return NFLExplorer(n_sims=n_sims, pop_size=pop_size, seed=seed)


# =============================================================================
# Monte Carlo Simulation Explorer
# =============================================================================

class MCExplorer(anywidget.AnyWidget):
    """
    Interactive explorer for Monte Carlo simulation.

    Assume data comes from Normal(μ, σ), generate synthetic samples,
    and build a sampling distribution. Watch the SE formula come alive.
    """

    _esm = """
    function render({ model, el }) {
        const c = document.createElement('div');
        c.style.fontFamily = 'system-ui, -apple-system, sans-serif';
        c.style.maxWidth = '900px';

        const title = document.createElement('div');
        title.innerHTML = '<strong>Monte Carlo Simulation</strong>: Assume a distribution → simulate → build sampling distribution';
        title.style.marginBottom = '12px';
        title.style.fontSize = '14px';
        c.appendChild(title);

        // --- Row 1: μ, σ, n sliders ---
        const row1 = document.createElement('div');
        row1.style.display = 'flex';
        row1.style.alignItems = 'center';
        row1.style.gap = '16px';
        row1.style.marginBottom = '8px';
        row1.style.flexWrap = 'wrap';

        function addSlider(parent, label, trait, min, max, step, isInt) {
            const g = document.createElement('div');
            g.style.display = 'flex';
            g.style.alignItems = 'center';
            g.style.gap = '6px';
            const l = document.createElement('span');
            l.textContent = label;
            l.style.fontSize = '13px';
            const s = document.createElement('input');
            s.type = 'range';
            s.min = min; s.max = max; s.step = step;
            s.value = model.get(trait);
            s.style.width = '120px';
            s.style.cursor = 'pointer';
            const v = document.createElement('span');
            v.textContent = model.get(trait);
            v.style.fontFamily = 'monospace';
            v.style.fontSize = '13px';
            v.style.minWidth = '30px';
            s.addEventListener('input', (e) => {
                const val = isInt ? parseInt(e.target.value) : parseFloat(e.target.value);
                model.set(trait, val);
                model.save_changes();
                v.textContent = val;
            });
            g.appendChild(l); g.appendChild(s); g.appendChild(v);
            parent.appendChild(g);
        }

        addSlider(row1, 'Assumed μ:', 'mu', '30', '70', '1', false);
        addSlider(row1, 'Assumed σ:', 'sigma', '5', '25', '1', false);
        addSlider(row1, 'Sample size (n):', 'n', '10', '200', '5', true);
        c.appendChild(row1);

        // --- Row 2: batch + buttons ---
        const row2 = document.createElement('div');
        row2.style.display = 'flex';
        row2.style.alignItems = 'center';
        row2.style.gap = '16px';
        row2.style.marginBottom = '8px';
        row2.style.flexWrap = 'wrap';

        addSlider(row2, 'Per click:', 'batch_size', '10', '200', '10', true);

        const runBtn = document.createElement('button');
        runBtn.textContent = '▶ Simulate';
        runBtn.style.cssText = 'font-size:13px;padding:4px 14px;cursor:pointer;background:#4a90d9;color:white;border:none;border-radius:4px';
        runBtn.addEventListener('click', () => {
            model.set('run_trigger', model.get('run_trigger') + 1);
            model.save_changes();
        });
        row2.appendChild(runBtn);

        const resetBtn = document.createElement('button');
        resetBtn.textContent = '↺ Reset';
        resetBtn.style.cssText = 'font-size:13px;padding:4px 10px;cursor:pointer;border:1px solid #ccc;border-radius:4px;background:#f5f5f5';
        resetBtn.addEventListener('click', () => {
            model.set('reset_trigger', model.get('reset_trigger') + 1);
            model.save_changes();
        });
        row2.appendChild(resetBtn);

        c.appendChild(row2);

        // Stats
        const stats = document.createElement('div');
        stats.style.cssText = 'font-family:monospace;font-size:13px;margin-bottom:12px;padding:8px;background:#f5f5f5;border-radius:4px';
        stats.innerHTML = model.get('stats_html');
        c.appendChild(stats);
        model.on('change:stats_html', () => { stats.innerHTML = model.get('stats_html'); });

        // Chart
        const img = document.createElement('img');
        img.src = 'data:image/png;base64,' + model.get('chart_base64');
        img.style.maxWidth = '100%';
        model.on('change:chart_base64', () => { img.src = 'data:image/png;base64,' + model.get('chart_base64'); });
        c.appendChild(img);

        el.appendChild(c);
    }
    export default { render };
    """

    # Synced traits
    mu = traitlets.Float(50.0).tag(sync=True)
    sigma = traitlets.Float(15.0).tag(sync=True)
    n = traitlets.Int(30).tag(sync=True)
    batch_size = traitlets.Int(50).tag(sync=True)
    run_trigger = traitlets.Int(0).tag(sync=True)
    reset_trigger = traitlets.Int(0).tag(sync=True)
    chart_base64 = traitlets.Unicode("").tag(sync=True)
    stats_html = traitlets.Unicode("").tag(sync=True)

    def __init__(self, seed: int = 42, **kwargs):
        super().__init__(**kwargs)

        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._accumulated: list[float] = []

        self._fig, self._axes = plt.subplots(1, 2, figsize=(10, 4))

        self._render()
        self.observe(self._on_run, names=['run_trigger'])
        self.observe(self._on_reset, names=['reset_trigger'])
        self.observe(self._on_settings_change, names=['mu', 'sigma', 'n'])

    def _on_run(self, change):
        """Add a batch of simulated means."""
        for _ in range(self.batch_size):
            sample = self._rng.normal(loc=self.mu, scale=self.sigma, size=self.n)
            self._accumulated.append(float(sample.mean()))
        self._render()

    def _on_reset(self, change):
        """Clear and re-render."""
        self._rng = np.random.default_rng(self._seed)
        self._accumulated = []
        self._render()

    def _on_settings_change(self, change):
        """Reset when parameters change."""
        self._rng = np.random.default_rng(self._seed)
        self._accumulated = []
        self._render()

    def _render(self):
        """Render the MC simulation visualization."""
        ax_dist, ax_samp = self._axes
        ax_dist.clear()
        ax_samp.clear()

        # Left: Assumed Normal distribution curve
        x = np.linspace(self.mu - 4 * self.sigma, self.mu + 4 * self.sigma, 200)
        y = (1 / (self.sigma * np.sqrt(2 * np.pi))) * \
            np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)
        ax_dist.plot(x, y, 'steelblue', linewidth=2,
                     label=f'Normal(μ={self.mu:.0f}, σ={self.sigma:.0f})')
        ax_dist.fill_between(x, y, alpha=0.2, color='steelblue')
        ax_dist.axvline(self.mu, color='red', linestyle='--', linewidth=1.5,
                        label=f'μ = {self.mu:.0f}')
        ax_dist.set_xlabel('Value', fontsize=10)
        ax_dist.set_ylabel('Density', fontsize=10)
        ax_dist.set_title(f'Assumed: Normal(μ={self.mu:.0f}, σ={self.sigma:.0f})', fontsize=11)
        ax_dist.legend(fontsize=9)

        # Right: Sampling distribution
        total = len(self._accumulated)
        if total == 0:
            ax_samp.text(
                0.5, 0.5, 'Click "Simulate"\nto start building\nthe sampling distribution',
                transform=ax_samp.transAxes, ha='center', va='center',
                fontsize=14, color='#888',
            )
            ax_samp.set_title('Sampling Distribution of the Mean', fontsize=11)
        else:
            stats_array = np.array(self._accumulated)
            ax_samp.hist(stats_array, bins=min(40, max(10, total // 5)),
                         density=True, alpha=0.7, color='coral', edgecolor='white')

            sim_mean = float(np.mean(stats_array))
            sim_std = float(np.std(stats_array))
            if sim_std > 0:
                x_norm = np.linspace(sim_mean - 4 * sim_std, sim_mean + 4 * sim_std, 100)
                y_norm = (1 / (sim_std * np.sqrt(2 * np.pi))) * \
                         np.exp(-0.5 * ((x_norm - sim_mean) / sim_std) ** 2)
                ax_samp.plot(x_norm, y_norm, 'k-', linewidth=2, alpha=0.6,
                             label='Normal fit')

            ax_samp.axvline(self.mu, color='red', linestyle='--', linewidth=2,
                            label=f'True μ = {self.mu:.0f}')
            ax_samp.set_title(f'Sampling Distribution ({total} sims, n={self.n})',
                              fontsize=11)
            ax_samp.legend(fontsize=9)

        ax_samp.set_xlabel('Sample Mean', fontsize=10)
        ax_samp.set_ylabel('Density', fontsize=10)

        self._fig.tight_layout()
        self.chart_base64 = fig_to_base64(self._fig)

        # Stats
        theoretical_se = self.sigma / np.sqrt(self.n)
        if total == 0:
            self.stats_html = (
                f"<b>Assumed</b>: Normal(μ={self.mu:.0f}, σ={self.sigma:.0f}) | "
                f"<b>n</b> = {self.n} | "
                f"<b>Theoretical SE</b> (σ/√n): <b>{theoretical_se:.3f}</b> | "
                "Simulations: <b>0</b> — click <b>Simulate</b> to begin"
            )
        else:
            sim_se = float(np.std(self._accumulated))
            self.stats_html = (
                f"<b>Assumed</b>: Normal(μ={self.mu:.0f}, σ={self.sigma:.0f}) | "
                f"<b>n</b> = {self.n} | "
                f"Sims: <b>{total}</b> | "
                f"Theoretical SE (σ/√n): <b>{theoretical_se:.3f}</b> | "
                f"Simulated SE: <b>{sim_se:.3f}</b>"
            )

    def __del__(self):
        plt.close(self._fig)


def mc_explorer(seed: int = 42) -> MCExplorer:
    """
    Create an interactive Monte Carlo simulation explorer.

    Set assumed μ, σ, and sample size, then click "Simulate" to draw
    synthetic samples and build a sampling distribution of the mean.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    MCExplorer widget
    """
    return MCExplorer(seed=seed)


# =============================================================================
# Bootstrap Explorer
# =============================================================================

class BootstrapExplorer(anywidget.AnyWidget):
    """
    Interactive explorer for bootstrap resampling.

    See how resampling WITH replacement from observed data builds a
    sampling distribution — no distributional assumptions needed.
    The left panel highlights which observations were selected in the
    last resample.
    """

    _esm = """
    function render({ model, el }) {
        const c = document.createElement('div');
        c.style.fontFamily = 'system-ui, -apple-system, sans-serif';
        c.style.maxWidth = '900px';

        const title = document.createElement('div');
        title.innerHTML = '<strong>Bootstrap Resampling</strong>: Resample with replacement to estimate uncertainty';
        title.style.marginBottom = '12px';
        title.style.fontSize = '14px';
        c.appendChild(title);

        // Controls row
        const row = document.createElement('div');
        row.style.display = 'flex';
        row.style.alignItems = 'center';
        row.style.gap = '16px';
        row.style.marginBottom = '8px';
        row.style.flexWrap = 'wrap';

        // Batch size slider
        const batchGroup = document.createElement('div');
        batchGroup.style.display = 'flex';
        batchGroup.style.alignItems = 'center';
        batchGroup.style.gap = '6px';
        const batchLabel = document.createElement('span');
        batchLabel.textContent = 'Per click:';
        batchLabel.style.fontSize = '13px';
        const batchSlider = document.createElement('input');
        batchSlider.type = 'range';
        batchSlider.min = '10';
        batchSlider.max = '200';
        batchSlider.step = '10';
        batchSlider.value = model.get('batch_size');
        batchSlider.style.width = '100px';
        batchSlider.style.cursor = 'pointer';
        const batchValue = document.createElement('span');
        batchValue.textContent = model.get('batch_size');
        batchValue.style.fontFamily = 'monospace';
        batchValue.style.fontSize = '13px';
        batchValue.style.minWidth = '30px';
        batchSlider.addEventListener('input', (e) => {
            const val = parseInt(e.target.value);
            model.set('batch_size', val);
            model.save_changes();
            batchValue.textContent = val;
        });
        batchGroup.appendChild(batchLabel);
        batchGroup.appendChild(batchSlider);
        batchGroup.appendChild(batchValue);
        row.appendChild(batchGroup);

        // Resample button
        const runBtn = document.createElement('button');
        runBtn.textContent = '▶ Resample';
        runBtn.style.cssText = 'font-size:13px;padding:4px 14px;cursor:pointer;background:#7b4dba;color:white;border:none;border-radius:4px';
        runBtn.addEventListener('click', () => {
            model.set('run_trigger', model.get('run_trigger') + 1);
            model.save_changes();
        });
        row.appendChild(runBtn);

        // Reset button
        const resetBtn = document.createElement('button');
        resetBtn.textContent = '↺ Reset';
        resetBtn.style.cssText = 'font-size:13px;padding:4px 10px;cursor:pointer;border:1px solid #ccc;border-radius:4px;background:#f5f5f5';
        resetBtn.addEventListener('click', () => {
            model.set('reset_trigger', model.get('reset_trigger') + 1);
            model.save_changes();
        });
        row.appendChild(resetBtn);

        c.appendChild(row);

        // Stats
        const stats = document.createElement('div');
        stats.style.cssText = 'font-family:monospace;font-size:13px;margin-bottom:12px;padding:8px;background:#f5f5f5;border-radius:4px';
        stats.innerHTML = model.get('stats_html');
        c.appendChild(stats);
        model.on('change:stats_html', () => { stats.innerHTML = model.get('stats_html'); });

        // Chart
        const img = document.createElement('img');
        img.src = 'data:image/png;base64,' + model.get('chart_base64');
        img.style.maxWidth = '100%';
        model.on('change:chart_base64', () => { img.src = 'data:image/png;base64,' + model.get('chart_base64'); });
        c.appendChild(img);

        el.appendChild(c);
    }
    export default { render };
    """

    # Synced traits
    batch_size = traitlets.Int(50).tag(sync=True)
    run_trigger = traitlets.Int(0).tag(sync=True)
    reset_trigger = traitlets.Int(0).tag(sync=True)
    chart_base64 = traitlets.Unicode("").tag(sync=True)
    stats_html = traitlets.Unicode("").tag(sync=True)

    def __init__(self, data_size: int = 25, seed: int = 42, **kwargs):
        super().__init__(**kwargs)

        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._data_size = data_size

        # Generate fixed toy dataset
        self._sample_data = self._rng.normal(loc=50, scale=12, size=data_size)
        self._sample_mean = float(np.mean(self._sample_data))

        self._accumulated: list[float] = []
        self._last_counts: np.ndarray | None = None

        self._fig, self._axes = plt.subplots(1, 2, figsize=(10, 4))

        self._render()
        self.observe(self._on_run, names=['run_trigger'])
        self.observe(self._on_reset, names=['reset_trigger'])

    def _on_run(self, change):
        """Add a batch of bootstrap resamples."""
        for _ in range(self.batch_size):
            indices = self._rng.choice(self._data_size, size=self._data_size, replace=True)
            boot_sample = self._sample_data[indices]
            self._accumulated.append(float(boot_sample.mean()))
            self._last_counts = np.bincount(indices, minlength=self._data_size)
        self._render()

    def _on_reset(self, change):
        """Clear and regenerate fresh sample."""
        self._rng = np.random.default_rng(self._seed)
        self._sample_data = self._rng.normal(loc=50, scale=12, size=self._data_size)
        self._sample_mean = float(np.mean(self._sample_data))
        self._accumulated = []
        self._last_counts = None
        self._render()

    def _render(self):
        """Render the bootstrap visualization."""
        ax_data, ax_boot = self._axes
        ax_data.clear()
        ax_boot.clear()

        # Left: Original sample as strip plot with resample highlighting
        y_pos = np.zeros(self._data_size)

        if self._last_counts is not None:
            colors = []
            for count in self._last_counts:
                if count == 0:
                    colors.append('#cccccc')
                elif count == 1:
                    colors.append('#1f77b4')
                else:
                    colors.append('#d62728')

            sizes = [max(40, int(count * 60)) for count in self._last_counts]
            ax_data.scatter(self._sample_data, y_pos, c=colors, s=sizes,
                            edgecolors='white', linewidths=0.5, zorder=3)

            n_not_picked = int(np.sum(self._last_counts == 0))
            n_duplicated = int(np.sum(self._last_counts >= 2))
            ax_data.set_title(
                f'Last resample: {n_not_picked} skipped, {n_duplicated} duplicated',
                fontsize=11,
            )

            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#cccccc',
                       markersize=8, label='Not picked'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
                       markersize=8, label='Picked once'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
                       markersize=10, label='Picked 2+ times'),
            ]
            ax_data.legend(handles=legend_elements, fontsize=8, loc='upper right')
        else:
            ax_data.scatter(self._sample_data, y_pos, c='steelblue', s=60,
                            edgecolors='white', linewidths=0.5, zorder=3)
            ax_data.set_title(f'Original sample (n={self._data_size})', fontsize=11)

        ax_data.axvline(self._sample_mean, color='red', linestyle='--', linewidth=1.5,
                        label=f'Mean = {self._sample_mean:.1f}')
        ax_data.set_xlabel('Value', fontsize=10)
        ax_data.set_yticks([])
        ax_data.legend(fontsize=9, loc='upper left')

        # Right: Bootstrap distribution
        total = len(self._accumulated)
        if total == 0:
            ax_boot.text(
                0.5, 0.5, 'Click "Resample"\nto start building\nthe bootstrap distribution',
                transform=ax_boot.transAxes, ha='center', va='center',
                fontsize=14, color='#888',
            )
            ax_boot.set_title('Bootstrap Distribution of the Mean', fontsize=11)
        else:
            stats_array = np.array(self._accumulated)
            ax_boot.hist(stats_array, bins=min(40, max(10, total // 5)),
                         density=True, alpha=0.7, color='mediumpurple', edgecolor='white')

            ci_low = float(np.percentile(stats_array, 2.5))
            ci_high = float(np.percentile(stats_array, 97.5))

            ax_boot.axvline(self._sample_mean, color='red', linestyle='--', linewidth=2,
                            label=f'Sample mean = {self._sample_mean:.1f}')
            ax_boot.axvspan(ci_low, ci_high, alpha=0.2, color='green', label='95% CI')
            ax_boot.set_title(f'Bootstrap Distribution ({total} resamples)', fontsize=11)
            ax_boot.legend(fontsize=9)

        ax_boot.set_xlabel('Bootstrap Mean', fontsize=10)
        ax_boot.set_ylabel('Density', fontsize=10)

        self._fig.tight_layout()
        self.chart_base64 = fig_to_base64(self._fig)

        # Stats
        if total == 0:
            self.stats_html = (
                f"<b>Sample</b>: {self._data_size} observations, "
                f"mean = {self._sample_mean:.2f} | "
                "Resamples: <b>0</b> — click <b>Resample</b> to begin"
            )
        else:
            boot_se = float(np.std(self._accumulated))
            ci_low = float(np.percentile(self._accumulated, 2.5))
            ci_high = float(np.percentile(self._accumulated, 97.5))
            self.stats_html = (
                f"<b>Sample</b>: {self._data_size} observations | "
                f"Resamples: <b>{total}</b> | "
                f"Bootstrap SE: <b>{boot_se:.3f}</b> | "
                f"95% CI: [<b>{ci_low:.2f}</b>, <b>{ci_high:.2f}</b>]"
            )

    def __del__(self):
        plt.close(self._fig)


def bootstrap_explorer(data_size: int = 25, seed: int = 42) -> BootstrapExplorer:
    """
    Create an interactive bootstrap resampling explorer.

    Click "Resample" to draw bootstrap samples with replacement and build
    a bootstrap distribution. The left panel shows which observations were
    selected (and how many times) in the last resample.

    Parameters
    ----------
    data_size : int
        Number of observations in the toy dataset. Default: 25
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    BootstrapExplorer widget
    """
    return BootstrapExplorer(data_size=data_size, seed=seed)


# =============================================================================
# Permutation Explorer
# =============================================================================

class PermutationExplorer(anywidget.AnyWidget):
    """
    Interactive explorer for permutation tests.

    Control the effect size and sample size to generate two groups, then
    shuffle labels to build a null distribution. Watch the p-value change
    as the null distribution grows.
    """

    _esm = """
    function render({ model, el }) {
        const c = document.createElement('div');
        c.style.fontFamily = 'system-ui, -apple-system, sans-serif';
        c.style.maxWidth = '900px';

        const title = document.createElement('div');
        title.innerHTML = '<strong>Permutation Test</strong>: Shuffle labels to build a null distribution';
        title.style.marginBottom = '12px';
        title.style.fontSize = '14px';
        c.appendChild(title);

        // --- Row 1: effect size + sample size sliders ---
        const row1 = document.createElement('div');
        row1.style.display = 'flex';
        row1.style.alignItems = 'center';
        row1.style.gap = '16px';
        row1.style.marginBottom = '8px';
        row1.style.flexWrap = 'wrap';

        function addSlider(parent, label, trait, min, max, step, isInt) {
            const g = document.createElement('div');
            g.style.display = 'flex';
            g.style.alignItems = 'center';
            g.style.gap = '6px';
            const l = document.createElement('span');
            l.textContent = label;
            l.style.fontSize = '13px';
            const s = document.createElement('input');
            s.type = 'range';
            s.min = min; s.max = max; s.step = step;
            s.value = model.get(trait);
            s.style.width = '120px';
            s.style.cursor = 'pointer';
            const v = document.createElement('span');
            v.textContent = model.get(trait);
            v.style.fontFamily = 'monospace';
            v.style.fontSize = '13px';
            v.style.minWidth = '30px';
            s.addEventListener('input', (e) => {
                const val = isInt ? parseInt(e.target.value) : parseFloat(e.target.value);
                model.set(trait, val);
                model.save_changes();
                v.textContent = val;
            });
            g.appendChild(l); g.appendChild(s); g.appendChild(v);
            parent.appendChild(g);
        }

        addSlider(row1, 'Effect size (d):', 'effect_size', '0', '2', '0.25', false);
        addSlider(row1, 'n per group:', 'group_n', '20', '100', '10', true);
        c.appendChild(row1);

        // --- Row 2: batch + buttons ---
        const row2 = document.createElement('div');
        row2.style.display = 'flex';
        row2.style.alignItems = 'center';
        row2.style.gap = '16px';
        row2.style.marginBottom = '8px';
        row2.style.flexWrap = 'wrap';

        addSlider(row2, 'Per click:', 'batch_size', '100', '1000', '100', true);

        const runBtn = document.createElement('button');
        runBtn.textContent = '▶ Shuffle';
        runBtn.style.cssText = 'font-size:13px;padding:4px 14px;cursor:pointer;background:#d9534f;color:white;border:none;border-radius:4px';
        runBtn.addEventListener('click', () => {
            model.set('run_trigger', model.get('run_trigger') + 1);
            model.save_changes();
        });
        row2.appendChild(runBtn);

        const resetBtn = document.createElement('button');
        resetBtn.textContent = '↺ Reset';
        resetBtn.style.cssText = 'font-size:13px;padding:4px 10px;cursor:pointer;border:1px solid #ccc;border-radius:4px;background:#f5f5f5';
        resetBtn.addEventListener('click', () => {
            model.set('reset_trigger', model.get('reset_trigger') + 1);
            model.save_changes();
        });
        row2.appendChild(resetBtn);

        c.appendChild(row2);

        // Stats
        const stats = document.createElement('div');
        stats.style.cssText = 'font-family:monospace;font-size:13px;margin-bottom:12px;padding:8px;background:#f5f5f5;border-radius:4px';
        stats.innerHTML = model.get('stats_html');
        c.appendChild(stats);
        model.on('change:stats_html', () => { stats.innerHTML = model.get('stats_html'); });

        // Chart
        const img = document.createElement('img');
        img.src = 'data:image/png;base64,' + model.get('chart_base64');
        img.style.maxWidth = '100%';
        model.on('change:chart_base64', () => { img.src = 'data:image/png;base64,' + model.get('chart_base64'); });
        c.appendChild(img);

        el.appendChild(c);
    }
    export default { render };
    """

    # Synced traits
    effect_size = traitlets.Float(1.0).tag(sync=True)
    group_n = traitlets.Int(40).tag(sync=True)
    batch_size = traitlets.Int(200).tag(sync=True)
    run_trigger = traitlets.Int(0).tag(sync=True)
    reset_trigger = traitlets.Int(0).tag(sync=True)
    chart_base64 = traitlets.Unicode("").tag(sync=True)
    stats_html = traitlets.Unicode("").tag(sync=True)

    def __init__(self, seed: int = 42, **kwargs):
        super().__init__(**kwargs)

        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._accumulated: list[float] = []

        self._generate_data()
        self._fig, self._axes = plt.subplots(1, 2, figsize=(10, 4))

        self._render()
        self.observe(self._on_run, names=['run_trigger'])
        self.observe(self._on_reset, names=['reset_trigger'])
        self.observe(self._on_settings_change, names=['effect_size', 'group_n'])

    def _generate_data(self):
        """Generate two groups with the specified effect size."""
        rng = np.random.default_rng(self._seed)
        sigma = 15.0
        self._group_a = rng.normal(loc=50, scale=sigma, size=self.group_n)
        self._group_b = rng.normal(
            loc=50 + self.effect_size * sigma, scale=sigma, size=self.group_n,
        )
        self._observed_diff = float(self._group_b.mean() - self._group_a.mean())
        self._combined = np.concatenate([self._group_a, self._group_b])

    def _on_run(self, change):
        """Add a batch of shuffled differences."""
        n_a = len(self._group_a)
        for _ in range(self.batch_size):
            shuffled = self._rng.permutation(self._combined)
            perm_diff = float(shuffled[n_a:].mean() - shuffled[:n_a].mean())
            self._accumulated.append(perm_diff)
        self._render()

    def _on_reset(self, change):
        """Clear and re-render."""
        self._rng = np.random.default_rng(self._seed)
        self._accumulated = []
        self._generate_data()
        self._render()

    def _on_settings_change(self, change):
        """Regenerate data and reset when parameters change."""
        self._rng = np.random.default_rng(self._seed)
        self._accumulated = []
        self._generate_data()
        self._render()

    def _render(self):
        """Render the permutation test visualization."""
        ax_groups, ax_null = self._axes
        ax_groups.clear()
        ax_null.clear()

        # Left: Two overlapping histograms
        bins = np.linspace(
            min(self._group_a.min(), self._group_b.min()) - 5,
            max(self._group_a.max(), self._group_b.max()) + 5,
            30,
        )
        ax_groups.hist(self._group_a, bins=bins, alpha=0.6, color='#1f77b4',
                       edgecolor='white', label=f'Group A (μ={self._group_a.mean():.1f})')
        ax_groups.hist(self._group_b, bins=bins, alpha=0.6, color='#ff7f0e',
                       edgecolor='white', label=f'Group B (μ={self._group_b.mean():.1f})')
        ax_groups.axvline(self._group_a.mean(), color='#1f77b4', ls='--', lw=2)
        ax_groups.axvline(self._group_b.mean(), color='#ff7f0e', ls='--', lw=2)
        ax_groups.set_xlabel('Value', fontsize=10)
        ax_groups.set_ylabel('Count', fontsize=10)
        ax_groups.set_title(f'Two Groups (n={self.group_n} each, d={self.effect_size:.2f})',
                            fontsize=11)
        ax_groups.legend(fontsize=9)

        # Right: Null distribution
        total = len(self._accumulated)
        if total == 0:
            ax_null.text(
                0.5, 0.5, 'Click "Shuffle"\nto start building\nthe null distribution',
                transform=ax_null.transAxes, ha='center', va='center',
                fontsize=14, color='#888',
            )
            ax_null.set_title('Null Distribution', fontsize=11)
        else:
            perm_array = np.array(self._accumulated)
            ax_null.hist(perm_array, bins=min(50, max(15, total // 10)),
                         alpha=0.7, color='coral', edgecolor='white',
                         label='Shuffled diffs')

            ax_null.axvline(self._observed_diff, color='red', ls='--', lw=2,
                            label=f'Observed: {self._observed_diff:.2f}')
            ax_null.axvline(-self._observed_diff, color='red', ls=':', lw=1.5,
                            alpha=0.6, label=f'Mirror: {-self._observed_diff:.2f}')

            p_value = float(np.mean(np.abs(perm_array) >= abs(self._observed_diff)))
            ax_null.set_title(f'Null Distribution ({total} shuffles, p={p_value:.3f})',
                              fontsize=11)
            ax_null.legend(fontsize=9)

        ax_null.set_xlabel('Difference (B − A)', fontsize=10)
        ax_null.set_ylabel('Count', fontsize=10)

        self._fig.tight_layout()
        self.chart_base64 = fig_to_base64(self._fig)

        # Stats
        total = len(self._accumulated)
        if total == 0:
            self.stats_html = (
                f"<b>Effect size</b>: d={self.effect_size:.2f} | "
                f"<b>n</b> = {self.group_n} per group | "
                f"<b>Observed diff</b>: {self._observed_diff:.2f} | "
                "Shuffles: <b>0</b> — click <b>Shuffle</b> to begin"
            )
        else:
            perm_array = np.array(self._accumulated)
            p_value = float(np.mean(np.abs(perm_array) >= abs(self._observed_diff)))
            self.stats_html = (
                f"<b>Effect size</b>: d={self.effect_size:.2f} | "
                f"<b>n</b> = {self.group_n} per group | "
                f"<b>Observed diff</b>: {self._observed_diff:.2f} | "
                f"Shuffles: <b>{total}</b> | "
                f"<b>p-value</b> (two-tailed): <b>{p_value:.4f}</b>"
            )

    def __del__(self):
        plt.close(self._fig)


def permutation_explorer(seed: int = 42) -> PermutationExplorer:
    """
    Create an interactive permutation test explorer.

    Control the effect size to generate two groups, then click "Shuffle"
    to build a null distribution by randomly reassigning group labels.
    Watch the p-value update as the null distribution grows.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    PermutationExplorer widget
    """
    return PermutationExplorer(seed=seed)


# =============================================================================
# Cross-Validation Explorer (Bias-Variance Decomposition)
# =============================================================================

class CVExplorer(anywidget.AnyWidget):
    """
    Interactive explorer showing the classic bias-variance tradeoff.

    Generates synthetic data from a known function y = f(x) + noise, fits
    polynomials of increasing degree, and decomposes the test error into
    Bias², Variance, and Irreducible Error. Shows the characteristic U-shape
    where low complexity = high bias (underfitting) and high complexity =
    high variance (overfitting).
    """

    _esm = """
    function render({ model, el }) {
        const c = document.createElement('div');
        c.style.fontFamily = 'system-ui, -apple-system, sans-serif';
        c.style.maxWidth = '900px';

        const title = document.createElement('div');
        title.innerHTML = '<strong>Bias-Variance Tradeoff</strong>: MSE = Bias² + Variance + Irreducible Error';
        title.style.marginBottom = '12px';
        title.style.fontSize = '14px';
        c.appendChild(title);

        // --- Controls row ---
        const row = document.createElement('div');
        row.style.display = 'flex';
        row.style.alignItems = 'center';
        row.style.gap = '16px';
        row.style.marginBottom = '8px';
        row.style.flexWrap = 'wrap';

        function addSlider(parent, label, trait, min, max, step, suffix) {
            const g = document.createElement('div');
            g.style.display = 'flex';
            g.style.alignItems = 'center';
            g.style.gap = '6px';
            const l = document.createElement('span');
            l.textContent = label;
            l.style.fontSize = '13px';
            const s = document.createElement('input');
            s.type = 'range';
            s.min = min; s.max = max; s.step = step;
            s.value = model.get(trait);
            s.style.width = '100px';
            s.style.cursor = 'pointer';
            const v = document.createElement('span');
            v.textContent = model.get(trait) + (suffix || '');
            v.style.fontFamily = 'monospace';
            v.style.fontSize = '13px';
            v.style.minWidth = '40px';
            s.addEventListener('input', (e) => {
                const val = parseFloat(e.target.value);
                model.set(trait, val);
                model.save_changes();
                v.textContent = val + (suffix || '');
            });
            g.appendChild(l); g.appendChild(s); g.appendChild(v);
            parent.appendChild(g);
        }

        addSlider(row, 'Noise (σ):', 'noise_std', '0.1', '1.0', '0.1', '');
        addSlider(row, 'Sample size:', 'n_samples', '30', '200', '10', '');
        addSlider(row, 'Simulations:', 'n_sims', '50', '300', '50', '');

        c.appendChild(row);

        // Stats
        const stats = document.createElement('div');
        stats.style.cssText = 'font-family:monospace;font-size:13px;margin-bottom:12px;padding:8px;background:#f5f5f5;border-radius:4px';
        stats.innerHTML = model.get('stats_html');
        c.appendChild(stats);
        model.on('change:stats_html', () => { stats.innerHTML = model.get('stats_html'); });

        // Chart
        const img = document.createElement('img');
        img.src = 'data:image/png;base64,' + model.get('chart_base64');
        img.style.maxWidth = '100%';
        model.on('change:chart_base64', () => { img.src = 'data:image/png;base64,' + model.get('chart_base64'); });
        c.appendChild(img);

        el.appendChild(c);
    }
    export default { render };
    """

    # Synced traits
    noise_std = traitlets.Float(0.3).tag(sync=True)
    n_samples = traitlets.Int(80).tag(sync=True)
    n_sims = traitlets.Int(100).tag(sync=True)
    chart_base64 = traitlets.Unicode("").tag(sync=True)
    stats_html = traitlets.Unicode("").tag(sync=True)

    _MAX_DEGREE = 12

    def __init__(self, seed: int = 42, **kwargs):
        super().__init__(**kwargs)
        self._seed = seed
        self._fig, self._axes = plt.subplots(1, 2, figsize=(12, 5))
        self._render()
        self.observe(self._on_change, names=['noise_std', 'n_samples', 'n_sims'])

    def _true_function(self, x):
        """The true underlying function: y = sin(2πx)."""
        return np.sin(2 * np.pi * x)

    def _on_change(self, change):
        """Re-render when settings change."""
        self._render()

    def _compute_bias_variance(self):
        """Compute bias², variance, and total error for each polynomial degree."""
        rng = np.random.default_rng(self._seed)
        degrees = list(range(1, self._MAX_DEGREE + 1))

        # Fixed test grid for evaluation
        x_test = np.linspace(0, 1, 50)
        y_true = self._true_function(x_test)

        results = {
            'degree': degrees,
            'bias_sq': [],
            'variance': [],
            'total_mse': [],
            'train_mse': [],
        }

        for d in degrees:
            # Store predictions across simulations
            all_preds = np.zeros((self.n_sims, len(x_test)))
            train_mses = []

            for sim in range(self.n_sims):
                # Generate training data
                x_train = rng.uniform(0, 1, self.n_samples)
                y_train = self._true_function(x_train) + rng.normal(0, self.noise_std, self.n_samples)

                # Fit polynomial
                try:
                    coeffs = np.polyfit(x_train, y_train, d)
                    y_pred_train = np.polyval(coeffs, x_train)
                    y_pred_test = np.polyval(coeffs, x_test)
                except np.RankWarning:
                    y_pred_test = np.zeros_like(x_test)
                    y_pred_train = np.zeros_like(x_train)

                all_preds[sim] = y_pred_test
                train_mses.append(np.mean((y_train - y_pred_train) ** 2))

            # Compute bias² and variance at each test point
            mean_pred = np.mean(all_preds, axis=0)  # E[f_hat(x)]
            bias_sq_per_point = (mean_pred - y_true) ** 2  # (E[f_hat] - f)²
            var_per_point = np.var(all_preds, axis=0)  # Var[f_hat(x)]

            # Average over test points
            avg_bias_sq = np.mean(bias_sq_per_point)
            avg_variance = np.mean(var_per_point)
            avg_total_mse = avg_bias_sq + avg_variance + self.noise_std ** 2

            results['bias_sq'].append(avg_bias_sq)
            results['variance'].append(avg_variance)
            results['total_mse'].append(avg_total_mse)
            results['train_mse'].append(np.mean(train_mses))

        return results

    def _render(self):
        """Render the bias-variance decomposition."""
        ax_decomp, ax_fit = self._axes
        ax_decomp.clear()
        ax_fit.clear()

        r = self._compute_bias_variance()
        degrees = np.array(r['degree'])
        bias_sq = np.array(r['bias_sq'])
        variance = np.array(r['variance'])
        total_mse = np.array(r['total_mse'])
        irreducible = self.noise_std ** 2

        # Left panel: Bias-Variance decomposition
        ax_decomp.plot(degrees, bias_sq, 'o-', color='#1f77b4', lw=2, ms=6,
                       label='Bias²')
        ax_decomp.plot(degrees, variance, 's-', color='#ff7f0e', lw=2, ms=6,
                       label='Variance')
        ax_decomp.plot(degrees, total_mse, '^-', color='#2ca02c', lw=2.5, ms=7,
                       label='Total MSE')
        ax_decomp.axhline(irreducible, color='#888', ls='--', lw=1.5,
                          label=f'Irreducible (σ² = {irreducible:.2f})')

        # Find and mark optimal degree
        optimal_idx = np.argmin(total_mse)
        optimal_degree = degrees[optimal_idx]
        ax_decomp.axvline(optimal_degree, color='#d62728', ls=':', lw=1.5, alpha=0.7)
        ax_decomp.scatter([optimal_degree], [total_mse[optimal_idx]], s=150,
                          color='#d62728', marker='*', zorder=5,
                          label=f'Optimal (d={optimal_degree})')

        # Annotations for regions
        ax_decomp.annotate('← Underfitting\n   (High Bias)',
                           xy=(1.5, ax_decomp.get_ylim()[1] * 0.85),
                           fontsize=10, color='#1f77b4', ha='left', fontweight='bold')
        ax_decomp.annotate('Overfitting →\n(High Variance)',
                           xy=(self._MAX_DEGREE - 0.5, ax_decomp.get_ylim()[1] * 0.85),
                           fontsize=10, color='#ff7f0e', ha='right', fontweight='bold')

        ax_decomp.set_xlabel('Model Complexity (Polynomial Degree)', fontsize=11)
        ax_decomp.set_ylabel('Error (MSE)', fontsize=11)
        ax_decomp.set_title('Bias-Variance Decomposition', fontsize=12)
        ax_decomp.legend(fontsize=9, loc='upper center')
        ax_decomp.set_xticks(degrees)
        ax_decomp.set_xlim(0.5, self._MAX_DEGREE + 0.5)

        # Right panel: Example fits at low, optimal, and high complexity
        rng = np.random.default_rng(self._seed)
        x_train = rng.uniform(0, 1, self.n_samples)
        y_train = self._true_function(x_train) + rng.normal(0, self.noise_std, self.n_samples)
        x_plot = np.linspace(0, 1, 100)
        y_true_plot = self._true_function(x_plot)

        ax_fit.scatter(x_train, y_train, s=20, alpha=0.5, color='gray', label='Training data')
        ax_fit.plot(x_plot, y_true_plot, 'k-', lw=2, label='True function')

        # Fit and plot three example polynomials
        example_degrees = [1, optimal_degree, self._MAX_DEGREE]
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
        labels = ['Underfitting (d=1)', f'Optimal (d={optimal_degree})', f'Overfitting (d={self._MAX_DEGREE})']

        for d, c, lbl in zip(example_degrees, colors, labels):
            coeffs = np.polyfit(x_train, y_train, d)
            y_fit = np.polyval(coeffs, x_plot)
            ax_fit.plot(x_plot, y_fit, '--', color=c, lw=1.5, alpha=0.8, label=lbl)

        ax_fit.set_xlabel('x', fontsize=11)
        ax_fit.set_ylabel('y', fontsize=11)
        ax_fit.set_title('Example Polynomial Fits', fontsize=12)
        ax_fit.legend(fontsize=9, loc='upper right')
        ax_fit.set_ylim(-2, 2)

        self._fig.tight_layout()
        self.chart_base64 = fig_to_base64(self._fig)

        # Stats
        self.stats_html = (
            f"<b>True function</b>: y = sin(2πx) + ε | "
            f"<b>Noise</b>: σ = {self.noise_std:.1f} | "
            f"<b>n</b> = {self.n_samples} | "
            f"<b>Simulations</b>: {self.n_sims} | "
            f"<b>Optimal degree</b>: <span style='color:#d62728'><b>{optimal_degree}</b></span> "
            f"(MSE = {total_mse[optimal_idx]:.3f})"
        )

    def __del__(self):
        plt.close(self._fig)


def cv_explorer(seed: int = 42) -> CVExplorer:
    """
    Create an interactive bias-variance decomposition explorer.

    Generates synthetic data from y = sin(2πx) + noise, fits polynomials
    of varying degrees, and shows the decomposition of test error into
    Bias², Variance, and Irreducible Error. Demonstrates the classic
    U-shaped tradeoff curve.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    CVExplorer widget
    """
    return CVExplorer(seed=seed)


# =============================================================================
# Convenience: All explorers
# =============================================================================

__all__ = [
    'load_penguins',
    'load_tips',
    'loss_explorer',
    'lln_explorer',
    'clt_explorer',
    'nfl_explorer',
    'mc_explorer',
    'bootstrap_explorer',
    'permutation_explorer',
    'cv_explorer',
    'LossExplorer',
    'LLNExplorer',
    'CLTExplorer',
    'NFLExplorer',
    'MCExplorer',
    'BootstrapExplorer',
    'PermutationExplorer',
    'CVExplorer',
]
