import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import seaborn as sns
    from pathlib import Path
    import os
    from pathlib import Path
    import matplotlib.pyplot as plt
    from itertools import combinations
    return Path, combinations, mo, os, pl, plt, sns


@app.cell
def _(mo):
    mo.md(r"""
    # Constants
    """)
    return


@app.cell
def _(Path, os):
    ROOT_DIR = Path(os.environ["ROOT_DIR"])

    RESULTS_DIR = ROOT_DIR / "results"
    PLOTS_DIR = ROOT_DIR / "analysis" / "plots"
    HEATMAPS_DIR = PLOTS_DIR / "heatmaps"
    return HEATMAPS_DIR, PLOTS_DIR, RESULTS_DIR


@app.cell
def _():
    colors = {
        "NIM": "#E78AC3",  # pink
        "1": "#8DA0CB",    # blue
        "4": "#FC8D62",    # orange
        "8": "#66C2A5",    # green
    }
    return (colors,)


@app.cell
def _(mo):
    mo.md(r"""
    # Functions
    """)
    return


@app.cell
def _(Path, pl, plt, sns):
    def plot_inference_distributions(
        df: pl.DataFrame,
        violin_path: Path,
        kde_path: Path,
        colors: dict[str, str],
        fig_width: float = 8.0,
        fig_height: float = 6.0,
        alpha_violin: float = 0.7,
        alpha_kde: float = 0.4,
    ) -> None:

        order = ["NIM", "1", "4", "8"]
        palette = [colors[b] for b in order if b in colors]

        sns.set(style="whitegrid")

        # 1) Violin Plot
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        sns.violinplot(
            data=df,
            x="runtime_sec",
            y="batch_size",
            order=order,
            palette=palette,
            inner="box",
            cut=0,
            orient="h",
            alpha=alpha_violin,
            ax=ax,
        )
        ax.set_title("Distribution of Inference Times by Group")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(violin_path, dpi=300)
        plt.close(fig)

        # 2) KDE Plot
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        for batch in order:
            subdf = df.filter(pl.col("batch_size") == batch)
            sns.kdeplot(
                data=subdf,
                x="runtime_sec",
                label=batch,
                fill=True,
                alpha=alpha_kde,
                color=colors[batch],
                ax=ax,
                cut=0
            )

        ax.set_title("Distribution of Inference Times by Group")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Density")
        ax.legend(title="batch")
        fig.tight_layout()
        fig.savefig(kde_path, dpi=300)
        plt.close(fig)

    return (plot_inference_distributions,)


@app.cell
def _(Path, pl, plt, sns):
    def plot_scatter(
        df: pl.DataFrame,
        save_path: Path,
        colors: dict[str,str],
        fig_width: float = 10.0,
        fig_height: float = 8.0,
        alpha_points: float = 0.7,
        line_color: str = "grey",
        line_alpha: float = 0.7,
    ) -> None:
        baseline = (
            df
            .filter(pl.col("batch_size") == "1")
            .rename({'runtime_sec': 'baseline'})
        )

        target_batches = ["4", "8", "NIM"]
        batched = (
            df
            .filter(pl.col("batch_size").is_in(target_batches))
            .rename({'batch_size': 'batch', 'runtime_sec': 'batched'})
        )

        pair_long = batched.join(baseline, 
                                 on=['recycling_steps',
                                     'sampling_steps',
                                     'diffusion_samples',
                                     'sampling_steps_affinity',
                                     'diffusion_samples_affinity'], 
                                 how="inner")

        hue_order = [b for b in ["4", "8", "NIM"] if b in pair_long["batch"].unique().to_list()]
        palette = {b: colors[b] for b in hue_order if b in colors}

        min_v = float(
            min(
                pair_long.select(pl.min("baseline")).item(),
                pair_long.select(pl.min("batched")).item(),
            )
        )
        max_v = float(
            max(
                pair_long.select(pl.max("baseline")).item(),
                pair_long.select(pl.max("batched")).item(),
            )
        )

        sns.set(style="whitegrid")

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        sns.scatterplot(
            data=pair_long,
            x="baseline",
            y="batched",
            hue="batch",
            hue_order=hue_order,
            palette=palette,
            alpha=alpha_points,
            ax=ax,
        )

        ax.plot(
            [min_v, max_v],
            [min_v, max_v],
            linestyle="--",
            linewidth=1,
            color=line_color,
            alpha=line_alpha,
        )

        ax.set_title("Baseline vs Batched/NIM Inference Time")
        ax.set_xlabel("Baseline time (s)")
        ax.set_ylabel("Batched/NIM time (s)")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(title="Group")

        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

    return (plot_scatter,)


@app.cell
def _(Path, combinations, pl, plt, sns):
    def plot_mean_speedup_heatmaps(
        df: pl.DataFrame,
        save_dir: Path,
        heat_group: str,
        param_cols: list[str],
        fig_width: float = 6.0,
        fig_height: float = 5.0,
        cmap: str = "viridis",
    ) -> None:
        baseline = (
            df
            .filter(pl.col("batch_size") == "1")
            .select(["id", *param_cols, pl.col("runtime_sec").alias("baseline")])
        )

        group_df = (
            df
            .filter(pl.col("batch_size") == heat_group)
            .select(["id", *param_cols, pl.col("runtime_sec").alias("group_time")])
        )

        joined = baseline.join(
            group_df,
            on=param_cols,
            how="inner",
        )

        joined = joined.with_columns(
            (pl.col("baseline") / pl.col("group_time")).alias("speedup")
        )

        sns.set(style="whitegrid")

        for row_param, col_param in combinations(param_cols, 2):
            heat_df = (
                joined
                .group_by([row_param, col_param])
                .agg(pl.mean("speedup").alias("mean_speedup"))
                .pivot(
                    values="mean_speedup",
                    index=row_param,
                    on=col_param,
                    aggregate_function="first",
                    sort_columns=True,
                )
                .sort(row_param)
            )

            row_vals = heat_df[row_param].to_list()
            col_names = [c for c in heat_df.columns if c != row_param]
            heat_vals = heat_df.select(col_names).to_numpy()

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            sns.heatmap(
                heat_vals,
                annot=False,
                cmap=cmap,
                cbar_kws={"label": "Mean speedup (×)"},
                ax=ax,
            )

            ax.set_title(
                f"Mean Speedup (group={heat_group}) by "
                f"{row_param.replace('_', ' ')} × {col_param.replace('_', ' ')}"
            )
            ax.set_xlabel(col_param.replace("_", " "))
            ax.set_ylabel(row_param.replace("_", " "))

            ax.set_xticks([i + 0.5 for i in range(len(col_names))])
            ax.set_xticklabels(col_names, ha="right")

            ax.set_yticks([i + 0.5 for i in range(len(row_vals))])
            ax.set_yticklabels([str(v) for v in row_vals])

            plt.tight_layout()

            out_name = (
                f"heatmap_mean_speedup_{heat_group}_"
                f"{row_param}_vs_{col_param}.png"
            )
            fig.savefig(save_dir / out_name, dpi=300)
            plt.show()
            plt.close(fig)
    return (plot_mean_speedup_heatmaps,)


@app.cell
def _(mo):
    mo.md(r"""
    # Read results data
    """)
    return


@app.cell
def _(RESULTS_DIR, pl):
    results_df = pl.read_ndjson(RESULTS_DIR).unnest('params')
    results_df
    return (results_df,)


@app.cell
def _(results_df):
    results_df.head()
    return


@app.cell
def _(HEATMAPS_DIR, PLOTS_DIR):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)
    (HEATMAPS_DIR / "NIM").mkdir(parents=True, exist_ok=True)
    (HEATMAPS_DIR / "batch_4").mkdir(parents=True, exist_ok=True)
    (HEATMAPS_DIR / "batch_8").mkdir(parents=True, exist_ok=True)
    return


@app.cell
def _(PLOTS_DIR, colors, plot_inference_distributions, results_df):
    plot_inference_distributions(df=results_df,
                                violin_path=PLOTS_DIR / 'violin.png',
                                kde_path=PLOTS_DIR / 'kde.png',
                                 colors=colors)
    return


@app.cell
def _(PLOTS_DIR, colors, plot_scatter, results_df):
    plot_scatter(
        df = results_df,
        save_path = PLOTS_DIR / 'scatter.png',
        colors = colors
    )
    return


@app.cell
def _():
    param_cols=[
            "recycling_steps",
            "sampling_steps",
            "diffusion_samples",
            "sampling_steps_affinity",
            "diffusion_samples_affinity",
        ]
    return (param_cols,)


@app.cell
def _(HEATMAPS_DIR, param_cols, plot_mean_speedup_heatmaps, results_df):
    plot_mean_speedup_heatmaps(
        df=results_df,
        save_dir=HEATMAPS_DIR / "NIM",
        heat_group="NIM",
        param_cols=param_cols
    )

    plot_mean_speedup_heatmaps(
        df=results_df,
        save_dir=HEATMAPS_DIR / "batch_4",
        heat_group="4",
        param_cols=param_cols
    )

    plot_mean_speedup_heatmaps(
        df=results_df,
        save_dir=HEATMAPS_DIR / "batch_8",
        heat_group="8",
        param_cols=param_cols
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

