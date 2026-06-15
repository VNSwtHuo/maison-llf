from pathlib import Path
import argparse
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def numeric_sort_columns(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "TOP_X_SENSOR",
        "TOP_Y_DEMO",
        "n_sensor_union",
        "n_demo_union",
        "n_features",
        "epochs",
    ]
    for col in cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64")
    for col in frame.columns:
        if col not in cols:
            try:
                frame[col] = pd.to_numeric(frame[col])
            except (TypeError, ValueError):
                pass
    return frame


def load_snapshot(results_path: Path) -> pd.DataFrame:
    if not results_path.exists():
        raise FileNotFoundError(f"No results CSV found at {results_path}")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        snapshot_path = Path(tmp.name)
    shutil.copy2(results_path, snapshot_path)
    try:
        return numeric_sort_columns(pd.read_csv(snapshot_path))
    finally:
        snapshot_path.unlink(missing_ok=True)


def print_progress(results: pd.DataFrame) -> None:
    completed = len(results)
    expected = 42 * 8
    print(f"Completed: {completed}/{expected} ({completed / expected:.1%})")
    print(
        "Covered TOP_X_SENSOR:",
        f"{results['TOP_X_SENSOR'].min()}..{results['TOP_X_SENSOR'].max()}",
    )
    print(
        "Covered TOP_Y_DEMO:",
        sorted(results["TOP_Y_DEMO"].dropna().astype(int).unique().tolist()),
    )
    print()


def print_best(results: pd.DataFrame, metric: str, n: int) -> None:
    if metric not in results.columns:
        print(f"Metric not found: {metric}")
        return

    cols = [
        "TOP_X_SENSOR",
        "TOP_Y_DEMO",
        "n_features",
        "epochs",
        metric,
        "test_SIS_MAE",
        "test_OHS_MAE",
        "val_overall_MAE",
        "best_val_loss",
    ]
    cols = [c for c in cols if c in results.columns]
    print(f"Top {n} by {metric}")
    print(results.sort_values(metric, ascending=True)[cols].head(n).to_string(index=False))
    print()


def plot_heatmap(results: pd.DataFrame, metric: str, output_path: Path) -> None:
    table = (
        results.pivot_table(
            index="TOP_Y_DEMO",
            columns="TOP_X_SENSOR",
            values=metric,
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    masked = np.ma.masked_invalid(table.to_numpy(dtype=float))
    im = ax.imshow(masked, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=90)
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_yticklabels(table.index)
    ax.set_xlabel("TOP_X_SENSOR")
    ax.set_ylabel("TOP_Y_DEMO")
    ax.set_title(metric)
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_lines(results: pd.DataFrame, metric: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for top_y_demo, group in results.sort_values("TOP_X_SENSOR").groupby("TOP_Y_DEMO"):
        ax.plot(group["TOP_X_SENSOR"], group[metric], marker="o", linewidth=1.5, label=top_y_demo)
    ax.set_xlabel("TOP_X_SENSOR")
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.legend(title="TOP_Y_DEMO", ncol=4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_plots(results: pd.DataFrame, output_dir: Path, metrics: list[str]) -> None:
    plot_dir = output_dir / "partial_analysis_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        if metric not in results.columns:
            continue
        plot_heatmap(results, metric, plot_dir / f"{metric}_partial_heatmap.png")
        plot_lines(results, metric, plot_dir / f"{metric}_partial_lines.png")
    print(f"Saved partial plots to {plot_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("grid_search_results/grid_search_results.csv"),
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--metric", default="test_overall_MAE")
    parser.add_argument("--output-dir", type=Path, default=Path("grid_search_results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_snapshot(args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print_progress(results)
    print_best(results, args.metric, args.top)
    print_best(results, "val_overall_MAE", args.top)

    summary_path = args.output_dir / f"partial_top{args.top}_by_{args.metric}.csv"
    results.sort_values(args.metric, ascending=True).head(args.top).to_csv(summary_path, index=False)
    print(f"Saved top table to {summary_path}")

    save_plots(
        results,
        args.output_dir,
        [
            "test_overall_MAE",
            "test_SIS_MAE",
            "test_OHS_MAE",
            "test_overall_RMSE",
            "val_overall_MAE",
        ],
    )


if __name__ == "__main__":
    main()
