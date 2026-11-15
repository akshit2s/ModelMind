from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


def _get_model_name(result):
    """Extract the model name from a ranked result."""
    for key in ("model_name", "model", "name", "algorithm"):
        value = result.get(key)
        if isinstance(value, str):
            return value

    return "Unknown model"


def _get_metrics(result):
    """Extract the metrics dictionary."""
    for key in ("metrics", "evaluation_metrics", "scores"):
        value = result.get(key)
        if isinstance(value, dict):
            return value

    # Support results where metrics are stored directly in the dictionary.
    return {
        key: value
        for key, value in result.items()
        if isinstance(value, (int, float))
        and key not in ("rank", "index")
    }


def _normalise(value):
    return str(value).lower().replace("_", "").replace("-", "").replace(" ", "")


def _get_primary_score(metrics, primary_metric):
    """Find the primary metric, allowing common naming variations."""
    aliases = {
        "f1": ("f1", "f1score", "weightedf1", "f1weighted"),
        "rmse": ("rmse", "rootmeansquarederror"),
    }

    target = _normalise(primary_metric)
    possible_names = aliases.get(target, (target,))

    for key, value in metrics.items():
        normalised_key = _normalise(key)

        if normalised_key in possible_names:
            return value

    # Allow a partial match for names such as weighted_f1_score.
    for key, value in metrics.items():
        normalised_key = _normalise(key)

        if any(alias in normalised_key for alias in possible_names):
            return value

    return None


def _format_score(value):
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def render_leaderboard(
    ranked_results,
    problem_type,
    primary_metric=None,
    console=None
):
    """
    Display ranked model results in a Rich table.

    Parameters:
        ranked_results: Results returned by model_comparator.
        problem_type: classification or regression.
        primary_metric: Metric used to rank the models.
        console: Optional Rich Console instance.
    """

    console = console or Console()

    if not ranked_results:
        console.print(
            Panel("No model results are available yet.", title="Leaderboard")
        )
        return

    problem_type = problem_type.lower()

    if primary_metric is None:
        primary_metric = (
            "f1" if problem_type == "classification" else "rmse"
        )

    higher_is_better = primary_metric.lower() in (
        "f1", "accuracy", "precision", "recall", "r2"
    )

    direction = "Higher is better" if higher_is_better else "Lower is better"

    table = Table(
        title="ModelMind — Model Leaderboard",
        show_header=True,
        header_style="bold cyan",
        border_style="blue"
    )

    table.add_column("Rank", justify="center", width=7)
    table.add_column("Model", min_width=22)
    table.add_column(
        primary_metric.upper(),
        justify="right",
        min_width=12
    )
    table.add_column("Other Metrics", min_width=30)

    best_model = None
    best_score = None

    for position, result in enumerate(ranked_results, start=1):
        model_name = _get_model_name(result)
        metrics = _get_metrics(result)

        score = _get_primary_score(metrics, primary_metric)

        rank = result.get("rank", position)

        other_metrics = [
            f"{key}: {_format_score(value)}"
            for key, value in metrics.items()
            if _normalise(key) != _normalise(primary_metric)
        ]

        metric_summary = " | ".join(other_metrics) or "—"

        if position == 1:
            best_model = model_name
            best_score = score

        style = "bold green" if position == 1 else None

        table.add_row(
            str(rank),
            model_name,
            _format_score(score),
            metric_summary,
            style=style
        )

    console.print()
    console.print(table)

    if best_model is not None:
        summary = Text()
        summary.append("Best candidate: ", style="bold")
        summary.append(best_model, style="bold green")
        summary.append(
            f"\n{primary_metric.upper()}: {_format_score(best_score)}",
            style="cyan"
        )
        summary.append(f"\nRanking rule: {direction}")

        console.print(
            Panel(
                summary,
                title="Top-Ranked Model",
                border_style="green"
            )
        )

    console.print()