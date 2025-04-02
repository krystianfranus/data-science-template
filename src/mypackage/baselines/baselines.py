import pandas as pd
import torch
from clearml import Logger
from pandas import DataFrame
from torchmetrics.retrieval import RetrievalAUROC, RetrievalNormalizedDCG


def evaluate_baselines(train: DataFrame, val: DataFrame) -> None:
    """
    Computes metrics (AUROC & NDCG) for baseline approaches to recommendations.

    This function calculates performance metrics for a recommender system
    using various prediction strategies. It evaluates models based on random guessing,
    worst-case predictions, popularity-based approaches (by click count and CTR), and
    an ideal model. The function then logs these results along with item popularity
    statistics.

    Args:
        train (DataFrame): The training dataset containing item interaction data.
        val (DataFrame): The validation dataset used for evaluation.

    Returns:
        None: The function logs results but does not return any values.

    Logs:
        - A table comparing AUROC and NDCG scores for different baseline models.
        - A table showing the top 20 and bottom 20 most popular items based on click count.
    """
    stats_per_item = (
        train.groupby("item_idx")
        .agg({"target": ["sum", "mean"]})
        .droplevel(0, axis=1)
        .rename(columns={"sum": "n_clicks", "mean": "ctr"})
        .reset_index()
    )
    val = val.merge(stats_per_item, "inner", "item_idx")

    # Compute aucroc and ndcg for various strategies
    indexes = torch.tensor(val["list_id"])
    targets = torch.tensor(val["target"])
    worst_preds = torch.tensor((val["target"] + 1) % 2, dtype=torch.float32)
    random_preds = torch.rand(val.shape[0])
    top_clicks_preds = torch.tensor(val["n_clicks"], dtype=torch.float32)
    top_ctr_preds = torch.tensor(val["ctr"])
    best_preds = torch.tensor(val["target"], dtype=torch.float32)

    auroc = RetrievalAUROC(empty_target_action="error")
    worst_auroc = auroc(worst_preds, targets, indexes=indexes).item()
    random_auroc = auroc(random_preds, targets, indexes=indexes).item()
    top_clicks_auroc = auroc(top_clicks_preds, targets, indexes=indexes).item()
    top_ctr_auroc = auroc(top_ctr_preds, targets, indexes=indexes).item()
    best_auroc = auroc(best_preds, targets, indexes=indexes).item()

    ndcg = RetrievalNormalizedDCG(empty_target_action="error")
    worst_ndcg = ndcg(worst_preds, targets, indexes=indexes).item()
    random_ndcg = ndcg(random_preds, targets, indexes=indexes).item()
    top_clicks_ndcg = ndcg(top_clicks_preds, targets, indexes=indexes).item()
    top_ctr_ndcg = ndcg(top_ctr_preds, targets, indexes=indexes).item()
    best_ndcg = ndcg(best_preds, targets, indexes=indexes).item()

    # Log AUROC & NDCG
    baseline = pd.DataFrame()

    baseline.loc["Worst model", "auroc"] = worst_auroc
    baseline.loc["Random model", "auroc"] = random_auroc
    baseline.loc["Popularity-based model (by #clicks)", "auroc"] = top_clicks_auroc
    baseline.loc["Popularity-based model (by ctr)", "auroc"] = top_ctr_auroc
    baseline.loc["Best model", "auroc"] = best_auroc

    baseline.loc["Worst model", "ndcg"] = worst_ndcg
    baseline.loc["Random model", "ndcg"] = random_ndcg
    baseline.loc["Popularity-based model (by #clicks)", "ndcg"] = top_clicks_ndcg
    baseline.loc["Popularity-based model (by ctr)", "ndcg"] = top_ctr_ndcg
    baseline.loc["Best model", "ndcg"] = best_ndcg

    Logger.current_logger().report_table(
        "Baselines computed on validation data",
        "AUROC & NDCG",
        iteration=0,
        table_plot=baseline,
    )

    # Log top20 and bottom20 items based on popularity (by #clicks)
    tmp = stats_per_item.sort_values("n_clicks", ascending=False).reset_index(drop=True)
    Logger.current_logger().report_table(
        "Items popularity (based on #clicks)",
        "Top 20",
        iteration=0,
        table_plot=tmp[:20],
    )
    Logger.current_logger().report_table(
        "Items popularity (based on #clicks)",
        "Bottom 20",
        iteration=0,
        table_plot=tmp[-20:],
    )
