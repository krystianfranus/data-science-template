import pandas as pd
import torch
from clearml import Logger
from pandas import DataFrame
from torchmetrics.retrieval import RetrievalAUROC, RetrievalNormalizedDCG


def compute_baselines_simple(train: DataFrame, val: DataFrame) -> None:
    stats_per_item = (
        train.groupby("item")
        .agg({"target": ["sum", "mean"]})
        .droplevel(0, axis=1)
        .rename(columns={"sum": "n_clicks", "mean": "ctr"})
        .reset_index()
    )

    _common(stats_per_item, val)


def compute_baselines_bpr(train: DataFrame, val: DataFrame) -> None:
    clicks_per_user = train[["user", "item_pos"]].drop_duplicates()
    stats_pos = (
        clicks_per_user.groupby("item_pos")
        .count()
        .reset_index()
        .rename(columns={"item_pos": "item", "user": "n_clicks"})
    )

    impressions_per_user = train[["user", "item_neg"]].drop_duplicates()
    stats_neg = (
        impressions_per_user.groupby("item_neg")
        .count()
        .reset_index()
        .rename(columns={"item_neg": "item", "user": "n_impressions"})
    )

    stats_per_item = stats_pos.merge(stats_neg, "inner", "item")
    stats_per_item["ctr"] = stats_per_item["n_clicks"] / (
        stats_per_item["n_clicks"] + stats_per_item["n_impressions"]
    )

    _common(stats_per_item, val)


def _common(stats_per_item: DataFrame, val: DataFrame):
    val = val.merge(stats_per_item, "inner", "item")

    # Compute aucroc and ndcg for different scenarios
    indexes = torch.tensor(val["user"])
    targets = torch.tensor(val["target"])
    worst_preds = torch.tensor((val["target"] + 1) % 2, dtype=torch.float32)
    random_preds = torch.rand(val.shape[0])
    top_clicks_preds = torch.tensor(val["n_clicks"], dtype=torch.float32)
    top_ctr_preds = torch.tensor(val["ctr"])
    best_preds = torch.tensor(val["target"], dtype=torch.float32)

    auroc = RetrievalAUROC(empty_target_action="skip")
    worst_auroc = auroc(worst_preds, targets, indexes=indexes).item()
    random_auroc = auroc(random_preds, targets, indexes=indexes).item()
    top_clicks_auroc = auroc(top_clicks_preds, targets, indexes=indexes).item()
    top_ctr_auroc = auroc(top_ctr_preds, targets, indexes=indexes).item()
    best_auroc = auroc(best_preds, targets, indexes=indexes).item()

    ndcg = RetrievalNormalizedDCG(empty_target_action="skip")
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

    # Log top20 and bottom20 items based on popularity(by ctr)
    tmp = stats_per_item.sort_values("ctr", ascending=False).reset_index(drop=True)
    Logger.current_logger().report_table(
        "Items popularity (based on ctr)",
        "Top 20",
        iteration=0,
        table_plot=tmp[:20],
    )
    Logger.current_logger().report_table(
        "Items popularity (based on ctr)",
        "Bottom 20",
        iteration=0,
        table_plot=tmp[-20:],
    )
