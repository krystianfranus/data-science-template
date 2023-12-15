import pandas as pd
import torch
from clearml import Logger
from pandas import DataFrame
from torchmetrics.retrieval import RetrievalNormalizedDCG


def compute_baseline(train: DataFrame, val: DataFrame, data_type: str) -> None:
    # Add item popularity to val dataframe
    match data_type:
        case "simple":
            stats_per_item = (
                train.groupby("item")
                .agg({"target": ["sum", "mean"]})
                .droplevel(0, axis=1)
                .rename(columns={"sum": "n_clicks", "mean": "ctr"})
                .reset_index()
            )
        case "bpr":
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
        case _:
            raise ValueError(f"Invalid data type, you provided '{type}'")
    val = val.merge(stats_per_item, "inner", "item")

    # Compute ndcg for different scenarios
    indexes = torch.tensor(val["user"])
    target = torch.tensor(val["target"])
    worst_preds = torch.tensor((val["target"] + 1) % 2, dtype=torch.float32)
    random_preds = torch.rand(val.shape[0])
    top_clicks_preds = torch.tensor(val["n_clicks"], dtype=torch.float32)
    top_ctr_preds = torch.tensor(val["ctr"], dtype=torch.float32)
    best_preds = torch.tensor(val["target"], dtype=torch.float32)

    ndcg = RetrievalNormalizedDCG()
    worst_ndcg = ndcg(worst_preds, target, indexes=indexes).item()
    random_ndcg = ndcg(random_preds, target, indexes=indexes).item()
    top_clicks_ndcg = ndcg(top_clicks_preds, target, indexes=indexes).item()
    top_ctr_ndcg = ndcg(top_ctr_preds, target, indexes=indexes).item()
    best_ndcg = ndcg(best_preds, target, indexes=indexes).item()

    # Log NDCG
    baseline = pd.DataFrame()
    baseline.loc["Worst model", "value"] = worst_ndcg
    baseline.loc["Random model", "value"] = random_ndcg
    baseline.loc["Popularity-based model (by #clicks)", "value"] = top_clicks_ndcg
    baseline.loc["Popularity-based model (by ctr)", "value"] = top_ctr_ndcg
    baseline.loc["Best model", "value"] = best_ndcg
    Logger.current_logger().report_table(
        "Baselines of validation data",
        "NDCG",
        iteration=0,
        table_plot=baseline,
    )

    # Log top10 and bottom10 items based on popularity (by #clicks)
    tmp = stats_per_item.sort_values("n_clicks", ascending=False).reset_index(drop=True)
    Logger.current_logger().report_table(
        "Items popularity (based on #clicks)",
        "Top 10",
        iteration=0,
        table_plot=tmp[:10],
    )
    Logger.current_logger().report_table(
        "Items popularity (based on #clicks)",
        "Bottom 10",
        iteration=0,
        table_plot=tmp[-10:],
    )

    # Log top10 and bottom10 items based on popularity(by ctr)
    tmp = stats_per_item.sort_values("ctr", ascending=False).reset_index(drop=True)
    Logger.current_logger().report_table(
        "Items popularity (based on ctr)",
        "Top 10",
        iteration=0,
        table_plot=tmp[:10],
    )
    Logger.current_logger().report_table(
        "Items popularity (based on ctr)",
        "Bottom 10",
        iteration=0,
        table_plot=tmp[-10:],
    )
