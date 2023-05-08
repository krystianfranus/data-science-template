import pandas as pd
import torch
from clearml import Logger
from pandas import DataFrame
from torchmetrics import RetrievalNormalizedDCG


def compute_baseline(train: DataFrame, val: DataFrame, type: str) -> None:
    # Add item popularity to val dataframe
    match type:
        case "simple":
            n_clicks_per_item = (
                train.groupby("item")
                .agg({"target": "sum"})
                .rename(columns={"target": "popularity"})
                .reset_index()
            )
        case "bpr":
            clicked_items_by_users = train[["user", "item_pos"]].drop_duplicates()
            n_clicks_per_item = (
                clicked_items_by_users.groupby("item_pos")
                .count()
                .reset_index()
                .rename(columns={"item_pos": "item", "user": "popularity"})
            )
        case _:
            raise ValueError(f"Invalid data type, you provided '{type}'")
    val = val.merge(n_clicks_per_item, "inner", "item")

    # Compute ndcg for different scenarios
    indexes = torch.tensor(val["user"])
    target = torch.tensor(val["target"])
    worst_preds = torch.tensor((val["target"] + 1) % 2, dtype=torch.float32)
    random_preds = torch.rand(val.shape[0])
    pop_preds = torch.tensor(val["popularity"], dtype=torch.float32)
    best_preds = torch.tensor(val["target"], dtype=torch.float32)

    ndcg = RetrievalNormalizedDCG()
    worst_ndcg = ndcg(worst_preds, target, indexes=indexes).item()
    random_ndcg = ndcg(random_preds, target, indexes=indexes).item()
    pop_ndcg = ndcg(pop_preds, target, indexes=indexes).item()
    best_ndcg = ndcg(best_preds, target, indexes=indexes).item()

    # Log NDCG
    baseline = pd.DataFrame()
    baseline.loc["Worst model", "value"] = worst_ndcg
    baseline.loc["Random model", "value"] = random_ndcg
    baseline.loc["Popularity-based model", "value"] = pop_ndcg
    baseline.loc["Best model", "value"] = best_ndcg
    Logger.current_logger().report_table(
        "Baselines of validation data",
        "NDCG",
        iteration=0,
        table_plot=baseline,
    )

    # Log bottom10 and top10 items based on popularity
    n_clicks_per_item_sorted = n_clicks_per_item.sort_values(
        "popularity", ascending=False
    ).reset_index(drop=True)
    Logger.current_logger().report_table(
        "Items popularity (based on clicks)",
        "Bottom 10",
        iteration=0,
        table_plot=n_clicks_per_item_sorted[-10:],
    )
    Logger.current_logger().report_table(
        "Items popularity (based on clicks)",
        "Top 10",
        iteration=0,
        table_plot=n_clicks_per_item_sorted[:10],
    )
