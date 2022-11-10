from typing import Tuple

import pandas as pd


class MovieLens1M:
    def __init__(self, remote_data, data_type):
        if remote_data:
            prefix = "s3://kfranus-bucket"
            data_path = f"{prefix}/data-science-template/data/movielens/ratings.dat"
        else:
            data_path = "data/movielens/ratings.dat"

        self.data = pd.read_csv(
            data_path,
            sep="::",
            names=["user", "item", "target", "timestamp"],
            engine="python",
        )
        self.data_type = data_type

    def preprocess(self):
        self.data = self.data.sort_values("timestamp").reset_index(drop=True)
        self.data = self.data.drop(columns=["timestamp"])

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_data, val_data, test_data = None, None, None
        if self.data_type == "explicit":
            train_data, val_data, test_data = self._prepare_explicit()
        elif self.data_type == "implicit":
            train_data, val_data, test_data = self._prepare_implicit()
        elif self.data_type == "implicit_bpr":
            train_data, val_data, test_data = self._prepare_implicit_bpr()
        return train_data, val_data, test_data

    def _prepare_explicit(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Split data into train/val/test
        train_data = self.data[:900_000].reset_index(drop=True)
        val_data = self.data[900_000:950_000].reset_index(drop=True)
        test_data = self.data[950_000:].reset_index(drop=True)

        # Prepare unique train user and items
        train_users = train_data["user"].unique()
        train_items = train_data["item"].unique()

        # Filter val/test data
        val_data = val_data[val_data["user"].isin(train_users)]
        val_data = val_data[val_data["item"].isin(train_items)]
        val_data = val_data.reset_index(drop=True)
        test_data = test_data[test_data["user"].isin(train_users)]
        test_data = test_data[test_data["item"].isin(train_items)]
        test_data = test_data.reset_index(drop=True)

        # Map idx
        user_to_idx = {user: idx for idx, user in enumerate(train_users)}
        item_to_idx = {item: idx for idx, item in enumerate(train_items)}
        train_data["user"] = train_data["user"].map(user_to_idx)
        train_data["item"] = train_data["item"].map(item_to_idx)
        val_data["user"] = val_data["user"].map(user_to_idx)
        val_data["item"] = val_data["item"].map(item_to_idx)
        test_data["user"] = test_data["user"].map(user_to_idx)
        test_data["item"] = test_data["item"].map(item_to_idx)

        return train_data, val_data, test_data

    def _prepare_implicit(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.data.loc[self.data["target"] < 4, "target"] = 0
        self.data.loc[self.data["target"] >= 4, "target"] = 1

        # Split data into train/val/test
        train_data = self.data[:900_000].reset_index(drop=True)
        val_data = self.data[900_000:950_000].reset_index(drop=True)
        test_data = self.data[950_000:].reset_index(drop=True)

        # Prepare unique train user and items
        train_users = train_data["user"].unique()
        train_items = train_data["item"].unique()

        # Filter val/test data
        val_data = val_data[val_data["user"].isin(train_users)]
        val_data = val_data[val_data["item"].isin(train_items)]
        val_data = val_data.reset_index(drop=True)
        test_data = test_data[test_data["user"].isin(train_users)]
        test_data = test_data[test_data["item"].isin(train_items)]
        test_data = test_data.reset_index(drop=True)

        # Map idx
        user_to_idx = {user: idx for idx, user in enumerate(train_users)}
        item_to_idx = {item: idx for idx, item in enumerate(train_items)}
        train_data["user"] = train_data["user"].map(user_to_idx)
        train_data["item"] = train_data["item"].map(item_to_idx)
        val_data["user"] = val_data["user"].map(user_to_idx)
        val_data["item"] = val_data["item"].map(item_to_idx)
        test_data["user"] = test_data["user"].map(user_to_idx)
        test_data["item"] = test_data["item"].map(item_to_idx)

        return train_data, val_data, test_data

    def _prepare_implicit_bpr(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.data.loc[self.data["target"] < 4, "target"] = 0
        self.data.loc[self.data["target"] >= 4, "target"] = 1

        # Split data into train/val/test
        train_data = self.data[:900_000].reset_index(drop=True)
        tmp0 = train_data.loc[train_data["target"] == 0, ["user", "item"]]
        tmp1 = train_data.loc[train_data["target"] == 1, ["user", "item"]]
        train_data = tmp0.merge(tmp1, "inner", "user", suffixes=("_neg", "_pos"))
        train_data = train_data.sample(frac=0.05, random_state=0).reset_index(drop=True)
        val_data = self.data[900_000:950_000].reset_index(drop=True)
        test_data = self.data[950_000:].reset_index(drop=True)

        # Prepare unique train user and items
        train_users = train_data["user"].unique()
        item_neg_set = set(train_data["item_neg"])
        item_pos_set = set(train_data["item_pos"])
        train_items = pd.Series(list(item_neg_set | item_pos_set)).unique()

        # Filter val/test data
        val_data = val_data[val_data["user"].isin(train_users)]
        val_data = val_data[val_data["item"].isin(train_items)]
        val_data = val_data.reset_index(drop=True)
        test_data = test_data[test_data["user"].isin(train_users)]
        test_data = test_data[test_data["item"].isin(train_items)]
        test_data = test_data.reset_index(drop=True)

        # Map idx
        user_to_idx = {user: idx for idx, user in enumerate(train_users)}
        item_to_idx = {item: idx for idx, item in enumerate(train_items)}
        train_data["user"] = train_data["user"].map(user_to_idx)
        train_data["item_neg"] = train_data["item_neg"].map(item_to_idx)
        train_data["item_pos"] = train_data["item_pos"].map(item_to_idx)
        val_data["user"] = val_data["user"].map(user_to_idx)
        val_data["item"] = val_data["item"].map(item_to_idx)
        test_data["user"] = test_data["user"].map(user_to_idx)
        test_data["item"] = test_data["item"].map(item_to_idx)

        return train_data, val_data, test_data

    def save_data(self, train_data, val_data, test_data):
        train_data.to_csv(
            f"data/movielens/train_data_{self.data_type}.csv", index=False, header=False
        )
        val_data.to_csv(
            f"data/movielens/val_data_{self.data_type}.csv", index=False, header=False
        )
        test_data.to_csv(
            f"data/movielens/test_data_{self.data_type}.csv", index=False, header=False
        )


class ContentWise:
    def __init__(self, remote_data, data_type):
        if remote_data:  # TODO
            prefix = "s3://kfranus-bucket"
            data_path = f"{prefix}/data-science-template/data/movielens/ratings.dat"
            data_path2 = f"{prefix}/data-science-template/data/movielens/ratings.dat"
        else:
            prefix = "data/contentwise/data/contentwise"
            data_path = f"{prefix}/CW10M-CSV/interactions.csv.gz"
            data_path2 = f"{prefix}/CW10M-CSV/impressions-direct-link.csv.gz"

        self.data = pd.read_csv(data_path)
        self.data2 = pd.read_csv(data_path2)
        self.data_type = data_type

    def preprocess(self):
        pass

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_data, val_data, test_data = None, None, None
        if self.data_type == "implicit":
            train_data, val_data, test_data = self._prepare_implicit()
        elif self.data_type == "implicit_bpr":
            train_data, val_data, test_data = self._prepare_implicit_bpr()
        return train_data, val_data, test_data

    def _prepare_implicit(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.data = self.data[self.data["interaction_type"] == 0].reset_index(drop=True)
        columns = [
            "utc_ts_milliseconds",
            "user_id",
            "series_id",
            "recommendation_id",
            "vision_factor",
        ]
        self.data = self.data[columns]

        self.data2["recommended_series_list"] = (
            self.data2["recommended_series_list"]
            .str.replace(r"(\[|\])", "", regex=True)
            .str.split()
        )
        self.data2 = self.data2.explode("recommended_series_list").reset_index(
            drop=True
        )

        merged = self.data.merge(self.data2, "inner", "recommendation_id")
        merged["recommended_series_list"] = pd.to_numeric(
            merged["recommended_series_list"]
        )
        merged.loc[
            merged["series_id"] == merged["recommended_series_list"], "target"
        ] = 1
        merged.loc[
            merged["series_id"] != merged["recommended_series_list"], "target"
        ] = 0
        merged = merged[["user_id", "recommended_series_list", "target"]]
        merged["target"] = merged["target"].astype(int)
        merged.columns = ["user", "item", "target"]

        merged = merged.groupby(["user", "item"]).agg({"target": "sum"}).reset_index()
        merged.loc[merged["target"] > 0, "target"] = 1

        user_to_idx = {user: idx for idx, user in enumerate(merged["user"].unique())}
        item_to_idx = {item: idx for idx, item in enumerate(merged["item"].unique())}
        merged["user"] = merged["user"].map(user_to_idx)
        merged["item"] = merged["item"].map(item_to_idx)

        train_data = merged[:800_000].reset_index(drop=True)
        val_data = merged[800_000:1_000_000].reset_index(drop=True)
        test_data = merged[1_000_000:].reset_index(drop=True)

        return train_data, val_data, test_data

    def _prepare_implicit_bpr(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.data = self.data[self.data["interaction_type"] == 0].reset_index(drop=True)
        columns = [
            "utc_ts_milliseconds",
            "user_id",
            "series_id",
            "recommendation_id",
            "vision_factor",
        ]
        self.data = self.data[columns]

        self.data2["recommended_series_list"] = (
            self.data2["recommended_series_list"]
            .str.replace(r"(\[|\])", "", regex=True)
            .str.split()
        )
        self.data2 = self.data2.explode("recommended_series_list").reset_index(
            drop=True
        )

        merged = self.data.merge(self.data2, "inner", "recommendation_id")
        merged["recommended_series_list"] = pd.to_numeric(
            merged["recommended_series_list"]
        )
        merged.loc[
            merged["series_id"] == merged["recommended_series_list"], "target"
        ] = 1
        merged.loc[
            merged["series_id"] != merged["recommended_series_list"], "target"
        ] = 0
        merged = merged[["user_id", "recommended_series_list", "target"]]
        merged["target"] = merged["target"].astype(int)
        merged.columns = ["user", "item", "target"]

        merged = merged.groupby(["user", "item"]).agg({"target": "sum"}).reset_index()
        merged.loc[merged["target"] > 0, "target"] = 1

        user_to_idx = {user: idx for idx, user in enumerate(merged["user"].unique())}
        item_to_idx = {item: idx for idx, item in enumerate(merged["item"].unique())}
        merged["user"] = merged["user"].map(user_to_idx)
        merged["item"] = merged["item"].map(item_to_idx)

        train_data = merged[:800_000].reset_index(drop=True)
        tmp0 = train_data.loc[train_data["target"] == 0, ["user", "item"]]
        tmp1 = train_data.loc[train_data["target"] == 1, ["user", "item"]]
        train_data = tmp0.merge(tmp1, "inner", "user", suffixes=("_neg", "_pos"))
        train_data = train_data.sample(frac=0.2, random_state=0).reset_index(drop=True)
        val_data = merged[800_000:1_000_000].reset_index(drop=True)
        test_data = merged[1_000_000:].reset_index(drop=True)

        return train_data, val_data, test_data

    def save_data(self, train_data, val_data, test_data):
        train_data.to_csv(
            f"data/contentwise/train_data_{self.data_type}.csv",
            index=False,
            header=False,
        )
        val_data.to_csv(
            f"data/contentwise/val_data_{self.data_type}.csv", index=False, header=False
        )
        test_data.to_csv(
            f"data/contentwise/test_data_{self.data_type}.csv",
            index=False,
            header=False,
        )
