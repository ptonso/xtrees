from typing import Any, Dict, List, Optional, Union
import glob
import re
import ast

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from src.utils import show_df


def read_result_csv(experiment_type: str) -> pd.DataFrame:
    """
    Load and concatenate all CSV files matching '{experiment_type}_experiment*.csv'
    under 'data/results'. 'experiment_type' must be either 'reg' or 'class'.
    """
    if experiment_type not in {"reg", "class"}:
        raise ValueError("experiment_type must be 'reg' or 'class'")

    pattern = rf"{experiment_type}_experiment\d+"
    csv_paths = glob.glob("data/results/*.csv")
    matched_files = [p for p in csv_paths if re.search(pattern, p)]

    if not matched_files:
        return pd.DataFrame()

    dfs: List[pd.DataFrame] = []
    for path in matched_files:
        df = pd.read_csv(path, index_col=None)
        # Drop the unnamed index column if present
        if df.columns[0].startswith("Unnamed"):
            df = df.drop(df.columns[0], axis=1)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    if "experiment_id" in combined.columns:
        combined = combined.drop(columns=["experiment_id"])
    combined.reset_index(drop=True, inplace=True)
    combined.index += 1
    combined.index.name = "experiment_id"
    return combined



def _parse_data_params(raw: Any) -> Dict[str, int]:
    """
    Parse a 'data_params' field into {'n_samples':…, 'n_features':…}.
    Accepts:
      • real dicts
      • call-style strings  Foo(n_samples=…, n_features=…)
      • dict-style strings  "{'n_samples': np.int64(7000), 'n_features': …}"
    """
    if isinstance(raw, dict):
        return {k: int(v) for k, v in raw.items() if k in {"n_samples", "n_features"}}

    s = str(raw).strip()

    m = re.search(r"n_samples\s*=\s*(\d+)\s*,\s*n_features\s*=\s*(\d+)", s)
    if m:
        return {"n_samples": int(m.group(1)), "n_features": int(m.group(2))}

    if s.startswith("{") and s.endswith("}"):
        cleaned = re.sub(r"np\.(?:int|float)\d*\s*\((\d+(?:\.\d+)?)\)", r"\1", s)
        try:
            literal = ast.literal_eval(cleaned)
            return {
                "n_samples": int(literal["n_samples"]),
                "n_features": int(literal["n_features"]),
            }
        except Exception:
            pass
    raise ValueError(f"Cannot parse data_params: {raw!r}")


def extract_reg(df: pd.DataFrame) -> pd.DataFrame:
    """
    From a regression‐experiment results DataFrame, extract:
      - model_name
      - data_size (formatted as "n_samples x n_features")
      - mae, mse, r², train_time, pred_time.
    """
    records: List[Dict[str, Union[str, float]]] = []
    for _, row in df.iterrows():
        params = _parse_data_params(row["data_params"])
        size_str = f"{params['n_samples']}x{params['n_features']}"
        records.append({
            "model_name":  row["model_name"],
            "data_size":   size_str,
            "mae":         row["mae"],
            "mse":         row["mse"],
            "r²":          row["r2"],
            "train_time (s)":  row["train_time (s)"],
            "pred_time":   row["pred_time (s)"],
        })
    return pd.DataFrame(records)


def extract_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    From a classification‐experiment results DataFrame, extract:
      - model_name
      - data_size (formatted as "n_samples x n_features")
      - accuracy, train_time (s), pred_time (s).
    """
    records: List[Dict[str, Union[str, float]]] = []
    for _, row in df.iterrows():
        params = _parse_data_params(row["data_params"])
        size_str = f"{params['n_samples']}x{params['n_features']}"
        records.append({
            "model_name":      row["model_name"],
            "data_size":       size_str,
            "accuracy":        row["accuracy"],
            "train_time (s)":  row["train_time (s)"],
            "pred_time (s)":   row["pred_time (s)"],
        })
    return pd.DataFrame(records)


def average_reg_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group regression results by 'model_name' and report:
      - mean ± std for train_time, pred_time, normalized_mse, and r².
    """
    df = df.copy()
    df["normalized_mae"] = df["mae"] / df["target_mad"]
    df["normalized_mse"] = df["mse"] / df["target_var"]

    agg = df.groupby("model_name").agg({
        "train_time (s)":  ["mean", "std"],
        "pred_time (s)":   ["mean", "std"],
        "normalized_mse":  ["mean", "std"],
        "r2":              ["mean", "std"],
    }).reset_index()

    # Flatten MultiIndex columns
    agg.columns = ["_".join(col).strip("_") for col in agg.columns.values]

    # Format each metric as "mean ± std"
    for base in ["train_time (s)", "pred_time (s)", "normalized_mse", "r2"]:
        mean_col = f"{base}_mean"
        std_col = f"{base}_std"
        agg[base] = (
            agg[mean_col].round(4).astype(str)
            + " ± "
            + agg[std_col].round(2).astype(str)
        )
        agg = agg.drop(columns=[mean_col, std_col])

    # Re‐order columns
    return agg[[
        "model_name",
        "train_time (s)",
        "pred_time (s)",
        "normalized_mse",
        "r2",
    ]]


def average_class_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group classification results by 'model_name' and report:
      - mean ± std for train_time, pred_time, and accuracy.
    """
    agg = df.groupby("model_name").agg({
        "train_time (s)": ["mean", "std"],
        "pred_time (s)":  ["mean", "std"],
        "accuracy":       ["mean", "std"],
    }).reset_index()

    agg.columns = ["_".join(col).strip("_") for col in agg.columns.values]

    for base in ["train_time (s)", "pred_time (s)", "accuracy"]:
        mean_col = f"{base}_mean"
        std_col = f"{base}_std"
        agg[base] = (
            agg[mean_col].round(4).astype(str)
            + " ± "
            + agg[std_col].round(2).astype(str)
        )
        agg = agg.drop(columns=[mean_col, std_col])

    return agg[["model_name", "train_time (s)", "pred_time (s)", "accuracy"]]


class FitClass:
    """
    Helper class containing routines to train or tune classification models:
      - Fit a ForestBasedTree (FBT) on top of a RandomForestClassifier.
      - Optionally perform a RandomizedSearchCV to tune FBT hyperparameters.
      - Fit or tune vanilla RandomForestClassifier or DecisionTreeClassifier.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def fit_fbt_classifier(
        self,
        fbt_model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Any:
        """
        Train an FBT model by first fitting a RandomForestClassifier internally.
        Assumes fbt_model has a 'fit(rf, X, y, feature_types, feature_names, ...)' signature.
        """
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=self.seed
        )
        rf.fit(X_train, y_train)

        feature_names = list(X_train.columns)
        feature_types = pd.Series("float64", index=feature_names)

        fbt_model.fit(
            rf,
            X_train,
            y_train,
            feature_types=feature_types,
            feature_names=feature_names,
            minimal_forest_size=10,
            amount_of_branches_threshold=50,
            exclusion_threshold=0.8
        )
        return fbt_model

    def tune_fbt_classifier(
        self,
        fbt_model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Any:
        """
        Perform RandomizedSearchCV over FBT hyperparameters for classification.
        """
        param_dist = {
            "num_of_estimators":            [10, 50, 100, 200],
            "max_depth":                    [3, 5, 10, None],
            "minimal_forest_size":          [5, 10, 20],
            "amount_of_branches_threshold": [10, 20, 30, 40],
            "exclusion_threshold":          [0.5, 0.7, 0.8, 0.9],
        }
        search = RandomizedSearchCV(
            fbt_model,
            param_distributions=param_dist,
            n_iter=3,
            cv=3,
            scoring="accuracy",
            random_state=self.seed,
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        return search.best_estimator_

    def fit_rf_classifier(
        self,
        rf_model: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> BaseEstimator:
        """
        Fit a pre‐initialized RandomForestClassifier (or other sklearn estimator).
        """
        rf_model.fit(X_train, y_train)
        return rf_model

    def tune_dt_classifier(
        self,
        dt_model: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> BaseEstimator:
        """
        Perform RandomizedSearchCV over DecisionTreeClassifier hyperparameters.
        """
        param_grid = {
            "max_depth":       [None, 10, 30, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf":  [1, 4, 10],
            "max_features":      [None, "sqrt", "log2"],
            "max_leaf_nodes":    [None, 10, 20, 50],
            "ccp_alpha":         [0.0, 0.01, 0.1, 0.001],
        }
        search = RandomizedSearchCV(
            estimator=dt_model,
            param_distributions=param_grid,
            n_iter=10,
            scoring="accuracy",
            cv=3,
            n_jobs=-1,
            random_state=self.seed
        )
        search.fit(X_train, y_train)
        return search.best_estimator_


class FitReg:
    """
    Helper class containing routines to train or tune regression models:
      - Fit a ForestBasedTree (FBT) on top of a RandomForestRegressor.
      - Optionally perform a RandomizedSearchCV to tune FBT hyperparameters.
      - Fit or tune vanilla RandomForestRegressor or DecisionTreeRegressor.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def fit_fbt_regressor(
        self,
        fbt_model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Any:
        """
        Train an FBT model by first fitting a RandomForestRegressor internally.
        Assumes fbt_model has a 'fit(rf, X, y, feature_types, feature_names, ...)' signature.
        """
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=self.seed
        )
        rf.fit(X_train, y_train)

        feature_names = list(X_train.columns)
        feature_types = pd.Series("float64", index=feature_names)

        fbt_model.fit(
            rf,
            X_train,
            y_train,
            feature_types=feature_types,
            feature_names=feature_names,
            minimal_forest_size=10,
            amount_of_branches_threshold=50,
            exclusion_threshold=0.8
        )
        return fbt_model

    def tune_fbt_regressor(
        self,
        fbt_model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Any:
        """
        Perform RandomizedSearchCV over FBT hyperparameters for regression.
        """
        param_dist = {
            "num_of_estimators":            [10, 50, 100, 200],
            "max_depth":                    [3, 5, 10, None],
            "minimal_forest_size":          [5, 10, 20],
            "amount_of_branches_threshold": [10, 20, 30, 40],
            "exclusion_threshold":          [0.5, 0.7, 0.8, 0.9],
        }
        search = RandomizedSearchCV(
            fbt_model,
            param_distributions=param_dist,
            n_iter=10,
            cv=3,
            scoring="neg_mean_squared_error",
            random_state=self.seed,
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        return search.best_estimator_

    def fit_rf_regressor(
        self,
        rf_model: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> BaseEstimator:
        """
        Fit a pre‐initialized RandomForestRegressor (or other sklearn estimator).
        """
        rf_model.fit(X_train, y_train)
        return rf_model

    def tune_dt_regressor(
        self,
        dt_model: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> BaseEstimator:
        """
        Perform RandomizedSearchCV over DecisionTreeRegressor hyperparameters.
        """
        param_grid = {
            "max_depth":        [None, 10, 30, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf":  [1, 4, 10],
            "max_features":      [None, "sqrt", "log2"],
            "max_leaf_nodes":    [None, 10, 20, 50],
            "ccp_alpha":         [0.0, 0.01, 0.1, 0.001],
        }
        search = RandomizedSearchCV(
            estimator=dt_model,
            param_distributions=param_grid,
            n_iter=3,
            scoring="neg_mean_squared_error",
            cv=3,
            n_jobs=-1,
            random_state=self.seed
        )
        search.fit(X_train, y_train)
        return search.best_estimator_
    


if __name__ == "__main__":
    """
    Standalone demo:
    - Fit a small RandomForestClassifier and RandomForestRegressor
      into placeholder 'ForestBasedTree'-style instances (replace with your actual FBT class).
    """
    from sklearn.datasets import load_iris, fetch_california_housing
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    # Replace `ForestBasedTree` with the actual import from your package:
    try:
        from src.xtrees.model.fbt import ForestBasedTree
    except ImportError:
        ForestBasedTree = None

    # Classification demo
    iris = load_iris()
    Xc, yc = pd.DataFrame(iris.data, columns=iris.feature_names), iris.target
    rf_clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0)
    rf_clf.fit(Xc, yc)
    if ForestBasedTree is not None:
        fbt_cls = ForestBasedTree(random_state=0)
        fbt_cls.fit(rf_clf, Xc, yc, pd.Series("float64", index=Xc.columns), Xc.columns.tolist())
        print("Classification FBT cs_df head:")
        show_df(fbt_cls.cs_df)
    else:
        print("ForestBasedTree not found; skipping classification demo.")

    # Regression demo
    calif = fetch_california_housing()
    Xr, yr = pd.DataFrame(calif.data, columns=calif.feature_names), calif.target
    rf_reg = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=0)
    rf_reg.fit(Xr, yr)
    if ForestBasedTree is not None:
        fbt_reg = ForestBasedTree(random_state=0)
        fbt_reg.fit(rf_reg, Xr, yr, pd.Series("float64", index=Xr.columns), Xr.columns.tolist())
        print("\nRegression FBT cs_df head:")
        print(show_df(fbt_reg.cs_df))
    else:
        print("ForestBasedTree not found; skipping regression demo.")
