# src/experiments/supervised.py

from typing import Any, Dict, List, Optional, Union
import glob
import re
import ast
import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_iris, fetch_california_housing


def read_result_csv(experiment_type: str) -> pd.DataFrame:
    """
    Load and concatenate CSVs matching '{experiment_type}_experiment*.csv' in 'data/results'.
    'experiment_type' must be 'reg' or 'class'.
    """
    if experiment_type not in {"reg", "class"}:
        raise ValueError("experiment_type must be 'reg' or 'class'")
    pattern = rf"{experiment_type}_experiment\d+"
    csv_paths = glob.glob("data/results/*.csv")
    matched = [p for p in csv_paths if re.search(pattern, p)]
    if not matched:
        return pd.DataFrame()
    dfs: List[pd.DataFrame] = []
    for path in matched:
        df = pd.read_csv(path, index_col=None)
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
    Parse 'data_params' which may be:
      - a dict,
      - a literal string like "{'n_samples':100,'n_features':10}",
      - or "ClassName(n_samples=100,n_features=10)".
    Returns {'n_samples': int, 'n_features': int}.
    """
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()
    if s.startswith("{") and s.endswith("}"):
        return ast.literal_eval(s)  # type: ignore[arg-type]
    m = re.search(r"n_samples\s*=\s*(\d+)\s*,\s*n_features\s*=\s*(\d+)", s)
    if m:
        return {"n_samples": int(m.group(1)), "n_features": int(m.group(2))}
    raise ValueError(f"Cannot parse data_params: {raw!r}")


def extract_reg(df: pd.DataFrame) -> pd.DataFrame:
    """
    From regression-results DataFrame, extract:
      - model_name, data_size, mae, mse, r², train_time, pred_time.
    """
    records: List[Dict[str, Union[str, float]]] = []
    for _, row in df.iterrows():
        params = _parse_data_params(row["data_params"])
        size_str = f"{params['n_samples']}x{params['n_features']}"
        records.append({
            "model_name": row["model_name"],
            "data_size": size_str,
            "mae": row["mae"],
            "mse": row["mse"],
            "r²": row["r2"],
            "train_time": row["train_time"],
            "pred_time": row["pred_time"],
        })
    return pd.DataFrame(records)


def extract_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    From classification-results DataFrame, extract:
      - model_name, data_size, accuracy, train_time (s), pred_time (s).
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
    Group regression results by 'model_name' and format:
      - train_time (s), pred_time (s), normalized_mse, r² as "mean ± std".
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
    agg.columns = ["_".join(col).strip("_") for col in agg.columns.values]
    for base in ["train_time (s)", "pred_time (s)", "normalized_mse", "r2"]:
        mc = f"{base}_mean"
        sc = f"{base}_std"
        agg[base] = (
            agg[mc].round(4).astype(str) + " ± " + agg[sc].round(2).astype(str)
        )
        agg = agg.drop(columns=[mc, sc])
    return agg[[
        "model_name",
        "train_time (s)",
        "pred_time (s)",
        "normalized_mse",
        "r2",
    ]]


def average_class_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group classification results by 'model_name' and format:
      - train_time (s), pred_time (s), accuracy as "mean ± std".
    """
    agg = df.groupby("model_name").agg({
        "train_time (s)": ["mean", "std"],
        "pred_time (s)":  ["mean", "std"],
        "accuracy":       ["mean", "std"],
    }).reset_index()
    agg.columns = ["_".join(col).strip("_") for col in agg.columns.values]
    for base in ["train_time (s)", "pred_time (s)", "accuracy"]:
        mc = f"{base}_mean"
        sc = f"{base}_std"
        agg[base] = (
            agg[mc].round(4).astype(str) + " ± " + agg[sc].round(2).astype(str)
        )
        agg = agg.drop(columns=[mc, sc])
    return agg[["model_name", "train_time (s)", "pred_time (s)", "accuracy"]]


class FitClass:
    """
    Routines to train or tune classification models:
      - RF, DT, FBT.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def fit_fbt_classifier(
        self,
        fbt_model: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray]
    ) -> Any:
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=self.seed
        )
        rf.fit(X_train, y_train)
        cols = (
            X_train.columns.tolist()
            if isinstance(X_train, pd.DataFrame)
            else [f"feature {i}" for i in range(X_train.shape[1])]
        )
        types = pd.Series("float64", index=cols)
        fbt_model.fit(
            rf,
            X_train,
            y_train,
            feature_types=types,
            feature_names=cols,
            minimal_forest_size=10,
            amount_of_branches_threshold=50,
            exclusion_threshold=0.8
        )
        return fbt_model

    def tune_fbt_classifier(
        self,
        fbt_model: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray]
    ) -> Any:
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
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray]
    ) -> BaseEstimator:
        rf_model.fit(X_train, y_train)
        return rf_model

    def tune_dt_classifier(
        self,
        dt_model: BaseEstimator,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray]
    ) -> BaseEstimator:
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
    Routines to train or tune regression models:
      - RF, DT, FBT.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def fit_fbt_regressor(
        self,
        fbt_model: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray]
    ) -> Any:
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=self.seed
        )
        rf.fit(X_train, y_train)
        cols = (
            X_train.columns.tolist()
            if isinstance(X_train, pd.DataFrame)
            else [f"feature {i}" for i in range(X_train.shape[1])]
        )
        types = pd.Series("float64", index=cols)
        fbt_model.fit(
            rf,
            X_train,
            y_train,
            feature_types=types,
            feature_names=cols,
            minimal_forest_size=10,
            amount_of_branches_threshold=50,
            exclusion_threshold=0.8
        )
        return fbt_model

    def tune_fbt_regressor(
        self,
        fbt_model: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray]
    ) -> Any:
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
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray]
    ) -> BaseEstimator:
        rf_model.fit(X_train, y_train)
        return rf_model

    def tune_dt_regressor(
        self,
        dt_model: BaseEstimator,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray]
    ) -> BaseEstimator:
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


class Experiment:
    """
    Manages parameterized experiments: synthetic data generation and evaluation.
    """

    size_definitions = {
        "small":  {"n_samples": [70, 100, 130],      "n_features": [10, 15, 20]},
        "medium": {"n_samples": [700, 1000, 1300],   "n_features": [50, 70, 100]},
        "large":  {"n_samples": [7000, 10000, 13000],"n_features": [200, 300, 400]},
        "mixed": {
            "n_samples": [70, 100, 130, 700, 1000, 1300, 7000, 10000, 13000],
            "n_features": [10, 15, 20, 50, 70, 100, 200]
        },
    }

    information_levels = {
        "low":    {"n_informative": 0.1, "n_redundant": 0.05, "n_repeated": 0.05},
        "medium": {"n_informative": 0.4, "n_redundant": 0.2,  "n_repeated": 0.1},
        "high":   {"n_informative": 0.6, "n_redundant": 0.3,  "n_repeated": 0.2},
        "mixed": {
            "n_informative": [0.1, 0.3, 0.5],
            "n_redundant":   [0.05, 0.2, 0.3],
            "n_repeated":    [0.05, 0.1, 0.2],
        },
    }

    prediction_levels = {
        "narrow": {"n_classes": [2, 3], "tail_strength": [0.1, 0.2]},
        "medium": {"n_classes": [4, 5], "tail_strength": [0.3, 0.4]},
        "spread": {"n_classes": [6, 7], "tail_strength": [0.5, 0.6]},
        "mixed":  {
            "n_classes":    [2, 3, 4, 5, 6, 7],
            "tail_strength": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        },
    }

    def __init__(self, params: Dict[str, Any]) -> None:
        meta = params.get("meta-params", {})
        if "is_classification" not in meta:
            raise ValueError("The 'is_classification' key must be in 'meta-params'.")
        meta.setdefault("use_cross_validation", False)
        meta.setdefault("cv_folds", 5)
        meta.setdefault("random_state", None)
        self.params = {"meta-params": meta, "data-params": params.get("data-params", []), "model-params": {}}
        self.results: List[Dict[str, Any]] = []

    def _update_params(self, category: str, new_vals: Dict[str, Any], index: Optional[int] = None) -> None:
        if category not in self.params:
            return
        if index is None:
            for k, v in new_vals.items():
                self.params[category][k] = v
        else:
            if 0 <= index < len(self.params[category]):
                for k, v in new_vals.items():
                    self.params[category][index][k] = v

    def update_meta_params(self, new_meta: Dict[str, Any]) -> None:
        self._update_params("meta-params", new_meta)

    def update_data_params(self, idx: int, new_data: Dict[str, Any]) -> None:
        self._update_params("data-params", new_data, idx)

    def _flatten_dict(self, d: Dict[str, Any], parent: str = "") -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
        for k, v in d.items():
            key = f"{parent}{k}" if parent else k
            if isinstance(v, dict):
                flat.update(self._flatten_dict(v, key))
            else:
                flat[key] = v
        return flat

    def _format_pair(self, value: Any) -> str:
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    def print_dict(self, d: Dict[str, Any], horizontal: bool = True) -> None:
        flat = self._flatten_dict(d)
        max_key = max(len(k) for k in flat)
        key_w = min(max(max_key, 10), 20)
        if horizontal:
            hdr = " | ".join(f"{k:<{key_w}}" for k in flat)
            print(hdr)
            print("=" * len(hdr))
            row = " | ".join(f"{self._format_pair(v):<{key_w}}" for v in flat.values())
            print(row)
        else:
            print(f"{'Key':<{key_w}} {'Value':<10}")
            print("=" * (key_w + 10))
            for k, v in flat.items():
                val_str = self._format_pair(v)
                print(f"{k:<{key_w}} {val_str:<10}")

    def print_side_by_side(self, dataset_id: int) -> None:
        results = [r for r in self.results if r["dataset_id"] == dataset_id]
        if not results:
            return
        names = [r["model_name"] for r in results]
        keys = sorted(k for k in results[0].keys() if k not in {"meta_params", "data_params", "dataset_id", "model_name", "model_index"})
        col_w = max(max(len(k) for k in keys) + 2, 15)
        name_w = max(len(name) for name in names) + 10
        hdr = f"{'Metric':<{col_w}}" + " | ".join(f"{name:<{name_w}}" for name in names)
        print(hdr)
        print("=" * len(hdr))
        for k in keys:
            row = [f"{k:<{col_w}}"]
            for r in results:
                v = r[k]
                v = round(v, 4) if isinstance(v, float) else v
                row.append(f"{str(v):<{name_w}}")
            print(" | ".join(row))

    def assemble_results_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def perform_cross_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any,
        n_folds: int,
        is_classification: bool,
        fit_fn: Optional[Any],
        random_state: Optional[int],
    ) -> Dict[str, Any]:
        from sklearn.model_selection import KFold
        from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        ttimes, ptimes = [], []
        accs, mses, maes, r2s = [], [], [], []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            st = time.time()
            if fit_fn:
                model = fit_fn(model, X_train, y_train)
            else:
                model.fit(X_train, y_train)
            ttimes.append(time.time() - st)

            sp = time.time()
            y_pred = model.predict(X_test)
            ptimes.append(time.time() - sp)

            if is_classification:
                accs.append(accuracy_score(y_test, y_pred))
            else:
                mses.append(mean_squared_error(y_test, y_pred))
                maes.append(mean_absolute_error(y_test, y_pred))
                r2s.append(r2_score(y_test, y_pred))

        results: Dict[str, Any] = {
            "train_time (s)": np.mean(ttimes),
            "pred_time (s)":  np.mean(ptimes),
        }
        if is_classification:
            results["accuracy"] = np.mean(accs)
        else:
            results["mse"] = np.mean(mses)
            results["mae"] = np.mean(maes)
            results["r2"]  = np.mean(r2s)
        return results

    def perform_train_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any,
        is_classification: bool,
        fit_fn: Optional[Any],
        random_state: Optional[int],
    ) -> Dict[str, Any]:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        st = time.time()
        if fit_fn:
            model = fit_fn(model, X_train, y_train)
        else:
            model.fit(X_train, y_train)
        tt = time.time() - st

        sp = time.time()
        y_pred = model.predict(X_test)
        pt = time.time() - sp

        results: Dict[str, Any] = {
            "train_time (s)": tt,
            "pred_time (s)":  pt,
        }
        if is_classification:
            results["accuracy"] = accuracy_score(y_test, y_pred)
        else:
            results["mse"] = mean_squared_error(y_test, y_pred)
            results["mae"] = mean_absolute_error(y_test, y_pred)
            results["r2"]  = r2_score(y_test, y_pred)
        return results

    def run_single_experiment(
        self,
        meta: Dict[str, Any],
        data_params: Dict[str, Any],
        model: Any,
        dataset_id: Optional[int],
        fit_fn: Optional[Any],
    ) -> None:
        if meta["is_classification"]:
            from sklearn.datasets import make_classification
            X, y = make_classification(**data_params)
        else:
            from sklearn.datasets import make_regression
            X, y = make_regression(**data_params)

        if meta["use_cross_validation"]:
            res = self.perform_cross_validation(
                X, y, model,
                n_folds=meta["cv_folds"],
                is_classification=meta["is_classification"],
                fit_fn=fit_fn,
                random_state=meta["random_state"],
            )
        else:
            res = self.perform_train_test(
                X, y, model,
                is_classification=meta["is_classification"],
                fit_fn=fit_fn,
                random_state=meta["random_state"],
            )

        tgt_avg = np.mean(y)
        res["target_avg"] = tgt_avg
        res["target_var"] = np.mean((y - tgt_avg) ** 2)
        res["target_mad"] = np.mean(np.abs(y - tgt_avg))

        entry: Dict[str, Any] = {
            "dataset_id": dataset_id,
            "experiment_id": len(self.results) + 1
        }
        entry.update(res)
        entry.update({"meta_params": meta, "data_params": data_params})
        self.results.append(entry)

    def populate_data_params(
        self,
        num_datasets: int,
        overall_size: str,
        information: str,
        prediction: str
    ) -> None:
        if overall_size not in self.size_definitions:
            return
        if information not in self.information_levels:
            return
        if prediction not in self.prediction_levels:
            return

        sizes = self.size_definitions[overall_size]
        info = self.information_levels[information]
        pred = self.prediction_levels[prediction]

        def count_combos(d: Dict[str, Any]) -> int:
            prod = 1
            for v in d.values():
                prod *= len(v) if isinstance(v, list) else 1
            return prod

        max_combos = count_combos(sizes) * count_combos(info) * count_combos(pred)
        if num_datasets > max_combos:
            num_datasets = max_combos

        combos = set()
        rng = np.random.default_rng(self.params["meta-params"]["random_state"])
        while len(self.params["data-params"]) < num_datasets:
            n_samp = rng.choice(sizes["n_samples"])
            n_feat = rng.choice(sizes["n_features"])
            if information == "mixed":
                ni = int(round(rng.choice(info["n_informative"]) * n_feat))
                nr = int(round(rng.choice(info["n_redundant"]) * n_feat))
                npd= int(round(rng.choice(info["n_repeated"]) * n_feat))
            else:
                ni = int(round(info["n_informative"] * n_feat))
                nr = int(round(info["n_redundant"] * n_feat))
                npd= int(round(info["n_repeated"] * n_feat))
            total = ni + nr + npd
            if total > n_feat:
                factor = n_feat / total
                ni = int(round(ni * factor))
                nr = int(round(nr * factor))
                npd= int(round(npd * factor))

            dp: Dict[str, Any] = {
                "n_samples": n_samp,
                "n_features": n_feat,
                "n_informative": ni,
                "random_state": self.params["meta-params"]["random_state"],
            }
            if self.params["meta-params"]["is_classification"]:
                nc = rng.choice(pred["n_classes"]) if prediction != "mixed" else rng.choice(pred["n_classes"])
                if nc * 2 > 2**ni:
                    continue
                dp.update({
                    "n_classes": nc,
                    "n_redundant": nr,
                    "n_repeated": npd,
                })
            else:
                ts = rng.choice(pred["tail_strength"]) if prediction != "mixed" else rng.choice(pred["tail_strength"])
                dp["tail_strength"] = ts

            tpl = tuple(sorted(dp.items()))
            if tpl not in combos:
                combos.add(tpl)
                self.params["data-params"].append(dp)

    def perform_experiments(
        self,
        num_datasets: int,
        overall_size: str,
        information: str,
        prediction: str,
        model_instances: Union[List[Any], Any],
        fit_functions: Optional[Union[List[Any], Any]] = None,
    ) -> pd.DataFrame:
        self.populate_data_params(num_datasets, overall_size, information, prediction)
        meta = self.params["meta-params"]
        models = (
            model_instances
            if isinstance(model_instances, list)
            else [model_instances]
        )
        fits = (
            fit_functions
            if isinstance(fit_functions, list)
            else [fit_functions] * len(models)
            if fit_functions is not None
            else [None] * len(models)
        )

        for ds_id, data_params in enumerate(self.params["data-params"], start=1):
            print(f"\nDataset ID: {ds_id}")
            self.print_dict(data_params, horizontal=True)
            for idx, (mdl, fn) in enumerate(zip(models, fits)):
                name = type(mdl).__name__
                print(name)
                self.run_single_experiment(meta, data_params, mdl, ds_id, fn)
                self.results[-1]["model_name"] = name
                self.results[-1]["model_index"] = idx
            self.print_side_by_side(ds_id)
            print("\n")

        print(f"Completed {len(self.params['data-params'])} experiments.")
        return self.assemble_results_dataframe()


if __name__ == "__main__":
    """
    Demo of Experiment, FitClass, and FitReg workflows.
    """

    SEED = 0

    # 1) Classification example
    params_class = {
        "meta-params": {
            "is_classification": True,
            "random_state": SEED,
            "use_cross_validation": True,
            "cv_folds": 3,
        },
        "data-params": [],
        "model-params": {},
    }

    rf_clf = RandomForestClassifier(
        random_state=params_class["meta-params"]["random_state"],
        n_estimators=10,
        max_depth=5,
    )
    dt_clf = DecisionTreeClassifier(
        random_state=params_class["meta-params"]["random_state"]
    )

    try:
        from src.xtrees.model.fbt import ForestBasedTree
        fbt_clf = ForestBasedTree(random_state=SEED, verbose=False)
    except ImportError:
        fbt_clf = None

    fit_cls = FitClass(SEED)
    models_cls: List[Any] = [rf_clf, dt_clf]
    fits_cls: List[Any] = [fit_cls.fit_rf_classifier, fit_cls.tune_dt_classifier]

    if fbt_clf is not None:
        models_cls.append(fbt_clf)
        fits_cls.append(fit_cls.fit_fbt_classifier)

    exp_cls = Experiment(params_class)
    results_cls = exp_cls.perform_experiments(
        num_datasets=1,
        overall_size="small",
        information="low",
        prediction="narrow",
        model_instances=models_cls,
        fit_functions=fits_cls,
    )
    print("\nClassification Results:")
    print(results_cls.head())

    # 2) Regression example
    params_reg = {
        "meta-params": {
            "is_classification": False,
            "random_state": SEED,
            "use_cross_validation": True,
            "cv_folds": 3,
        },
        "data-params": [],
        "model-params": {},
    }

    rf_reg = RandomForestRegressor(
        random_state=params_reg["meta-params"]["random_state"],
        n_estimators=10,
        max_depth=5,
    )
    dt_reg = DecisionTreeRegressor(
        random_state=params_reg["meta-params"]["random_state"]
    )

    try:
        from src.xtrees.model.fbt import ForestBasedTree
        fbt_reg = ForestBasedTree(random_state=SEED, verbose=False)
    except ImportError:
        fbt_reg = None

    fit_reg = FitReg(SEED)
    models_reg: List[Any] = [rf_reg, dt_reg]
    fits_reg: List[Any] = [fit_reg.fit_rf_regressor, fit_reg.tune_dt_regressor]

    if fbt_reg is not None:
        models_reg.append(fbt_reg)
        fits_reg.append(fit_reg.fit_fbt_regressor)

    exp_reg = Experiment(params_reg)
    results_reg = exp_reg.perform_experiments(
        num_datasets=1,
        overall_size="small",
        information="low",
        prediction="narrow",
        model_instances=models_reg,
        fit_functions=fits_reg,
    )
    print("\nRegression Results:")
    print(results_reg.head())
