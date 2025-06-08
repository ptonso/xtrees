from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import roc_curve, auc
from statsmodels.distributions.empirical_distribution import ECDF

from src.logger import setup_logger
from src.utils import setup_toy_classifier, setup_toy_regressor

logger = setup_logger("api.log")
EPSILON = 1e-6


class Tree_:
    """
    Lightweight tree container holding node arrays for a custom tree.
    """

    def __init__(self, node_count: int, n_outputs: int, is_classifier: bool) -> None:
        self.node_count: int = node_count
        self.feature: np.ndarray = np.full(node_count, -2, dtype=np.int32)
        self.threshold: np.ndarray = np.full(node_count, -2.0, dtype=np.float64)
        self.children_left: np.ndarray = np.full(node_count, -1, dtype=np.int32)
        self.children_right: np.ndarray = np.full(node_count, -1, dtype=np.int32)
        self.n_node_samples: np.ndarray = np.zeros(node_count, dtype=np.int32)
        if is_classifier:
            # shape: (node_count, 1, n_outputs) for probability vectors
            self.value: np.ndarray = np.zeros((node_count, 1, n_outputs), dtype=np.float64)
        else:
            # shape: (node_count, n_outputs) for regression values
            self.value: np.ndarray = np.zeros((node_count, n_outputs), dtype=np.float64)


class ForestBasedTree:
    """
    Builds a single interpretable tree by merging branches from a random forest.
    """

    def __init__(
        self,
        minimal_forest_size: int = 10,
        amount_of_branches_threshold: int = 100,
        exclusion_threshold: float = 0.8,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self.minimal_forest_size: int = minimal_forest_size
        self.amount_of_branches_threshold: int = amount_of_branches_threshold
        self.exclusion_threshold: float = exclusion_threshold
        self.verbose: bool = verbose
        self.random_state: Optional[int] = random_state
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Will be set in fit
        self.rf: Any = None
        self.is_classifier: bool = False
        self.classes_: Optional[np.ndarray] = None
        self.feature_names_in_: Optional[Union[List[str], pd.Index]] = None
        self.feature_types: Optional[pd.Series] = None
        self.X_train: np.ndarray = np.array([])
        self.y_train: np.ndarray = np.array([])

        self.conjunction_set: Optional["ConjunctionSet"] = None
        self.cs_df: Optional[pd.DataFrame] = None

        self.root: Optional["Node"] = None
        self.node_count: int = 0
        self.tree_: Optional[Tree_] = None
        self.current_node_id: int = 0

    def fit(
        self,
        random_forest: Any,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        feature_types: Optional[Union[List[str], pd.Series]] = None,
        feature_names: Optional[Union[List[str], pd.Index]] = None,
        minimal_forest_size: int = 10,
        amount_of_branches_threshold: int = 100,
        exclusion_threshold: float = 0.8,
        random_state: Optional[int] = None,
    ) -> "ForestBasedTree":
        """
        Fit the ForestBasedTree by extracting and merging forest branches.
        """
        self._initialize_parameters(
            random_forest,
            X_train,
            y_train,
            feature_types,
            feature_names,
            minimal_forest_size,
            amount_of_branches_threshold,
            exclusion_threshold,
            random_state,
        )
        self._validate_forest_dimensions()
        self._build_conjunction_set()
        self._build_internal_tree()
        return self

    def _initialize_parameters(
        self,
        random_forest: Any,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        feature_types: Optional[Union[List[str], pd.Series]],
        feature_names: Optional[Union[List[str], pd.Index]],
        minimal_forest_size: int,
        amount_of_branches_threshold: int,
        exclusion_threshold: float,
        random_state: Optional[int],
    ) -> None:
        self.rf = random_forest
        self.is_classifier = hasattr(random_forest, "classes_")
        if self.is_classifier:
            self.classes_ = random_forest.classes_

        # Convert DataFrame to numpy, capture names/types
        if isinstance(X_train, pd.DataFrame):
            if feature_names is None:
                feature_names = X_train.columns
            if feature_types is None:
                feature_types = X_train.dtypes
            X_train = X_train.to_numpy()
            y_train = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train

        self.feature_names_in_ = feature_names
        self.feature_types = feature_types
        self.X_train = X_train
        self.y_train = y_train

        self.minimal_forest_size = minimal_forest_size
        self.amount_of_branches_threshold = amount_of_branches_threshold
        self.exclusion_threshold = exclusion_threshold

        if not self.random_state:
            self.random_state = random_state
            if self.random_state is not None:
                np.random.seed(self.random_state)

    def _validate_forest_dimensions(self) -> None:
        """
        Check that every tree in the forest has the same number of features as the data.
        """
        n_features = len(self.feature_names_in_)
        for idx, estimator in enumerate(self.rf.estimators_):
            tree = estimator.tree_
            if tree.n_features != n_features:
                raise ValueError(
                    f"Estimator {idx} has {tree.n_features} features; "
                    f"expected {n_features} based on X_train."
                )

    def _build_conjunction_set(self) -> None:
        """
        Build and filter the conjunction set by merging branches and pruning.
        """
        from functools import reduce  # local import to break any circularity

        self.conjunction_set = ConjunctionSet(
            self.feature_names_in_,
            self.X_train,
            self.y_train,
            self.rf,
            self.feature_types,
            self.amount_of_branches_threshold,
            self.random_state,
            exclusion_starting_point=5,
            minimal_forest_size=self.minimal_forest_size,
            exclusion_threshold=self.exclusion_threshold,
            verbose=self.verbose,
        )
        self.cs_df = self.conjunction_set.get_conjunction_set_df().round(decimals=5)        
        if self.is_classifier:
            for i, cls in enumerate(self.rf.classes_):
                self.cs_df[cls] = [probas[i] for probas in self.cs_df["probas"]]
            self.cs_df.columns = self.cs_df.columns.map(str)
        self.root = Node([True] * len(self.cs_df), random_state=self.random_state)
        self.root.split(self.cs_df)


    def _build_internal_tree(self) -> None:
        """
        Count nodes, initialize Tree_ container, and populate it recursively.
        """
        self.node_count = self._count_nodes(self.root)
        n_outputs = len(self.rf.classes_) if self.is_classifier else 1
        self.tree_ = Tree_(self.node_count, n_outputs, self.is_classifier)

        # Reinitialize values array if classification
        if self.is_classifier:
            self.tree_.value = np.zeros((self.node_count, 1, len(self.rf.classes_)))
        else:
            self.tree_.value = np.zeros((self.node_count, 1, 1))

        self.current_node_id = 0
        self._populate_tree(self.root)

    def _count_nodes(self, node: Optional["Node"]) -> int:
        """
        Recursively count the number of nodes in the built tree structure.
        """
        if node is None:
            return 0
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    def _populate_tree(self, node: Optional["Node"]) -> int:
        """
        Recursively fill Tree_ arrays: feature, threshold, children, values, sample counts.
        Returns the assigned node ID.
        """
        if node is None:
            return -1

        node_id = self.current_node_id
        self.current_node_id += 1

        left_id = self._populate_tree(node.left)
        right_id = self._populate_tree(node.right)

        # If leaf, assign node.value directly
        if left_id == -1 and right_id == -1:
            if self.is_classifier:
                self.tree_.value[node_id, 0, :] = (
                    node.value if node.value is not None else np.zeros(len(self.rf.classes_))
                )
            else:
                self.tree_.value[node_id, 0, 0] = (
                    node.value if node.value is not None else np.array([0.0])
                )
        else:
            # Compute weighted value from children
            if self.is_classifier:
                left_val = (
                    self.tree_.value[left_id, 0, :]
                    if left_id != -1
                    else np.zeros(len(self.rf.classes_))
                )
                right_val = (
                    self.tree_.value[right_id, 0, :]
                    if right_id != -1
                    else np.zeros(len(self.rf.classes_))
                )
            else:
                left_val = (
                    self.tree_.value[left_id, 0, 0] if left_id != -1 else np.array([0.0])
                )
                right_val = (
                    self.tree_.value[right_id, 0, 0] if right_id != -1 else np.array([0.0])
                )

            left_samples = (
                self.tree_.n_node_samples[left_id] if left_id != -1 else 0
            )
            right_samples = (
                self.tree_.n_node_samples[right_id] if right_id != -1 else 0
            )
            total_samples = left_samples + right_samples

            if total_samples > 0:
                weighted = (left_val * left_samples + right_val * right_samples) / total_samples
            else:
                weighted = (
                    np.zeros(len(self.rf.classes_)) if self.is_classifier else np.array([0.0])
                )

            if self.is_classifier:
                wsum = weighted.sum()
                if wsum > 0:
                    weighted = weighted / wsum
                else:
                    logger.warning(f"Weighted sum zero at internal node {node_id}")
                self.tree_.value[node_id, 0, :] = weighted
            else:
                self.tree_.value[node_id, 0, 0] = weighted

        # Set feature, threshold, and sample counts
        self.tree_.feature[node_id] = node.feature if node.feature is not None else -2
        self.tree_.threshold[node_id] = node.threshold if node.threshold is not None else -2
        self.tree_.n_node_samples[node_id] = np.sum(node.mask)

        self.tree_.children_left[node_id] = left_id
        self.tree_.children_right[node_id] = right_id

        return node_id

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict labels or values for each instance using the built conjunction tree.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        preds: List[Any] = []
        for inst in X:
            val = self.root.predict_value(inst, self.cs_df)
            if self.is_classifier:
                preds.append(np.argmax(val))
            else:
                preds.append(val)
        return np.array(preds).flatten()

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict probability vectors for a classifier.
        """
        if not self.is_classifier:
            raise ValueError("predict_proba only for classifiers.")
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        probs: List[np.ndarray] = []
        for inst in X:
            probs.append(self.root.predict_value(inst, self.cs_df))
        return np.array(probs)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Return parameters for compatibility.
        """
        return {
            "minimal_forest_size": self.minimal_forest_size,
            "amount_of_branches_threshold": self.amount_of_branches_threshold,
            "exclusion_threshold": self.exclusion_threshold,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    def set_params(self, **params: Any) -> "ForestBasedTree":
        """
        Set parameters for compatibility.
        """
        for k, v in params.items():
            setattr(self, k, v)
        return self



class Branch:
    """
    Represents a single path (branch) from root to leaf in a conjunction tree.
    """

    def __init__(
        self,
        feature_names: List[str],
        feature_types: Union[pd.Series, List[str]],
        output: List[float],
        number_of_samples: Optional[int] = None,
    ) -> None:
        self.feature_types: Union[pd.Series, List[str]] = feature_types
        self.number_of_features: int = len(feature_names)
        self.feature_names: List[str] = feature_names
        self.features_upper: List[float] = [np.inf] * self.number_of_features
        self.features_lower: List[float] = [-np.inf] * self.number_of_features
        self.output: List[float] = output
        self.number_of_samples: Optional[int] = number_of_samples
        self.categorical_features_dict: Dict[str, str] = {}
        self.leaves_indexes: List[str] = []

    def addCondition(self, feature: int, threshold: float, bound: str) -> None:
        """
        Update upper/lower bounds for a feature based on split.
        """
        if feature >= self.number_of_features:
            raise IndexError(f"Feature index {feature} out of range.")

        if bound == "lower":
            if self.features_lower[feature] < threshold:
                self.features_lower[feature] = threshold
                if "=" in self.feature_names[feature] and threshold >= 0:
                    key, val = self.feature_names[feature].split("=")
                    self.categorical_features_dict[key] = val
        else:
            if self.features_upper[feature] > threshold:
                self.features_upper[feature] = threshold

    def contradictBranch(self, other_branch):
        for categorical_feature in self.categorical_features_dict:
            if (
                categorical_feature in other_branch.categorical_features_dict
                and self.categorical_features_dict[categorical_feature]
                != other_branch.categorical_features_dict[categorical_feature]
            ):
                return True

        for i in range(self.number_of_features):
            if self.feature_types is not None:
                if (
                    self.feature_types.iloc[i] == 'int'
                    and min(self.features_upper[i], other_branch.features_upper[i]) % 1 > 0
                    and min(self.features_upper[i], other_branch.features_upper[i])
                    - max(self.features_lower[i], other_branch.features_lower[i])
                    < 1
                ):
                    return True
            if (
                self.features_upper[i] <= other_branch.features_lower[i] + EPSILON
                or self.features_lower[i] + EPSILON >= other_branch.features_upper[i]
            ):
                return True

        return False


    def mergeBranch(self, other: "Branch", is_classifier: bool) -> "Branch":
        """
        Merge two non-contradictory branches by averaging or combining outputs.
        """
        if is_classifier:
            new_output = [a + b for a, b in zip(self.output, other.output)]
            total = sum(new_output)
            if total > 0:
                new_output = [o / total for o in new_output]
        else:
            tot_samples = (self.number_of_samples or 0) + (other.number_of_samples or 0)
            self_out = np.array(self.output)
            other_out = np.array(other.output)
            new_output = (
                (self_out * (self.number_of_samples or 0) + other_out * (other.number_of_samples or 0))
                / tot_samples
                if tot_samples > 0
                else self_out
            )

        new_samples = int(np.sqrt((self.number_of_samples or 1) * (other.number_of_samples or 1)))
        new_branch = Branch(self.feature_names, self.feature_types, new_output, new_samples)
        new_branch.features_upper = self.features_upper.copy()
        new_branch.features_lower = self.features_lower.copy()
        for f in range(self.number_of_features):
            new_branch.addCondition(f, other.features_upper[f], "upper")
            new_branch.addCondition(f, other.features_lower[f], "lower")

        new_branch.categorical_features_dict = {**self.categorical_features_dict, **other.categorical_features_dict}
        new_branch.leaves_indexes = self.leaves_indexes + other.leaves_indexes
        return new_branch

    def printBranch(self) -> None:
        """
        Print human-readable branch conditions and output.
        """
        s = ""
        for i, thr in enumerate(self.features_lower):
            if thr != -np.inf:
                s += f"{i} > {thr:.3f}, "
        for i, thr in enumerate(self.features_upper):
            if thr != np.inf:
                s += f"{i} <= {thr:.3f}, "
        s += f"output: [{', '.join(map(str, self.output))}]"
        s += f" samples: {self.number_of_samples}"
        print(s)

    def containsInstance(self, instance: np.ndarray) -> bool:
        """
        Check if an instance satisfies all conditions in this branch.
        """
        return bool(
            np.all(np.array(self.features_lower) < instance)
            and np.all(instance <= np.array(self.features_upper))
        )

    def getLabel(self) -> int:
        """
        Return the class label for this branch (for classifiers).
        """
        return int(np.argmax(self.output))

    def get_branch_dict(self, ecdf_dict: Dict[int, ECDF]) -> Dict[str, Any]:
        """
        Return a dictionary representation of the branch constraints, probability, output.
        """
        d: Dict[str, Any] = {}
        for i, (up, low) in enumerate(zip(self.features_upper, self.features_lower)):
            d[f"{i}_upper"] = up
            d[f"{i}_lower"] = low
        d["n_samples"] = self.number_of_samples
        d["branch_prob"] = self.calculate_branch_probability_by_ecdf(ecdf_dict)
        d["output"] = np.array(self.output)
        return d

    def calculate_branch_probability_by_ecdf(self, ecdf_dict: Dict[int, ECDF]) -> float:
        """
        Compute the probability of this branch under ECDF for each feature.
        """
        probs: List[float] = []
        delta = 1e-9
        for idx in range(len(ecdf_dict)):
            low = self.features_lower[idx]
            up = self.features_upper[idx]
            p_low, p_up = ecdf_dict[idx]([low, up])
            probs.append(max(p_up - p_low + delta, 0.0))
        return float(np.prod(probs))

    def is_excludable_branch(self, threshold: float) -> bool:
        """
        Determine if branch should be excluded based on output ratio (for classifiers/regressors).
        """
        if self.output:
            if max(self.output) / sum(self.output) > threshold:
                return True
        return False


class ConjunctionSet:
    """
    Builds a conjunction (merged branch) set by pruning and merging branches from a forest.
    """

    def __init__(
        self,
        feature_names: List[str],
        X_train: np.ndarray,
        y_train: np.ndarray,
        model: Any,
        feature_types: Union[pd.Series, List[str]],
        amount_of_branches_threshold: int,
        random_state: Optional[int] = None,
        exclusion_starting_point: int = 5,
        minimal_forest_size: int = 10,
        exclusion_threshold: float = 0.8,
        verbose: bool = True,
    ) -> None:
        self.feature_names: List[str] = feature_names
        self.feature_types: Union[pd.Series, List[str]] = feature_types
        self.amount_of_branches_threshold: int = amount_of_branches_threshold
        self.random_state: Optional[int] = random_state
        self.logger = setup_logger("conjunction_set.log")
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.model: Any = model
        self.is_classifier: bool = hasattr(model, "classes_")
        self.label_names: Optional[np.ndarray] = model.classes_ if self.is_classifier else None
        self.relevant_indexes: List[int] = reduce_error_pruning_cached(
            self.model, X_train, y_train, minimal_forest_size
        )
        self.feature_types = feature_types
        self.exclusion_starting_point: int = exclusion_starting_point
        self.exclusion_threshold: float = exclusion_threshold
        self.verbose: bool = verbose

        self._set_ecdf(X_train)
        self._generate_branches()
        self.number_of_branches_per_iteration: List[int] = []
        self._build_conjunction_set()

    def _set_ecdf(self, data: np.ndarray) -> None:
        """
        Prepare ECDFs for each feature.
        """
        self.ecdf_dict: Dict[int, ECDF] = {i: ECDF(data[:, i]) for i in range(len(self.feature_names))}

    def _generate_branches(self) -> None:
        """
        Extract branches (leaf-to-root paths) from each relevant tree in the forest.
        """
        trees = [estimator.tree_ for estimator in self.model.estimators_]
        self.branches_lists: List[List[Branch]] = []
        for idx, tree_ in enumerate(trees):
            if idx not in self.relevant_indexes:
                continue
            leaf_idxs = [
                i for i in range(tree_.node_count)
                if tree_.children_left[i] == -1 and tree_.children_right[i] == -1
            ]
            branch_list: List[Branch] = [
                self._get_branch_from_leaf(tree_, leaf_idx) for leaf_idx in leaf_idxs
            ]
            for leaf_idx, br in enumerate(branch_list):
                br.leaves_indexes = [f"{idx}_{leaf_idx}"]
            self.branches_lists.append(branch_list)

    def _get_branch_from_leaf(self, tree_: Any, leaf_idx: int) -> Branch:
        """
        Build a Branch object tracing conditions from a leaf up to the root.
        """
        if self.is_classifier:
            probas = tree_.value[leaf_idx][0]
            total = probas.sum()
            output = (probas / total).tolist() if total > 0 else [0.0] * len(probas)
        else:
            output = [tree_.value[leaf_idx][0][0]]  # regression value

        new_branch = Branch(
            self.feature_names,
            self.feature_types,
            output,
            number_of_samples=int(tree_.n_node_samples[leaf_idx]),
        )

        node_id = leaf_idx
        while node_id != 0:
            parent_idx = np.where(tree_.children_left == node_id)[0]
            bound = "upper"
            if parent_idx.size == 0:
                bound = "lower"
                parent_idx = np.where(tree_.children_right == node_id)[0]
            pid = int(parent_idx[0])
            new_branch.addCondition(tree_.feature[pid], tree_.threshold[pid], bound)
            node_id = pid

        return new_branch

    def _build_conjunction_set(self) -> None:
        """
        Merge branches iteratively, apply exclusion and filtering.
        """
        conjunction_set = self.branches_lists[0] if self.branches_lists else []
        excluded: List[Branch] = []

        for i, branch_list in enumerate(self.branches_lists[1:], start=1):
            if self.verbose:
                self.logger.info(f"Iteration {i}: {len(conjunction_set)} conjunctions")
            filter_flag = i < len(self.branches_lists) - 1
            conjunction_set = self._merge_and_filter(branch_list, conjunction_set, filter_flag)
            if (
                i >= self.exclusion_starting_point
                and len(conjunction_set) > 0.8 * self.amount_of_branches_threshold
            ):
                conjunction_set, ex_br = self._exclude_branches(conjunction_set)
                excluded.extend(ex_br)

        self.conjunctionSet: List[Branch] = excluded + conjunction_set
        if self.verbose:
            self.logger.info(f"Final CS size: {len(self.conjunctionSet)}")

    def _merge_and_filter(
        self,
        branch_list: List[Branch],
        conj_set: List[Branch],
        filter_flag: bool,
    ) -> List[Branch]:
        """
        Merge new branches with existing set and optionally filter by probability.
        """
        new_set: List[Branch] = []
        for b1 in conj_set:
            for b2 in branch_list:
                if not b1.contradictBranch(b2):
                    new_set.append(b1.mergeBranch(b2, self.is_classifier))

        if filter_flag:
            new_set = self._filter_conjunction_set(new_set)
        self.number_of_branches_per_iteration.append(len(new_set))
        return new_set

    def _filter_conjunction_set(self, cs: List[Branch]) -> List[Branch]:
        """
        If too many branches, keep top branches by probability.
        """
        if len(cs) <= self.amount_of_branches_threshold:
            return cs

        metrics: List[float] = [b.calculate_branch_probability_by_ecdf(self.ecdf_dict) for b in cs]
        threshold_value = sorted(metrics, reverse=True)[ self.amount_of_branches_threshold - 1 ]
        filtered = [b for b, m in zip(cs, metrics) if m >= threshold_value]
        return filtered[: self.amount_of_branches_threshold]

    def _exclude_branches(self, cs: List[Branch]) -> Tuple[List[Branch], List[Branch]]:
        """
        Split out branches that exceed exclusion threshold.
        """
        keep: List[Branch] = []
        exclude: List[Branch] = []
        for b in cs:
            if b.is_excludable_branch(self.exclusion_threshold):
                exclude.append(b)
            else:
                keep.append(b)
        return keep, exclude

    def get_conjunction_set_df(self) -> pd.DataFrame:
        """
        Return DataFrame of branch dictionaries, with proper column naming.
        """
        df = pd.DataFrame([b.get_branch_dict(self.ecdf_dict) for b in self.conjunctionSet])
        if self.is_classifier:
            return df.rename(columns={"output": "probas"})
        df = df.rename(columns={"output": "regressions"})
        df["regressions"] = df["regressions"].apply(
            lambda x: float(x[0]) if isinstance(x, np.ndarray) and x.shape == (1,) else float(x)
        )
        return df

    def predict(self, X: np.ndarray) -> List[Any]:
        """
        For each instance, find the first branch that contains it and return that branch’s label/output.
        """
        preds: List[Any] = []
        for inst in X:
            for conj in self.conjunctionSet:
                if conj.containsInstance(inst):
                    preds.append(self.label_names[conj.getLabel()] if self.is_classifier else conj.output)
                    break
        return preds


def predict_instance_with_included_tree(
    model: Any, included_indexes: List[int], inst: np.ndarray
) -> np.ndarray:
    """
    Ensemble prediction from a subset of trees in a random forest for one instance.
    """
    if hasattr(model, "n_classes_"):
        v = np.zeros(model.n_classes_)
        for i in included_indexes:
            v += model.estimators_[i].predict_proba(inst.reshape(1, -1))[0]
        return v / v.sum() if v.sum() > 0 else v
    else:
        v = 0.0
        for i in included_indexes:
            v += model.estimators_[i].predict(inst.reshape(1, -1))[0]
        return np.array([v / len(included_indexes)])


def get_auc(Y: np.ndarray, y_score: np.ndarray, classes: np.ndarray) -> float:
    """
    Compute multiclass AUC by flattening predictions and true labels.
    """
    y_bin = np.array([[1 if yi == c else 0 for c in classes] for yi in Y])
    fpr, tpr, _ = roc_curve(y_bin.ravel(), y_score.ravel())
    return auc(fpr, tpr)


def select_index(
    rf: Any, current_indexes: List[int], X_train: np.ndarray, y_train: np.ndarray
) -> Tuple[float, List[int]]:
    """
    Greedy step: evaluate adding each unused tree to the subset, choose best.
    """
    options: Dict[int, float] = {}
    for i in range(len(rf.estimators_)):
        if i in current_indexes:
            continue
        preds = predict_with_included_trees(rf, current_indexes + [i], X_train)
        if hasattr(rf, "classes_"):
            options[i] = get_auc(y_train, preds, rf.classes_)
        else:
            options[i] = -np.mean(np.abs(y_train - preds))

    if not options:
        raise ValueError("No valid tree index to select.")

    if hasattr(rf, "classes_"):
        best = max(options, key=options.get)
    else:
        best = min(options, key=options.get)

    return options[best], current_indexes + [best]


def reduce_error_pruning_cached(
    model: Any, X_train: np.ndarray, y_train: np.ndarray, min_size: int
) -> List[int]:
    """
    Reduction with caching to avoid redundant predictions.
    (Uses model.n_estimators instead of model.n_estimators.)
    """
    cache: Dict[int, np.ndarray] = {}
    best_metric, current_indexes = select_index_cached(model, [], X_train, y_train, cache)
    while len(current_indexes) <= model.n_estimators:
        if len(current_indexes) == len(model.estimators_):
            break
        new_metric, new_indexes = select_index_cached(model, current_indexes, X_train, y_train, cache)
        if (
            hasattr(model, "classes_")
            and new_metric <= best_metric
            and len(new_indexes) > min_size
        ) or (
            not hasattr(model, "classes_")
            and new_metric >= best_metric
            and len(new_indexes) > min_size
        ):
            break
        best_metric, current_indexes = new_metric, new_indexes
    return current_indexes




def predict_with_included_trees(
    model: Any, included_indexes: List[int], X: np.ndarray
) -> np.ndarray:
    """
    Ensemble predictions for multiple instances using selected tree indexes.
    """
    preds: List[np.ndarray] = []
    for inst in X:
        preds.append(predict_instance_with_included_tree(model, included_indexes, inst))
    return np.array(preds)



def select_index_cached(
    rf: Any,
    current_indexes: List[int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cache: Dict[int, np.ndarray],
) -> Tuple[float, List[int]]:
    """
    Like select_index, but reuse cached predictions for speed.
    (Uses rf.n_estimators instead of rf.n_estimators.)
    """
    options: Dict[int, float] = {}
    for i in range(len(rf.estimators_)):
        if i in current_indexes:
            continue
        preds = predict_with_included_trees_cached(rf, current_indexes + [i], X_train, cache)
        if hasattr(rf, "classes_"):
            options[i] = get_auc(y_train, preds, rf.classes_)
        else:
            options[i] = -np.mean(np.abs(y_train - preds))

    if not options:
        raise ValueError("No valid index found.")

    if hasattr(rf, "classes_"):
        best = max(options, key=options.get)
    else:
        best = min(options, key=options.get)

    return options[best], current_indexes + [best]





def predict_with_included_trees_cached(
    model: Any, included_indexes: List[int], X: np.ndarray, cache: Dict[int, np.ndarray]
) -> np.ndarray:
    """
    Cached ensemble predictions for speed: store per-tree predictions in cache.
    """
    is_clf = hasattr(model, "n_classes_")
    if is_clf:
        preds = np.zeros((X.shape[0], model.n_classes_))
    else:
        preds = np.zeros(X.shape[0])

    count = 0
    for i in included_indexes:
        if i not in cache:
            if is_clf:
                cache[i] = np.array(
                    [model.estimators_[i].predict_proba(x.reshape(1, -1))[0] for x in X]
                )
            else:
                cache[i] = np.array(
                    [model.estimators_[i].predict(x.reshape(1, -1))[0] for x in X]
                )
        preds += cache[i]
        count += 1

    preds = preds / count if count > 0 else preds
    return preds



def reduce_error_pruning_cached(
    model: Any, X_train: np.ndarray, y_train: np.ndarray, min_size: int
) -> List[int]:
    """
    Reduction with caching to avoid redundant predictions.
    """
    cache: Dict[int, np.ndarray] = {}
    best_metric, current_indexes = select_index_cached(model, [], X_train, y_train, cache)
    while len(current_indexes) <= model.n_estimators:
        if len(current_indexes) == len(model.estimators_):
            break
        new_metric, new_indexes = select_index_cached(model, current_indexes, X_train, y_train, cache)
        if (
            hasattr(model, "classes_")
            and new_metric <= best_metric
            and len(new_indexes) > min_size
        ) or (
            not hasattr(model, "classes_")
            and new_metric >= best_metric
            and len(new_indexes) > min_size
        ):
            break
        best_metric, current_indexes = new_metric, new_indexes
    return current_indexes


class Node:
    """
    Represents a node in the conjunction tree being built.
    """

    def __init__(self, mask: List[bool], random_state: Optional[Union[int, np.random.RandomState]]) -> None:
        self.mask: np.ndarray = np.array(mask, dtype=bool)
        self.feature: Optional[int] = None
        self.threshold: Optional[float] = None
        self.value: Optional[Union[np.ndarray, float]] = None
        if isinstance(random_state, np.random.RandomState):
            self.random_state: np.random.RandomState = random_state
        else:
            self.random_state = np.random.RandomState(random_state)

        self.left: Optional["Node"] = None
        self.right: Optional["Node"] = None
        self.left_mask: Optional[np.ndarray] = None
        self.right_mask: Optional[np.ndarray] = None
        self.both_mask: Optional[np.ndarray] = None

    def split(self, df: pd.DataFrame) -> None:
        """
        Recursively split this node by selecting the best feature/threshold or become a leaf.
        """
        if self._is_leaf(df):
            self.left = None
            self.right = None
            self.value = self._node_value(df)
            return

        self._select_feature_and_value(df)
        self._create_masks(df)

        if not self._is_splitable():
            self.left = None
            self.right = None
            self.value = self._node_value(df)
            return

        self.left = Node(
            list(np.logical_and(self.mask, np.logical_or(self.left_mask, self.both_mask))), 
            self.random_state,
        )
        self.right = Node(
            list(np.logical_and(self.mask, np.logical_or(self.right_mask, self.both_mask))),
            self.random_state,
        )
        self.left.split(df)
        self.right.split(df)

    def _is_leaf(self, df: pd.DataFrame) -> bool:
        """
        Check if this node should be a leaf (only one sample or homogeneous labels).
        """
        if np.sum(self.mask) == 1:
            return True
        return self._has_same_label(df)

    def _has_same_label(self, df: pd.DataFrame) -> bool:
        """
        Determine if all instances in this node have the same class/regression value.
        """
        if "probas" in df.columns:
            labels = {int(np.argmax(l)) for l in df["probas"][self.mask]}
        else:
            labels = set(df["regressions"][self.mask])
        return len(labels) == 1

    def _node_value(self, df: pd.DataFrame) -> Union[np.ndarray, float]:
        """
        Compute the output value for this node (mean probability vector or mean regression).
        """
        if "probas" in df.columns:
            return df["probas"][self.mask].mean(axis=0)
        return float(df["regressions"][self.mask].mean())


    def _select_feature_and_value(self, df: pd.DataFrame) -> None:
        """
        Choose a feature and threshold that minimizes entropy or variance.
        """
        # Only consider column names that are strings ending with "_upper"
        self.features = [
            int(k.split("_")[0])
            for k in df.keys()
            if isinstance(k, str) and k.endswith("_upper")
        ]

        feature_to_val: Dict[int, float] = {}
        feature_to_metric: Dict[int, float] = {}

        for feat in self.features:
            val, metric = self._check_feature_split_value(df, feat)
            feature_to_val[feat] = val
            feature_to_metric[feat] = metric

        self.feature = min(feature_to_metric, key=feature_to_metric.get)
        self.threshold = feature_to_val[self.feature]


    def _check_feature_split_value(
        self, df: pd.DataFrame, feature: int
    ) -> Tuple[float, float]:
        """
        Evaluate a few candidate thresholds for a feature, return best (threshold, metric).
        """
        vals = list(set(
            list(df[f"{feature}_upper"][self.mask]) + 
            list(df[f"{feature}_lower"][self.mask])
        ))
        self.random_state.shuffle(vals)
        vals = vals[:3]

        metrics: Dict[float, float] = {}
        for v in vals:
            left_m = [u <= v for u in df[f"{feature}_upper"]]
            right_m = [l >= v for l in df[f"{feature}_lower"]]
            both_m = [
                (l < v < u) 
                for l, u in zip(df[f"{feature}_lower"], df[f"{feature}_upper"])
            ]
            metrics[v] = self._value_metric(df, left_m, right_m, both_m)

        best_v = min(metrics, key=metrics.get)
        return best_v, metrics[best_v]

    def _value_metric(
        self,
        df: pd.DataFrame,
        left_m: List[bool],
        right_m: List[bool],
        both_m: List[bool],
    ) -> float:
        """
        Compute weighted entropy (for classification) or variance (for regression) of a potential split.
        """
        l_mask = np.logical_and(np.logical_or(left_m, both_m), self.mask)
        r_mask = np.logical_and(np.logical_or(right_m, both_m), self.mask)

        if not (np.sum(l_mask) and np.sum(r_mask)):
            return float("inf")

        l_prop = np.sum(l_mask) / len(self.mask)
        r_prop = np.sum(r_mask) / len(self.mask)

        if "probas" in df.columns:
            l_ent = self._calculate_entropy(df, l_mask)
            r_ent = self._calculate_entropy(df, r_mask)
            return l_ent * l_prop + r_ent * r_prop
        l_var = self._calculate_variance(df, l_mask)
        r_var = self._calculate_variance(df, r_mask)
        return l_var * l_prop + r_var * r_prop

    def _calculate_entropy(self, df: pd.DataFrame, mask: np.ndarray) -> float:
        """
        Entropy of average probability vector under the node.
        """
        dist = df["probas"][mask].mean()
        s = dist.sum()
        return float(entropy(dist / s)) if s > 0 else 0.0

    def _calculate_variance(self, df: pd.DataFrame, mask: np.ndarray) -> float:
        """
        Variance of regression values in the node.
        """
        return float(np.var(df["regressions"][mask]))

    def _create_masks(self, df: pd.DataFrame) -> None:
        """
        Build boolean masks for left, right, and overlapping intervals for splits.
        """
        self.left_mask = df[f"{self.feature}_upper"] <= self.threshold
        self.right_mask = df[f"{self.feature}_lower"] >= self.threshold
        self.both_mask = (
            (df[f"{self.feature}_lower"] < self.threshold)
            & (df[f"{self.feature}_upper"] > self.threshold)
        )

    def _is_splitable(self) -> bool:
        """
        Ensure that both child masks have at least one sample and not entire node.
        """
        l_count = np.sum(np.logical_and(self.mask, np.logical_or(self.left_mask, self.both_mask)))
        r_count = np.sum(np.logical_and(self.mask, np.logical_or(self.right_mask, self.both_mask)))
        total = np.sum(self.mask)

        if l_count == 0 or r_count == 0:
            return False
        if l_count == total or r_count == total:
            return False
        return True

    def predict_value(self, inst: np.ndarray, df: pd.DataFrame) -> Union[np.ndarray, float]:
        """
        Recursively traverse tree for one instance to return node value.
        """
        if self.left is None and self.right is None:
            return self._node_value(df)
        if inst[self.feature] <= self.threshold:
            return self.left.predict_value(inst, df)  # type: ignore
        return self.right.predict_value(inst, df)  # type: ignore

    def get_node_prediction(self, df: pd.DataFrame) -> Union[np.ndarray, float]:
        """
        Return prediction for the node as a label or regression mean.
        """
        if "probas" in df.columns:
            v = df["probas"][self.mask][0]
            total = v.sum()
            return (v / total) if total > 0 else v
        return float(df["regressions"][self.mask].mean())

    def count_depth(self) -> int:
        """
        Depth of this node’s subtree.
        """
        if self.right is None:
            return 1
        return 1 + max(self.left.count_depth(), self.right.count_depth())  # type: ignore

    def number_of_children(self) -> int:
        """
        Number of nodes in this subtree.
        """
        if self.right is None:
            return 1
        return 1 + self.left.number_of_children() + self.right.number_of_children()  # type: ignore


if __name__ == "__main__":
    # Classification example
    Xc, yc, class_names_c, rf_c = setup_toy_classifier(
        n_samples=50, n_features=3, n_classes=2, random_state=42
    )
    fbt_c = ForestBasedTree(random_state=42, verbose=True)
    fbt_c.fit(rf_c, Xc, yc, feature_types=None, feature_names=[f"feat_{i}" for i in range(Xc.shape[1])])
    preds_c = fbt_c.predict(Xc)
    logger.info(f"Classification predictions: {preds_c[:5]}")

    # Regression exampleF
    Xr, yr, rf_r = setup_toy_regressor(n_samples=50, n_features=3, random_state=42)
    fbt_r = ForestBasedTree(random_state=42, verbose=True)
    fbt_r.fit(rf_r, Xr, yr, feature_types=None, feature_names=[f"feat_{i}" for i in range(Xr.shape[1])])
    preds_r = fbt_r.predict(Xr)
    logger.info(f"Regression predictions: {preds_r[:5]}")





