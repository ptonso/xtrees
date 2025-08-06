from typing import *
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.tree import _tree

from logger import setup_logger
from xtrees.utils import _fmt
from xtrees.stats.node import NodeStatistics


class VisNode:
    def __init__(
        self,
        id: int,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        value: Optional[Union[np.ndarray, float]] = None,
        parent: Optional[int] = None,
        is_left: Optional[bool] = None,
    ) -> None:
        self.id = id
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.parent = parent
        self.is_left = is_left

        self.left: Optional[int] = None
        self.right: Optional[int] = None

        self.n_train: int = 0
        self.n_samples: int = 0


class VisTree:
    """
    Thin wrapper that converts a fitted scikit-learn decision tree into an
    easily navigable object graph and offers helpers for pruning,
    propagation, branch extraction, etc.
    """

    def __init__(
        self,
        model: Any,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        class_names: Optional[List[str]] = None,
        log_coloring: bool = False,
        rf: Optional[Any] = None,
    ) -> None:
        self.logger = setup_logger("api.log")
        self.model = model
        self.nodes: Dict[int, VisNode] = {}

        self.feature_names: Optional[List[str]] = (
            list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
        )

        self.feature_prob_df: Optional[pd.DataFrame] = None
        if rf is not None and self.feature_names is not None:
            self.feature_prob_df = NodeStatistics(self.feature_names).extract_df(rf)

        # classifier / regressor metadata
        if hasattr(model, "is_classifier"):
            self.is_classifier = model.is_classifier
        else:
            self.is_classifier = hasattr(model, "classes_")

        if class_names is not None:
            self.class_names = class_names
            self.n_classes = len(class_names) if self.is_classifier else 0
        elif self.is_classifier:
            raise Exception("For classifiers, pass class_names")

        self.log_coloring = log_coloring

        self.n_train: int = 0
        self.max_depth: int = 0
        self.possible_values: set = set()  # regression colour scale

        self._populate_from_model()
        self._generate_color_struct()

        if X is not None:
            arr = X.to_numpy() if isinstance(X, pd.DataFrame) else X
            self.populate_ns(arr, prune=True)
            self.propagate_values(consider_proba=True)
            self.prune_redundants(consider_proba=True)


    def _generate_color_struct(self, n: int = 100, opacity: float = 0.8) -> None:
        if self.is_classifier:
            palette = px.colors.qualitative.Plotly[: self.n_classes]
            self.color_struct = {i: c for i, c in enumerate(palette)}
        else:
            if not self.possible_values:
                self.color_struct = []
                return
            mn, mx = min(self.possible_values), max(self.possible_values)
            ticks = np.linspace(mn, mx, n)
            colors = px.colors.sample_colorscale("Viridis", n)
            self.color_struct = [
                (ticks[i], colors[i].replace("rgb", "rgba").replace(")", f", {opacity})"))
                for i in range(n)
            ]


    def _add_node(
        self,
        id: int,
        feature: Optional[int],
        threshold: Optional[float],
        value: Union[np.ndarray, float],
        parent: Optional[int],
        is_left: Optional[bool],
    ) -> None:
        self.nodes[id] = VisNode(id, feature, threshold, value, parent, is_left)


    def _populate_from_model(self) -> None:
        tree = self.model.tree_

        def walk(idx: int, parent: Optional[int], is_left: Optional[bool], depth: int) -> Optional[int]:
            if idx == _tree.TREE_LEAF:
                return None

            val: Union[np.ndarray, float]
            if self.is_classifier:
                val = tree.value[idx].flatten()
            else:
                val = float(tree.value[idx][0, 0])
                self.possible_values.add(val)

            feat = tree.feature[idx] if tree.feature[idx] != _tree.TREE_UNDEFINED else None
            thr = tree.threshold[idx] if feat is not None else None

            self._add_node(idx, feat, thr, val, parent, is_left)
            self.max_depth = max(self.max_depth, depth)

            l = walk(tree.children_left[idx], idx, True, depth + 1)
            r = walk(tree.children_right[idx], idx, False, depth + 1)
            self.nodes[idx].left, self.nodes[idx].right = l, r
            return idx

        walk(0, None, None, 0)

    def update_max_depth(self, nid: int = 0, depth: int = 0) -> None:
        node = self.nodes.get(nid)
        if node is None:
            return
        self.max_depth = max(self.max_depth, depth)
        if node.left is not None:
            self.update_max_depth(node.left, depth + 1)
        if node.right is not None:
            self.update_max_depth(node.right, depth + 1)


    def get_nodes_depth_list(self) -> List[List[int]]:
        layers: Dict[int, List[int]] = {}

        def rec(idx: int, depth: int) -> None:
            layers.setdefault(depth, []).append(idx)
            node = self.nodes[idx]
            for child in (node.left, node.right):
                if child is not None:
                    rec(child, depth + 1)

        rec(0, 0)
        self.max_depth = max(layers)
        return [layers.get(i, []) for i in range(self.max_depth + 1)]


    def predict(self, X: Union[pd.DataFrame, np.ndarray], predict_probas: bool = False) -> np.ndarray:
        arr = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        n = arr.shape[0]
        if self.is_classifier:
            y = np.zeros((n, self.n_classes)) if predict_probas else np.zeros(n, int)
        else:
            y = np.zeros(n)

        def walk(idx: int, mask: np.ndarray) -> None:
            node = self.nodes[idx]
            if node.left is None and node.right is None:
                if self.is_classifier:
                    if predict_probas:
                        y[mask] = node.value / node.value.sum()
                    else:
                        y[mask] = int(np.argmax(node.value))
                else:
                    y[mask] = float(node.value)
                return
            lm = mask & (arr[:, node.feature] <= node.threshold)  # type: ignore[arg-type]
            rm = mask & ~lm
            if node.left is not None:
                walk(node.left, lm)
            if node.right is not None:
                walk(node.right, rm)

        walk(0, np.ones(n, bool))
        return y


    def populate_ns(self, samples: np.ndarray, prune: bool = False) -> None:
        self.n_train = samples.shape[0]

        def walk(idx: int, mask: np.ndarray) -> None:
            node = self.nodes[idx]
            if node.left is None and node.right is None:
                node.n_train = int(mask.sum())
                return
            lm = mask & (samples[:, node.feature] <= node.threshold)  # type: ignore[arg-type]
            rm = mask & ~lm
            if node.left is not None:
                walk(node.left, lm)
            if node.right is not None:
                walk(node.right, rm)
            node.n_train = int(mask.sum())

        walk(0, np.ones(self.n_train, bool))
        if prune:
            self._prune_by_samples()

    def _prune_by_samples(self) -> None:
        def prune_layer(layer: List[int]) -> bool:
            changed = False
            for idx in layer:
                node = self.nodes[idx]
                if node.left is None and node.right is None and node.n_train == 0:
                    parent = self.nodes[node.parent]
                    self.prune_branch(self, parent, left=node if node.is_left else None, right=node if not node.is_left else None)
                    changed = True
            return changed

        layers = self.get_nodes_depth_list()
        while layers and prune_layer(layers[-1]):
            layers = self.get_nodes_depth_list()


    @staticmethod
    def prune_branch(
            tree: "VisTree",
            parent: VisNode, 
            left: Optional[VisNode] = None, 
            right: Optional[VisNode] = None
        ) -> None:
        """
        Remove children from *parent* and delete their entries from tree.nodes.
        """
        if left and right:
            parent.feature = parent.threshold = None
        if left:
            parent.left = None
            del tree.nodes[left.id]
        if right:
            parent.right = None
            del tree.nodes[right.id]


    def prune(self, max_depth: int) -> "VisTree":
        """
        Return a deep-copy pruned to *max_depth* (keeps API identical to original).
        """
        from copy import deepcopy

        clone = deepcopy(self)
        layers = clone.get_nodes_depth_list()
        cut = max(0, clone.max_depth - max_depth)
        for d in range(cut):
            for idx in list(layers[-(d + 1)]):
                if idx not in clone.nodes:
                    continue
                node = clone.nodes[idx]
                parent = clone.nodes[node.parent]
                sib_id = parent.right if node.is_left else parent.left
                sib = clone.nodes.get(sib_id)
                if node.is_left:
                    VisTree.prune_branch(clone, parent, left=node, right=sib)
                else:
                    VisTree.prune_branch(clone, parent, left=sib, right=node)
        clone.prune_redundants(consider_proba=True)
        clone.update_max_depth()
        return clone


    def propagate_values(self, consider_proba: bool = True) -> None:
        if self.is_classifier:
            self._propagate_class(consider_proba)
        else:
            self._propagate_reg()

    def _propagate_class(self, consider_proba: bool) -> None:
        layers = self.get_nodes_depth_list()
        for depth in range(self.max_depth, -1, -1):
            for idx in layers[depth]:
                node = self.nodes[idx]
                if node.left is None and node.right is None:
                    continue
                accum = np.zeros_like(node.value)
                for cid in (node.left, node.right):
                    if cid is None:
                        continue
                    child = self.nodes[cid]
                    if consider_proba:
                        accum += child.value * child.n_train
                    else:
                        accum[int(np.argmax(child.value))] += child.n_train
                node.value = accum / node.n_train if node.n_train else accum

    def _propagate_reg(self) -> None:
        layers = self.get_nodes_depth_list()
        for depth in range(self.max_depth, -1, -1):
            for idx in layers[depth]:
                node = self.nodes[idx]
                if node.left is None and node.right is None:
                    continue
                tot, cnt = 0.0, 0
                for cid in (node.left, node.right):
                    if cid is None:
                        continue
                    child = self.nodes[cid]
                    tot += child.value * child.n_train
                    cnt += child.n_train
                node.value = tot / cnt if cnt else tot
                self.possible_values.add(node.value)
        self._generate_color_struct()


    def prune_redundants(self, consider_proba: bool = True, tol: float = 1e-4) -> None:
        def homog(idx: int) -> Tuple[bool, Union[np.ndarray, float], int]:
            leaves, ntot = [], 0
            stack = [idx]
            while stack:
                cur = self.nodes[stack.pop()]
                if cur.left is None and cur.right is None:
                    leaves.append(cur.value)
                    ntot += cur.n_train
                else:
                    if cur.left:
                        stack.append(cur.left)
                    if cur.right:
                        stack.append(cur.right)
            if self.is_classifier:
                lbl = int(np.argmax(leaves[0]))
                if not all(int(np.argmax(v)) == lbl for v in leaves):
                    return False, 0, 0
                agg = sum(v * self.nodes[idx].n_train for v in leaves)
                val = agg / ntot if consider_proba and ntot else leaves[0]
            else:
                v0 = float(leaves[0])
                if not all(abs(float(v) - v0) < tol for v in leaves):
                    return False, 0.0, 0
                agg = sum(float(v) * self.nodes[idx].n_train for v in leaves)
                val = agg / ntot if ntot else v0
            return True, val, ntot

        layers = self.get_nodes_depth_list()
        for depth in range(len(layers) -1, -1, -1):
            for idx in layers[depth]:
                ok, new_val, ntot = homog(idx)
                if ok:
                    n = self.nodes[idx]
                    n.left = n.right = None
                    n.feature = n.threshold = None
                    n.value, n.n_train = new_val, ntot
        self.update_max_depth()


    def _build_branch(self, leaf_id: int) -> Dict[str, Any]:
        conds: Dict[str, Any] = {}

        # init bounds
        if self.feature_names:
            for f in self.feature_names:
                conds[f"{f}_upper"] = np.inf
                conds[f"{f}_lower"] = -np.inf
        else:
            mx = max((n.feature for n in self.nodes.values() if n.feature is not None), default=-1)
            for i in range(mx + 1):
                conds[f"{i}_upper"] = np.inf
                conds[f"{i}_lower"] = -np.inf

        node = self.nodes[leaf_id]
        while node.parent is not None:
            parent = self.nodes[node.parent]
            fname = self.feature_names[parent.feature] if self.feature_names else str(parent.feature)
            if node.is_left:
                conds[f"{fname}_upper"] = min(conds[f"{fname}_upper"], parent.threshold)
            else:
                conds[f"{fname}_lower"] = max(conds[f"{fname}_lower"], parent.threshold)
            node = parent

        importance = node.n_train / self.n_train if self.n_train else 0.0
        conds["branch_importance"] = importance
        conds["value"] = self.nodes[leaf_id].value
        return conds

    def extract_df(self, tree_id: int = 0) -> pd.DataFrame:
        rows = [
            dict(branch_id=i, **self._build_branch(idx), tree_id=tree_id)
            for i, idx in enumerate(self.nodes)
            if self.nodes[idx].left is None and self.nodes[idx].right is None
        ]
        return pd.DataFrame(rows)


    def print_tree(self) -> None:
        def walk(idx: int, depth: int, tag: str) -> None:
            ind = "  " * depth
            n = self.nodes[idx]
            if n.left is None and n.right is None:
                print(f"{ind}{tag} Leaf {idx} | value={_fmt(n.value)} | n_train={n.n_train}")
            if n.feature is not None:
                fname = self.feature_names[n.feature] if self.feature_names else n.feature
                print(f"{ind}  feat={fname}, thr={n.threshold:.2f}, left={n.left}, right={n.right}")
            if n.left is not None:
                walk(n.left, depth + 1, "L")
            if n.right is not None:
                walk(n.right, depth + 1, "R")

        walk(0, 0, "Root")

    def print_nodes(self, state: str = "") -> None:
        """
        Print every attribute of each node, in a recursive/tree‐shaped format,
        so you can see parent/child relationships, features, thresholds, etc.
        """
        def _walk(idx: int, depth: int) -> None:
            node = self.nodes[idx]
            indent = "  " * depth

            parent_str = str(node.parent) if node.parent is not None else "None"
            is_left_str = str(node.is_left) if node.is_left is not None else "None"

            if node.left is None and node.right is None:
                print(f"{indent}Leaf {idx} "
                    f"(parent={parent_str}, is_left={is_left_str}) | "
                    f"value={_fmt(node.value)} | n_train={node.n_train}")
                return

            print(f"{indent}Node {idx} "
                f"(parent={parent_str}, is_left={is_left_str}) | "
                f"value={_fmt(node.value)} | n_train={node.n_train}")

            feat_name = (
                self.feature_names[node.feature]
                if (node.feature is not None and self.feature_names is not None)
                else str(node.feature)
            )
            print(f"{indent}  feature:   {feat_name}")
            thr_str = f"{node.threshold:.2f}" if isinstance(node.threshold, (float, int)) else str(node.threshold)
            print(f"{indent}  threshold: {thr_str}")
            print(f"{indent}  left_id:   {node.left}")
            print(f"{indent}  right_id:  {node.right}")
            print(f"{indent}  n_samples: {node.n_samples}")

            # Recurse into children (if any)
            if node.left is not None:
                _walk(node.left, depth + 1)
            if node.right is not None:
                _walk(node.right, depth + 1)

        # If the user passed a state (label), print it first
        if state:
            print(state)
        # Kick off the recursion at root (id=0)
        if 0 in self.nodes:
            _walk(0, 0)
        else:
            print("(no nodes to print)")




if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier

    iris = load_iris()
    X, y = iris.data, iris.target
    clf = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X, y)

    tree = VisTree(clf, X=pd.DataFrame(X), class_names=iris.target_names.tolist())

    print("\n--- FULL TREE ---")
    tree.print_tree()

    print("\n--- BRANCHES (head) ---")
    branches_df = tree.extract_df()
    branches_df["value"] = branches_df["value"].apply(_fmt)
    print(branches_df.head())

    pruned = tree.prune(max_depth=2)
    print("\n--- PRUNED (depth≤2) ---")
    pruned.print_tree()
