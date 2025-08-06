from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from logger import setup_logger


class NodeStatistics:
    """
    Compute feature-split probabilities per tree node for a forest model.
    """

    def __init__(self, feature_names: List[str]) -> None:
        """
        Initialize feature/node counters and logger.
        """
        self.feature_names: List[str] = feature_names
        self.feature_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.node_counts: Dict[str, int] = defaultdict(int)
        self.logger = setup_logger("api.log")

    def extract_df(self, model: Any) -> pd.DataFrame:
        """
        Traverse each tree in the forest and return a DataFrame of feature probabilities per node.
        """
        for idx, estimator in enumerate(model.estimators_):
            tree = self._get_tree(estimator, idx)
            if tree is None:
                continue

            self._traverse_tree_bfs(tree)

        return self._create_dataframe()

    def _get_tree(self, estimator: Any, idx: int) -> Optional[Any]:
        """
        Safely retrieve the tree_ attribute from an estimator.
        """
        try:
            return estimator.tree_
        except AttributeError:
            self.logger.error(f"Estimator at index {idx} has no 'tree_' attribute.")
            return None

    def _traverse_tree_bfs(self, tree: Any) -> None:
        """
        Breadth-first traversal of a single decision tree to update feature/node counts.
        """
        queue: deque = deque([(0, "A")])  # (node_id, node_label)
        while queue:
            node_id, node_label = queue.popleft()
            self.node_counts[node_label] += 1

            feature_index: int = tree.feature[node_id]
            if feature_index != -2:  # not a leaf
                feature_name = self.feature_names[feature_index]
                self.feature_counts[node_label][feature_name] += 1

                left_child: int = tree.children_left[node_id]
                right_child: int = tree.children_right[node_id]
                if left_child != -1:
                    queue.append((left_child, f"{node_label}L"))
                if right_child != -1:
                    queue.append((right_child, f"{node_label}R"))

    def _create_dataframe(self) -> pd.DataFrame:
        """
        Build a DataFrame where each row is a node label and columns are split probabilities.
        """
        data: List[Dict[str, Optional[float]]] = []
        all_features = set(self.feature_names)

        for node_label, total in self.node_counts.items():
            probs = {
                feat: self.feature_counts[node_label].get(feat, 0) / total
                for feat in all_features
            }
            row = {"node_name": node_label, **probs, "node_count": total}
            data.append(row)

        df = pd.DataFrame(data)
        df.set_index("node_name", inplace=True)
        return df


if __name__ == "__main__":
    import numpy as np
    seed = 123

    from src.utils import setup_toy_classifier, setup_toy_regressor
    X_c, y_c, class_names_c, rf_c = setup_toy_classifier(random_state=seed)
    X_r, y_r, rf_r                = setup_toy_regressor (random_state=seed)

    feature_names = [f"feat_{i}" for i in range(X_c.shape[1])]

    stats = NodeStatistics(feature_names)
    df = stats.extract_df(rf_c)
    print(df)
