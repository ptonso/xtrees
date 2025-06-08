import pandas as pd
import numpy as np
from collections import defaultdict, deque

class NodeStatistics:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.feature_counts = defaultdict(lambda: defaultdict(int))  # {node_name: {feature: count}}
        self.node_counts = defaultdict(int)  # {node_name: count}

    def extract_df(self, model):
        for tree_idx, estimator in enumerate(model.estimators_):
            tree = estimator.tree_
            self._traverse_tree_bfs(tree, tree_idx)

        return self._create_dataframe()

    def _traverse_tree_bfs(self, tree, tree_idx):
        queue = deque([(0, "A")])  # (node_id, node_name)

        while queue:
            node_id, node_name = queue.popleft()

            # Count this node at the current position
            self.node_counts[(node_name)] += 1

            # Check if the node is a split node
            feature_index = tree.feature[node_id]
            if feature_index != -2:  # -2 indicates a leaf node
                feature_name = self.feature_names[feature_index]
                self.feature_counts[(node_name)][feature_name] += 1

                # Add children to the queue
                left_child = tree.children_left[node_id]
                right_child = tree.children_right[node_id]

                if left_child != -1:
                    queue.append((left_child, f"{node_name}L"))
                if right_child != -1:
                    queue.append((right_child, f"{node_name}R"))

    def _create_dataframe(self):
        data = []
        all_features = set(self.feature_names)

        for (node_name), total_nodes in self.node_counts.items():
            feature_probs = {
                feature: self.feature_counts[(node_name)].get(feature, 0) / total_nodes
                for feature in all_features
            }
            data.append({"node_name": f"{node_name}", **feature_probs, "node_count": total_nodes})

        df = pd.DataFrame(data)
        df.set_index("node_name", inplace=True)
        return df


# ps = PathStatistics(feature_names)
# node_probs_df = ps.extract_df(rf)


# ---

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from src.logger import setup_logger


class NodeStatistics:
    """
    Compute feature-split probabilities per tree node for a forest model.
    """

    def __init__(self, feature_names: List[str]) -> None:
        """
        Initialize counters and logger.
        """
        self.feature_names: List[str] = feature_names
        self.feature_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.node_counts: Dict[str, int] = defaultdict(int)
        self.logger = setup_logger("api.log")

    def extract_df(
        self,
        model: Any,
    ) -> pd.DataFrame:
        """
        Traverse each tree in the forest and return a DataFrame of feature probabilities by node.
        """
        for tree_idx, estimator in enumerate(model.estimators_):
            try:
                tree = estimator.tree_
            except AttributeError:
                self.logger.error(f"Estimator at index {tree_idx} has no attribute 'tree_'.")
                continue

            self._traverse_tree_bfs(tree, tree_idx)

        return self._create_dataframe()

    def _traverse_tree_bfs(
        self,
        tree: Any,
        tree_idx: int,
    ) -> None:
        """
        Breadth-first traversal of a single decision tree to update counts.
        """
        queue: deque = deque([(0, "A")])  # (node_id, node_label)

        while queue:
            node_id, node_label = queue.popleft()

            self.node_counts[node_label] += 1

            feature_index: int = tree.feature[node_id]
            if feature_index != -2:
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
        Build a DataFrame where each row is a node label and columns are feature split probabilities.
        """
        data: List[Dict[str, Optional[float]]] = []
        all_features = set(self.feature_names)

        for node_label, total in self.node_counts.items():
            probs = {
                feature: self.feature_counts[node_label].get(feature, 0) / total
                for feature in all_features
            }
            row = {"node_name": node_label, **probs, "node_count": total}
            data.append(row)

        df = pd.DataFrame(data)
        df.set_index("node_name", inplace=True)
        return df
