import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import (
    List,
    Dict,
    Any,
    Optional
)


class PathStatistics:
    """Extract decision-path statistics from a fitted tree or forest."""

    def __init__(self, feature_names: List[str]) -> None:
        self.feature_names = feature_names
        self.default_conditions = {
            **{f"{f}_upper": np.inf for f in feature_names},
            **{f"{f}_lower": -np.inf for f in feature_names},
        }
        self.classes_: Optional[np.ndarray] = None
        self.branch_df: Optional[pd.DataFrame] = None

    def extract_df(self, model: Any) -> pd.DataFrame:
        """Build a DataFrame of paths, their coverage, and predicted labels."""
        if hasattr(model, "classes_"):
            self.classes_ = model.classes_

        branch_data: List[Dict[str, Any]] = []
        try:
            if hasattr(model, "estimators_"):
                for tree_id, est in enumerate(model.estimators_):
                    tree = est.tree_
                    for branch in self._extract_branches_from_tree(tree):
                        self._merge_branch_data(branch, tree_id, branch_data)
            else:
                tree = model.tree_
                for branch in self._extract_branches_from_tree(tree):
                    self._merge_branch_data(branch, 0, branch_data)
        except AttributeError:
            raise

        df = pd.DataFrame(branch_data)
        if self.classes_ is not None:
            df["value_dist"] = df["value"]
            df["predicted_label"] = df["value_dist"].apply(
                lambda v: self.classes_[int(np.argmax(v))]
            )
            df["_classes"] = [list(self.classes_)] * len(df)
        else:
            df.rename(columns={"value": "predicted_label"}, inplace=True)

        # Sort by raw path coverage (branch_importance)
        df.sort_values("branch_importance", ascending=False, inplace=True)

        self.branch_df = df
        return df

    def _merge_branch_data(self, branch: Dict[str, Any], tree_id: int,
                           data: List[Dict[str, Any]]) -> None:
        """Combine branches with identical feature constraints."""
        for existing in data:
            if all(
                branch[f"{f}_upper"] == existing[f"{f}_upper"] and
                branch[f"{f}_lower"] == existing[f"{f}_lower"]
                for f in self.feature_names
            ):
                existing["tree_id"] += f",{tree_id}"
                return
        branch["tree_id"] = str(tree_id)
        data.append(branch)

    def _extract_branches_from_tree(self, tree: Any) -> List[Dict[str, Any]]:
        """Return a list of leaf-node branches from a tree."""
        branches: List[Dict[str, Any]] = []
        for node in range(tree.node_count):
            if tree.children_left[node] == -1 and tree.children_right[node] == -1:
                branches.append(self._build_branch(tree, node))
        return branches

    def _build_branch(self, tree: Any, leaf_id: int) -> Dict[str, Any]:
        """Trace conditions from leaf up to the root."""
        conds = self.default_conditions.copy()
        node = leaf_id
        while node != 0:
            left = np.where(tree.children_left == node)[0]
            right = np.where(tree.children_right == node)[0]
            if left.size:
                parent = left[0]
                is_left = True
            elif right.size:
                parent = right[0]
                is_left = False
            else:
                break

            feat = self.feature_names[tree.feature[parent]]
            thr = tree.threshold[parent]
            if is_left:
                conds[f"{feat}_upper"] = min(conds[f"{feat}_upper"], thr)
            else:
                conds[f"{feat}_lower"] = max(conds[f"{feat}_lower"], thr)

            node = parent

        importance = tree.n_node_samples[leaf_id] / tree.n_node_samples[0]
        val = tree.value[leaf_id]
        if hasattr(val, "shape") and len(val.shape) > 1:
            val = val.mean(axis=0)
        else:
            val = val[0]
        return {**conds, "branch_importance": importance, "value": val}


class Visualizer:
    """Plot top decision-paths with unified, larger barrier glyphs per feature."""

    def __init__(self, features: List[str], data: pd.DataFrame) -> None:
        self.features = features
        self.data = data

    def plot_paths(self, df: pd.DataFrame, top_m: int = 10) -> None:
        """Render top-M branches with enhanced barrier glyphs per feature."""
        df_top = df.head(top_m)
        n_feats = len(self.features)
        n_cols = n_feats + 1
        y_off = 0.03
        y_off_label = 0.10

        # feature ranges
        data_ranges = {
            feat: (self.data[feat].min(), self.data[feat].max())
            for feat in self.features
        }

        # enhanced glyph settings
        spine_len = 0.5     # taller bar
        half = spine_len / 2
        offset = 0.15

        # colors and sizes
        forbit_color = 'dimgray'
        allow_color = 'lightblue'  # lighter shade
        dot_size = 8
        forbid_width = 4
        allow_width = 2

        fig, ax = plt.subplots(
            figsize=(n_cols * 2.2, top_m * 1.05)
        )
        fig.subplots_adjust(left=0.15, top=0.82, bottom=0.15)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_ylim(0.5, top_m + 0.5)
        ax.set_xlim(-0.6, n_cols - 0.4)
        ax.set_yticks([])
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([''] * n_feats + ['Prediction'], fontsize=10)

        # headers
        pc_x = -0.32
        ax.text(pc_x, top_m + 0.5, 'Path coverage',
                ha='center', va='bottom', fontsize=12)
        header = '[' + ', '.join(map(str, df_top['_classes'].iloc[0])) + ']'
        ax.text(n_cols - 1, top_m + 0.5, header,
                ha='center', va='bottom', fontsize=12)
        ax.set_title('Top Decision Paths', fontsize=20,
                     fontweight='bold', pad=30)

        for idx, (_, row) in enumerate(df_top.iterrows()):
            y = top_m - idx
            # dotted row line
            ax.hlines(y, 0, n_cols - 1,
                      color='#B0B0B0', linestyle='dotted', linewidth=1.2)
            # coverage label
            cov = row['branch_importance']
            ax.text(pc_x, y + y_off, f'{cov:.1%}',
                    ha='center', va='bottom', fontsize=10, color='dimgray')

            # features
            for col, feat in enumerate(self.features):
                lo, hi = row[f'{feat}_lower'], row[f'{feat}_upper']
                if lo > -np.inf or hi < np.inf:
                    # feature name above
                    ax.text(col, y + y_off, feat,
                            ha='center', va='bottom', fontsize=10)

                    # threshold text below
                    parts = []
                    if lo > -np.inf: parts.append(f'> {lo:.2f}')
                    if hi < np.inf:  parts.append(f'< {hi:.2f}')
                    cond_text = "\n".join(parts)
                    ax.text(col, y - y_off, cond_text,
                            ha='center', va='top', fontsize=9)

                    # coords for glyph
                    fmin, fmax = data_ranges[feat]
                    y0, y1 = y - half, y + half
                    y_lo = ((lo - fmin) / (fmax - fmin)) * spine_len + (y - half) \
                            if lo > -np.inf else y0
                    y_hi = ((hi - fmin) / (fmax - fmin)) * spine_len + (y - half) \
                            if hi < np.inf else y1
                    xg = col + offset

                    # forbidden segments (thicker)
                    if lo > -np.inf:
                        ax.plot([xg, xg], [y0, y_lo],
                                color=forbit_color, linewidth=forbid_width, solid_capstyle='butt')
                    if hi < np.inf:
                        ax.plot([xg, xg], [y_hi, y1],
                                color=forbit_color, linewidth=forbid_width, solid_capstyle='butt')
                    # allowed segment (dotted)
                    ax.plot([xg, xg], [y_lo, y_hi],
                            linestyle='dotted', linewidth=allow_width, color=allow_color)

                    # extremity dots (no border)
                    bottom_color = allow_color if lo <= -np.inf else forbit_color
                    ax.plot(xg, y0, 'o', ms=dot_size,
                            color=bottom_color)
                    top_color = allow_color if hi >= np.inf else forbit_color
                    ax.plot(xg, y1, 'o', ms=dot_size,
                            color=top_color)

            # prediction
            ax.plot(n_cols - 1, y, 'o', ms=10, color='C3')
            ax.text(n_cols - 1, y + y_off_label, str(row['predicted_label']),
                    ha='center', va='bottom', fontsize=12,
                    color='C3', fontweight='bold')
            dist = np.array(row['value_dist'], dtype=float)
            vectext = '[' + ', '.join(f'{p:.2f}' for p in dist) + ']'
            ax.text(n_cols - 1, y - y_off_label, vectext,
                    ha='center', va='top', fontsize=8, color='C3')

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    np.random.seed(42)
    X_train = pd.DataFrame(
        np.random.randn(100, 4),
        columns=["feature1", "feature2", "feature3", "feature4"]
    )
    y_train = np.random.choice(["label1", "label2"], size=100)
    rf = RandomForestClassifier(random_state=123)
    rf.fit(X_train, y_train)

    ps = PathStatistics(X_train.columns.tolist())
    branch_df = ps.extract_df(rf)

    vis = Visualizer(features=X_train.columns.tolist(), data=X_train)
    vis.plot_paths(branch_df, top_m=5)
