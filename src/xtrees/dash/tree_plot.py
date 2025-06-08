from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from src.logger import setup_logger
from src.xtrees.dash.vis_tree import VisTree


class SankeyTreePlot:
    """
    Create a Sankey plot representing a decision tree.
    """

    def __init__(
        self,
        vis_tree: VisTree,
        show_text: bool = True,
        show_label: bool = False,
    ) -> None:
        self.logger = setup_logger("api.log")
        self.vis_tree: VisTree = vis_tree
        self.show_text: bool = show_text
        self.show_label: bool = show_label
        self.max_depth: int = vis_tree.max_depth
        self.feature_prob_df = vis_tree.feature_prob_df

        (
            labels,
            colors,
            sources,
            targets,
            values,
            x,
            y,
            customdata,
            edge_labels,
        ) = self._tree_to_sankey_data()
        self.fig = self._create_figure(
            labels, colors, sources, targets, values, x, y, customdata, edge_labels
        )

    def show(self) -> None:
        """
        Display the Sankey plot.
        """
        if self.show_label:
            self._add_color_label()
        self.fig.show()

    def save(self, path: str) -> None:
        """
        Save the Sankey plot to a file.
        """
        if self.show_label:
            self._add_color_label()
        self.fig.write_image(path)

    def _create_figure(
        self,
        labels: List[str],
        colors: List[str],
        sources: List[int],
        targets: List[int],
        values: List[int],
        x: List[float],
        y: List[float],
        customdata: List[List[str]],
        edge_labels: List[str],
    ) -> go.Figure:
        """
        Build the Plotly Sankey figure given node/link data.
        """
        hovertemplate = (
            '<b style="font-size:14px;">%{customdata[0]}</b><br>'
            '<b style="font-size:14px;">n_train:</b> %{value}<br>'
            '<b style="font-size:14px;">Value:</b> %{customdata[1]}<extra></extra>'
        )
        node_labels = labels if self.show_text else [""] * len(labels)

        sankey = go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=colors,
                x=x,
                y=y,
                customdata=customdata,
                hovertemplate=hovertemplate,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                hovertemplate='<b style="font-size:14px;">n_train:</b> %{value}<extra></extra>',
                label=edge_labels,
            ),
        )

        fig = go.Figure(data=[sankey])
        fig.update_layout(
            font=dict(family="Courier New", size=14, color="black")
        )
        return fig

    def _tree_to_sankey_data(
        self,
    ) -> Tuple[
        List[str],
        List[str],
        List[int],
        List[int],
        List[int],
        List[float],
        List[float],
        List[List[str]],
        List[str],
    ]:
        """
        Traverse the vis_tree and collect data for Sankey: node labels, colors, positions, and edges.
        """
        nodes: List[int] = []
        labels: List[str] = []
        colors: List[str] = []
        sources: List[int] = []
        targets: List[int] = []
        values: List[int] = []
        x: List[float] = []
        y: List[float] = []
        customdata: List[List[str]] = []
        edge_labels: List[str] = []

        positions = self._compute_node_positions()
        max_x, max_y = self._find_max_positions(positions)

        def _traverse(node_id: int, node_name: str = "A") -> None:
            node = self.vis_tree.nodes[node_id]
            label, customlabel = self._get_node_labels(node_id)
            color = self._get_color_for_value(node.value)

            if node_id not in nodes:
                if (
                    self.feature_prob_df is not None
                    and node_name in self.feature_prob_df.index
                ):
                    feature_probs = self.feature_prob_df.loc[node_name]
                    if node.feature is not None:
                        feature_name = self.vis_tree.feature_names[node.feature]
                        fprob = feature_probs[feature_name]
                        label += f"<br>fp: {fprob:.2f}"
                        customlabel += f"<br>feature_prob: {fprob:.2f}"

                nodes.append(node_id)
                labels.append(label)
                colors.append(color)
                customdata.append(
                    [
                        customlabel,
                        (
                            f"{np.round(node.value, 2)}"
                            if self.vis_tree.is_classifier
                            else f"{node.value:.4f}"
                        ),
                    ]
                )

                x_pos, y_pos = positions[node_id]
                x.append(x_pos / max_x)
                y.append((y_pos + max_y) / (2 * max_y))

            if node.left is not None:
                sources.append(nodes.index(node_id))
                targets.append(len(nodes))
                left_n = int(self.vis_tree.nodes[node.left].n_train)
                values.append(left_n)
                edge_labels.append(str(left_n))
                _traverse(node.left, f"{node_name}L")

            if node.right is not None:
                sources.append(nodes.index(node_id))
                targets.append(len(nodes))
                right_n = int(self.vis_tree.nodes[node.right].n_train)
                values.append(right_n)
                edge_labels.append(str(right_n))
                _traverse(node.right, f"{node_name}R")

        _traverse(0)

        if len(colors) < len(labels):
            colors += ["rgba(0, 0, 0, 0.8)"] * (len(labels) - len(colors))

        return labels, colors, sources, targets, values, x, y, customdata, edge_labels

    def _get_node_labels(self, node_id: int) -> Tuple[str, str]:
        """
        Generate display label and custom label for a node.
        """
        node = self.vis_tree.nodes[node_id]
        label = ""
        customlabel = ""

        if node.feature is not None and node.threshold is not None:
            if self.vis_tree.feature_names is not None:
                fname = self.vis_tree.feature_names[node.feature]
            else:
                fname = f"Feature {node.feature}"
            label = f"{fname[:6]}<br>≤ {node.threshold:.2f}"
            customlabel = f"{fname}<br>≤ {node.threshold:.2f}"
        else:
            if self.vis_tree.is_classifier:
                idx = int(np.argmax(node.value))
                if (
                    self.vis_tree.class_names is not None
                    and idx < len(self.vis_tree.class_names)
                ):
                    label = self.vis_tree.class_names[idx]
                else:
                    label = f"Class {idx}"
            else:
                label = f"{node.value:.4f}"

        return label, customlabel

    def _get_color_for_value(self, value: Any) -> str:
        """
        Map a node value to a color.
        """
        color_struct = self.vis_tree.color_struct
        if self.vis_tree.is_classifier:
            return color_struct[value.argmax()]
        else:
            for i in range(len(color_struct) - 1):
                if color_struct[i][0] <= value < color_struct[i + 1][0]:
                    return color_struct[i][1]
            return color_struct[-1][1]

    def _add_color_label(self) -> None:
        """
        Add a legend or hue bar to the figure depending on classifier/regressor.
        """
        if self.vis_tree.is_classifier:
            self._add_classification_color_label()
        else:
            self._add_regression_hue_bar()

    def _add_classification_color_label(self) -> None:
        """
        Annotate the figure with class-color mappings for classifiers.
        """
        color_struct = self.vis_tree.color_struct
        class_names = self.vis_tree.class_names or []
        labels_with_colors: List[str] = []

        for idx, name in enumerate(class_names):
            color = color_struct[idx]
            labels_with_colors.append(f'{name}: <span style="color:{color};">&#9608;</span>')

        label_text = "<br>".join(labels_with_colors)
        self.fig.add_annotation(
            text=label_text,
            xref="paper",
            yref="paper",
            x=1.05,
            y=1,
            showarrow=False,
            align="left",
        )

    def _add_regression_hue_bar(self) -> None:
        """
        Add a color hue bar for regression values.
        """
        color_struct = self.vis_tree.color_struct
        cs = [c[0] for c in color_struct]

        hue_bar = go.Heatmap(
            z=[[c] for c in cs],
            colorscale=[[i / (len(color_struct) - 1), color[1]] for i, color in enumerate(color_struct)],
            showscale=False,
            x=[0.98],
            y=cs,
            xaxis="x2",
            yaxis="y2",
        )
        self.fig.add_trace(hue_bar)
        self.fig.update_layout(
            xaxis2=dict(
                range=[0, 0.5],
                domain=[0.75, 1.0],
                anchor="free",
                overlaying="x",
                side="right",
                position=0.95,
                visible=False,
            ),
            yaxis2=dict(
                range=[min(cs), max(cs)],
                domain=[0.40, 0.90],
                anchor="free",
                overlaying="y",
                side="right",
                position=1,
                showgrid=False,
            ),
            plot_bgcolor="rgba(0,0,0,0)",
        )

    def _compute_node_positions(self) -> Dict[int, Tuple[float, float]]:
        """
        Compute (x, y) positions for all nodes using a recursive heuristic.
        """
        tree = self.vis_tree
        class_order: List[Any] = []

        def _recur(
            node: Any,
            x_pos: float = 0,
            y_pos: float = 0,
            pos: Optional[Dict[int, Tuple[float, float]]] = None,
            level: int = 0,
        ) -> Dict[int, Tuple[float, float]]:
            if pos is None:
                pos = {}
            pos[node.id] = (x_pos, y_pos)
            node_height = 1 + (node.n_train / tree.n_train) * 0.4
            vertical_spacing = node_height + 2 ** (self.max_depth / 2 - level - 1)
            next_x = x_pos + 1

            if node.left is not None and node.right is not None:
                left_y = y_pos + vertical_spacing / 2
                right_y = y_pos - vertical_spacing / 2
            elif node.left is None and node.right is None:
                class_order.insert(0, node)
                left_y = y_pos
                right_y = y_pos
            else:
                left_y = y_pos + vertical_spacing
                right_y = y_pos - vertical_spacing

            if node.left is not None:
                pos = _recur(tree.nodes[node.left], next_x, left_y, pos, level + 1)
            if node.right is not None:
                pos = _recur(tree.nodes[node.right], next_x, right_y, pos, level + 1)

            pos[node.id] = (x_pos, y_pos)
            return pos

        root = tree.nodes[0]
        return _recur(root)

    def _find_max_positions(
        self, positions: Dict[int, Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        Determine maximum absolute x and y from positions to normalize plotting.
        """
        max_x = max(abs(px) for px, _ in positions.values())
        max_y = max(abs(py) for _, py in positions.values())
        return max_x, max_y


class GoTreePlot:
    """
    Create a basic node-edge plot of a decision tree using Plotly.
    """

    def __init__(self, vis_tree: VisTree, show_text: bool = True) -> None:
        self.logger = setup_logger("api.log")
        self.tree: VisTree = vis_tree
        self.show_text: bool = show_text
        self._update_edge_widths()
        positions = self._compute_layout()
        self.fig = self._build_figure(positions)

    def show(self) -> None:
        """
        Display the node-edge plot.
        """
        self.fig.show()

    def save(self, path: str) -> None:
        """
        Save the node-edge plot to a file.
        """
        self.fig.write_image(path)

    def _update_edge_widths(self) -> None:
        """
        Assign a uniform width to each edge based on sample counts.
        """
        max_n = self.tree.n_train
        for node_id, node in self.tree.nodes.items():
            if node.left is not None:
                left_node = self.tree.nodes[node.left]
                left_node.parent_edge_width = 5
            if node.right is not None:
                right_node = self.tree.nodes[node.right]
                right_node.parent_edge_width = 5

    def _compute_layout(self) -> Dict[int, Tuple[float, float]]:
        """
        Recursively compute (x, y) positions for plotting nodes in GoTreePlot.
        """
        def _recur(node: Any, x: float = 0, y: float = 0, pos: Optional[Dict[int, Tuple[float, float]]] = None, level: int = 0) -> Dict[int, Tuple[float, float]]:
            if pos is None:
                pos = {}
            pos[node.id] = (x, y)

            if node.left is not None:
                left_x = x - (2 ** (5 - level))
                left_y = y - 1
                pos = _recur(self.tree.nodes[node.left], left_x, left_y, pos, level + 1)
            if node.right is not None:
                right_x = x + (2 ** (5 - level))
                right_y = y - 1
                pos = _recur(self.tree.nodes[node.right], right_x, right_y, pos, level + 1)

            return pos

        root = self.tree.nodes[0]
        return _recur(root)

    def _build_figure(self, positions: Dict[int, Tuple[float, float]]) -> go.Figure:
        """
        Use node positions to draw lines and labels in Plotly.
        """
        labels: List[str] = []
        for node_id in self.tree.nodes:
            node = self.tree.nodes[node_id]
            if node.feature is not None and node.threshold is not None:
                fname = self.tree.feature_names[node.feature] if self.tree.feature_names else f"Feature {node.feature}"
                labels.append(f"{fname}<br>≤ {node.threshold:.2f}")
            else:
                if self.tree.is_classifier:
                    class_name = (
                        self.tree.class_names[node.value.argmax()]
                        if self.tree.class_names
                        else f"Class {node.value.argmax()}"
                    )
                    labels.append(class_name)
                else:
                    labels.append(f"{node.value:.2f}")

        # Normalize y so minimum is zero
        ys = [pos[1] for pos in positions.values()]
        min_y = min(ys)
        for k in positions:
            positions[k] = (positions[k][0], positions[k][1] - min_y)

        xe: List[float] = []
        ye: List[float] = []
        edge_colors: List[str] = []
        edge_widths: List[int] = []

        for node_id, node in self.tree.nodes.items():
            if node.left is not None:
                parent = positions[node_id]
                child = positions[node.left]
                xe += [parent[0], child[0], None]
                ye += [parent[1], child[1], None]
                color = (
                    self.tree.color_struct[int(node.value.argmax())]
                    if self.tree.is_classifier
                    else self.tree.color_struct[node.value]
                )
                edge_colors.append(color)
                edge_widths.append(self.tree.nodes[node.left].parent_edge_width)

            if node.right is not None:
                parent = positions[node_id]
                child = positions[node.right]
                xe += [parent[0], child[0], None]
                ye += [parent[1], child[1], None]
                color = self.tree.color_struct[node.value.argmax()] if self.tree.is_classifier else self.tree.color_dict[node.value]
                edge_colors.append(color)
                edge_widths.append(self.tree.nodes[node.right].parent_edge_width)

        def _make_annotations(
            pos: Dict[int, Tuple[float, float]],
            text: List[str],
            font_size: int = 10,
            font_color: str = "rgb(250,250,250)",
        ) -> List[Dict[str, Any]]:
            if len(pos) != len(text):
                raise ValueError("pos and text must have the same length")

            return [
                dict(
                    text=str(text[k]),
                    x=pos[node_id][0],
                    y=pos[node_id][1],
                    xref="x1",
                    yref="y1",
                    font=dict(color=font_color, size=font_size),
                    showarrow=False,
                    bgcolor="rgb(31, 119, 180)",
                    opacity=1,
                )
                for k, node_id in enumerate(pos)
            ]

        fig = go.Figure()

        for i_edge in range(len(xe) // 3):
            fig.add_trace(
                go.Scatter(
                    x=[xe[3 * i_edge], xe[3 * i_edge + 1], None],
                    y=[ye[3 * i_edge], ye[3 * i_edge + 1], None],
                    mode="lines",
                    line=dict(color=edge_colors[i_edge], width=edge_widths[i_edge]),
                    hoverinfo="none",
                )
            )

        annotations = _make_annotations(positions, labels)
        fig.update_layout(
            annotations=annotations,
            font_size=12,
            showlegend=False,
            xaxis=dict(showline=False, zeroline=False, showgrid=False),
            yaxis=dict(showline=False, zeroline=False, showgrid=False),
            margin=dict(l=40, r=40, b=85, t=100),
            hovermode="closest",
            plot_bgcolor="rgb(255,255,255)",
        )

        return fig


if __name__ == "__main__":
    # Dummy example to test SankeyTreePlot and GoTreePlot
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    # Create random data
    X = np.random.rand(50, 3)
    y = np.random.randint(0, 2, size=50)
    class_names = ["Class 0", "Class 1"]

    # Train a small random forest
    rf = RandomForestClassifier(n_estimators=1, max_depth=3)
    rf.fit(X, y)

    # Build a VisTree for the first estimator (must supply class_names)
    vis_tree = VisTree(rf.estimators_[0], X, class_names=class_names)

    # Sankey plot example
    sankey_plot = SankeyTreePlot(vis_tree, show_text=True, show_label=True)
    sankey_plot.show()

    # GoTree plot example
    go_plot = GoTreePlot(vis_tree, show_text=True)
    go_plot.show()
