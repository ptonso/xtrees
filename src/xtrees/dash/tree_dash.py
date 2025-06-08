from typing import Any, List, Optional, Tuple, Dict

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error

from src.logger import setup_logger
from src.xtrees.dash.tree_plot import SankeyTreePlot
from src.xtrees.dash.vis_tree import VisTree


class VisTreeDashboard:
    """
    Dashboard for visualizing a single decision tree via a Sankey plot.
    """

    def __init__(
        self,
        viz_tree: VisTree,
        X_test: Any,
        y_test: Any,
        show_text: bool = False,
    ) -> None:
        self.logger = setup_logger("api.log")
        self.viz_tree: VisTree = viz_tree
        self.is_classifier: bool = self.viz_tree.is_classifier
        self.X_test: Any = X_test
        self.y_test: Any = y_test
        self.show_text: bool = show_text
        self.max_depth: int = self.viz_tree.max_depth

        self.app: dash.Dash = dash.Dash(__name__)
        self.initial_sankey: SankeyTreePlot = SankeyTreePlot(self.viz_tree)

        self._setup_layout()
        self._setup_callbacks()

    def _build_sidebar(self) -> html.Div:
        """
        Build the sidebar containing the depth slider and metric display.
        """
        return html.Div(
            [
                html.Div(
                    id="metric-display",
                    style={
                        "fontSize": 18,
                        "fontWeight": "bold",
                        "margin": "10px",
                    },
                ),
                html.Label(
                    "Depth Slider",
                    style={
                        "fontSize": 16,
                        "margin": "10px",
                        "textAlign": "center",
                    },
                ),
                dcc.Slider(
                    id="depth-slider",
                    min=1,
                    max=self.max_depth,
                    value=self.max_depth,
                    marks={str(i): str(i) for i in range(1, self.max_depth + 1)},
                    step=None,
                    vertical=True,
                    tooltip={"always_visible": True, "placement": "right"},
                    updatemode="drag",
                    verticalHeight=400,
                ),
            ],
            style={
                "padding": "0px",
                "backgroundColor": "white",
                "textAlign": "center",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "flexDirection": "column",
                "marginLeft": "20px",
            },
        )

    def _build_graph_container(self) -> dcc.Graph:
        """
        Build the Graph component for displaying the Sankey plot.
        """
        return dcc.Graph(
            id="sankey-plot",
            figure=self.initial_sankey.fig,
            style={"flex": "4"},
        )

    def _setup_layout(self) -> None:
        """
        Assemble the overall layout of the dashboard.
        """
        sidebar = self._build_sidebar()
        graph = self._build_graph_container()

        self.app.layout = html.Div(
            [
                html.Div(
                    [sidebar, graph],
                    style={
                        "display": "flex",
                        "flex-direction": "row",
                        "backgroundColor": "white",
                        "flex": "1",
                        "minHeight": "600px",
                    },
                )
            ],
            style={
                "display": "flex",
                "flex-direction": "column",
                "backgroundColor": "white",
                "height": "100vh",
                "padding": "0px",
                "boxSizing": "border-box",
            },
        )

    def _setup_callbacks(self) -> None:
        """
        Define callbacks to update the Sankey plot and metrics based on depth slider.
        """
        @self.app.callback(
            [Output("sankey-plot", "figure"), Output("metric-display", "children")],
            Input("depth-slider", "value"),
        )
        def update_sankey_plot(max_depth: int) -> Tuple[Any, str]:
            pruned_tree = self.viz_tree.prune(max_depth)
            show = (max_depth < 3 and self.show_text is None) or self.show_text
            sankey_obj = SankeyTreePlot(pruned_tree, show_text=show)

            y_pred = pruned_tree.predict(self.X_test)
            if self.is_classifier:
                metric_val = accuracy_score(self.y_test, y_pred)
                metric_text = f"Accuracy: {metric_val:.2f}"
            else:
                metric_val = mean_absolute_error(self.y_test, y_pred)
                metric_text = f"MAE: {metric_val:.2f}"

            return sankey_obj.fig, metric_text

    def run(self, port: int = 8060) -> None:
        """
        Launch the Dash server.
        """
        self.app.run_server(port=port, debug=True)


class RFDashboard:
    """
    Dashboard for visualizing individual trees in a random forest.
    """

    def __init__(
        self,
        X: Any,
        X_test: Any,
        y_test: Any,
        rf: Any,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.logger = setup_logger("api.log")
        self.class_names: Optional[List[str]] = class_names
        self.X: Any = X
        self.X_test: Any = X_test
        self.y_test: Any = y_test
        self.rf: Any = rf

        self.viz_trees: List[VisTree] = [
            VisTree(tree, X, class_names) for tree in rf.estimators_
        ]
        self.is_classifier: bool = self.viz_trees[0].is_classifier

        self.app: dash.Dash = dash.Dash(__name__)
        self.initial_tree_id: int = 0
        self.initial_sankey: SankeyTreePlot = SankeyTreePlot(
            self.viz_trees[self.initial_tree_id], rf=self.rf
        )
        self.initial_max_depth: int = self.viz_trees[self.initial_tree_id].max_depth

        self._setup_layout()
        self._setup_callbacks()

    def _build_sidebar(self) -> html.Div:
        """
        Build the sidebar containing the depth slider and metric display.
        """
        return html.Div(
            [
                html.Div(
                    id="metric-display",
                    style={
                        "fontSize": 18,
                        "fontWeight": "bold",
                        "margin": "10px",
                    },
                ),
                html.Label(
                    "Depth Slider",
                    style={
                        "fontSize": 16,
                        "margin": "10px",
                        "textAlign": "center",
                    },
                ),
                dcc.Slider(
                    id="depth-slider",
                    min=1,
                    max=self.initial_max_depth,
                    value=self.initial_max_depth,
                    marks={str(i): str(i) for i in range(1, self.initial_max_depth + 1)},
                    step=None,
                    vertical=True,
                    tooltip={"always_visible": True, "placement": "right"},
                    updatemode="drag",
                    verticalHeight=400,
                ),
            ],
            style={
                "padding": "0px",
                "backgroundColor": "white",
                "textAlign": "center",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "flexDirection": "column",
                "marginLeft": "20px",
            },
        )

    def _build_tree_selector_and_graph(self) -> html.Div:
        """
        Build the tree-ID slider and Sankey plot container.
        """
        return html.Div(
            [
                dcc.Slider(
                    id="tree-id-slider",
                    min=0,
                    max=len(self.viz_trees) - 1,
                    value=self.initial_tree_id,
                    marks={str(i): str(i) for i in range(len(self.viz_trees))},
                    step=1,
                    tooltip={"always_visible": True, "placement": "bottom"},
                    updatemode="drag",
                ),
                dcc.Graph(
                    id="sankey-plot",
                    figure=self.initial_sankey.fig,
                    style={"flex": "4", "marginTop": "10px"},
                ),
            ],
            style={
                "width": "100%",
                "display": "flex",
                "flexDirection": "column",
                "justifyContent": "center",
                "maxHeight": "600px",
                "marginTop": "30px",
            },
        )

    def _setup_layout(self) -> None:
        """
        Assemble the overall layout of the random forest dashboard.
        """
        sidebar = self._build_sidebar()
        tree_selector_and_graph = self._build_tree_selector_and_graph()

        self.app.layout = html.Div(
            [
                html.Div(
                    [sidebar, tree_selector_and_graph],
                    style={
                        "display": "flex",
                        "flex-direction": "row",
                        "backgroundColor": "white",
                        "flex": "1",
                        "minHeight": "600px",
                    },
                )
            ],
            style={
                "display": "flex",
                "flex-direction": "column",
                "backgroundColor": "white",
                "height": "100vh",
                "padding": "0px",
                "boxSizing": "border-box",
            },
        )

    def _setup_callbacks(self) -> None:
        """
        Define callbacks to update Sankey plot/metrics and depth slider based on tree ID.
        """
        @self.app.callback(
            [Output("sankey-plot", "figure"), Output("metric-display", "children")],
            [Input("depth-slider", "value"), Input("tree-id-slider", "value")],
        )
        def update_sankey_plot(max_depth: int, tree_id: int) -> Tuple[Any, str]:
            selected = self.viz_trees[tree_id]
            pruned = selected.prune(max_depth)
            if max_depth > 3:
                sankey_obj = SankeyTreePlot(pruned, show_text=False)
            else:
                sankey_obj = SankeyTreePlot(pruned, show_text=True)

            y_pred = pruned.predict(self.X_test)
            if self.is_classifier:
                metric_val = accuracy_score(self.y_test, y_pred)
                metric_text = f"Accuracy: {metric_val:.2f}"
            else:
                metric_val = mean_absolute_error(self.y_test, y_pred)
                metric_text = f"MAE: {metric_val:.2f}"

            return sankey_obj.fig, metric_text

        @self.app.callback(
            [Output("depth-slider", "max"), Output("depth-slider", "marks"), Output("depth-slider", "value")],
            Input("tree-id-slider", "value"),
        )
        def update_depth_slider(tree_id: int) -> Tuple[int, Dict[str, str], int]:
            selected = self.viz_trees[tree_id]
            max_depth = selected.max_depth
            marks = {str(i): str(i) for i in range(1, max_depth + 1)}
            return max_depth, marks, max_depth

    def run(self, port: int = 8061) -> None:
        """
        Launch the Dash server.
        """
        self.app.run_server(debug=True, port=port)





class CombinedDashboard:
    """
    Dashboard combining a single tree and the full random forest,
    with all dimensions hard-coded in pixels.
    """

    def __init__(
        self,
        viz_tree: VisTree,
        X_test: Any,
        y_test: Any,
        X: Any,
        rf: Any,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.logger = setup_logger("api.log")
        self.viz_tree = viz_tree
        self.X_test = X_test
        self.y_test = y_test
        self.max_depth = viz_tree.max_depth
        self.class_names = class_names
        self.X = X
        self.rf = rf
        self.rf_pred = rf.predict(X_test)

        # build list of VisTree for each RF estimator
        feature_names = rf.feature_names_in_
        self.viz_trees: List[VisTree] = []
        for tree in rf.estimators_:
            tree.feature_names_in_ = feature_names
            self.viz_trees.append(VisTree(tree, X, class_names))

        # initial Sankey objects
        self.initial_sankey_tree = SankeyTreePlot(viz_tree)
        first_rf = self.viz_trees[0]
        self.initial_sankey_rf = SankeyTreePlot(first_rf)
        self.initial_max_depth_rf = first_rf.max_depth

        # Dash app
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()

    def _build_tree_section(self) -> html.Div:
        """Build left-column: controls + single-tree Sankey."""
        return html.Div(
            [
                # controls
                html.Div(
                    [
                        html.H2("Forest Based Tree", style={"textAlign": "center", "marginLeft": "20px"}),
                        html.Div(id="metric-display-tree", style={"fontSize": 14, "fontWeight": "bold", "margin": "10px"}),
                        html.Label("Tree Depth Slider", style={"fontSize": 16, "margin": "10px", "textAlign": "center"}),
                        dcc.Slider(
                            id="depth-slider-tree",
                            min=1,
                            max=self.max_depth,
                            value=self.max_depth,
                            marks={str(i): str(i) for i in range(1, self.max_depth + 1)},
                            step=None,
                            vertical=True,
                            tooltip={"always_visible": True, "placement": "right"},
                            updatemode="drag",
                            verticalHeight=300,
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "width": "15%",
                    },
                ),
                # graph
                html.Div(
                    dcc.Graph(
                        id="sankey-plot-tree",
                        figure=self.initial_sankey_tree.fig,
                        style={
                            "width": "600px",
                            "height": "500px",
                            "marginTop": "150px",
                            "marginBottom": "150px",
                        },
                    ),
                    style={"width": "100%", "display": "flex", "justifyContent": "center"},
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "row",
                "backgroundColor": "white",
                "flex": "1",
            },
        )

    def _build_rf_section(self) -> html.Div:
        """Build right-column: controls + RF-tree-ID slider + RF Sankey."""
        return html.Div(
            [
                # controls
                html.Div(
                    [
                        html.H2("Original Random Forest", style={"textAlign": "center"}),
                        html.Div(id="metric-display-rf", style={"fontSize": 14, "fontWeight": "bold", "margin": "10px"}),
                        html.Label("RF Depth Slider", style={"fontSize": 16, "margin": "10px", "textAlign": "center"}),
                        dcc.Slider(
                            id="depth-slider-rf",
                            min=1,
                            max=self.initial_max_depth_rf,
                            value=self.initial_max_depth_rf,
                            marks={str(i): str(i) for i in range(1, self.initial_max_depth_rf + 1)},
                            step=None,
                            vertical=True,
                            tooltip={"always_visible": True, "placement": "right"},
                            updatemode="drag",
                            verticalHeight=300,
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "width": "15%",
                    },
                ),
                # tree-ID + graph
                html.Div(
                    [
                        dcc.Slider(
                            id="tree-id-slider",
                            min=0,
                            max=len(self.viz_trees) - 1,
                            value=0,
                            marks={str(i): str(i) for i in range(len(self.viz_trees))},
                            step=1,
                            tooltip={"always_visible": True, "placement": "bottom"},
                            updatemode="drag",
                        ),
                        dcc.Graph(
                            id="sankey-plot-rf",
                            figure=self.initial_sankey_rf.fig,
                            style={"width": "600px", "height": "500px", "marginTop": "40px"},
                        ),
                    ],
                    style={
                        "width": "100%",
                        "display": "flex",
                        "flexDirection": "column",
                        "justifyContent": "center",
                        "maxHeight": "600px",
                        "marginTop": "60px",
                    },
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "row",
                "backgroundColor": "white",
                "flex": "1",
                "minHeight": "600px",
            },
        )

    def _setup_layout(self) -> None:
        """Assemble the two columns into the page layout."""
        self.app.layout = html.Div(
            [
                self._build_tree_section(),
                self._build_rf_section(),
            ],
            style={
                "display": "flex",
                "flexDirection": "row",
                "backgroundColor": "white",
                "height": "100vh",
                "padding": "0px",
                "boxSizing": "border-box",
            },
        )

    def metric_text(self, pruned_ypred: Any) -> List[html.Span]:
        """Compute the accuracy or MAE text for display."""
        if self.viz_tree.is_classifier:
            val = accuracy_score(self.y_test, pruned_ypred)
            txt = f"Acc.: {val:.2f}"
        else:
            val = mean_absolute_error(self.y_test, pruned_ypred)
            txt = f"MAE: {val:.2f}"
        return [html.Span(txt)]

    def _setup_callbacks(self) -> None:
        """Wire up the slider callbacks to regenerate both Sankey plots."""
        @self.app.callback(
            [Output("sankey-plot-tree", "figure"), Output("metric-display-tree", "children")],
            Input("depth-slider-tree", "value"),
        )
        def update_tree(max_depth: int) -> Tuple[Any, List[html.Span]]:
            pruned = self.viz_tree.prune(max_depth)
            fig = SankeyTreePlot(pruned, show_text=(max_depth <= 3)).fig
            preds = pruned.predict(self.X_test)
            return fig, self.metric_text(preds)

        @self.app.callback(
            [Output("sankey-plot-rf", "figure"), Output("metric-display-rf", "children")],
            [Input("depth-slider-rf", "value"), Input("tree-id-slider", "value")],
        )
        def update_rf(max_depth: int, tree_id: int) -> Tuple[Any, List[html.Span]]:
            vt = self.viz_trees[tree_id].prune(max_depth)
            fig = SankeyTreePlot(vt, show_text=(max_depth <= 3)).fig
            preds = vt.predict(self.X_test)
            return fig, self.metric_text(preds)

    def run(self, port: int = 8062) -> None:
        """Launch the Dash server."""
        self.app.run_server(port=port, debug=True)




if __name__ == "__main__":
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, size=100)
    class_names = ["Class 0", "Class 1"]

    rf = RandomForestClassifier(n_estimators=2, max_depth=3)
    rf.fit(X, y)

    viz_tree = VisTree(rf.estimators_[0], X, class_names=class_names)

    tree_dashboard = VisTreeDashboard(viz_tree, X, y)
    tree_dashboard.app.run(debug=True, port=8060)

    # To test the RF dashboard, uncomment:
    # rf_dashboard = RFDashboard(X, X, y, rf, class_names=class_names)
    # rf_dashboard.app.run(debug=True, port=8061)

    # To test the combined dashboard, uncomment:
    # combined_dashboard = CombinedDashboard(viz_tree, X, y, X, rf, class_names=class_names)
    # combined_dashboard.app.run(debug=True, port=8062)
