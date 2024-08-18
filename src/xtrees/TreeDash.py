import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import accuracy_score, mean_absolute_error

from .TreePlot import SankeyTreePlot
from .VizTree import VizTree


class VizTreeDashboard:
    def __init__(self, viz_tree, X_test, y_test, show_text=False):
        self.viz_tree = viz_tree
        self.is_classifier = self.viz_tree.is_classifier
        self.X_test = X_test
        self.y_test = y_test
        self.show_text = show_text
        self.max_depth = viz_tree.max_depth

        self.app = dash.Dash(__name__)

        self.initial_sankey = SankeyTreePlot(viz_tree)

        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
            self.app.layout = html.Div([
                html.Div([
                    html.Div([
                        html.Div(id='metric-display', style={'fontSize': 18, 'fontWeight': 'bold', 'margin': '10px'}),
                        html.Label('Depth Slider', style={'fontSize': 16, 'margin': '10px', 'textAlign': 'center'}),
                        dcc.Slider(
                            id='depth-slider',
                            min=1,
                            max=self.max_depth,
                            value=self.max_depth,
                            marks={str(i): str(i) for i in range(1, self.max_depth + 1)},
                            step=None,
                            vertical=True,
                            tooltip={'always_visible': True, 'placement': 'right'},
                            updatemode='drag',
                            verticalHeight=400
                        )
                    ], style={'padding': '0px', 'backgroundColor': 'white', 'textAlign': 'center', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'flexDirection': 'column', 'marginLeft': '20px'}),
                    dcc.Graph(id='sankey-plot', figure=self.initial_sankey.fig, style={'flex': '4'})
                ], style={'display': 'flex', 'flex-direction': 'row', 'backgroundColor': 'white', 'flex': '1', 'minHeight': '600px'})
            ], style={'display': 'flex', 'flex-direction': 'column', 'backgroundColor': 'white', 'height': '100vh', 'padding': '0px', 'boxSizing':'border-box'})
     
    def setup_callbacks(self):
        @self.app.callback(
            [Output('sankey-plot', 'figure'),
             Output('metric-display', 'children')],
            Input('depth-slider', 'value')
        )
        def update_sankey_plot(max_depth):
            pruned_viz_tree = self.viz_tree.prune(max_depth)
            if max_depth < 3 and self.show_text is None or self.show_text == True:
                pruned_sankey = SankeyTreePlot(pruned_viz_tree, show_text=True)
            else:
                pruned_sankey = SankeyTreePlot(pruned_viz_tree, show_text=False)
            pruned_ypred = pruned_viz_tree.predict(self.X_test)
            if self.is_classifier:
                metric = accuracy_score(self.y_test, pruned_ypred)
                metric_text = f"Accuracy: {metric:.2f}"
            else:
                metric = mean_absolute_error(self.y_test, pruned_ypred)
                metric_text = f"MAE: {metric:.2f}"
            return pruned_sankey.fig, metric_text

    def run(self, port=8060):
        self.app.run_server(port=port, debug=True)



class RFDashboard:
    def __init__(self, X, X_test, y_test, rf, class_names=None):
        self.class_names = class_names
        self.X = X
        self.X_test = X_test
        self.y_test = y_test
        self.rf = rf

        self.viz_trees = [VizTree(tree, X, class_names) for tree in rf.estimators_]
        self.is_classifier = self.viz_trees[0].is_classifier

        self.app = dash.Dash(__name__)

        self.initial_tree_id = 0
        self.initial_sankey = SankeyTreePlot(self.viz_trees[self.initial_tree_id])
        self.initial_max_depth = self.viz_trees[self.initial_tree_id].max_depth

        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        self.app.layout = html.Div([
            html.Div([
                html.Div([
                    html.Div(id='metric-display', style={'fontSize': 18, 'fontWeight': 'bold', 'margin': '10px'}),
                    html.Label('Depth Slider', style={'fontSize': 16, 'margin': '10px', 'textAlign': 'center'}),
                    dcc.Slider(
                        id='depth-slider',
                        min=1,
                        max=self.initial_max_depth,
                        value=self.initial_max_depth,
                        marks={str(i): str(i) for i in range(1, self.initial_max_depth + 1)},
                        step=None,
                        vertical=True,
                        tooltip={'always_visible': True, 'placement': 'right'},
                        updatemode='drag',
                        verticalHeight=400
                    )
                ], style={'padding': '0px', 'backgroundColor': 'white', 'textAlign': 'center', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'flexDirection': 'column', 'marginLeft': '20px'}),
            

            html.Div([
                dcc.Slider(
                    id='tree-id-slider',
                    min=0,
                    max=len(self.viz_trees) - 1,
                    value=self.initial_tree_id,
                    marks={str(i): str(i) for i in range(len(self.viz_trees))},
                    step=1,
                    tooltip={'always_visible': True, 'placement': 'bottom'},
                    updatemode='drag'
                ),
                dcc.Graph(id='sankey-plot', figure=self.initial_sankey.fig, style={'flex': '4', 'marginTop': '10px'}) 
            ], style={'width': '100%', 'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'maxHeight': '600px', 'marginTop': '30px'})
            ], style={'display': 'flex', 'flex-direction': 'row', 'backgroundColor': 'white' 'white', 'flex': '1', 'minHeight': '600px'})            
        ], style={'display': 'flex', 'flex-direction': 'column', 'backgroundColor': 'white', 'height': '100vh', 'padding': '0px', 'boxSizing':'border-box'})

    def setup_callbacks(self):
        @self.app.callback(
            [Output('sankey-plot', 'figure'),
             Output('metric-display', 'children')],
            [Input('depth-slider', 'value'),
             Input('tree-id-slider', 'value')]
        )
        def update_sankey_plot(max_depth, tree_id):
            selected_viz_tree = self.viz_trees[tree_id]
            pruned_viz_tree = selected_viz_tree.prune(max_depth)
            if max_depth > 3:
                pruned_sankey = SankeyTreePlot(pruned_viz_tree, show_text=False)
            else:
                pruned_sankey = SankeyTreePlot(pruned_viz_tree, show_text=True)
            pruned_ypred = pruned_viz_tree.predict(self.X_test)
            if self.is_classifier:
                metric = accuracy_score(self.y_test, pruned_ypred)
                metric_text = f"Accuracy: {metric:.2f}"
            else:
                metric = mean_absolute_error(self.y_test, pruned_ypred)
                metric_text = f"MAE: {metric:.2f}"
            return pruned_sankey.fig, metric_text

        @self.app.callback(
            [Output('depth-slider', 'max'),
             Output('depth-slider', 'marks'),
             Output('depth-slider', 'value')],
            Input('tree-id-slider', 'value')
        )
        def update_depth_slider(tree_id):
            selected_viz_tree = self.viz_trees[tree_id]
            max_depth = selected_viz_tree.max_depth
            marks = {str(i): str(i) for i in range(1, max_depth + 1)}
            return max_depth, marks, max_depth

    def run(self, port=8061):
        self.app.run_server(debug=True, port=port)


class CombinedDashboard:
    def __init__(self, viz_tree, X_test, y_test, X, rf, class_names=None):
        self.viz_tree = viz_tree
        self.is_classifier = self.viz_tree.is_classifier
        self.X_test = X_test
        self.y_test = y_test
        self.max_depth = viz_tree.max_depth

        self.class_names = class_names
        self.X = X
        self.rf = rf
        self.rf_pred = rf.predict(X_test)

        feature_names = rf.feature_names_in_
        self.viz_trees = []
        for tree in rf.estimators_:
            tree.feature_names_in_ = feature_names
            viz_tree = VizTree(tree, X, class_names)
            self.viz_trees.append(viz_tree)

        self.initial_tree_id = 0
        self.initial_sankey_rf = SankeyTreePlot(self.viz_trees[self.initial_tree_id])
        self.initial_max_depth_rf = self.viz_trees[self.initial_tree_id].max_depth

        self.app = dash.Dash(__name__)

        self.initial_sankey_tree = SankeyTreePlot(viz_tree)

        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        self.app.layout = html.Div([
            # Div for the first plot and its slider
            html.Div([
                html.Div([
                    html.H2('Forest Based Tree', style={'textAlign': 'center', 'marginLeft': '20px'}), 
                    html.Div([
                        html.Div(id='metric-display-tree', style={'fontSize': 14, 'fontWeight': 'bold', 'margin': '10px'}),
                        html.Label('Tree Depth Slider', style={'fontSize': 16, 'margin': '10px', 'textAlign': 'center'}),
                        dcc.Slider(
                            id='depth-slider-tree',
                            min=1,
                            max=self.max_depth,
                            value=self.max_depth,
                            marks={str(i): str(i) for i in range(1, self.max_depth + 1)},
                            step=None,
                            vertical=True,
                            tooltip={'always_visible': True, 'placement': 'right'},
                            updatemode='drag',
                            verticalHeight=300
                        )
                    ], style={'padding': '0px', 'backgroundColor': 'white', 'textAlign': 'center', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'flexDirection': 'column', 'marginLeft': '0px'}), 
                ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'justifyContent': 'center', 'width': '15%'}),
                html.Div([
                    dcc.Graph(id='sankey-plot-tree', figure=self.initial_sankey_tree.fig, style={'width': '600px', 'height': '500px', 'marginTop': '150px', 'marginBotton': '150px'})
                ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'center'})
            ], style={'display': 'flex', 'flexDirection': 'row', 'backgroundColor': 'white', 'flex': '1'}),
            
            # Div for the second plot and its sliders
            html.Div([
                html.Div([
                    html.H2('Original Random Forest', style={'textAlign': 'center'}),
                    html.Div([
                        html.Div(id='metric-display-rf', style={'fontSize': 14, 'fontWeight': 'bold', 'margin': '10px'}),
                        html.Label('RF Depth Slider', style={'fontSize': 16, 'margin': '10px', 'textAlign': 'center'}),
                        dcc.Slider(
                            id='depth-slider-rf',
                            min=1,
                            max=self.initial_max_depth_rf,
                            value=self.initial_max_depth_rf,
                            marks={str(i): str(i) for i in range(1, self.initial_max_depth_rf + 1)},
                            step=None,
                            vertical=True,
                            tooltip={'always_visible': True, 'placement': 'right'},
                            updatemode='drag',
                            verticalHeight=300
                        )
                    ], style={'padding': '0px', 'backgroundColor': 'white', 'textAlign': 'center', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'flexDirection': 'column'}),
                ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'justifyContent': 'center', 'width': '15%'}),
                html.Div([
                    dcc.Slider(
                        id='tree-id-slider',
                        min=0,
                        max=len(self.viz_trees) - 1,
                        value=self.initial_tree_id,
                        marks={str(i): str(i) for i in range(len(self.viz_trees))},
                        step=1,
                        tooltip={'always_visible': True, 'placement': 'bottom'},
                        updatemode='drag'
                    ),
                    dcc.Graph(id='sankey-plot-rf', figure=self.initial_sankey_rf.fig, style={'width': '600px', 'height': '500px', 'marginTop': '40px'}) 
                ], style={'width': '100%', 'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'maxHeight': '600px', 'marginTop': '60px'})
            ], style={'display': 'flex', 'flexDirection': 'row', 'backgroundColor': 'white', 'flex': '1', 'minHeight': '600px'})
        ], style={'display': 'flex', 'flexDirection': 'row', 'backgroundColor': 'white', 'height': '100vh', 'padding': '0px', 'boxSizing': 'border-box'})

    def metric_text(self, pruned_ypred):
        if self.is_classifier:
            y_metric = accuracy_score(self.y_test, pruned_ypred)
            cong_metric = accuracy_score(self.rf_pred, pruned_ypred)
            ymetric_text = f"Acc.: {y_metric:.2f}"
        else:
            y_metric = mean_absolute_error(self.y_test, pruned_ypred)
            cong_metric = mean_absolute_error(self.rf_pred, pruned_ypred)
            ymetric_text = f"MAE:  {y_metric:.2f}"
        return [html.Span(ymetric_text)]

    def setup_callbacks(self):
        @self.app.callback(
            [Output('sankey-plot-tree', 'figure'),
             Output('metric-display-tree', 'children')],
            Input('depth-slider-tree', 'value')
        )
        def update_sankey_plot_tree(max_depth):
            pruned_viz_tree = self.viz_tree.prune(max_depth)
            if max_depth > 3:
                pruned_sankey = SankeyTreePlot(pruned_viz_tree, show_text=False)
            else:
                pruned_sankey = SankeyTreePlot(pruned_viz_tree, show_text=True)
            pruned_ypred = pruned_viz_tree.predict(self.X_test)
            metric_display_text = self.metric_text(pruned_ypred)
            return pruned_sankey.fig, metric_display_text

        @self.app.callback(
            [Output('sankey-plot-rf', 'figure'),
             Output('metric-display-rf', 'children')],
            [Input('depth-slider-rf', 'value'),
             Input('tree-id-slider', 'value')]
        )
        def update_sankey_plot_rf(max_depth, tree_id):
            selected_viz_tree = self.viz_trees[tree_id]
            pruned_viz_tree = selected_viz_tree.prune(max_depth)
            if max_depth > 3:
                pruned_sankey = SankeyTreePlot(pruned_viz_tree, show_text=False)
            else:
                pruned_sankey = SankeyTreePlot(pruned_viz_tree, show_text=True)
            pruned_ypred = pruned_viz_tree.predict(self.X_test)
            metric_display_text = self.metric_text(pruned_ypred)
            return pruned_sankey.fig, metric_display_text

        @self.app.callback(
            [Output('depth-slider-rf', 'max'),
             Output('depth-slider-rf', 'marks'),
             Output('depth-slider-rf', 'value')],
            Input('tree-id-slider', 'value')
        )
        def update_depth_slider_rf(tree_id):
            selected_viz_tree = self.viz_trees[tree_id]
            max_depth = selected_viz_tree.max_depth
            marks = {str(i): str(i) for i in range(1, max_depth + 1)}
            return max_depth, marks, max_depth

    def run(self, port=8062):
        self.app.run_server(port=port, debug=True)


