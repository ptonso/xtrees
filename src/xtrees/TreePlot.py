import plotly.graph_objects as go
import plotly.express as px
import numpy as np


class SankeyTreePlot:
    def __init__(self, viz_tree, show_text=True, show_label=False):
        self.viz_tree = viz_tree
        self.show_text = show_text
        self.show_label = show_label
        self.max_depth = viz_tree.max_depth
        labels, colors, sources, targets, values, x, y, customdata, edge_labels = self.tree_to_sankey()
        self.fig = self.create_fig(labels, colors, sources, targets, values, x, y, customdata, edge_labels)

    def show(self):
        if self.show_label:
            self.add_color_label()
        self.fig.show()

    def save(self, path):
        if self.show_label:
            self.add_color_label()
        self.fig.write_image(path)

    def create_fig(self, labels, colors, sources, targets, values, x, y, customdata, edge_labels):
        hovertemplate = (
            '<b style="font-size:14px;">%{customdata[0]}</b><br>'
            '<b style="font-size:14px;">n_train:</b> %{value}<br>'
            '<b style="font-size:14px;">Value:</b> %{customdata[1]}<extra></extra>'
        )
        
        node_labels = labels if self.show_text else [''] * len(labels)

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=colors,
                x=x,
                y=y,
                customdata=customdata,
                hovertemplate=hovertemplate
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                hovertemplate='<b style "font-size:14px;">n_train:</b> %{value}<extra></extra>',
                label=edge_labels
            )
        )])

        fig.update_layout(
            font=dict(
                family="Courier New",
                size=14,
                color='black'
            )
        )

        return fig

    def tree_to_sankey(self):
        nodes = []
        labels = []
        colors = []
        sources = []
        targets = []
        values = []
        x = []
        y = []
        customdata = []
        edge_labels = []
        viz_tree = self.viz_tree

        positions = self.compute_node_positions()

        max_x = float('-inf')
        max_y = float('-inf')
        for key, (x_pos,y_pos) in positions.items():
            x_pos = abs(x_pos)
            y_pos = abs(y_pos)
            if x_pos > max_x:
                max_x = x_pos
            if y_pos > max_y:
                max_y = y_pos

        def traverse(node_id):
            node = viz_tree.nodes[node_id]
            label = ""
            customlabel = ""

            color = self._get_color_for_value(node.value)

            if node.feature is not None and node.threshold is not None:
                feature_label = viz_tree.feature_names[node.feature] if viz_tree.feature_names is not None else f"Feature {node.feature}"
                label = f"{feature_label[:6]}<br>≤ {node.threshold:.2f}"
                customlabel = f"{feature_label}<br>≤ {node.threshold:.2f}"
            else:
                if viz_tree.is_classifier:
                    if viz_tree.class_names is not None:
                        label = viz_tree.class_names[node.value.argmax()]
                    else:
                        label = f"Class {node.value.argmax()}"
                else:
                    label = f"{node.value:.4f}"

            if node_id not in nodes:
                nodes.append(node_id)
                labels.append(label)
                colors.append(color)

                customdata.append([customlabel, f"{np.round(node.value, 2)}" if self.viz_tree.is_classifier else f"{node.value:.4f}"])

                x_pos, y_pos = positions[node_id]
                x.append(x_pos / max_x)
                y.append((y_pos + max_y) / (2 * max_y))


            if node.left is not None:
                sources.append(nodes.index(node_id))
                targets.append(len(nodes))
                values.append(int(viz_tree.nodes[node.left].n_train))
                edge_labels.append(str(viz_tree.nodes[node.left].n_train))
                traverse(node.left)

            if node.right is not None:
                sources.append(nodes.index(node_id))
                targets.append(len(nodes))
                values.append(int(viz_tree.nodes[node.right].n_train))
                edge_labels.append(str(viz_tree.nodes[node.right].n_train))
                traverse(node.right)

        traverse(0)

        if len(colors) < len(labels):
            colors += ['rgba(0, 0, 0, 0.8)'] * (len(labels) - len(colors))

        return labels, colors, sources, targets, values, x, y, customdata, edge_labels


    def _get_color_for_value(self, value):
        color_struct = self.viz_tree.color_struct
        if self.viz_tree.is_classifier:
            return color_struct[value.argmax()]
        else:
            for i in range(len(color_struct) - 1):
                if color_struct[i][0] <= value < color_struct[i + 1][0]:
                    return color_struct[i][1]
            return color_struct[-1][1] 


    def add_color_label(self):
        if self.viz_tree.is_classifier:
            self.add_classification_color_label()
        else:
            self.add_regression_hue_bar()


    def add_classification_color_label(self):
        color_struct = self.viz_tree.color_struct
        class_names = self.viz_tree.class_names
        
        labels_with_colors = []
        for name, color in zip(class_names, color_struct):
            labels_with_colors.append(
                f'{name}: <span style="color:{color_struct[color]};">&#9608;</span>'
            )
        
        label_text = "<br>".join(labels_with_colors)
        self.fig.add_annotation(
            text=label_text,
            xref="paper", yref="paper",
            x=1.05, y=1,
            showarrow=False,
            align="left"
        )


    def add_regression_hue_bar(self):
        color_struct = self.viz_tree.color_struct
        cs = [c[0] for c in color_struct]
        hue_bar = go.Heatmap(        
            z=[[c[0]] for c in color_struct],
            colorscale=[[i/(len(color_struct)-1), color[1]] for i, color in enumerate(color_struct)], 
            showscale=False,
            x=[0.98],
            y=cs,  
            xaxis="x2",
            yaxis="y2"
        )
        self.fig.add_trace(hue_bar)
        self.fig.update_layout(
            xaxis2=dict(
                range=[0,0.5],
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
                showgrid=False
            ),
            plot_bgcolor='rgba(0,0,0,0)',
        )




    def compute_node_positions_linspace(self):
        # MVP
        nodes_list = self.viz_tree.get_nodes_depth_list()
        positions = {}

        max_depth = self.viz_tree.max_depth

        x_ticks = list(np.linspace(0, 1, max_depth+1))
        y_ticks = list(np.linspace(0, 1, len(nodes_list[max_depth])))

        for i, layer in enumerate(nodes_list):
            y_layer_ticks = y_ticks.copy()
            n_layer = len(layer)
            drop_nodes = max_depth - n_layer

            remove_from_front = True    
            for j in range(drop_nodes):
                if remove_from_front:
                    y_layer_ticks.pop(0)
                else:
                    y_layer_ticks.pop()
                remove_from_front = not remove_from_front
            for j, node in enumerate(layer):
                positions[node] = (x_ticks[i], y_layer_ticks[j])


    # _traverse_heuristics
    def compute_node_positions(self):
        tree = self.viz_tree
        self.class_order = []

        def compute_positions(node, x=0, y=0, pos=None, level=0, spacing=1.5):
            if pos is None:
                pos = {}
            node_height = 1 + (node.n_train / tree.n_train) * 0.4
            vertical_spacing = node_height + (2 ** (self.max_depth / 2 - level - 1))
            x_pos = x + 1
            if node.left and node.right:
                y_left = y + vertical_spacing / 2
                y_right = y - vertical_spacing / 2
            elif node.left is None and node.right is None:
                self.class_order.insert(0, node)
            else:
                y_left = y + vertical_spacing
                y_right = y - vertical_spacing
            if node.left:
                pos = compute_positions(tree.nodes[node.left], x_pos, y_left, pos, level + 1, spacing)
            if node.right:
                pos = compute_positions(tree.nodes[node.right], x_pos, y_right, pos, level + 1, spacing)
            pos[node.id] = (x, y)
            return pos
        
        root = tree.nodes[0]
        return compute_positions(root, spacing=1)


    def compute_node_positions_layer(self):
        nodes_list = self.viz_tree.get_nodes_depth_list()
        positions = {}

        max_depth = self.viz_tree.max_depth
        max_n_train = self.viz_tree.n_train

        x_ticks = list(np.linspace(0.1, 0.9, max_depth+1))
        
        cuts = self.compute_cuts()

        positions[0] = (0, 0.5)

        for i, layer_cuts in enumerate(cuts):

            for j, node_id in enumerate(nodes_list[i+1]):
                x_pos = x_ticks[i+1]
                node = self.viz_tree.nodes[node_id]

                cut = layer_cuts[j//2]
                displacement = node.n_train / max_n_train / 2

                if node.is_left:
                    y_pos = cut - displacement
                else:
                    y_pos = cut + displacement

                positions[node_id] = (x_pos, y_pos)

        return positions
    

    def compute_cuts(self):
        nodes_list = self.viz_tree.get_nodes_depth_list()
        max_n_train = self.viz_tree.n_train
        cuts = [[0.33], [0.66], [0.60, 0.7]]
        cuts = [[] for _ in range(self.viz_tree.max_depth)]

        prev_right_cut = 0
        prev_left_cut = 0

        for i in range(0, len(nodes_list)):
            for node_id in nodes_list[i]:
                node = self.viz_tree.nodes[node_id]
                if node.feature is not None:
                    if node.is_left:
                        if node.right:
                            right_child = self.viz_tree.nodes[node.right]
                            value = (right_child.n_train / max_n_train)
                            next_cut = prev_right_cut - value
                            cuts[i].append(next_cut)
                    else:
                        if node.left:
                            left_child = self.viz_tree.nodes[node.left]
                            value = (left_child.n_train / max_n_train)
                            next_cut = prev_right_cut + value
                            cuts[i].append(next_cut)
                            prev_right_cut = next_cut

        return cuts




class GoTreePlot():
    def __init__(self, viz_tree, show_text):
        self.tree = viz_tree
        self.show_text =  show_text
        self.update_edge_widths_based_on_samples(self.tree)     
        position = self.compute_layout(viz_tree)
        self.fig = self.tree2plot(viz_tree, position)

    def show(self):
        self.fig.show()

    def save(self, path):
        self.fig.write_image(path)


    def compute_layout(self, tree):
        def compute_positions(node, x=0, y=0, pos=None, level=0):
            if pos is None:
                pos = {}
                pos[node.id] = (x, y)

            if node is not None:
                if node.left is not None:
                    pos = compute_positions(tree.nodes[node.left], x - (2 ** (5 - level)), y - 1, pos, level + 1)
                if node.right is not None:
                    pos = compute_positions(tree.nodes[node.right], x + (2 ** (5 - level)), y - 1, pos, level + 1)
            return pos

        root = tree.nodes[0]
        return compute_positions(root)

    def update_edge_widths_based_on_samples(self, tree):
        max_n = tree.n_train
        for node_id, node in tree.nodes.items():
            if node.left:
                left_child = tree.nodes[node.left]
                left_child.parent_edge_width = 5
                # left_child.parent_edge_width = max(1, 20 * left_child.n_train / max_n)

            if node.right:
                right_child = tree.nodes[node.right]                
                # right_child.parent_edge_width = max(1, 20 * right_child.n_train / max_n)
                right_child.parent_edge_width = 5

    def tree2plot(self, viz_tree, position):
        feature_names = viz_tree.feature_names
        class_names = viz_tree.class_names
        labels = []
        for node_id in viz_tree.nodes:
            node = viz_tree.nodes[node_id]
            if node.feature is not None and node.threshold is not None:
                if feature_names is not None:
                    feature_label = feature_names[node.feature]
                else:
                    feature_label = f"Feature {node.feature}"                
                labels.append(f"{feature_label}<br>≤ {node.threshold:.2f}")
            else:
                if viz_tree.is_classifier():
                    class_name = class_names[node.value.argmax()] if viz_tree.class_names is not None else f"Class {node.value.argmax()}"
                    labels.append(class_name)
                else:
                    labels.append(f"Value {node.value:.2f}")

        Y = [position[node][1] for node in position]
        min_Y = min(Y)

        for k in position:
            position[k] = (position[k][0], position[k][1] - min_Y)

        Xn = [position[node_id][0] for node_id in viz_tree.nodes]
        Yn = [position[node_id][1] for node_id in viz_tree.nodes]

        Xe = []
        Ye = []
        edge_colors = []
        edge_widths = []

        for node_id, node in viz_tree.nodes.items():
            if node.left is not None:
                Xe += [position[node_id][0], position[node.left][0], None]
                Ye += [position[node_id][1], position[node.left][1], None]
                edge_colors.append(viz_tree.color_dict[node.value if not viz_tree.is_classifier() else node.value.argmax()])
                edge_widths.append(viz_tree.nodes[node.left].parent_edge_width)
            if node.right is not None:
                Xe += [position[node_id][0], position[node.right][0], None]
                Ye += [position[node_id][1], position[node.right][1], None]
                edge_colors.append(viz_tree.color_dict[node.value if not viz_tree.is_classifier() else node.value.argmax()])
                edge_widths.append(viz_tree.nodes[node.right].parent_edge_width)

        def make_annotations(pos, text, font_size=10, font_color='rgb(250,250,250)'):
            L = len(pos)
            if len(text) != L:
                raise ValueError('The lists pos and text must have the same length')

            annotations = []
            for k in range(L):
                annotations.append(
                    dict(
                        text=str(text[k]),
                        x=pos[k][0], y=pos[k][1],
                        xref='x1', yref='y1',
                        font=dict(color=font_color, size=font_size),
                        showarrow=False,
                        bgcolor='rgb(31, 119, 180)',
                        opacity=1
                    )
                )
            return annotations

        fig = go.Figure()

        for i in range(len(Xe) // 3):
            fig.add_trace(go.Scatter(
                x=[Xe[3 * i], Xe[3 * i + 1], None],
                y=[Ye[3 * i], Ye[3 * i + 1], None],
                mode='lines',
                line=dict(color=edge_colors[i], width=edge_widths[i]),
                hoverinfo='none'
            ))

        annotations = make_annotations(position, labels)
        fig.update_layout(
            annotations=annotations,
            font_size=12,
            showlegend=False,
            xaxis=dict(showline=False, zeroline=False, showgrid=False),
            yaxis=dict(showline=False, zeroline=False, showgrid=False),
            margin=dict(l=40, r=40, b=85, t=100),
            hovermode='closest',
            plot_bgcolor='rgb(255,255,255)'
        )

        return fig
