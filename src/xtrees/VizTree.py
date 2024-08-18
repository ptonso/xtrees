from sklearn.tree import _tree
import plotly.express as px
import numpy as np
import pandas as pd

class VizNode:
    def __init__(self, id, feature=None, threshold=None, value=None, parent=None, is_left=None):
        self.id = id
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.parent = parent
        self.is_left = is_left

        self.left = None
        self.right = None

        self.n_train = 0
        self.n_samples = 0


class VizTree():
    def __init__(self, model, X=None, class_names=None, log_coloring=False):
        self.model = model
        self.nodes = {}
        self.feature_names = model.feature_names_in_ if hasattr(model, "feature_names_in_") else None
        self.is_classifier = hasattr(model, "classes_")
        
        if class_names is not None:
            self.class_names = class_names
            self.n_classes = len(self.class_names) if self.is_classifier else 0
        elif self.is_classifier:
            raise Exception("should provide class_names for classifier")

        self.log_coloring = log_coloring

        self.n_samples = 0
        self.n_train = 0
        self.max_depth = 0

        self.populate_tree_from_model()
        self._generate_color_struct()

        if X is not None:
            if isinstance(X, pd.DataFrame):
                X = X.to_numpy()
            self.populate_ns(X, prune=True)
            self.propagate_values(consider_proba=True)
            self.prune_redundants(consider_proba=True)

    def _generate_color_struct(self, num_colors=100, opacity=0.8):
        """Generate a dictionary with labels as keys and color strings as values."""
        if self.is_classifier:
            colors = px.colors.qualitative.Plotly[:self.n_classes]
            self.color_struct = {i: color for i, color in enumerate(colors)}
        else:
            possible_values = sorted(self.possible_values)
            min_val, max_val = min(possible_values), max(possible_values)
            color_ranges = np.linspace(min_val, max_val, num_colors)
            colors = px.colors.sample_colorscale('Viridis', num_colors)
            self.color_struct = [(color_ranges[i], colors[i].replace('rgb', 'rgba').replace(')', f', {opacity})')) for i in range(num_colors)]



    def _add_node(self, id, feature=None, threshold=None, value=None, parent=None, is_left=None):
        node = VizNode(id, feature, threshold, value, parent, is_left)
        self.nodes[id] = node
        return node

    def populate_tree_from_model(self):
        """Populate the tree attributes from the model."""
        tree = self.model.tree_
        feature = tree.feature
        threshold = tree.threshold
        children_left = tree.children_left
        children_right = tree.children_right
        value = tree.value

        self.max_depth = 0
        self.possible_values = set()

        def add_tree_node(node_id, parent=None, is_left=None, depth=0):
            if node_id == _tree.TREE_LEAF:
                return None

            if self.is_classifier:
                node_value = value[node_id].flatten()                
            else:
                node_value = value[node_id][0, 0]
                self.possible_values.add(node_value)

            node = self._add_node(
                id=node_id,
                feature=feature[node_id] if feature[node_id] != _tree.TREE_UNDEFINED else None,
                threshold=threshold[node_id] if feature[node_id] != _tree.TREE_UNDEFINED else None,
                value=node_value,
                parent = parent.id if parent else None,
                is_left=is_left
            )

            self.max_depth = max(self.max_depth, depth)

            left_child_id = children_left[node_id]
            right_child_id = children_right[node_id]

            node.left = add_tree_node(left_child_id, node, is_left=True, depth=depth+1)
            node.right = add_tree_node(right_child_id, node, is_left=False, depth=depth+1)
            return node_id

        add_tree_node(0)


    def update_max_depth(self, node_id=0, depth=0):
        if depth == 0:
            self.max_depth = 0
        if node_id is None:
            return
        node = self.nodes[node_id]
        self.max_depth = max(depth, self.max_depth)
        if node.left is not None:
            self.update_max_depth(node.left, depth + 1)
        if node.right is not None:
            self.update_max_depth(node.right, depth + 1)


    def get_nodes_depth_list(self):
        """
        Return a list of lists of node objects based on their depth position.
        """
        if not self.nodes:
            return []
        
        nodes_by_depth = {}
        
        def traverse(node, depth):
            if node is None:
                return
            
            if depth not in nodes_by_depth:
                nodes_by_depth[depth] = []
            
            nodes_by_depth[depth].append(node.id)
            self.max_depth = max(self.max_depth, depth)

            if node.left is not None:
                traverse(self.nodes[node.left], depth + 1)
            if node.right is not None:
                traverse(self.nodes[node.right], depth + 1)

        traverse(self.nodes[0], 0)

        list_nodes = [nodes_by_depth.get(depth, []) for depth in range(self.max_depth+1)]

        return list_nodes


    def predict(self, X, predict_probas=False):
        """Predict the class or value for input samples X based on the VizTree."""
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        n_samples = X.shape[0]

        if self.is_classifier:
            if predict_probas:
                predictions = np.zeros((n_samples, self.n_classes))
            else:
                predictions = np.zeros(n_samples, dtype=int)
        else:
            predictions = np.zeros(n_samples)

        masks = np.ones(n_samples, dtype=bool)

        def traverse(node_id, masks):
            node = self.nodes[node_id]
            if node.left is None and node.right is None:
                if self.is_classifier:
                    if predict_probas:
                        predictions[masks] = node.value / node.value.sum()
                    else:
                        predictions[masks] = node.value.argmax()
                else:
                    predictions[masks] = node.value
                return
            
            left_masks = masks & (X[:, node.feature] <= node.threshold)
            right_masks = masks & (X[:, node.feature] > node.threshold)

            if node.left is not None:
                traverse(node.left, left_masks)
            if node.right is not None:
                traverse(node.right, right_masks)

        traverse(0, masks)
        return predictions
                

    def populate_ns(self, samples, prune=False):
        """Predict and adjust the number of samples that pass through each node."""
        if isinstance(samples, pd.DataFrame):
            samples = samples.to_numpy()

        self.n_train = samples.shape[0]
        masks = np.ones((samples.shape[0],), dtype=bool)

        def traverse(node_id, masks):
            node = self.nodes[node_id]
        
            if node.left is None and node.right is None:
                self.nodes[node_id].n_train = masks.sum()
                return
        
            left_masks = masks & (samples[:, node.feature] <= node.threshold)
            right_masks = masks & (samples[:, node.feature] > node.threshold)
        
            if node.left is not None:
                traverse(node.left, left_masks)
            if node.right is not None:
                traverse(node.right, right_masks)

            self.nodes[node_id].n_train = masks.sum()
        
        traverse(0, masks)
        if prune:
            self.prune_by_samples()
            self.update_max_depth()

    def prune_by_samples(self):

        def prune_leaf_nodes(nodes):
            pruned = False
            for node_id in nodes:
                node = self.nodes[node_id]
                if node.left is None and node.right is None and node.n_train == 0:
                    parent = self.nodes[node.parent]
                    if node.is_left:
                        VizTree.prune_branch(self, parent, left=node)
                    else:
                        VizTree.prune_branch(self, parent, right=node)
                    pruned = True
            return pruned

        nodes_list = self.get_nodes_depth_list()

        while nodes_list:
            pruned = prune_leaf_nodes(nodes_list[-1])
            nodes_list.pop()
            if pruned:            
                nodes_list = self.get_nodes_depth_list()
                    
        # Ensure no leaf nodes with n_train == 0 remain
        for depth_nodes in nodes_list:
            for node_id in depth_nodes:
                node = self.nodes[node_id]
                if node.left is None and node.right is None and node.n_train == 0:
                    print(f"Warning: Node {node_id} is a leaf node with n_train = 0.")



    def print_tree(self):
        """Print the tree structure in a readable format."""
        def print_node(node, depth=0, prefix="Root"):
            indent = "  " * depth
            print(f"{indent}{prefix}: Node {node.id}")
            if node.feature is not None:
                print(f"{indent}  Feature:      {self.feature_names[node.feature] if self.feature_names is not None else node.feature}")
                print(f"{indent}  Threshold:    {node.threshold:.2f}")
                print(f"{indent}  N_train:      {node.n_train}")
                print(f"{indent}  N_samples:    {node.n_samples}")
                print(f"{indent}  Left child:   {node.left}")
                print(f"{indent}  Right child:  {node.right}")

            print(f"{indent}  Value: {node.value}")
            if node.left is not None:
                print_node(self.nodes[node.left], depth + 1, prefix="L")
            if node.right is not None:
                print_node(self.nodes[node.right], depth + 1, prefix="R")

        if 0 in self.nodes:
            print_node(self.nodes[0])

    def print_nodes(self, state=''):
        print(state)
        for id, node in self.nodes.items():
            print(f"\nnode {id}")
            for attr, value in node.__dict__.items():
                print(f"{attr:<20}: {value}")


    @staticmethod
    def prune_branch(tree, parent, left=None, right=None):
        if left is not None and right is not None:
            parent.threshold = None
            parent.feature = None
        if left:
            parent.left = None
            del tree.nodes[left.id]

        if right:
            parent.right = None
            del tree.nodes[right.id]


    def prune(self, max_depth):
        """Return a pruned copy of the tree."""
        from copy import deepcopy
        pruned_tree = deepcopy(self)
        nodes_list = pruned_tree.get_nodes_depth_list()

        n_layers_to_prune = self.max_depth - max_depth
        n_layers_to_prune = 0 if n_layers_to_prune < 0 else n_layers_to_prune

        for i in range(n_layers_to_prune):
            last_layer = nodes_list[-i-1]
            while last_layer:
                node1_id = last_layer[0]
                node1 = pruned_tree.nodes[node1_id]
                is_left = node1.is_left
                parent = pruned_tree.nodes[node1.parent]
                node2_id = parent.right if is_left else parent.left
                node2 = pruned_tree.nodes[node2_id] if node2_id else None            
                if is_left:
                    VizTree.prune_branch(pruned_tree, parent, node1, node2)
                else:
                    VizTree.prune_branch(pruned_tree, parent, node2, node1)
                last_layer.remove(node1_id)
                if node2:
                    last_layer.remove(node2_id)

        pruned_tree.prune_redundants(consider_proba=True)
        pruned_tree.update_max_depth()

        return pruned_tree

    def propagate_values(self, consider_proba=True):
        if self.is_classifier:
            self.propagate_class(consider_proba)
        else:
            self.propagate_regvalue()

    def propagate_class(self, consider_proba=True):
        nodes_by_depth = self.get_nodes_depth_list()
        
        for depth in range(self.max_depth, -1, -1):
            for node_id in nodes_by_depth[depth]:
                node = self.nodes[node_id]
                
                if node.left is None and node.right is None:
                    continue

                value_accumulator = np.zeros_like(node.value)
                
                if node.left is not None:
                    left_node = self.nodes[node.left]
                    if consider_proba:
                        value_accumulator += left_node.value * left_node.n_train
                    else:
                        value_accumulator[np.argmax(left_node.value)] += left_node.n_train
                
                if node.right is not None:
                    right_node = self.nodes[node.right]
                    if consider_proba:
                        value_accumulator += right_node.value * right_node.n_train
                    else:
                        value_accumulator[np.argmax(right_node.value)] += right_node.n_train

                if node.n_train > 0:
                    if consider_proba:
                        node.value = value_accumulator / node.n_train
                    else:
                        node.value = np.zeros_like(node.value)
                        node.value[np.argmax(value_accumulator)] = 1.0
                else:
                    node.value = value_accumulator

    def propagate_regvalue(self):
        nodes_by_depth = self.get_nodes_depth_list()

        for depth in range(self.max_depth, -1, -1):
            for node_id in nodes_by_depth[depth]:
                node = self.nodes[node_id]

                if node.left is None and node.right is None:
                    continue

                value_accumulator= 0.0

                if node.left is not None:
                    left_node = self.nodes[node.left]
                    value_accumulator += left_node.value * left_node.n_train

                if node.right is not None:
                    right_node = self.nodes[node.right]
                    value_accumulator += right_node.value * right_node.n_train

                node.value = value_accumulator / node.n_train
                self.possible_values.add(node.value)
        
        self._generate_color_struct()



    def prune_redundants(self, consider_proba=True, tolerance=1e-4):
        def is_homogeneous_subtree(node_id):
            stack = [node_id]
            leaf_values = []
            total_n_train = 0

            while stack:
                current_id = stack.pop()
                current_node = self.nodes[current_id]

                if current_node.left is None and current_node.right is None:
                    leaf_values.append((current_node.value, current_node.n_train))
                else:
                    if current_node.left is not None:
                        stack.append(current_node.left)
                    if current_node.right is not None:
                        stack.append(current_node.right)

            if self.is_classifier:
                first_label = np.argmax(leaf_values[0][0])
                homogeneous = all(np.argmax(value) == first_label for value, _ in leaf_values)
                combined_value = np.zeros_like(leaf_values[0][0])

                if homogeneous:
                    for value, n_train in leaf_values:
                        combined_value += value * n_train
                        total_n_train += n_train
                    if consider_proba and total_n_train > 0:
                        combined_value /= total_n_train
                    else:
                        combined_value = np.zeros_like(combined_value)
                        combined_value[first_label] = 1.0
            
            else:
                first_value = leaf_values[0][0]
                homogeneous = all(abs(value - first_value) < tolerance for value, _ in leaf_values)
                combined_value = 0.0

                if homogeneous:
                    for value, n_train in leaf_values:
                        combined_value += value * n_train
                        total_n_train += n_train
                    if total_n_train > 0:
                        combined_value /= total_n_train
                    else:
                        combined_value = first_value
            return homogeneous, combined_value, total_n_train


        def prune_node(node_id):
            stack = [node_id]

            while stack:
                current_id = stack.pop()
                current_node = self.nodes[current_id]

                if current_node.left is not None:
                    stack.append(current_node.left)
                if current_node.right is not None:
                    stack.append(current_node.right)

                is_homog, combined_value, total_n_train = is_homogeneous_subtree(current_id)
                if is_homog:
                    current_node.left = None
                    current_node.right = None
                    current_node.feature = None
                    current_node.threshold = None
                    current_node.value = combined_value
                    current_node.n_train = total_n_train

        nodes_by_depth = self.get_nodes_depth_list()

        # Traverse from the maximum depth to the root
        for depth in range(self.max_depth, -1, -1):
            for node_id in nodes_by_depth[depth]:
                prune_node(node_id)

        self.update_max_depth()
