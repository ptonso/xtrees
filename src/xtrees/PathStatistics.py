import numpy as np
import pandas as pd


class PathStatistics:
    def __init__(self, feature_names):
        self.feature_names = feature_names


    def extract_df(self, model):
        self.model = model
        branch_data = []

        # is forest
        if hasattr(model, 'estimators_'):
            rf = model
            for tree_id, estimator in enumerate(rf.estimators_):
                tree = estimator.tree_
                branches = self._extract_branches_from_tree(tree)
                for branch in branches:
                    branch['tree_id'] = tree_id
                    branch_data.append(branch)
        # is tree
        else:
            tree = model.tree_
            branches = self._extract_branches_from_tree(tree)
            for branch in branches:
                branch['tree_id'] = 0
                branch_data.append(branch)
        
        self.branch_df = pd.DataFrame(branch_data)
        return self.branch_df
    

    def _extract_branches_from_tree(self, tree):
        branches = []
        for node_id in range(tree.node_count):
            # if leaf node
            if tree.children_left[node_id] == -1 and tree.children_right[node_id] == -1:
                branch = self._build_branch(tree, node_id)
                branches.append(branch)
        return branches
    
    def _build_branch(self, tree, leaf_id):
        node_id = leaf_id
        conditions = {f'{feat}_upper': np.inf for feat in self.feature_names}
        conditions.update({f'{feat}_lower': -np.inf for feat in self.feature_names})

        # backtrack path to root
        while node_id != 0:
            parent_id = np.where(tree.children_left == node_id)[0] if node_id in tree.children_left else np.where(tree.children_right == node_id)[0]
            parent_id = parent_id[0] if len(parent_id) > 0 else None
            if parent_id is None:
                break

            feature = self.feature_names[tree.feature[parent_id]]
            threshold = tree.threshold[parent_id]

            if node_id == tree.children_left[parent_id]:
                conditions[f'{feature}_upper'] = min(conditions[f'{feature}_upper'], threshold)
            else:
                conditions[f'{feature}_lower'] = max(conditions[f'{feature}_lower'], threshold)

            node_id = parent_id

        branch_importance = tree.n_node_samples[leaf_id] / tree.n_node_samples[0]
        
        value = tree.value[leaf_id].mean(axis=0) if len(tree.value[leaf_id].shape) > 1 else tree.value[leaf_id][0]

        branch_dict = {**conditions, 'branch_importance': branch_importance, 'value': value}
        return branch_dict
    


if __name__ == "__main__":

    # example
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd

    X_train = pd.DataFrame(np.random.randn(100, 4), columns=["feature1", "feature2", "feature3", "feature4"])
    y_train = np.random.randint(0, 2, 100)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    fs = ForestStatistics(rf, X_train.columns)
    branch_df = fs.extract_df()

    print(branch_df)
