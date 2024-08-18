import numpy as np
import pandas as pd

from scipy.stats import entropy
from sklearn.metrics import roc_curve, auc

from statsmodels.distributions.empirical_distribution import ECDF

class Tree_:
    def __init__(self, node_count, n_outputs, is_classifier):
        self.node_count = node_count
        self.feature = np.full(node_count, -2, dtype=np.int32)         # default: -2 for leaves
        self.threshold = np.full(node_count, -2.0, dtype=np.float64)   # default: -2 for leaves
        self.children_left = np.full(node_count, -1, dtype=np.int32)   # default: -1 for leaves
        self.children_right = np.full(node_count, -1, dtype=np.int32)  # default: -1 for leaves
        self.n_node_samples = np.zeros(node_count, dtype=np.int32)
        if is_classifier:
            self.value = np.zeros((node_count, 1, n_outputs), dtype=np.float64)  # shape for multi-class classification
        else:
            self.value = np.zeros((node_count, n_outputs), dtype=np.float64)     # shape for regression

class ForestBasedTree:
    def __init__(self, minimal_forest_size=10, amount_of_branches_threshold=100, exclusion_threshold=0.8, random_state=None, verbose=False):
        self.minimal_forest_size = minimal_forest_size
        self.amount_of_branches_threshold = amount_of_branches_threshold
        self.exclusion_threshold = exclusion_threshold
        self.verbose = verbose
        self.conjunction_set = None
        self.root = None
        self.node_count = 0
        self.tree_ = None
        self.current_node_id = 0

        self.random_state = random_state
        if self.random_state is not None:
            np.random.seed(self.random_state)


    def fit(self, random_forest, X_train, y_train, 
            feature_types=None, feature_names=None, minimal_forest_size=10, 
            amount_of_branches_threshold=100, 
            exclusion_threshold=0.8, random_state=None):
        
        self.rf = random_forest
        self.is_classifier = hasattr(random_forest, "classes_")

        if self.is_classifier:
            self.classes_ = random_forest.classes_

        if isinstance(X_train, pd.DataFrame):
            if feature_names is None:
                feature_names = X_train.columns
            if feature_types is None:
                feature_types = X_train.dtypes
            
            X_train = X_train.to_numpy()
            y_train = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train

        self.feature_names_in_ = feature_names
        self.feature_types = feature_types
        self.X_train = X_train

        self.X_train = X_train
        self.y_train = y_train

        if not self.random_state:
            self.random_state = random_state
            if self.random_state is not None:
                np.random.seed(self.random_state)

        n_features = len(feature_names)
        for estimator in self.rf.estimators_:
            tree = estimator.tree_
            if tree.n_features != n_features:
                raise ValueError(f"Tree in random forest has {tree.n_features} features, but dataset has {n_features} features")


        n_outputs = len(self.rf.classes_) if self.is_classifier else 1
        
        self.conjunction_set = ConjunctionSet(
            feature_names, X_train, y_train,
            random_forest, feature_types, amount_of_branches_threshold, self.random_state,
            exclusion_threshold,minimal_forest_size, exclusion_threshold, self.verbose
            )

        self.cs_df = self.conjunction_set.get_conjunction_set_df().round(decimals=5)

        if self.is_classifier:
            for i in range(len(self.rf.classes_)):
                self.cs_df[self.rf.classes_[i]] = [probas[i] for probas in self.cs_df['probas']]

        self.root = Node([True] * len(self.cs_df), random_state=self.random_state)
        self.root.split(self.cs_df)
        self.node_count = self._count_nodes(self.root)
        self.tree_ = Tree_(self.node_count, n_outputs, self.is_classifier)
        
        if self.is_classifier:
            self.tree_.value = np.zeros((self.node_count, 1, len(self.rf.classes_)))
        else:
            self.tree_.value = np.zeros((self.node_count, 1, 1))
        self.current_node_id = 0
        self._populate_tree(self.root)

    def _count_nodes(self, node):
        if node is None:
            return 0
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    def _populate_tree(self, node):
        if node is None:
            return -1

        node_id = self.current_node_id
        self.current_node_id += 1

        left_child_id = self._populate_tree(node.left)
        right_child_id = self._populate_tree(node.right)

        if left_child_id == -1 and right_child_id == -1:
            # It's a leaf node
            if self.is_classifier:
            
                self.tree_.value[node_id, 0, :] = node.value if node.value is not None else np.zeros(len(self.rf.classes_))
            else:
                self.tree_.value[node_id, 0, 0] = node.value if node.value is not None else np.array([0.0])
        else:
            if self.is_classifier:
                left_value = self.tree_.value[left_child_id, 0, :] if left_child_id != -1 else np.zeros(len(self.rf.classes_))
                right_value = self.tree_.value[right_child_id, 0, :] if right_child_id != -1 else np.zeros(len(self.rf.classes_))
            else:
                left_value = self.tree_.value[left_child_id, 0, 0] if left_child_id != -1 else np.array([0.0])
                right_value = self.tree_.value[right_child_id, 0, 0] if right_child_id != -1 else np.array([0.0])

            left_samples = self.tree_.n_node_samples[left_child_id] if left_child_id != -1 else 0
            right_samples = self.tree_.n_node_samples[right_child_id] if right_child_id != -1 else 0
            total_samples = left_samples + right_samples

            if total_samples > 0:
                weighted_value = (left_value * left_samples + right_value * right_samples) / total_samples
            else:
                weighted_value = np.zeros(len(self.rf.classes_)) if self.is_classifier else np.array([0.0])

            if self.is_classifier:
                weighted_sum = weighted_value.sum()
                if weighted_sum > 0:
                    weighted_value /= weighted_sum
                else:
                    print(f"Warning: Weighted sum of probabilities is zero for node {node_id}")

            if self.is_classifier:
                self.tree_.value[node_id, 0, :] = weighted_value
            else:
                self.tree_.value[node_id, :] = weighted_value


        self.tree_.feature[node_id] = node.feature if node.feature is not None else -2
        self.tree_.threshold[node_id] = node.threshold if node.threshold is not None else -2
        self.tree_.n_node_samples[node_id] = np.sum(node.mask)

        self.tree_.children_left[node_id] = left_child_id
        self.tree_.children_right[node_id] = right_child_id

        return node_id

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        predictions = []
        for inst in X:
            prediction = self.root.predict_value(inst, self.cs_df)
            if self.is_classifier:
                predictions.append(np.argmax(prediction))
            else:
                predictions.append(prediction)
        return np.array(predictions).flatten()

    def predict_proba(self, X):
        if not self.is_classifier:
            raise ValueError("Probability prediction is only available for classifiers.")
        probabilities = []
        for inst in X:
            prediction = self.root.predict_value(inst, self.cs_df)
            probabilities.append(prediction)
        return np.array(probabilities)
    
    def get_params(self, deep=True):
        return {
            'minimal_forest_size': self.minimal_forest_size,
            'amount_of_branches_threshold': self.amount_of_branches_threshold,
            'exclusion_threshold': self.exclusion_threshold,
            'random_state': self.random_state,
            'verbose': self.verbose
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


EPSILON=0.001

def get_prob(i,features_upper,features_lower,ecdf):
    return ecdf[i]([features_lower[i], features_upper[i]])

class Branch:
    def __init__(self, feature_names, feature_types, output, number_of_samples=None):
        self.feature_types = feature_types
        self.number_of_features = len(feature_names)
        self.feature_names = feature_names
        self.features_upper = [np.inf]*self.number_of_features #upper bound of the feature for the given rule
        self.features_lower = [-np.inf]*self.number_of_features #lower bound of the feature for the given rule
        self.output = output
        self.number_of_samples=number_of_samples #save number of samples in leaf (not relevant for the current model)
        self.categorical_features_dict={}

    def addCondition(self, feature, threshold, bound):
        if feature >= len(self.features_upper):
            print(f"Error: Feature index {feature} is out of bounds for features_upper with length {len(self.features_upper)}")
            raise IndexError(f"Feature index {feature} out of range in addCondition")

        if bound == 'lower':
            if self.features_lower[feature] < threshold:
                self.features_lower[feature] = threshold
                if '=' in self.feature_names[feature] and threshold >= 0:
                    splitted = self.feature_names[feature].split('=')
                    self.categorical_features_dict[splitted[0]]=splitted[1]
        else:
            if self.features_upper[feature] > threshold:
                self.features_upper[feature] = threshold
    
    def contradictBranch(self, other_branch):
        for categorical_feature in self.categorical_features_dict:
            if (categorical_feature in other_branch.categorical_features_dict and 
                self.categorical_features_dict[categorical_feature] != other_branch.categorical_features_dict[categorical_feature]):
                return True
        for i in range(self.number_of_features):
            if (self.features_upper[i] <= other_branch.features_lower[i] + EPSILON or 
                self.features_lower[i] + EPSILON >= other_branch.features_upper[i]):
                return True
            if (self.feature_types.iloc[i] == 'int' and 
                min(self.features_upper[i], other_branch.features_upper[i]) % 1 > 0 and 
                min(self.features_upper[i], other_branch.features_upper[i]) - max(self.features_lower[i], other_branch.features_lower[i]) < 1):
                return True
        return False

    def mergeBranch(self, other_branch, is_classifier):
        if is_classifier:
            new_output = [k + v for k, v in zip(self.output, other_branch.output)]
            new_output = [o / sum(new_output) for o in new_output]
        else:
            total_samples = self.number_of_samples + other_branch.number_of_samples
            self_output = np.array(self.output)
            other_output = np.array(other_branch.output)
            new_output = (self_output * self.number_of_samples + other_output * other_branch.number_of_samples) / total_samples

        new_number_of_samples=np.sqrt(self.number_of_samples * other_branch.number_of_samples)
        new_b = Branch(self.feature_names, self.feature_types, new_output, new_number_of_samples)
        new_b.features_upper, new_b.features_lower = list(self.features_upper), list(self.features_lower)
        for feature in range(self.number_of_features):
            new_b.addCondition(feature, other_branch.features_upper[feature], 'upper')
            new_b.addCondition(feature, other_branch.features_lower[feature], 'lower')
        new_b.categorical_features_dict = dict(self.categorical_features_dict)
        new_b.categorical_features_dict.update(dict(other_branch.categorical_features_dict))
        new_b.leaves_indexes = self.leaves_indexes + other_branch.leaves_indexes
        return new_b
    
    def printBranch(self):
        """
        This function creates a string representation of the branch (only for demonstration purposes)
        """
        s = ""
        for feature, threshold in enumerate(self.features_lower):
            if threshold != (-np.inf):
                s += str(feature) + ' > ' + str(np.round(threshold, 3)) + ", "
        for feature, threshold in enumerate(self.features_upper):
            if threshold != np.inf:
                s += str(feature) + ' <= ' + str(np.round(threshold, 3)) + ", "
        s += 'output: [' + ', '.join(map(str, self.output)) + ']'
        s += ' Number of samples: '+str(self.number_of_samples)
        print(s)

    def containsInstance(self, instance):
        np.sum(self.features_upper >= instance) == len(instance) and np.sum(self.features_lower < instance) == len(instance)
        
    def getLabel(self):
        return np.argmax(self.output)
    
    def get_branch_dict(self,ecdf):
        features={}
        for feature, upper_value, lower_value in zip(range(len(self.features_upper)), self.features_upper, self.features_lower):
            features[str(feature)+'_upper'] = upper_value
            features[str(feature)+'_lower'] = lower_value
        features['number_of_samples'] = self.number_of_samples
        features['branch_probability'] = self.calculate_branch_probability_by_ecdf(ecdf)
        features['output']=np.array(self.output)
        return  features

    def calculate_branch_probability_by_ecdf(self, ecdf):
        features_probabilities = []
        delta = 0.000000001
        for i in range(len(ecdf)):
            probs = ecdf[i]([self.features_lower[i], self.features_upper[i]])
            features_probabilities.append((probs[1] - probs[0] + delta))
        return np.prod(features_probabilities)
        
    def is_excludable_branch(self, threshold):
        
        # Think for regression!
        return max(self.output) / np.sum(self.output) > threshold
        
    
class ConjunctionSet():
    def __init__(self, feature_names, X_train, y_train,
                 model, feature_types, amount_of_branches_threshold, random_state=None,
                 exclusion_starting_point=5,
                 minimal_forest_size=10, exclusion_threshold=0.8, verbose=True):
        self.amount_of_branches_threshold = amount_of_branches_threshold
        self.model = model
        self.verbose = verbose
        self.feature_names = feature_names
        self.exclusion_threshold=exclusion_threshold
        self.is_classifier = hasattr(model, 'classes_')
        self.label_names = model.classes_ if self.is_classifier else None
        self.relevant_indexes = reduce_error_pruning_cached(self.model, X_train, y_train, minimal_forest_size)        
        self.feature_types = feature_types
        self.exclusion_starting_point = exclusion_starting_point
        self.set_ecdf(X_train)
        self.random_state = random_state
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.generateBranches()
        self.number_of_branches_per_iteration = []
        self.buildConjunctionSet()

    def generateBranches(self):
        trees = [estimator.tree_ for estimator in self.model.estimators_]
        self.branches_lists = [self.get_tree_branches(tree_) for i, tree_ in enumerate(trees) if i in self.relevant_indexes]
        for list_indx, branch_list in enumerate(self.branches_lists):
            for leaf_index, branch in enumerate(branch_list):
                branch.leaves_indexes = [str(list_indx) + '_' + str(leaf_index)]

    def get_tree_branches(self, tree_):
        leaf_indexes = [i for i in range(tree_.node_count) if tree_.children_left[i] == -1 and tree_.children_right[i] == -1]
        branches = [self.get_branch_from_leaf_index(tree_,leaf_index) for leaf_index in leaf_indexes]
        return branches
    
    def get_branch_from_leaf_index(self, tree_, leaf_index):
        if self.is_classifier:
            sum_of_probas=np.sum(tree_.value[leaf_index][0])
            output = [i / sum_of_probas for i in tree_.value[leaf_index][0]]
        else:
            output = [tree_.value[leaf_index][0][0]]

        new_branch=Branch(self.feature_names, self.feature_types, output,
                          number_of_samples=tree_.n_node_samples[leaf_index])

        node_id=leaf_index
        while node_id: #iterate over all nodes in branch
            ancestor_index = np.where(tree_.children_left==node_id)[0] #assuming left is the default for efficiency purposes
            bound = 'upper'
            if len(ancestor_index) == 0:
                bound = 'lower'
                ancestor_index = np.where(tree_.children_right == node_id)[0]
            new_branch.addCondition(tree_.feature[ancestor_index[0]], tree_.threshold[ancestor_index[0]], bound)
            node_id=ancestor_index[0]
        return new_branch
    
    def buildConjunctionSet(self):
        conjunctionSet = self.branches_lists[0]
        excluded_branches = []
        for i, branch_list in enumerate(self.branches_lists[1:]):
            if self.verbose:
                print('Iteration ' + str(i+1) + ": " + str(len(conjunctionSet)) + " conjunctions")
            filter = False if i == len(self.branches_lists[1:]) else True
            conjunctionSet = self.merge_branch_with_conjunctionSet(branch_list, conjunctionSet, filter=filter)
            if i >= self.exclusion_starting_point and len(conjunctionSet) > 0.8 * self.amount_of_branches_threshold:
                conjunctionSet, this_iteration_exclusions = self.exclude_branches_from_cs(conjunctionSet, self.exclusion_threshold)
                excluded_branches.extend(this_iteration_exclusions)

        self.conjunctionSet = excluded_branches + conjunctionSet
        if self.verbose:
            print('Final CS size: ' + str(len(self.conjunctionSet)))

    def exclude_branches_from_cs(self, cs, threshold):
        filtered_cs = []
        excludable_brances = []
        for branch in cs:
            if branch.is_excludable_branch(threshold):
                excludable_brances.append(branch)
            else:
                filtered_cs.append(branch)
        return filtered_cs, excludable_brances

    def filter_conjunction_set(self, cs):
        if len(cs) <= self.amount_of_branches_threshold:
            return cs
        # filter by branch probability
        branches_metrics=[b.calculate_branch_probability_by_ecdf(self.ecdf_dict) for b in cs]
        threshold = sorted(branches_metrics,reverse=True)[self.amount_of_branches_threshold-1]
        return [b for b, metric in zip(cs, branches_metrics) if metric >= threshold][:self.amount_of_branches_threshold]

    def merge_branch_with_conjunctionSet(self, branch_list, conjunctionSet, filter=True):
        new_conjunction_set = []
        for b1 in conjunctionSet:
            new_conjunction_set.extend([b1.mergeBranch(b2, self.is_classifier) for b2 in branch_list if not b1.contradictBranch(b2)])
        if filter:
            new_conjunction_set = self.filter_conjunction_set(new_conjunction_set)
        self.number_of_branches_per_iteration.append(len(new_conjunction_set))
        return new_conjunction_set
    
    def get_conjunction_set_df(self):            
        df = pd.DataFrame([b.get_branch_dict(self.ecdf_dict) for b in self.conjunctionSet])
        if self.is_classifier:
            return df.rename(columns={"output": "probas"})
        else:
            df = df.rename(columns={"output": "regressions"})
            df['regressions'] = df['regressions'].apply(lambda x: float(x[0]) if isinstance(x, np.ndarray) and x.shape == (1,) else float(x))
            return df

    def predict(self,X):
        predictions=[]
        for inst in X:
            for conjunction in self.conjunctionSet:
                if conjunction.containsInstance(inst):
                   predictions.append(self.label_names[conjunction.getLabel()])
        return predictions
    
    def set_ecdf(self,data):
        self.ecdf_dict={i: ECDF(data[:, i]) for i in range(len(self.feature_names))}
    
    def group_by_label_probas(self, conjunctionSet):
        probas_hashes={}
        for i, b in enumerate(conjunctionSet):
            probas_hash = hash(tuple(b.label_probas))
            if probas_hash not in probas_hashes:
                probas_hashes[probas_hash] = []
            probas_hashes[probas_hash].append(i)
        return probas_hashes
    



def predict_instance_with_included_tree(model, included_indexes, inst):
    if hasattr(model, 'n_classes_'):    
        v = np.array([0] * model.n_classes_)
        for i, t in enumerate(model.estimators_):
            if i in included_indexes:
                v = v + t.predict_proba(inst.reshape(1, -1))[0]
        return v / np.sum(v)
    else:
        v = np.array([0.0])
        for i, t in enumerate(model.estimators_):
            if i in included_indexes:
                v = v + t.predict(inst.reshape(1, -1))[0]
        return v / len(included_indexes)


def get_auc(Y, y_score, classes):
    y_test_binarize = np.array([[1 if i == c else 0 for c in classes] for i in Y])
    fpr, tpr, _ = roc_curve(y_test_binarize.ravel(), y_score.ravel())
    return auc(fpr, tpr)


def select_index(rf, current_indexes, X_train, y_train):
    options_metric = {}
    for i in range(len(rf.estimators_)):
        if i in current_indexes:
            continue
        predictions = predict_with_included_trees(rf, current_indexes + [i], X_train)
        if hasattr(rf, 'classes_'):    
            options_metric[i] = get_auc(y_train, predictions, rf.classes_)
        else:
            options_metric[i] = -np.mean(np.abs(y_train - predictions))
    if not options_metric:
        raise ValueError("options_metric is empty. No valid indexes found for selection.")

    if hasattr(rf, 'classes_'):
        best_index = max(options_metric, key=options_metric.get)
    else:
        best_index = min(options_metric, key=options_metric.get)

    best_metric = options_metric[best_index]
    return best_metric, current_indexes + [best_index]

def reduce_error_pruning(model, X_train, y_train, min_size):
    best_metric, current_indexes = select_index(model, [], X_train, y_train)
    while len(current_indexes) <= model.n_estimators:
        if len(current_indexes) == len(model.estimators_):
            break
        new_metric, new_current_indexes = select_index(model, current_indexes, X_train, y_train)
        if hasattr(model, 'classes_'):        
            if new_metric <= best_metric and len(new_current_indexes) > min_size:
                break
        else:
            if new_metric >= best_metric and len(new_current_indexes) > min_size:
                break
        
        best_metric, current_indexes = new_metric, new_current_indexes
    return current_indexes

def predict_with_included_trees(model, included_indexes, X):
    predictions = []
    for inst in X:
        predictions.append(predict_instance_with_included_tree(model, included_indexes, inst))
    return np.array(predictions)

def reduce_error_pruning_cached(model, X_train, y_train, min_size):
    cache = {}
    best_metric, current_indexes = select_index_cached(model, [], X_train, y_train, cache)
    while len(current_indexes) <= model.n_estimators:
        if len(current_indexes) == len(model.estimators_):
            break
        new_metric, new_current_indexes = select_index_cached(model, current_indexes, X_train, y_train, cache)
        if hasattr(model, 'classes_'):        
            if new_metric <= best_metric and len(new_current_indexes) > min_size:
                break
        else:
            if new_metric >= best_metric and len(new_current_indexes) > min_size:
                break
        
        best_metric, current_indexes = new_metric, new_current_indexes
    return current_indexes

def select_index_cached(rf, current_indexes, X_train, y_train, cache):
    options_metric = {}
    for i in range(len(rf.estimators_)):
        if i in current_indexes:
            continue
        predictions = predict_with_included_trees_cached(rf, current_indexes + [i], X_train, cache)
        if hasattr(rf, 'classes_'):
            options_metric[i] = get_auc(y_train, predictions, rf.classes_)
        else:
            options_metric[i] = -np.mean(np.abs(y_train - predictions))
    
    if not options_metric:
        raise ValueError("options_metric is empty. No valid indexes found for selection.")

    if hasattr(rf, 'classes_'):
        best_index = max(options_metric, key=options_metric.get)
    else:
        best_index = min(options_metric, key=options_metric.get)

    best_metric = options_metric[best_index]
    return best_metric, current_indexes + [best_index]

def predict_with_included_trees_cached(model, included_indexes, X, cache):
    is_classifier = hasattr(model, 'n_classes_')
    if is_classifier:
        predictions = np.zeros((X.shape[0], model.n_classes_))
    else:
        predictions = np.zeros(X.shape[0])
    count = 0
    for i in included_indexes:
        if i not in cache:
            if is_classifier:
                cache[i] = np.array([model.estimators_[i].predict_proba(x.reshape(1, -1))[0] for x in X])
            else:
                cache[i] = np.array([model.estimators_[i].predict(x.reshape(1, -1))[0] for x in X])
        predictions += cache[i]
        count += 1
    predictions /= count
    return predictions


EPSILON=0.000001

class Node():
    def __init__(self, mask, random_state):
        self.mask = mask
        self.feature = None
        self.threshold = None
        self.value = None

        self.random_state = random_state
        if self.random_state is not None and isinstance(self.random_state, np.random.RandomState):
            self.random_state = self.random_state
        elif self.random_state is not None:
            self.random_state = np.random.RandomState(self.random_state)
        else:
            self.random_state = np.random
        
    def split(self, df):
        if np.sum(self.mask)==1 or self.has_same_class(df):
            self.left=None
            self.right=None
            self.value = self.node_value(df)
            return
        self.features = [int(i.split('_')[0]) for i in df.keys() if 'upper' in str(i)]

        self.split_feature, self.split_value = self.select_split_feature(df)
        self.create_mask(df)
        is_splitable=self.is_splitable()
        if not is_splitable:
            self.left = None
            self.right = None
            self.value = self.node_value(df)
            return
        self.feature = self.split_feature
        self.threshold = self.split_value
        self.left=Node(list(np.logical_and(self.mask,np.logical_or(self.left_mask,self.both_mask))), self.random_state)
        self.right = Node(list(np.logical_and(self.mask,np.logical_or(self.right_mask,self.both_mask))), self.random_state)
        self.left.split(df)
        self.right.split(df)

    def node_value(self, df):
        if 'probas' in df.columns:
            return df['probas'][self.mask].mean(axis=0)
        return np.array([[[df['regressions'][self.mask].mean()]]])

    def has_same_class(self, df):
        if 'probas' in df.columns:
            labels = set([np.argmax(l) for l in df['probas'][self.mask]])
        else:
            labels = set(df['regressions'][self.mask])
        return len(labels) == 1


    def is_splitable(self):
        if np.sum(np.logical_and(self.mask, np.logical_or(self.left_mask, self.both_mask))) == 0 or np.sum(
            np.logical_and(self.mask, np.logical_or(self.right_mask, self.both_mask))) == 0:
            return False
        if np.sum(np.logical_and(self.mask, np.logical_or(self.left_mask, self.both_mask))) == np.sum(self.mask) or np.sum(
            np.logical_and(self.mask, np.logical_or(self.right_mask, self.both_mask))) == np.sum(self.mask):
            return False
        return True

    def create_mask(self, df):
        self.left_mask = df[str(self.split_feature) + "_upper"] <= self.split_value
        self.right_mask = df[str(self.split_feature) + '_lower'] >= self.split_value
        self.both_mask = ((df[str(self.split_feature) + '_lower'] < self.split_value) & 
                          (df[str(self.split_feature) + "_upper"] > self.split_value))
        

    def select_split_feature(self, df):
        feature_to_value = {}
        feature_to_metric = {}
        for feature in self.features:
           value, metric = self.check_feature_split_value(df,feature)
           feature_to_value[feature] = value
           feature_to_metric[feature] = metric
        feature = min(feature_to_metric, key=feature_to_metric.get)
        return feature, feature_to_value[feature]

    def check_feature_split_value(self, df, feature):
        value_to_metric = {}
        values = list(set(list(df[str(feature)+'_upper'][self.mask]) + list(df[str(feature)+'_lower'][self.mask])))
        self.random_state.shuffle(values)
        values = values[:3]
        for value in values:
            left_mask = [True if upper <= value  else False for upper in df[str(feature) + "_upper"]]
            right_mask = [True if lower >= value else False for lower in df[str(feature) + '_lower']]
            both_mask = [True if value < upper and value > lower else False for lower, upper in zip(df[str(feature) + '_lower'],df[str(feature) + "_upper"])]
            value_to_metric[value] = self.get_value_metric(df, left_mask, right_mask, both_mask)
        val = min(value_to_metric, key = value_to_metric.get)
        return val, value_to_metric[val]

    def get_value_metric(self, df, left_mask, right_mask, both_mask):
        l_df_mask = np.logical_and(np.logical_or(left_mask, both_mask), self.mask)
        r_df_mask = np.logical_and(np.logical_or(right_mask, both_mask), self.mask)
        if np.sum(l_df_mask) == 0 or np.sum(r_df_mask) == 0:
            return np.inf

        l_prop = np.sum(l_df_mask) / len(l_df_mask)
        r_prop = np.sum(r_df_mask) / len(r_df_mask)

        if 'probas' in df.columns:
            l_entropy, r_entropy = self.calculate_entropy(df, l_df_mask), self.calculate_entropy(df, r_df_mask)
            return l_entropy * l_prop + r_entropy * r_prop
        else:
            l_variance, r_variance = self.calculate_variance(df, l_df_mask), self.calculate_variance(df, r_df_mask)
            return l_variance * l_prop + r_variance * r_prop

    def calculate_entropy(self, df, mask):
        x = df['probas'][mask].mean()
        return entropy(x/x.sum()) if len(x.shape) > 0 and x.sum() > 0 else 0

    def calculate_variance(self, df, mask):
        return np.var(df['regressions'][mask])


    def predict_value(self, inst, df):
        if self.left is None and self.right is None:
            return self.node_value(df)
        if inst[self.split_feature] <= self.split_value:
            return self.left.predict_value(inst, df)
        else:
            return self.right.predict_value(inst, df)
        
    def get_node_prediction(self, df):
        if 'probas' in df.columns:
            v = df['probas'][self.mask][0]
            v = [i / np.sum(v) for i in v]
            return np.array(v) / np.sum(v) if len(v.shape) > 0 and np.sum(v) > 0 else np.array(v)
        else:
            return df['regressions'][self.mask].mean()
    
    
    def count_depth(self):
        if self.right == None:
            return 1
        return max(self.left.count_depth(), self.right.count_depth()) + 1
    
    def number_of_children(self):
        if self.right == None:
            return 1
        return 1 + self.right.number_of_children() + self.left.number_of_children()




