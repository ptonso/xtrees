import numpy as np
from operator import mul
from operator import mul
from functools import reduce

EPSILON=0.001
def get_prob(i,features_upper,features_lower,ecdf):
    return ecdf[i]([features_lower[i], features_upper[i]])

class Branch:
    def __init__(self,feature_names,feature_types,label_names,label_probas=None,number_of_samples=None):
        """Branch inatance can be initialized in 2 ways. One option is to initialize an empty branch
        (only with a global number of features and number of class labels) and gradually add
        conditions - this option is relevant for the merge implementation.
        Second option is to get the number of samples in branch and the labels
        probability vector - relevant for creating a branch out of an existing tree leaf.
        """
        self.feature_types=feature_types
        self.label_names=label_names
        self.number_of_features=len(feature_names)
        self.feature_names=feature_names
        self.features_upper=[np.inf]*self.number_of_features #upper bound of the feature for the given rule
        self.features_lower=[-np.inf]*self.number_of_features #lower bound of the feature for the given rule
        self.label_probas=label_probas
        self.number_of_samples=number_of_samples #save number of samples in leaf (not relevant for the current model)
        self.categorical_features_dict={}
    def addCondition(self, feature, threshold, bound):
        """
        This function gets feature index, its threshold for the condition and whether
        it is upper or lower bound. It updates the features thresholds for the given rule.
        """
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
        """
        check wether Branch b can be merged with the "self" Branch. Returns Boolean answer.
        """
        for categorical_feature in self.categorical_features_dict:
            if categorical_feature in other_branch.categorical_features_dict and self.categorical_features_dict[categorical_feature] != other_branch.categorical_features_dict[categorical_feature]:
                return True
        for i in range(self.number_of_features):
            if self.features_upper[i] <= other_branch.features_lower[i] + EPSILON or self.features_lower[i] + EPSILON >= other_branch.features_upper[i]:
                return True
            if self.feature_types[i]=='int' and min(self.features_upper[i],other_branch.features_upper[i])%1>0 and \
                                    min(self.features_upper[i],other_branch.features_upper[i])-max(self.features_lower[i],other_branch.features_lower[i])<1:
                return True

        return False
    def mergeBranch(self, other_branch):
        """
        This method gets Branch b and create a new branch which is a merge of the "self" object
        with b. As describe in the algorithm.
        """
        new_label_probas=[k+v for k,v in zip(self.label_probas,other_branch.label_probas)]
        new_number_of_samples=np.sqrt(self.number_of_samples * other_branch.number_of_samples)
        new_b = Branch(self.feature_names,self.feature_types,self.label_names,new_label_probas,new_number_of_samples)
        new_b.features_upper, new_b.features_lower = list(self.features_upper), list(self.features_lower)
        for feature in range(self.number_of_features):
            new_b.addCondition(feature, other_branch.features_upper[feature], 'upper')
            new_b.addCondition(feature, other_branch.features_lower[feature], 'lower')
        new_b.categorical_features_dict = dict(self.categorical_features_dict)
        new_b.categorical_features_dict.update(dict(other_branch.categorical_features_dict))
        new_b.leaves_indexes = self.leaves_indexes + other_branch.leaves_indexes
        return new_b
    def toString(self):
        """
        This function creates a string representation of the branch (only for demonstration purposes)
        """
        s = ""
        for feature, threshold in enumerate(self.features_lower):
            if threshold != (-np.inf):
                #s +=  self.feature_names[feature] + ' > ' + str(np.round(threshold,3)) + ", "
                s += str(feature) + ' > ' + str(np.round(threshold, 3)) + ", "
        for feature, threshold in enumerate(self.features_upper):
            if threshold != np.inf:
                #s +=  self.feature_names[feature] + ' <= ' + str(np.round(threshold,3)) + ", "
                s += str(feature) + ' <= ' + str(np.round(threshold, 3)) + ", "
        s += 'labels: ['
        for k in range(len(self.label_probas)):
            s+=str(self.label_names[k])+' : '+str(self.label_probas[k])+' '
        s+=']'
        s+=' Number of samples: '+str(self.number_of_samples)
        return s
    def printBranch(self):
        # print the branch by using tostring()
        print(self.toString())
    def containsInstance(self, instance):
        """This function gets an ibservation as an input. It returns True if the set of rules
        that represented by the branch matches the instance and false otherwise.
        """
        if np.sum(self.features_upper >= instance)==len(instance) and np.sum(self.features_lower < instance)==len(instance):
            return True
        return False
    def getLabel(self):
        # Return the predicted label accordint to the branch
        return np.argmax(self.label_probas)
    def containsInstance(self, v):
        for i,lower,upper in zip(range(len(v)),self.features_lower,self.features_upper):
            if v[i]>upper or v[i]<=lower:
                return False
        return True
    def get_branch_dict(self,ecdf):
        features={}
        for feature,upper_value,lower_value in zip(range(len(self.features_upper)),self.features_upper,self.features_lower):
            features[str(feature)+'_upper']=upper_value
            features[str(feature)+'_lower']=lower_value
        features['number_of_samples']=self.number_of_samples
        features['branch_probability'] = self.calculate_branch_probability_by_ecdf(ecdf)
        features['probas']=np.array(self.label_probas)
        return  features

    def calculate_branch_probability_by_ecdf(self, ecdf):
        features_probabilities=[]
        delta = 0.000000001
        for i in range(len(ecdf)):
            probs=ecdf[i]([self.features_lower[i],self.features_upper[i]])
            features_probabilities.append((probs[1]-probs[0]+delta))
        return np.prod(features_probabilities)
    def calculate_branch_probability_by_range(self, ranges):
        features_probabilities = 1
        for range, lower, upper in zip(ranges, self.features_lower, self.features_upper):
            probs = min(1,(upper-lower)/range)
        features_probabilities = features_probabilities*probs
        return features_probabilities
    def is_excludable_branch(self,threshold):
        if max(self.label_probas)/np.sum(self.label_probas)>threshold:
            return True
        return False
    def is_addable(self,other):
        for feature in range(self.number_of_features):
            if self.features_upper[feature] + EPSILON < other.features_lower[feature] or other.features_upper[feature] + EPSILON < self.features_lower[feature]:
                return False
        return True
    def is_valid_association(self,associative_leaves):
        for leaf1 in self.leaves_indexes:
            for leaf2 in self.leaves_indexes:
                if leaf1 == leaf2:
                    continue
                if associative_leaves[leaf1+'|'+leaf2]==0:
                    return False
        return True
    def number_of_unseen_pairs(self,associative_leaves):
        count=0
        for leaf1 in self.leaves_indexes:
            for leaf2 in self.leaves_indexes:
                if leaf1 == leaf2:
                    continue
                if associative_leaves[leaf1+'|'+leaf2]==0:
                    count+=1
        return count*(-1)


import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import entropy
import random

class ConjunctionSet():
    def __init__(self,feature_names,original_data, pruning_x, pruning_y,
                 model, feature_types, amount_of_branches_threshold,filter_approach='probability', exclusion_starting_point=5,
                 minimal_forest_size=10,exclusion_threshold=0.8):
        self.amount_of_branches_threshold = amount_of_branches_threshold
        self.model = model
        self.feature_names = feature_names
        self.exclusion_threshold=exclusion_threshold
        self.label_names = model.classes_
        self.relevant_indexes = reduce_error_pruning(self.model,pruning_x,pruning_y,minimal_forest_size)
        self.feature_types = feature_types
        self.filter_approach = filter_approach
        self.exclusion_starting_point = exclusion_starting_point
        self.set_ecdf(original_data)
        self.get_ranges(original_data)
        self.generateBranches()
        #self.get_associative_leaves(np.concatenate((original_data,pruning_x)))
        self.number_of_branches_per_iteration = []
        self.buildConjunctionSet()

    def generateBranches(self):
        trees=[estimator.tree_ for estimator in self.model.estimators_]
        self.branches_lists=[self.get_tree_branches(tree_,i) for i,tree_ in enumerate(trees) if i in self.relevant_indexes]
        for list_indx,branch_list in enumerate(self.branches_lists):
            for leaf_index,branch in enumerate(branch_list):
                branch.leaves_indexes=[str(list_indx)+'_'+str(leaf_index)]
    def get_tree_branches(self,tree_,tree_index):
        leaf_indexes = [i for i in range(tree_.node_count) if tree_.children_left[i] == -1 and tree_.children_right[i] == -1]
        branches=[self.get_branch_from_leaf_index(tree_,leaf_index) for leaf_index in leaf_indexes]
        return branches
    def get_branch_from_leaf_index(self,tree_,leaf_index):
        sum_of_probas=np.sum(tree_.value[leaf_index][0])
        label_probas=[i/sum_of_probas for i in tree_.value[leaf_index][0]]
        new_branch=Branch(self.feature_names,self.feature_types,self.label_names,label_probas=label_probas,
                          number_of_samples=tree_.n_node_samples[leaf_index])#initialize branch
        node_id=leaf_index
        while node_id: #iterate over all nodes in branch
            ancesor_index=np.where(tree_.children_left==node_id)[0] #assuming left is the default for efficiency purposes
            bound='upper'
            if len(ancesor_index)==0:
                bound='lower'
                ancesor_index = np.where(tree_.children_right == node_id)[0]
            new_branch.addCondition(tree_.feature[ancesor_index[0]], tree_.threshold[ancesor_index[0]], bound)
            node_id=ancesor_index[0]
        return new_branch
    def buildConjunctionSet(self):
        conjunctionSet=self.branches_lists[0]
        excluded_branches=[]
        for i,branch_list in enumerate(self.branches_lists[1:]):
            print('Iteration '+str(i+1)+": "+str(len(conjunctionSet))+" conjunctions")
            filter = False if i==len(self.branches_lists[1:]) else True
            conjunctionSet=self.merge_branch_with_conjunctionSet(branch_list,conjunctionSet,filter=filter)
            #print('i='+str(i))
            if i >= self.exclusion_starting_point and len(conjunctionSet)>0.8*self.amount_of_branches_threshold:
                conjunctionSet,this_iteration_exclusions=self.exclude_branches_from_cs(conjunctionSet,self.exclusion_threshold)
                excluded_branches.extend(this_iteration_exclusions)

        self.conjunctionSet=excluded_branches+conjunctionSet
        print('Final CS size: '+str(len(self.conjunctionSet)))
    def exclude_branches_from_cs(self,cs,threshold):
        filtered_cs=[]
        excludable_brancehs=[]
        for branch in cs:
            if branch.is_excludable_branch(threshold):
                excludable_brancehs.append(branch)
            else:
                filtered_cs.append(branch)
        return filtered_cs,excludable_brancehs
    def filter_conjunction_set(self,cs):
        if len(cs) <= self.amount_of_branches_threshold:
            return cs
        if self.filter_approach=='association':
            cs = [b for b in cs if b.is_valid_association(self.association_leaves)]
            return cs
        if self.filter_approach=='probability':
            branches_metrics=[b.calculate_branch_probability_by_ecdf(self.ecdf_dict) for b in cs]
        elif self.filter_approach=='number_of_samples':
            branches_metrics = [b.number_of_samples for b in cs]
        elif self.filter_approach=='probability_entropy':
            branches_metrics = [b.calculate_branch_probability_by_ecdf(self.ecdf_dict)*(1-entropy(b.label_probas)) for b in cs]
        elif self.filter_approach=='entropy':
            branches_metrics = [-entropy(b.label_probas) for b in cs]
        elif self.filter_approach=='range':
            branches_metrics = [b.calculate_branch_probability_by_range(self.ranges) for b in cs]
        elif self.filter_approach=='association_probability':
            branches_metrics = [b.is_valid_association(self.association_leaves)*b.calculate_branch_probability_by_ecdf(self.ecdf_dict) for b in cs]
        threshold=sorted(branches_metrics,reverse=True)[self.amount_of_branches_threshold-1]
        return [b for b,metric in zip(cs,branches_metrics) if metric >= threshold][:self.amount_of_branches_threshold]

    def merge_branch_with_conjunctionSet(self,branch_list,conjunctionSet,filter=True):
        new_conjunction_set=[]
        for b1 in conjunctionSet:
            new_conjunction_set.extend([b1.mergeBranch(b2) for b2 in branch_list if b1.contradictBranch(b2)==False])
        #print('number of branches before filterring: '+str(len(new_conjunction_set)))
        if filter:
            new_conjunction_set=self.filter_conjunction_set(new_conjunction_set)
        #print('number of branches after filterring: ' + str(len(new_conjunction_set)))
        self.number_of_branches_per_iteration.append(len(new_conjunction_set))
        return new_conjunction_set
    def get_conjunction_set_df(self):
        return pd.DataFrame([b.get_branch_dict(self.ecdf_dict) for b in self.conjunctionSet])
    def predict(self,X):
        predictions=[]
        for inst in X:
            for conjunction in self.conjunctionSet:
                if conjunction.containsInstance(inst):
                   predictions.append(self.label_names[conjunction.getLabel()])
        return predictions
    def get_instance_branch(self,inst):
        for conjunction in self.conjunctionSet:
            if conjunction.containsInstance(inst):
                return conjunction
    def set_ecdf(self,data):
        self.ecdf_dict={i:ECDF(data[:, i])for i in range(len(self.feature_names))}
    def group_by_label_probas(self,conjunctionSet):
        probas_hashes={}
        for i,b in enumerate(conjunctionSet):
            probas_hash = hash(tuple(b.label_probas))
            if probas_hash not in probas_hashes:
                probas_hashes[probas_hash]=[]
            probas_hashes[probas_hash].append(i)
        return probas_hashes
    def get_ranges(self,original_data):
        self.ranges = [max(v)-min(v) for v in original_data.transpose()]
    def get_associative_leaves(self,X):
        association_dict ={}
        for indx,inst in enumerate(X):
            association_dict[indx]=[]
            for tree_indx,tree_ in enumerate(self.branches_lists):
                for leaf_indx,leaf in enumerate(tree_):
                    if leaf.containsInstance(inst):
                        association_dict[indx].append(str(tree_indx)+'_'+str(leaf_indx))
        association_dict
        self.association_leaves={}
        for tree_indx1,tree_1 in enumerate(self.branches_lists):
            for tree_indx2,tree_2 in enumerate(self.branches_lists):
                if tree_indx1==tree_indx2:
                    continue
                for leaf_index1,leaf1 in enumerate(tree_1):
                    for leaf_index2,leaf2 in enumerate(tree_2):
                        self.association_leaves[str(tree_indx1)+'_'+str(leaf_index1)+'|'+str(tree_indx2)+'_'+str(leaf_index2)]=0
        for inst in association_dict:
            for leaf1 in association_dict[inst]:
                for leaf2 in association_dict[inst]:
                    if leaf1 == leaf2:
                        continue
                    else:
                        self.association_leaves[leaf1+'|'+leaf2]+=1



from scipy.stats import entropy
EPSILON=0.000001
class Node():
    def __init__(self,mask):
        self.mask = mask
    def split(self,df):
        #if np.sum(self.mask)==1 or self.has_same_class(df):
        if np.sum(self.mask) == 1:
            self.left=None
            self.right=None
            return
        self.features = [int(i.split('_')[0]) for i in df.keys() if 'upper' in str(i)]

        self.split_feature, self.split_value = self.select_split_feature(df)
        self.create_mask(df)
        is_splitable=self.is_splitable()
        if is_splitable==False:
            self.left = None
            self.right = None
            return
        self.left=Node(list(np.logical_and(self.mask,np.logical_or(self.left_mask,self.both_mask))))
        self.right = Node(list(np.logical_and(self.mask,np.logical_or(self.right_mask,self.both_mask))))
        self.left.split(df)
        self.right.split(df)

    def is_splitable(self):
        if np.sum(np.logical_and(self.mask, np.logical_or(self.left_mask, self.both_mask))) == 0 or np.sum(np.logical_and(self.mask, np.logical_or(self.right_mask, self.both_mask))) == 0:
            return False
        if np.sum(np.logical_and(self.mask, np.logical_or(self.left_mask, self.both_mask))) == np.sum(self.mask) or np.sum(
            np.logical_and(self.mask, np.logical_or(self.right_mask, self.both_mask))) == np.sum(self.mask):
            return False
        return True

    def create_mask(self,df):

        self.left_mask = df[str(self.split_feature) + "_upper"] <= self.split_value
        self.right_mask = df[str(self.split_feature) + '_lower'] >= self.split_value
        self.both_mask = ((df[str(self.split_feature) + '_lower'] < self.split_value) & (df[str(self.split_feature) + "_upper"] > self.split_value))
        #self.both_mask = [True if self.split_value < upper and self.split_value > lower else False for lower, upper in
        #             zip(df[str(self.split_feature) + '_lower'], df[str(self.split_feature) + "_upper"])]

    def select_split_feature(self,df):
        feature_to_value={}
        feature_to_metric={}
        for feature in self.features:
           value,metric=self.check_feature_split_value(df,feature)
           feature_to_value[feature] = value
           feature_to_metric[feature] = metric
        feature = min(feature_to_metric, key=feature_to_metric.get)
        return feature,feature_to_value[feature]

    def check_feature_split_value(self,df,feature):
        value_to_metric={}
        values=list(set(list(df[str(feature)+'_upper'][self.mask])+list(df[str(feature)+'_lower'][self.mask])))
        np.random.shuffle(values)
        values=values[:3]
        for value in values:
            left_mask=[True if upper <= value  else False for upper in df[str(feature)+"_upper"]]
            right_mask=[True if lower>= value else False for lower in df[str(feature)+'_lower']]
            both_mask=[True if value < upper and value> lower else False for lower,upper in zip(df[str(feature)+'_lower'],df[str(feature)+"_upper"])]
            value_to_metric[value]=self.get_value_metric(df,left_mask,right_mask,both_mask)
        val=min(value_to_metric,key=value_to_metric.get)
        return val,value_to_metric[val]

    def get_value_metric(self,df,left_mask,right_mask,both_mask):
        l_df_mask=np.logical_and(np.logical_or(left_mask,both_mask),self.mask)
        r_df_mask=np.logical_and(np.logical_or(right_mask,both_mask),self.mask)
        if np.sum(l_df_mask)==0 or np.sum(r_df_mask)==0:
            return np.inf
        l_entropy,r_entropy=self.calculate_entropy(df,l_df_mask),self.calculate_entropy(df,r_df_mask)
        l_prop=np.sum(l_df_mask)/len(l_df_mask)
        r_prop=np.sum(r_df_mask)/len(l_df_mask)
        return l_entropy*l_prop+r_entropy*r_prop

    def predict_probas_and_depth(self,inst,training_df):
        if self.left is None and self.right is None:
            return self.node_probas(training_df),1
        if inst[self.split_feature] <= self.split_value:
            prediction,depth = self.left.predict_probas_and_depth(inst,training_df)
            return prediction,depth + 1
        else:
            prediction, depth = self.right.predict_probas_and_depth(inst, training_df)
            return prediction, depth + 1

    def node_probas(self, df):
        x = df['probas'][self.mask].mean()
        return x/x.sum()
    def get_node_prediction(self,training_df):
        v=training_df['probas'][self.mask][0]
        v=[i/np.sum(v) for i in v]
        return np.array(v)
    def opposite_col(self,s):
        if 'upper' in s:
            return s.replace('upper','lower')
        else:
            return s.replace('lower', 'upper')
    def calculate_entropy(self,test_df,test_df_mask):
        x = test_df['probas'][test_df_mask].mean()
        return entropy(x/x.sum())
    def count_depth(self):
        if self.right==None:
            return 1
        return max(self.left.count_depth(),self.right.count_depth())+1
    def number_of_children(self):
        if self.right==None:
            return 1
        return 1+self.right.number_of_children()+self.left.number_of_children()
    def has_same_class(self,df):
        labels=set([np.argmax(l) for l in df['probas'][self.mask]])
        if len(labels)>1:
            return False
        return True
    

from sklearn.metrics import roc_curve, auc
import numpy as np

def get_auc(Y,y_score,classes):
    y_test_binarize=np.array([[1 if i == c else 0 for c in classes] for i in Y])
    fpr, tpr, _ = roc_curve(y_test_binarize.ravel(), y_score.ravel())
    return auc(fpr, tpr)
def predict_with_included_trees(model,included_indexes,X):
    predictions=[]
    for inst in X:
        predictions.append(predict_instance_with_included_tree(model,included_indexes,inst))
    return np.array(predictions)
def predict_instance_with_included_tree(model,included_indexes,inst):
    v=np.array([0]*model.n_classes_)
    for i,t in enumerate(model.estimators_):
        if i in included_indexes:
            v = v + t.predict_proba(inst.reshape(1, -1))[0]
    return v/np.sum(v)
def select_index(rf,current_indexes,validation_x,validation_y):
    options_auc = {}
    for i in range(len(rf.estimators_)):
        if i in current_indexes:
            continue
        predictions = predict_with_included_trees(rf,current_indexes+[i],validation_x)
        options_auc[i] = get_auc(validation_y,predictions,rf.classes_)
    best_index = max(options_auc, key=options_auc.get)
    best_auc = options_auc[best_index]
    return best_auc,current_indexes+[best_index]
def reduce_error_pruning(model,validation_x,validation_y,min_size):
    best_auc,current_indexes = select_index(model,[],validation_x,validation_y)
    while len(current_indexes) <= model.n_estimators:
        if len(current_indexes) == len(model.estimators_):
            break
        new_auc, new_current_indexes = select_index(model, current_indexes,validation_x,validation_y)
        if new_auc <= best_auc and len(new_current_indexes) > min_size:
            break
        best_auc, current_indexes = new_auc, new_current_indexes
        print(best_auc, current_indexes)
    print('Finish pruning')
    return current_indexes



import numpy as np
from scipy.stats import entropy
from joblib import delayed, Parallel

def select_split_feature_parallel(df, features, mask):
    feature_to_value = {}
    feature_to_metric = {}
    features_values_list = Parallel(n_jobs=-1, verbose=0)(delayed(check_feature_split_value)(df, feature,mask) for feature in features)
    for item in features_values_list:
        feature, value, metric = item[0],item[1],item[2]
        feature_to_value[feature] = value
        feature_to_metric[feature] = metric
    feature = min(feature_to_metric, key=feature_to_metric.get)
    return feature, feature_to_value[feature]
def check_feature_split_value(df, feature, mask):
    value_to_metric = {}
    values = list(set(list(df[str(feature) + '_upper'][mask]) + list(df[str(feature) + '_lower'][mask])))
    np.random.shuffle(values)
    values = values[:3]
    class_probas = np.array([np.array(l) / np.sum(l) for l in df['probas'][mask]])
    classes = set(np.array([np.argmax(x) for x in class_probas]))
    has_same_class = len(classes)==1
    for value in values:
        left_mask = [True if upper <= value  else False for upper in df[str(feature) + "_upper"]]
        right_mask = [True if lower >= value else False for lower in df[str(feature) + '_lower']]
        both_mask = [True if value < upper and value > lower else False for lower, upper in
                     zip(df[str(feature) + '_lower'], df[str(feature) + "_upper"])]
        if has_same_class:
            value_to_metric[value] = get_value_metric(df, left_mask, right_mask, both_mask, mask)
        else:
            value_to_metric[value] = get_value_metric_accuracy(df, left_mask, right_mask, both_mask, mask)
    val = min(value_to_metric, key=value_to_metric.get)
    return feature,val, value_to_metric[val]

def get_value_metric(df,left_mask,right_mask,both_mask,original_mask):
    l_df_mask=np.logical_and(np.logical_or(left_mask,both_mask),original_mask)
    r_df_mask=np.logical_and(np.logical_or(right_mask,both_mask),original_mask)
    if np.sum(l_df_mask)==0 or np.sum(r_df_mask)==0:
        return np.inf
    l_entropy,r_entropy=calculate_entropy(df,l_df_mask),calculate_entropy(df,r_df_mask)
    l_prop=np.sum(l_df_mask)/len(l_df_mask)
    r_prop=np.sum(r_df_mask)/len(l_df_mask)
    return l_entropy*l_prop+r_entropy*r_prop

def get_value_metric_accuracy(df,left_mask,right_mask,both_mask,original_mask):
    l_df_mask=np.logical_and(np.logical_or(left_mask,both_mask),original_mask)
    r_df_mask=np.logical_and(np.logical_or(right_mask,both_mask),original_mask)
    if np.sum(l_df_mask)==0 or np.sum(r_df_mask)==0:
        return np.inf
    l_entropy,r_entropy=calculate_entropy_accuracy(df,l_df_mask),calculate_entropy_accuracy(df,r_df_mask)
    l_prop=np.sum(l_df_mask)/len(l_df_mask)
    r_prop=np.sum(r_df_mask)/len(l_df_mask)
    return l_entropy*l_prop+r_entropy*r_prop

def calculate_entropy(test_df,test_df_mask):
    class_probas = np.array([np.array(l) / np.sum(l) for l in test_df['probas'][test_df_mask]])

    class_probas = class_probas.mean(axis=0)
    probas_sum = np.sum(class_probas)
    class_probas = [i / probas_sum for i in class_probas]
    return entropy(class_probas)

def calculate_entropy_accuracy(test_df,test_df_mask):
    class_probas = np.array([np.array(l) / np.sum(l) for l in test_df['probas'][test_df_mask]])
    print(type(class_probas))
    print(class_probas)
    print('*' * 100)
    values = np.array([np.argmax(x) for x in class_probas])
    values, counts = np.unique(values, return_counts=True)
    probas = counts / np.sum(counts)
    return entropy(probas)



from sklearn.ensemble import RandomForestClassifier
class PrevPaperClassifier:
    def __init__(self, minimal_forest_size=10, max_number_of_branches=50, exclusion_threshold=0.8):
        self.minimal_forest_size = minimal_forest_size
        self.max_number_of_branches = max_number_of_branches
        self.exclusion_threshold = exclusion_threshold
        self.conjunction_set = None
        self.node_model = None

    def predict(self, X):
        probas, depths = [], []
        for inst in X:
            prob, depth = self.node_model.predict_probas_and_depth(inst, self.conjunction_set.get_conjunction_set_df())
            probas.append(prob)
            depths.append(depth)
        predictions = [self.conjunction_set.label_names[i] for i in np.array([np.argmax(prob) for prob in probas])]
        return predictions


def fit_paper_fbt(model_instance, X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)

    feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    feature_types = pd.Series({name: 'float64' for name in feature_names})

    model_instance.conjunction_set = ConjunctionSet(
        feature_names, X_train, X_train, y_train, rf,
        feature_types, model_instance.max_number_of_branches,
        exclusion_threshold=model_instance.exclusion_threshold,
        minimal_forest_size=model_instance.minimal_forest_size
    )

    branches_df = model_instance.conjunction_set.get_conjunction_set_df().round(decimals=5)
    for i in range(len(rf.classes_)):
        branches_df[rf.classes_[i]] = [probas[i] for probas in branches_df['probas']]

    df_dict = {col: branches_df[col].values for col in branches_df.columns}
    model_instance.node_model = Node([True] * len(branches_df))
    model_instance.node_model.split(df_dict)

    return model_instance
