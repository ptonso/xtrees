
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
import glob
import re
import ast

def read_result_csv(search_name):
    if search_name not in ['reg', 'class']:
        print("search name have to be 'reg' or 'class'")
        pass
    csv_files = glob.glob('data/results/*.csv')
    regex = rf'{search_name}_experiment\d+'
    filtered_files = [file for file in csv_files if re.search(regex, file)]

    dfs = []
    for file in filtered_files:
        df = pd.read_csv(file, index_col=None)
        df.drop(df.columns[0], axis=1, inplace=True)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df.drop(columns=['experiment_id'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    df.index.name= 'experiment_id'
    return df


def extract_reg(df):
    extracted_data = []
    for _, row in df.iterrows():
        data_params = ast.literal_eval(row['data_params'])
        data_size = f"{data_params['n_samples']}x{data_params['n_features']}"
        extracted_data.append({
            'model_name': row['model_name'],
            'data_size': data_size,
            'mae': row['mae'],
            'mse': row['mse'],
            'r²': row['r2'],
            'train_time': row['train_time'],
            'pred_time': row['pred_time']
        })
    return pd.DataFrame(extracted_data)

def extract_class(df):
    extracted_data = []
    for _, row in df.iterrows():
        data_params = ast.literal_eval(row['data_params'])
        data_size = f"{data_params['n_samples']}x{data_params['n_features']}"
        extracted_data.append({
            'model_name': row['model_name'],
            'data_size': data_size,
            'accuracy': row['accuracy'],
            'train_time': row['train_time'],
            'pred_time': row['pred_time']
        })

    return pd.DataFrame(extracted_data)


def average_reg_metrics(df):
    df['normalized_mae'] = df['mae'] / df['target_mad']
    df['normalized_mse'] = df['mse'] / df['target_var']
    
    grouped = df.groupby('model_name').agg({
        'train_time': ['mean', 'std'],
        'pred_time': ['mean', 'std'],
        'normalized_mse': ['mean', 'std'],
        'r2': ['mean', 'std']
    }).reset_index()

    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    for col in ['train_time', 'pred_time', 'normalized_mse', 'r2']:
        grouped[col] = grouped[f'{col}_mean'].round(4).astype(str) + ' ± ' + grouped[f'{col}_std'].round(2).astype(str)
        grouped.drop([f'{col}_mean', f'{col}_std'], axis=1, inplace=True)
    grouped.columns = ["model_name", "train_time", "pred_time", "norm_mse",	"r2"]
    return grouped



def average_class_metrics(df):
    grouped = df.groupby('model_name').agg({
        'train_time': ['mean', 'std'],
        'pred_time': ['mean', 'std'],
        'accuracy': ['mean', 'std']
    }).reset_index()

    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    for col in ['train_time', 'pred_time', 'accuracy']:
        grouped[col] = grouped[f'{col}_mean'].round(4).astype(str) + ' ± ' + grouped[f'{col}_std'].round(2).astype(str)
        grouped.drop([f'{col}_mean', f'{col}_std'], axis=1, inplace=True)

    return grouped


class FitClass:

    def __init__(self, SEED):
        self.seed = SEED

    def fit_fbt_class(self, model_instance, X_train, y_train):
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        
        num_of_estimators = 100
        max_depth = 5

        rf = RandomForestClassifier(n_estimators=num_of_estimators, max_depth=max_depth, random_state=self.seed)

        rf.fit(X_train, y_train)

        feature_names = [f'feature {i}' for i in range(X_train.shape[1])]
        feature_types = pd.Series('float64', index=feature_names)

        minimal_forest_size=10
        max_number_of_branches=50
        exclusion_threshold=0.8

        model_instance.fit(rf, X_train, y_train, feature_types=feature_types, feature_names=feature_names, 
            minimal_forest_size=minimal_forest_size, amount_of_branches_threshold=max_number_of_branches, exclusion_threshold=exclusion_threshold)

        return model_instance

    def fit_fbtrand_class(self, model_instance, X_train, y_train):
        from sklearn.model_selection import RandomizedSearchCV
        param_distributions = {
            'num_of_estimators': [10, 50, 100, 200],
            'max_depth': [3, 5, 10, None],
            'minimal_forest_size': [5, 10, 20],
            'amount_of_branches_threshold': [10, 20, 30, 40],
            'exclusion_threshold': [0.5, 0.7, 0.8, 0.9]
        }
        random_search = RandomizedSearchCV(model_instance, 
                                        param_distributions=param_distributions, 
                                        n_iter=3,
                                        cv=3,
                                        scoring='accuracy',
                                        random_state=self.seed,
                                        n_jobs=-1)
        random_search.fit(X_train, y_train)
        print(random_search.best_params_)
        return random_search.best_estimator_


    def fit_rf_class(self, model_instance, X_train, y_train):
        model_instance.fit(X_train, y_train)
        return model_instance

    def fit_dtrand_class(self, model_instance, X_train, y_train):
        from sklearn.model_selection import RandomizedSearchCV
        
        param_grid = {
            'max_depth': [None, 10, 30, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 4, 10],
            'max_features': [None, 'sqrt', 'log2'],
            'max_leaf_nodes': [None, 10, 20, 50],
            'ccp_alpha': [0.0, 0.01, 0.1, 0.001],
        }

        randomized_search = RandomizedSearchCV(estimator=model_instance, 
                                        param_distributions=param_grid, 
                                        n_iter=10, 
                                        scoring='accuracy', 
                                        cv=3, 
                                        n_jobs=-1, 
                                        random_state=self.seed)


        randomized_search.fit(X_train, y_train)
        return randomized_search.best_estimator_
    


class FitReg:

    def __init__(self, SEED):
        self.seed = SEED

    def fit_fbt_reg(self, model_instance, X_train, y_train):
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor

        num_of_estimators = 100
        max_depth = 5
        rf = RandomForestRegressor(n_estimators=num_of_estimators, max_depth=max_depth, random_state=self.seed)
        rf.fit(X_train, y_train)
        feature_names = [f'feature {i}' for i in range(X_train.shape[1])]
        feature_types = pd.Series('float64', index=feature_names)

        minimal_forest_size=10
        max_number_of_branches=50
        exclusion_threshold=0.8

        model_instance.fit(rf, X_train, y_train, feature_types=feature_types, feature_names=feature_names,
            minimal_forest_size=minimal_forest_size, amount_of_branches_threshold=max_number_of_branches, 
            exclusion_threshold=exclusion_threshold)

        return model_instance

    def fit_fbtrand_reg(self, model_instance, X_train, y_train):
        from sklearn.model_selection import RandomizedSearchCV
        param_distributions = {
            'num_of_estimators': [10, 50, 100, 200],
            'max_depth': [3, 5, 10, None],
            'minimal_forest_size': [5, 10, 20],
            'amount_of_branches_threshold': [10, 20, 30, 40],
            'exclusion_threshold': [0.5, 0.7, 0.8, 0.9]
        }
        random_search = RandomizedSearchCV(model_instance, 
                                            param_distributions=param_distributions, 
                                            n_iter=10,
                                            cv=3,
                                            scoring='neg_mean_squared_error', 
                                            random_state=self.seed,
                                            n_jobs=-1)
        random_search.fit(X_train, y_train)
        print(random_search.best_params_)
        return random_search.best_estimator_


    def fit_rf_reg(self, model_instance, X_train, y_train):
        model_instance.fit(X_train, y_train)
        return model_instance


    def fit_dtrand_reg(self, model_instance, X_train, y_train):
        from sklearn.model_selection import RandomizedSearchCV

        param_grid = {
            'max_depth': [None, 10, 30, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 4, 10],
            'max_features': [None, 'sqrt', 'log2'],
            'max_leaf_nodes': [None, 10, 20, 50],
            'ccp_alpha': [0.0, 0.01, 0.1, 0.001],
        }

        randomized_search = RandomizedSearchCV(estimator=model_instance, 
                                            param_distributions=param_grid, 
                                            n_iter=3, 
                                            scoring='neg_mean_squared_error', 
                                            cv=3, 
                                            n_jobs=-1, 
                                            random_state=self.seed)


        randomized_search.fit(X_train, y_train)
        return randomized_search.best_estimator_


