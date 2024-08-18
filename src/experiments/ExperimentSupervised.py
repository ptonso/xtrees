import json
import time
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, log_loss
import numpy as np
import pandas as pd

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class Experiment:
    
    size_definitions = {
        'small': {'n_samples': [70, 100, 130], 'n_features': [10, 15, 20]},
        'medium': {'n_samples': [700, 1000, 1300], 'n_features': [50, 70, 100]},
        'large': {'n_samples': [7000, 10000, 13000], 'n_features': [200, 300, 400]},
        'mixed': {
            'n_samples': [70, 100, 130, 700, 1000, 1300, 7000, 10000, 13000],
            'n_features': [10, 15, 20, 50, 70, 100, 200]
        }
    }
    
    information_levels = {
        'low': {'n_informative': 0.1, 'n_redundant': 0.05, 'n_repeated': 0.05},
        'medium': {'n_informative': 0.4, 'n_redundant': 0.2, 'n_repeated': 0.1},
        'high': {'n_informative': 0.6, 'n_redundant': 0.3, 'n_repeated': 0.2},
        'mixed': {'n_informative': [0.1, 0.3, 0.5], 'n_redundant': [0.05, 0.2, 0.3], 'n_repeated': [0.05, 0.1, 0.2]}
    }
    
    prediction_levels = {
        'narrow': {'n_classes': [2, 3], 'tail_strength': [0.1, 0.2]},
        'medium': {'n_classes': [4, 5], 'tail_strength': [0.3, 0.4]},
        'spread': {'n_classes': [6, 7], 'tail_strength': [0.5, 0.6]},
        'mixed': {'n_classes': [2, 3, 4, 5, 6, 7], 'tail_strength': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
    }
    
    def __init__(self, params):
        if 'random_state' not in params['meta-params']:
            print("No random state!")
            params['meta-params']['random_state'] = None
        if 'is_classification' not in params['meta-params']:
            raise ValueError("The 'is_classification' key must be present in 'meta-params'.")
        self.params = params
        self.results = []

    def update_meta_params(self, new_meta_params):
        self._update_params('meta-params', new_meta_params)

    def update_data_params(self, index, new_data_params):
        if index < len(self.params['data-params']):
            self._update_params('data-params', new_data_params, index)
        else:
            print(f"Index {index} out of range for data-params.")


    def _update_params(self, param_class, new_params, index=None):
        if param_class in self.params:
            if index is not None:
                for key, value in new_params.items():
                    self.params[param_class][index][key] = value
                print(f"Updated {param_class}[{index}] with {new_params}")
            else:
                for key, value in new_params.items():
                    self.params[param_class][key] = value
                print(f"Updated {param_class} with {new_params}")
            print(f"Current {param_class}: {self.params[param_class]}")
        else:
            print(f"{param_class} not found in parameters.")

    def save_params_as_json(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.params, f, cls=NumpyEncoder)
        print(f"Parameters saved to {filepath}")

    def save_results_as_json(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.results, f, cls=NumpyEncoder)
        print(f"Results saved to {filepath}")


    def flatten_dict(self, d, parent_key=''):
        flat_dict = {}
        for k, v in d.items():
            new_key = f"{parent_key}" if parent_key else k
            if isinstance(v, dict):
                flat_dict.update(self.flatten_dict(v, new_key))
            else:
                if isinstance(v, (np.integer, np.floating, np.bool_)):
                    v = v.item()
                flat_dict[new_key] = v
        return flat_dict

    def print_dict(self, d, horizontal=True):
        flat_dict = self.flatten_dict(d)
        max_key_len = max(len(key) for key in flat_dict.keys())
        key_len = min(max(max_key_len, 10), 20)

        def format_value(value):
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)

        if horizontal:    
            headers = ' | '.join(f"{key:<{key_len}}" for key in flat_dict.keys())
            print(headers)
            print("=" * len(headers))

            row = ' | '.join(f"{format_value(value):<{key_len}}" for value in flat_dict.values())
            print(row)
        else:
            print(f"{'Key':<{key_len}} {'Value':<10}")
            print("=" * (key_len + 10))
        
            for key, value in flat_dict.items():
                key_lines = [key[i:i+key_len] for i in range(0, len(key), key_len)]
                value_str = format_value(value)
                value_lines = [value_str[i:i+10] for i in range(0, len(value_str), 10)]
                
                for i in range(max(len(key_lines), len(value_lines))):
                    key_part = key_lines[i] if i < len(key_lines) else ""
                    value_part = value_lines[i] if i < len(value_lines) else ""
                    print(f"{key_part:<{key_len}} {value_part:<10}")


    def print_side_by_side(self, dataset_id):
        results = [result for result in self.results if result['dataset_id'] == dataset_id]
        model_names = [result['model_name'] for result in results]
        keys = sorted(results[0].keys())
        keys = [k for k in keys if k not in ['meta_params', 'data_params', 'dataset_id', 'model_name', 'model_index']]

        col_width = max(max(len(k) for k in keys) + 2, 15)
        model_col_width = max(len(name) for name in model_names) + 10 

        header = f"{'Metric':<{col_width}}" + " | ".join([f"{name:<{model_col_width}}" for name in model_names])
        print(header)
        print("=" * len(header))

        for key in keys:
            row = [f"{key:<{col_width}}"]
            for result in results:
                value = result[key]
                if isinstance(value, float):
                    value = round(value, 4)
                row.append(f"{str(value):<{model_col_width}}")
            print(" | ".join(row))


    def assemble_results_dataframe(self):
        flattened_results = []
        for result_dict in self.results:
            flattened_results.append(result_dict)        
        df = pd.DataFrame(flattened_results)
        return df


    def run_single_experiment(self, meta_params, data_params, model_instance, dataset_id=None, fit_function=None):
        if meta_params['is_classification']:
            X, y = make_classification(**data_params)
        else:
            X, y = make_regression(**data_params)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=meta_params['train_test_split'], random_state=meta_params['random_state']
        )

        start_train_time = time.time()
        if fit_function:
            model_instance = fit_function(model_instance, X_train, y_train)
        else:
            model_instance.fit(X_train, y_train)
        train_time = time.time() - start_train_time

        start_pred_time = time.time()
        y_pred = model_instance.predict(X_test)
        pred_time = time.time() - start_pred_time

        experiment_result = {
            'dataset_id': dataset_id,
            'experiment_id': len(self.results) + 1,
            'train_time': train_time,
            'pred_time': pred_time
        }

        if meta_params['is_classification']:
            experiment_result['accuracy'] = accuracy_score(y_test, y_pred)
            
            if hasattr(model_instance, 'predict_proba'):
                y_proba = model_instance.predict_proba(X_test)
                unique_classes = np.unique(y_train)
                # experiment_result['log_loss'] = log_loss(y_test, y_proba, labels=unique_classes)

        else:
            experiment_result['mse'] = mean_squared_error(y_test, y_pred)
            experiment_result['mae'] = mean_absolute_error(y_test, y_pred)
            experiment_result['r2'] = r2_score(y_test, y_pred)
            target_avg = y.mean()
            experiment_result['target_avg'] = target_avg
            experiment_result['target_var'] = np.mean((y - target_avg) ** 2)
            experiment_result['target_mad'] = np.mean(np.abs(y - target_avg))

        experiment_result.update({
            'meta_params': meta_params,
            'data_params': data_params
        })
        self.results.append(experiment_result)
        return

    def populate_data_params(self, num_datasets, overall_size, information, prediction):
        if overall_size not in self.size_definitions:
            print(f"Invalid overall size {overall_size}. Choose from 'small', 'medium', 'large', 'mixed'.")
            return
        
        if information not in self.information_levels:
            print(f"Invalid information level {information}. Choose from 'low', 'medium', 'high', 'mixed'.")
            return
        
        if prediction not in self.prediction_levels:
            print(f"Invalid prediction level {prediction}. Choose from 'narrow', 'medium', 'spread', 'mixed'.")
            return

        sizes = self.size_definitions[overall_size]
        chosen_information = self.information_levels[information]
        chosen_prediction = self.prediction_levels[prediction]

        def compute_max_combinations(params_dict):
            max_combinations = 1
            for values in params_dict.values():
                if isinstance(values, list):
                    max_combinations *= len(values)
                return max_combinations
            
        max_possible_combinations = compute_max_combinations(sizes)
        max_possible_combinations *= compute_max_combinations(chosen_information)
        max_possible_combinations *= compute_max_combinations(chosen_prediction)

        if num_datasets > max_possible_combinations:
            print(f"Requested {num_datasets} datasets, but only {max_possible_combinations} unique combinations are possible. Generating {max_possible_combinations} datasets instead.")
            num_datasets = max_possible_combinations

        dataset_combinations = set()

        rng = np.random.default_rng(self.params['meta-params']['random_state'])

        while len(self.params['data-params']) < num_datasets:
            chosen_size = rng.choice(sizes['n_samples'])
            chosen_features = rng.choice(sizes['n_features'])
            n_informative_prob = chosen_information['n_informative']
            n_redundant_prob = chosen_information['n_redundant']
            n_repeated_prob = chosen_information['n_repeated']

            if information == 'mixed':
                n_informative_prob = rng.choice(chosen_information['n_informative'])
                n_redundant_prob = rng.choice(chosen_information['n_redundant'])
                n_repeated_prob = rng.choice(chosen_information['n_repeated'])

            n_informative = int(round(n_informative_prob * chosen_features))
            n_redundant = int(round(n_redundant_prob * chosen_features))
            n_repeated = int(round(n_repeated_prob * chosen_features))

            total = n_informative + n_redundant + n_repeated
            if total > chosen_features:
                factor = chosen_features / total
                n_informative = int(round(n_informative * factor))
                n_redundant = int(round(n_redundant * factor))
                n_repeated = int(round(n_repeated * factor))

            data_params = {
                'n_samples': chosen_size, 
                'n_features': chosen_features, 
                'n_informative': n_informative
            }

            if self.params['meta-params']['is_classification']:
                n_classes = rng.choice(chosen_prediction['n_classes'])

                if n_classes * 2 > 2**n_informative:
                    continue

                data_params['n_classes'] = n_classes
                data_params.update({
                    'n_classes': n_classes,
                    'n_informative': n_informative,
                    'n_redundant': n_redundant,
                    'n_repeated': n_repeated
                })
            else:
                tail_strength = rng.choice(chosen_prediction['tail_strength'])
                data_params['tail_strength'] = tail_strength
            
            data_params['random_state'] = self.params['meta-params']['random_state']

            if tuple(data_params.items()) not in dataset_combinations:
                dataset_combinations.add(tuple(data_params.items()))
                self.params['data-params'].append(data_params)
        
        print(f"Populated data-params with {num_datasets} datasets of overall size {overall_size}, information level {information}, and prediction level {prediction}.")


    def perform_experiments(self, num_datasets, overall_size, information, prediction, model_instances, fit_functions=None):
        self.populate_data_params(num_datasets, overall_size, information, prediction)
        meta_params = self.params['meta-params']

        if not isinstance(model_instances, list):
            model_instances = [model_instances]
        if fit_functions is None:
            fit_functions = [None] * len(model_instances)
        elif not isinstance(fit_functions, list):
            fit_functions = [fit_functions]
        
        for dataset_id, data_params in enumerate(self.params['data-params'], start=1):
            print(f"\nDataset ID: {dataset_id}")
            self.print_dict(self.params['data-params'][dataset_id-1], horizontal=True)
            print()
            for model_idx, (model_instance, fit_function) in enumerate(zip(model_instances, fit_functions)):
                model_name = type(model_instance).__name__

                print(model_name)

                self.run_single_experiment(meta_params, data_params, model_instance, dataset_id, fit_function)
                self.results[-1]['model_name'] = model_name
                self.results[-1]['model_index'] = model_idx

            self.print_side_by_side(dataset_id)
            print('\n'*2)

        print(f"Performed {num_datasets} experiments with overall size {overall_size}, information level {information}, and prediction level {prediction}.")
        df = self.assemble_results_dataframe()
        return df


