import numpy as np
import pandas as pd
from typing import *

try:
    from IPython.display import display
    from IPython import get_ipython
except ImportError:
    display = None
    get_ipython = None

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.logger import setup_logger


logger = setup_logger("api.log")

# add to file for setup
    # import numpy as np
    # seed = 123

    # from src.utils import setup_toy_classifier, setup_toy_regressor
    # X_c, y_c, class_names_c, rf_c = setup_toy_classifier(random_state=seed)
    # X_r, y_r, rf_r                = setup_toy_regressor (random_state=seed)


# df["value"] = df["value"].apply(_fmt)

def _fmt(val: Union[float, np.ndarray, List[float]]) -> str:
    """
    Format scalars or 1-D arrays/lists.
    >>>_fmt([0.333333, 0.666])
    '[0.33, 0.67]'
    """
    if isinstance(val, (list, np.ndarray)):
        arr = np.asarray(val).flatten()
        return "[" + ", ".join(f"{x:.2f}" for x in arr) + "]"
    if isinstance(val, (float, np.floating)):
        return f"{float(val):.2f}"
    if isinstance(val, (int, np.integer)):
        return f"{int(val)}"
    return str(val)


def show_df(df: pd.DataFrame, n: int = 5, max_col_width: int = 12) -> pd.DataFrame:
    """
    Display (and return) a formatted version of df.head(n):
      • Numeric columns are run through _fmt
      • Columns containing list/ndarray samples are run through _fmt
      • Column names longer than max_col_width are truncated to fit
    """
    head = df.head(n).copy()

    numeric_cols = head.select_dtypes(include=["number"]).columns.tolist()

    list_cols: List[str] = []
    for col in head.columns:
        if col in numeric_cols:
            continue
        nonnull = head[col].dropna()
        if not nonnull.empty and isinstance(nonnull.iloc[0], (list, np.ndarray)):
            list_cols.append(col)

    for col in numeric_cols + list_cols:
        head[col] = head[col].apply(_fmt)

    def _truncate(name: str) -> str:
        return name if len(name) <= max_col_width else name[: max_col_width - 3] + "..."

    truncated_map = {col: _truncate(col) for col in head.columns}
    head = head.rename(columns=truncated_map)

    print(head.to_string(index=True))
    return head


def setup_toy_classifier(
    n_samples: int = 100,
    n_features: int = 4,
    n_classes: int = 2,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], RandomForestClassifier]:
    """
    Generate a random classification dataset and train a RandomForestClassifier.

    Returns:
        X: feature matrix of shape (n_samples, n_features)
        y: integer labels of shape (n_samples,)
        class_names: list of class name strings
        rf: trained RandomForestClassifier
    """
    rng = np.random.RandomState(random_state)
    X: np.ndarray = rng.rand(n_samples, n_features)
    y: np.ndarray = rng.randint(0, n_classes, size=n_samples)
    class_names: List[str] = [f"Class {i}" for i in range(n_classes)]

    rf = RandomForestClassifier(
        n_estimators=3, max_depth=4, random_state=random_state
    )
    rf.fit(X, y)

    logger.info(
        f"Trained RandomForestClassifier on {n_samples} samples, "
        f"{n_features} features, {n_classes} classes."
    )
    return X, y, class_names, rf


def setup_toy_regressor(
    n_samples: int = 100,
    n_features: int = 4,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, RandomForestRegressor]:
    """
    Generate a random regression dataset and train a RandomForestRegressor.

    Returns:
        X: feature matrix of shape (n_samples, n_features)
        y: continuous targets of shape (n_samples,)
        rf: trained RandomForestRegressor
    """
    rng = np.random.RandomState(random_state)
    X: np.ndarray = rng.rand(n_samples, n_features)
    y: np.ndarray = rng.rand(n_samples)

    rf = RandomForestRegressor(
        n_estimators=3, max_depth=4, random_state=random_state
    )
    rf.fit(X, y)

    logger.info(
        f"Trained RandomForestRegressor on {n_samples} samples, "
        f"{n_features} features."
    )
    return X, y, rf


if __name__ == "__main__":
    # Example usage for testing utilities

    # Classifier example
    X_clf, y_clf, class_names, rf_clf = setup_toy_classifier(
        n_samples=50, n_features=3, n_classes=2, random_state=42
    )
    print("Classifier data shapes:", X_clf.shape, y_clf.shape)
    print("Classifier class names:", class_names)
    print("Classifier first 5 predictions:", rf_clf.predict(X_clf[:5]))

    # Regressor example
    X_reg, y_reg, rf_reg = setup_toy_regressor(
        n_samples=50, n_features=3, random_state=42
    )
    print("Regressor data shapes:", X_reg.shape, y_reg.shape)
    print("Regressor first 5 predictions:", rf_reg.predict(X_reg[:5]))





