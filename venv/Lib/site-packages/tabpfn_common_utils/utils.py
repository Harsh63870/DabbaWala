import time
import typing

from functools import lru_cache, wraps

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from typing import Literal, Union, Dict, Any
from dataclasses import dataclass
from typing_extensions import override


def serialize_to_csv_formatted_bytes(
    data: typing.Union[pd.DataFrame, pd.Series, np.ndarray],
) -> bytes:
    if type(data) not in [pd.DataFrame, pd.Series, np.ndarray]:
        raise TypeError(f"({type(data)}) is not supported for serialization")

    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    # data is now of type pd.DataFrame
    csv_bytes = data.to_csv(index=False).encode("utf-8")

    return csv_bytes


FileName = str
FileContent = bytes
FileCategory = str
FileUpload = typing.Tuple[FileCategory, FileName, FileContent]


def to_httpx_post_file_format(file_uploads: typing.List[FileUpload]) -> typing.Dict:
    ret = {}
    for file_upload in file_uploads:
        file_category, filename, content = file_upload
        ret[file_category] = (filename, content)

    return ret


def to_oauth_request_form(username: str, password: str) -> Dict[str, str]:
    return {"grant_type": "password", "username": username, "password": password}


class Singleton:
    def __new__(cls):
        raise TypeError("Cannot instantiate this class. This is a singleton.")


def singleton(cls):
    """
    Decorator to make a class a singleton.

    Args:
        cls: The class to make a singleton.

    Returns:
        The singleton instance of the class.
    """
    instance = [None]

    def wrapper(*args, **kwargs):
        if instance[0] is None:
            instance[0] = cls(*args, **kwargs)
        return instance[0]

    return wrapper


def ttl_cache(ttl_seconds: int = 60, max_size: int = 1):
    """Decorator to cache the result of a function using a TTL.

    Args:
        ttl_seconds: The time to live for the cached result.
        max_size: The maximum size of the cache.

    Returns:
        The decorator.
    """

    def decorator(func):
        # Set up the LRU cache properties
        func = lru_cache(maxsize=max_size)(func)
        func.ttl_seconds = ttl_seconds  # type: ignore
        func.expires_at = time.time() + ttl_seconds  # type: ignore

        @wraps(func)
        def wrapper(*args, **kwargs):
            if time.time() >= func.expires_at:  # type: ignore
                func.cache_clear()
                func.expires_at = time.time() + func.ttl_seconds  # type: ignore
            return func(*args, **kwargs)

        return wrapper

    return decorator


def shape_of(X: Any) -> tuple[int, int]:
    """Get the input dimension of the data.

    Supports numpy, pandas, torch, sklearn array-likes, and generic sequences.
    """
    # Objects with .shape (numpy, pandas, torch, scipy, etc.)
    try:
        shape = X.shape
        # Scalar types
        if len(shape) == 0:
            return 1, 1

        # 1D arrays
        if len(shape) == 1:  # 1D array
            return shape[0], 1

        # Default to 2D array
        return shape[0], shape[1]
    except AttributeError:
        pass
    except Exception:
        return 0, 0

    # Generic sequences like lists, tuples, etc.
    try:
        n_rows = len(X)
        if n_rows == 0:
            return 0, 0

        first = X[0]
        if hasattr(first, "__len__"):
            return n_rows, len(first)

        return n_rows, 1
    except Exception:
        return 0, 0


def get_example_dataset(
    dataset_name: typing.Literal["iris", "breast_cancer", "digits", "diabetes"],
) -> typing.Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    load_dataset_fn = {
        "iris": load_iris,
        "breast_cancer": load_breast_cancer,
        "digits": load_digits,
        "diabetes": load_diabetes,
    }
    x_train, y_train = load_dataset_fn[dataset_name](return_X_y=True, as_frame=True)

    # shuffle and get 10 examples
    # shuffle is needed because we will might get examples with only 1 class
    # use fixed seed for reproducibility
    rng = np.random.RandomState(46)
    indices = rng.permutation(len(x_train))[:10]
    x_train = x_train.iloc[indices]
    y_train = y_train.iloc[indices]

    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, test_size=0.33, random_state=42
    )

    return typing.cast(
        typing.Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        (x_train, x_test, y_train, y_test),
    )


def get_dataset_with_specific_size(
    num_examples: int = 10_000, num_columns: int = 100
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.random.RandomState(42).rand(num_examples, num_columns)
    y_train = np.random.RandomState(42).randint(0, 2, size=num_examples)

    return x_train, x_train, y_train, y_train


def assert_y_pred_proba_is_valid(x_test, y_pred_proba):
    if isinstance(y_pred_proba, list):
        y_pred_proba = np.array(y_pred_proba)

    proba_shape = y_pred_proba.shape
    assert proba_shape[0] == len(x_test)
    assert proba_shape[1] >= 2
    assert np.allclose(y_pred_proba.sum(axis=1), np.ones(proba_shape[0]))


@dataclass
class PreprocessorConfig:
    """Configuration for data preprocessors.

    Attributes:
        name: Name of the preprocessor.
        categorical_name:
            Name of the categorical encoding method.
            Options: "none", "numeric", "onehot", "ordinal", "ordinal_shuffled", "none".
        append_original: Whether to append original features to the transformed features
        subsample_features: Fraction of features to subsample. -1 means no subsampling.
        global_transformer_name: Name of the global transformer to use.
    """

    name: Literal[
        "per_feature",  # a different transformation for each feature
        "power",  # a standard sklearn power transformer
        "safepower",  # a power transformer that prevents some numerical issues
        "power_box",
        "safepower_box",
        "quantile_uni_coarse",  # quantile transformations with few quantiles up to many
        "quantile_norm_coarse",
        "quantile_uni",
        "quantile_norm",
        "quantile_uni_fine",
        "quantile_norm_fine",
        "robust",  # a standard sklearn robust scaler
        "kdi",
        "none",  # no transformation (only standardization in transformer)
        "kdi_random_alpha",
        "kdi_uni",
        "kdi_random_alpha_uni",
        "adaptive",
        "norm_and_kdi",
        # KDI with alpha collection
        "kdi_alpha_0.3_uni",
        "kdi_alpha_0.5_uni",
        "kdi_alpha_0.8_uni",
        "kdi_alpha_1.0_uni",
        "kdi_alpha_1.2_uni",
        "kdi_alpha_1.5_uni",
        "kdi_alpha_2.0_uni",
        "kdi_alpha_3.0_uni",
        "kdi_alpha_5.0_uni",
        "kdi_alpha_0.3",
        "kdi_alpha_0.5",
        "kdi_alpha_0.8",
        "kdi_alpha_1.0",
        "kdi_alpha_1.2",
        "kdi_alpha_1.5",
        "kdi_alpha_2.0",
        "kdi_alpha_3.0",
        "kdi_alpha_5.0",
    ]
    categorical_name: Literal[
        # categorical features are pretty much treated as ordinal, just not resorted
        "none",
        # categorical features are treated as numeric,
        # that means they are also power transformed for example
        "numeric",
        # "onehot": categorical features are onehot encoded
        "onehot",
        # "ordinal": categorical features are sorted and encoded as
        # integers from 0 to n_categories - 1
        "ordinal",
        # "ordinal_shuffled": categorical features are encoded as integers
        # from 0 to n_categories - 1 in a random order
        "ordinal_shuffled",
        "ordinal_very_common_categories_shuffled",
    ] = "none"
    append_original: bool = False
    subsample_features: float = -1
    global_transformer_name: Union[str, None] = None

    @override
    def __str__(self) -> str:
        return (
            f"{self.name}_cat:{self.categorical_name}"
            + ("_and_none" if self.append_original else "")
            + (
                f"_subsample_feats_{self.subsample_features}"
                if self.subsample_features > 0
                else ""
            )
            + (
                f"_global_transformer_{self.global_transformer_name}"
                if self.global_transformer_name is not None
                else ""
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the PreprocessorConfig instance to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing the configuration parameters.
        """
        return {
            "name": self.name,
            "categorical_name": self.categorical_name,
            "append_original": self.append_original,
            "subsample_features": self.subsample_features,
            "global_transformer_name": self.global_transformer_name,
        }
