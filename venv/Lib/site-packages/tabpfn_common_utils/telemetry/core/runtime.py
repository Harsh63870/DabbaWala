"""Runtime environment detection for telemetry."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Mapping, Optional, Sequence, Tuple


# The type of kernel the code is running in
KernelType = Literal["ipython", "jupyter", "tty"]

# The type of environment the code is running in
EnvironmentType = Literal[
    "kaggle",
    "colab",
    "gcp",
    "aws",
    "azure",
    "databricks",
    "modal",
    "mlflow",
    "kubernetes",
]

# Static list of environment hints, purely heuristic based on env variables.
# This information is used purely for detecting purposes, and the values are
# not propagated or stored in the telemetry state.
ENV_TYPE_HINTS: Mapping[EnvironmentType, Sequence[str]] = {
    # Notebook providers
    "kaggle": [
        # Kaggle kernels
        "KAGGLE_KERNEL_RUN_TYPE",
        "KAGGLE_URL_BASE",
        "KAGGLE_KERNEL_INTEGRATIONS",
        "KAGGLE_USER_SECRETS_TOKEN",
        "KAGGLE_GCP_PROJECT",
        "KAGGLE_GCP_ZONE",
    ],
    "colab": [
        # Google Colab
        "COLAB_GPU",
        "COLAB_TPU_ADDR",
        "COLAB_JUPYTER_TRANSPORT",
        "COLAB_BACKEND_VERSION",
    ],
    "databricks": [
        # Databricks clusters and runtime
        "DATABRICKS_RUNTIME_VERSION",
        "DATABRICKS_CLUSTER_ID",
        "DATABRICKS_HOST",
        "DATABRICKS_WORKSPACE_URL",
        "DB_IS_DRIVER",
    ],
    "modal": ["MODAL_CLOUD_PROVIDER", "MODAL_IMAGE_ID", "MODAL_REGION"],
    "mlflow": [
        "MLFLOW_RUN_ID",
        "MLFLOW_RUN_NAME",
    ],
    "kubernetes": [
        "KUBERNETES_SERVICE_HOST",
        "KUBERNETES_POD_NAME",
    ],
    # Cloud providers
    "aws": [
        # Generic AWS environment hints
        "AWS_EXECUTION_ENV",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        # SageMaker
        "SM_MODEL_DIR",
        "SM_NUM_GPUS",
        "SM_HOSTS",
        "SM_CURRENT_HOST",
        "TRAINING_JOB_NAME",
    ],
    "gcp": [
        # Project hints
        "GOOGLE_CLOUD_PROJECT",
        "GCP_PROJECT",
        "GCLOUD_PROJECT",
        "CLOUDSDK_CORE_PROJECT",
        # Cloud Run and Cloud Functions
        "K_SERVICE",
        "K_REVISION",
        "K_CONFIGURATION",
        "CLOUD_RUN_JOB",
        # Vertex AI
        "AIP_MODEL_DIR",
        "AIP_DATA_FORMAT",
        "AIP_TRAINING_DATA_URI",
        "CLOUD_ML_JOB_ID",
        # Cloud Shell
        "CLOUD_SHELL",
    ],
    "azure": [
        # Azure ML
        "AZUREML_RUN_ID",
        "AZUREML_ARM_SUBSCRIPTION",
        "AZUREML_ARM_RESOURCEGROUP",
        "AZUREML_ARM_WORKSPACE_NAME",
    ],
}


@dataclass
class ExecutionContext:
    """The execution context of the current environment."""

    interactive: bool
    """Whether the code is running in an interactive environment."""

    kernel: Optional[KernelType] = None
    """Low-level Python frontend or shell (e.g. IPython, Jupyter, TTY)"""

    environment: Optional[EnvironmentType] = None
    """Higher-level hosted environment or notebook platform."""

    ci: bool = False
    """Whether the code is running in a CI environment."""


def get_execution_context() -> ExecutionContext:
    """Get the execution context of the current environment.

    Returns:
        The execution context of the current environment.
    """
    # First check for environment
    environment = _get_environment()

    # Next, get kernel information
    interactive, kernel = _get_kernel()

    context = ExecutionContext(
        interactive=interactive, kernel=kernel, environment=environment, ci=_is_ci()
    )
    return context


def _get_kernel() -> Tuple[bool, Optional[KernelType]]:
    """Get the kernel the code is running in.

    Returns:
        A tuple of (whether the kernel is interactive, the kernel type).
    """
    mapping: Dict[KernelType, Callable[[], bool]] = {
        "ipython": _is_ipy,
        "jupyter": _is_jupyter_kernel,
        "tty": _is_tty,
    }
    for kernel, func in mapping.items():
        if func():
            return True, kernel
    return False, None


def _get_environment() -> Optional[EnvironmentType]:
    """Get the environment the code is running in.

    An environment is a higher-level hosted environment or notebook platform.
    This is about where the code is running (Kaggle, Colab, AWS, GCP, ...).

    Returns:
        The environment the code is running in.
    """
    for env_type, hints in ENV_TYPE_HINTS.items():
        if any(k in os.environ for k in hints):
            return env_type

    return None


def _is_ipy() -> bool:
    """Check if the current environment is an IPython notebook.

    Returns:
        True if the environment is an IPython notebook, False otherwise.
    """
    try:
        from IPython import get_ipython  # type: ignore[import-untyped]

        return get_ipython() is not None
    except ImportError:
        return False


def _is_jupyter_kernel() -> bool:
    """Check if the current environment is a Jupyter kernel.

    Returns:
        True if the current environment is a Jupyter kernel, False otherwise.
    """
    if "ipykernel" in sys.modules:
        return True

    # Common hints used by Jupyter frontends
    jupyter_env_vars = {
        "JPY_PARENT_PID",
        "JUPYTERHUB_API_URL",
        "JUPYTERHUB_USER",
        "COLAB_RELEASE_TAG",
    }
    return any(os.environ.get(k) for k in jupyter_env_vars)


def _is_tty() -> bool:
    """Check if the current environment is a TTY.

    Returns:
        True if the current environment is a TTY, False otherwise.
    """
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (OSError, AttributeError, IndexError):
        return False


def _is_ci() -> bool:
    """Check if the current environment is a CI environment.

    Returns:
        True if the current environment is a CI environment, False otherwise.
    """
    # Common CI environment variables
    ci_env_vars = {
        # GitHub Actions
        "GITHUB_ACTIONS",
        # GitLab CI
        "GITLAB_CI",
        # Jenkins
        "JENKINS_URL",
        "JENKINS_HOME",
        # Travis CI
        "TRAVIS",
        # CircleCI
        "CIRCLECI",
        # Azure DevOps
        "TF_BUILD",
        "AZURE_DEVOPS",
        # AWS CodeBuild
        "CODEBUILD_BUILD_ID",
        # Google Cloud Build
        "BUILD_ID",
    }
    return any(os.environ.get(var) for var in ci_env_vars)
