"""Application version derived from the project package metadata."""

from importlib.metadata import version


# setuptools generates this metadata from ``project.version`` in pyproject.toml.
APP_VERSION = version("niconavi-app")
APP_VERSION_LABEL = f"NicoNavi v{APP_VERSION}"
