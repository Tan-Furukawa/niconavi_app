from setuptools import setup, find_packages

setup(
    name="niconavi-app",
    version="0.0.1",
    description="Packaged niconavi application.",
    author="Furukawa Tan",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "niconavi_app": ["assets/*.png", "assets/icons/*.png"],
        "niconavi_app.niconavi": ["package_data/*.npy"],
    },
)
