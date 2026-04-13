from setuptools import setup, find_packages

setup(
    name="inventory-optimization-platform",
    version="0.1.0",
    description=(
        "Production-ready inventory optimization system combining demand "
        "forecasting, OR-Tools optimization, and supply chain simulation "
        "on the M5 Forecasting dataset."
    ),
    author="Harish Sivasub",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyarrow>=14.0.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "click>=8.1.7",
    ],
    extras_require={
        "ml": ["xgboost>=2.0.0", "lightgbm>=4.0.0"],
        "opt": ["ortools>=9.8.3296"],
        "dashboard": ["streamlit>=1.32.0", "plotly>=5.18.0"],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "run-pipeline=scripts.run_data_pipeline:main",
        ]
    },
)
