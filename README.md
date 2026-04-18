# AutoFEAT

### The framework for automatically generating features with metadata to improve explainability

[![PyPI version](https://badge.fury.io/py/autofe-vsu-project.svg)](https://pypi.org/project/autofe-grass/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

..in development..

## Installation

```bash
pip install autofe-grass
```

## Quick Start

```python
import pandas as pd
from autofe import GroupAggregationFeatures, StatisticalFeatureGenerator

# Load your data
df = pd.read_csv('your_data.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Generate group-based features
group_feats1 = GroupAggregationFeatures(
        numeric_cols=numeric_cols,
        group_cols=group_cols,
        aggs=['mean', 'std' ...],  
        add_deviations=True,
        add_rank=False
)
X_train_grouped = group_features.fit_transform(X_train)

# Generate statistical features
stat_gen = StatisticalFeatureGenerator(
        numeric_cols=numeric_cols,
        unary=['log', 'sqrt'],
        pairwise=['ratio', 'diff'],
        max_features=20,
        corr_th=0.95,
        min_var=1e-5
    )
X_train_stats = stat_features.fit_transform(X_train_grouped, y_train)
```

The ```transform``` method allows you to view the meaning of any generated feature by using ```meta_usage=True``` flag (default value is False)

```python
X_train_stats = stat_features.transform(X_train_grouped, meta_usage=True)
```

## Learn more
Usage examples: [notebooks](https://github.com/deola-q/AutoFE/tree/dc4b0e21bb1a143bae1a73dadc5f99d441c0bfbe/notebooks)

## Key Features

- **Group Aggregations** - Mean, std, min, max, sum, count by categories
- **Statistical Transforms** - Log, sqrt, ratio, difference between features
- **Sklearn-Compatible** - Works with sklearn Pipeline


## 🔧 Requirements

- Python >= 3.8
- numpy >= 1.19.0
- pandas >= 1.2.0
- scikit-learn >= 0.24.0
