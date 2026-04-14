# AutoFEAT

### Framework for automated feature generation based on statistics and AI tools

[![PyPI version](https://badge.fury.io/py/autofe-vsu-project.svg)](https://pypi.org/project/autofe-vsu-project/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Installation

```bash
pip install autofe-vsu-project
```

## Quick Start

```python
import pandas as pd
from autofe import GroupAggregationFeatures, StatisticalFeatureGenerator

# Load your data
df = pd.read_csv('your_data.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Generate group-based features
group_features = GroupAggregationFeatures(
    numeric_cols=['age', 'fare'],
    group_cols=['pclass', 'sex'],
    aggs=['mean', 'std'],
    add_deviation=True
)
X_train_grouped = group_features.fit_transform(X_train)

# Generate statistical features
stat_features = StatisticalFeatureGenerator(
    numeric_cols=['age', 'fare'],
    unary=['log', 'sqrt'],
    pairwise=['diff']
)
X_train_stats = stat_features.fit_transform(X_train_grouped, y_train)
```

## Key Features

- **Group Aggregations** - Mean, std, min, max, sum, count by categories
- **Statistical Transforms** - Log, sqrt, ratio, difference between features
- **Sklearn-Compatible** - Works with Pipeline, GridSearchCV


## 🔧 Requirements

- Python >= 3.8
- numpy >= 1.19.0
- pandas >= 1.2.0
- scikit-learn >= 0.24.0

**Made with ❤️ for automated machine learning**
