from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import mutual_info_regression



class FeatureUtils:
    @staticmethod
    def safe_log(x):
        return np.log1p(np.clip(x, 0, None))

    @staticmethod
    def safe_sqrt(x):
        return np.sqrt(np.clip(x, 0, None))

    @staticmethod
    def check_columns(df, required):
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")



class AutoFETBase(BaseEstimator, TransformerMixin):
    def _validate_is_fitted(self):
        check_is_fitted(self, "is_fitted_")


class GroupAggregationFeatures(AutoFETBase):
    
    DEFAULT_AGGS = [
        "mean", 
        "std", 
        "min", 
        "max", 
        "median", 
        "sum", 
        "count"
    ]

    def __init__(
        self,
        numeric_cols: List[str],
        group_cols: List[str],
        aggs: Union[str, List[str]] = "default",
        add_deviation: bool = True,
        add_rank: bool = False,
    ):
        self.numeric_cols = numeric_cols
        self.group_cols = group_cols
        self.aggs = self.DEFAULT_AGGS if aggs == "default" else aggs
        self.add_deviation = add_deviation
        self.add_rank = add_rank

    def fit(self, X: pd.DataFrame, y=None):
        FeatureUtils.check_columns(X, self.numeric_cols + self.group_cols)
        
        self.stats_ = {}
        grouped = X.groupby(self.group_cols)
        
        for agg in self.aggs:
            self.stats_[agg] = grouped[self.numeric_cols].agg(agg).reset_index()
        
        self.global_mean_ = grouped[self.numeric_cols].mean().reset_index()
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame, meta_usage: bool = False):
        self._validate_is_fitted()
        df = X.copy()
        
        # Store metadata as a separate dictionary if needed
        metadata_dict = {}
        feature_lineage_dict = {}
        
        if meta_usage:
            # Initialize metadata structures as attributes of the DataFrame
            # Using a wrapper approach - we'll return a tuple or add as attributes
            pass
        
        # Групповые агрегации
        for agg, stats in self.stats_.items():
            stats = stats.copy()
            new_cols = [f"{agg}_{c}" for c in self.numeric_cols]
            stats.columns = self.group_cols + new_cols
            df = df.merge(stats, on=self.group_cols, how="left")
            
            if meta_usage:
                for c in self.numeric_cols:
                    col_name = f"{agg}_{c}"
                    metadata_dict[col_name] = {
                        'operation': agg,
                        'base_column': c,
                        'type': 'aggregation',
                        'group_cols': self.group_cols,
                        'transformer': 'GroupAggregationFeatures'
                    }
                    
                    # Отслеживаем lineage
                    if c not in feature_lineage_dict:
                        feature_lineage_dict[c] = []
                    feature_lineage_dict[c].append(col_name)
        
        # Добавление отклонений
        if self.add_deviation:
            for c in self.numeric_cols:
                mean_col = f"mean_{c}"
                
                # Разница
                diff_col = f"diff_{c}"
                df[diff_col] = df[c] - df[mean_col]
                
                # Отношение
                ratio_col = f"ratio_{c}"
                df[ratio_col] = df[c] / (df[mean_col].replace(0, np.nan).clip(lower=1e-10))
                
                if meta_usage:
                    # Метаданные для diff
                    metadata_dict[diff_col] = {
                        'operation': 'difference',
                        'base_column': c,
                        'reference_column': mean_col,
                        'type': 'deviation',
                        'formula': f'{c} - {mean_col}',
                        'transformer': 'GroupAggregationFeatures'
                    }
                    
                    # Метаданные для ratio
                    metadata_dict[ratio_col] = {
                        'operation': 'ratio',
                        'base_column': c,
                        'reference_column': mean_col,
                        'type': 'deviation',
                        'formula': f'{c} / {mean_col}',
                        'transformer': 'GroupAggregationFeatures'
                    }
                    
                    # Lineage
                    if c not in feature_lineage_dict:
                        feature_lineage_dict[c] = []
                    feature_lineage_dict[c].extend([diff_col, ratio_col])
        
        # Добавление рангов
        if self.add_rank:
            for c in self.numeric_cols:
                rank_col = f"rank_{c}"
                df[rank_col] = df.groupby(self.group_cols)[c].rank()
                
                if meta_usage:
                    metadata_dict[rank_col] = {
                        'operation': 'rank',
                        'base_column': c,
                        'type': 'ranking',
                        'group_cols': self.group_cols,
                        'method': 'average',
                        'transformer': 'GroupAggregationFeatures'
                    }
                    
                    if c not in feature_lineage_dict:
                        feature_lineage_dict[c] = []
                    feature_lineage_dict[c].append(rank_col)
        
        # Сохраняем список сгенерированных признаков
        self.generated_features_ = [c for c in df.columns if c not in X.columns]
        
        if meta_usage:
            # Attach metadata as attributes to the DataFrame
            # Note: This might still raise warnings but will work
            df.metadata = metadata_dict
            df.feature_lineage = feature_lineage_dict
            
            # Добавляем дополнительную статистику по метаданным
            df.metadata_stats = {
                'total_features_generated': len(self.generated_features_),
                'aggregation_features': sum(1 for v in metadata_dict.values() if v.get('type') == 'aggregation'),
                'deviation_features': sum(1 for v in metadata_dict.values() if v.get('type') == 'deviation'),
                'ranking_features': sum(1 for v in metadata_dict.values() if v.get('type') == 'ranking'),
                'timestamp': pd.Timestamp.now()
            }
        
        return df


class StatisticalFeatureGenerator(AutoFETBase):
    def __init__(
        self,
        numeric_cols: Optional[List[str]] = None,
        unary: Optional[List[str]] = None,
        pairwise: Optional[List[str]] = None,
        max_features: int = 200,
        corr_th: float = 0.95,
        min_var: float = 1e-5,
    ):
        self.numeric_cols = numeric_cols
        self.unary = unary or []
        self.pairwise = pairwise or []
        self.max_features = max_features
        self.corr_th = corr_th
        self.min_var = min_var

    def _get_cols(self, X):
        return self.numeric_cols or X.select_dtypes(include=np.number).columns.tolist()

    def fit(self, X, y=None):
        X = X.copy()
        self.cols_ = self._get_cols(X)

        X_new = self._make_features(X)
        X_new = self._filter_variance(X_new)
        X_new = self._filter_corr(X_new)

        if y is not None:
            mi = mutual_info_regression(X_new.fillna(0), y)
            self.selected_ = X_new.columns[np.argsort(mi)[-self.max_features:]].tolist()
        else:
            self.selected_ = X_new.columns[:self.max_features].tolist()

        self.is_fitted_ = True
        return self

    def transform(self, X, meta_usage: bool = False):
        """Transform with optional metadata generation"""
        self._validate_is_fitted()
        X_base = X.copy()
        X_new = self._make_features(X_base)
        
        # Generate metadata if requested
        metadata_dict = {}
        feature_lineage_dict = {}
        
        if meta_usage:
            # Generate metadata for all generated features
            for col in X_new.columns:
                metadata_dict[col] = self._get_feature_metadata(col, X_base)
                
                # Track lineage for base columns
                base_cols = self._get_base_columns(col)
                for base_col in base_cols:
                    if base_col not in feature_lineage_dict:
                        feature_lineage_dict[base_col] = []
                    feature_lineage_dict[base_col].append(col)
        
        # Select only the best features
        selected_new = X_new[self.selected_]
        result = pd.concat([X_base, selected_new], axis=1)
        
        if meta_usage:
            # Attach metadata as attributes
            object.__setattr__(result, 'metadata', metadata_dict)
            object.__setattr__(result, 'feature_lineage', feature_lineage_dict)
            
            # Add metadata statistics
            object.__setattr__(result, 'metadata_stats', {
                'total_features_generated': len(X_new.columns),
                'selected_features': len(self.selected_),
                'unary_features': sum(1 for col in X_new.columns if self._is_unary_feature(col)),
                'pairwise_features': sum(1 for col in X_new.columns if self._is_pairwise_feature(col)),
                'timestamp': pd.Timestamp.now()
            })
        
        return result

    def _make_features(self, X):
        """Generate all possible features"""
        X_new = pd.DataFrame(index=X.index)
        cols = self.cols_

        # Unary operations
        for c in cols:
            x = X[c]

            if "log" in self.unary:
                X_new[f"log_{c}"] = FeatureUtils.safe_log(x)
            if "sqrt" in self.unary:
                X_new[f"sqrt_{c}"] = FeatureUtils.safe_sqrt(x)

        # Pairwise operations
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                a, b = cols[i], cols[j]

                if "ratio" in self.pairwise:
                    X_new[f"ratio_{a}_{b}"] = X[a] / (X[b].replace(0, np.nan).clip(lower=1e-10))
                    X_new[f"ratio_{b}_{a}"] = X[b] / (X[a].replace(0, np.nan).clip(lower=1e-10))

                if "diff" in self.pairwise:
                    X_new[f"diff_{a}_{b}"] = X[a] - X[b]
                    X_new[f"diff_{b}_{a}"] = X[b] - X[a]

        return X_new

    def _get_feature_metadata(self, feature_name: str, X: pd.DataFrame) -> dict:
        """Generate metadata for a single feature with correct parsing"""
        metadata = {
            'type': 'unknown',
            'transformer': 'StatisticalFeatureGenerator'
        }
        
        # Check if it's a unary feature
        if self._is_unary_feature(feature_name):
            parts = feature_name.split('_', 1)  # Split only on first underscore
            operation = parts[0]
            base_col = parts[1]
            metadata.update({
                'type': 'unary',
                'operation': operation,
                'base_column': base_col,
                'formula': f"{operation}({base_col})"
            })
        
        # Check if it's a pairwise feature
        elif self._is_pairwise_feature(feature_name):
            # Find the operation (first part)
            parts = feature_name.split('_', 1)
            operation = parts[0]
            rest = parts[1]
            
            # Now need to split the rest into two column names
            # We need to find where the first column ends and second begins
            # Strategy: try all possible splits and check if both parts are in cols_
            possible_splits = []
            for i in range(1, len(rest)):
                col1_candidate = rest[:i]
                col2_candidate = rest[i+1:]  # +1 to skip the underscore
                
                # Check if both candidates are in the list of columns
                if col1_candidate in self.cols_ and col2_candidate in self.cols_:
                    possible_splits.append((col1_candidate, col2_candidate))
            
            # Use the first valid split (should be unambiguous if column names don't contain underscores)
            if possible_splits:
                col1, col2 = possible_splits[0]
                metadata.update({
                    'type': 'pairwise',
                    'operation': operation,
                    'columns': [col1, col2],
                    'formula': f"{col1} {self._get_operator_symbol(operation)} {col2}"
                })
            else:
                # Fallback: store raw name
                metadata.update({
                    'type': 'pairwise',
                    'operation': operation,
                    'raw_name': feature_name,
                    'note': 'Could not parse column names'
                })
        
        return metadata

    def _is_unary_feature(self, feature_name: str) -> bool:
        """Check if feature is from unary operation"""
        if '_' not in feature_name:
            return False
        operation = feature_name.split('_')[0]
        return operation in self.unary

    def _is_pairwise_feature(self, feature_name: str) -> bool:
        """Check if feature is from pairwise operation"""
        if '_' not in feature_name:
            return False
        operation = feature_name.split('_')[0]
        return operation in self.pairwise

    def _get_base_columns(self, feature_name: str) -> List[str]:
        """Extract base column names from generated feature"""
        if self._is_unary_feature(feature_name):
            return [feature_name.split('_', 1)[1]]
        elif self._is_pairwise_feature(feature_name):
            # Try to parse pairwise columns
            parts = feature_name.split('_', 1)
            if len(parts) == 2:
                rest = parts[1]
                for i in range(1, len(rest)):
                    col1 = rest[:i]
                    col2 = rest[i+1:]
                    if col1 in self.cols_ and col2 in self.cols_:
                        return [col1, col2]
        return []

    def _get_operator_symbol(self, operation: str) -> str:
        """Get mathematical symbol for operation"""
        symbols = {
            'ratio': '/',
            'diff': '-',
            'product': '*',
            'sum': '+',
            'log': 'log()',
            'sqrt': '√()'
        }
        return symbols.get(operation, operation)

    def _filter_variance(self, X):
        return X.loc[:, X.var() > self.min_var]

    def _filter_corr(self, X):
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
        drop = [c for c in upper.columns if any(upper[c] > self.corr_th)]
        return X.drop(columns=drop)



class FeatureSelector(AutoFETBase):
    def __init__(self, max_features=200):
        self.max_features = max_features


    def fit(self, X, y):
        mi = mutual_info_regression(X.fillna(0), y)
        self.selected_ = list(X.columns[np.argsort(mi)[-self.max_features:]])
        self.is_fitted_ = True
        return self


    def transform(self, X):
        self._validate_is_fitted()
        return X[self.selected_]

