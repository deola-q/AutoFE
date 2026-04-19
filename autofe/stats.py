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
    def safe_divide(num, denom, eps=1e-10):
        """Безопасное деление с защитой от нуля и бесконечности"""
        denom = denom.replace(0, np.nan).clip(lower=eps)
        return num / denom
    
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
        "mean", "std", "min", "max", "median", "sum", "count"
    ]
    
    # Доступные типы отклонений
    DEVIATION_TYPES = [
        "diff", 
        "ratio", 
        "zscore", 
        "normalized", 
        "proportion", 
        "distance_to_boundary"
        ]
    
    def __init__(
        self,
        numeric_cols: List[str],
        group_cols: List[str],
        aggs: Union[str, List[str]] = "default",
        add_deviations: Union[bool, List[str]] = True,      # True="all", False="none", или список ["mean", "median", "min", "max"]
        add_ratios: Union[bool, List[str]] = True,          # True="all", False="none", или список ["mean", "median", "min", "max"]
        deviation_types: List[str] = None,                  # ["diff", "ratio", "zscore", "normalized", "proportion", "distance_to_boundary"]
        add_zscore: bool = False,                           # z-score = (x - mean)/std
        add_normalized: bool = False,                       # (x - min)/(max - min)
        add_proportion: bool = False,                       # доля от суммы
        add_distance_to_boundary: bool = False,             # расстояние до min/max
        add_rank: bool = False,
        add_cumulative: bool = False,                       # кумулятивные суммы/проценты
        handle_infinity: str = "clip",                      # "clip", "drop", "fillna"
        eps: float = 1e-10,                                 
    ):
        self.numeric_cols = numeric_cols
        self.group_cols = group_cols
        self.aggs = self.DEFAULT_AGGS if aggs == "default" else aggs
        self.add_deviations = add_deviations
        self.add_ratios = add_ratios
        self.deviation_types = deviation_types or []
        self.add_zscore = add_zscore
        self.add_normalized = add_normalized
        self.add_proportion = add_proportion
        self.add_distance_to_boundary = add_distance_to_boundary
        self.add_rank = add_rank
        self.add_cumulative = add_cumulative
        self.handle_infinity = handle_infinity
        self.eps = eps
        
        # Автоматически добавляем типы отклонений на основе флагов
        if add_zscore and "zscore" not in self.deviation_types:
            self.deviation_types.append("zscore")
        if add_normalized and "normalized" not in self.deviation_types:
            self.deviation_types.append("normalized")
        if add_proportion and "proportion" not in self.deviation_types:
            self.deviation_types.append("proportion")
        if add_distance_to_boundary and "distance_to_boundary" not in self.deviation_types:
            self.deviation_types.append("distance_to_boundary")
            
        # Если deviation_types пуст, но add_deviations или add_ratios активны,
        # добавляем базовые типы
        if not self.deviation_types and (self.add_deviations or self.add_ratios):
            self.deviation_types = ["diff", "ratio"]
        
        # Определяем, для каких агрегаций создавать отклонения
        if add_deviations is True:
            self.deviation_base_cols = ["mean", "median", "min", "max"]
        elif isinstance(add_deviations, list):
            self.deviation_base_cols = add_deviations
        else:
            self.deviation_base_cols = []
            
        # Для отношений
        if add_ratios is True:
            self.ratio_base_cols = ["mean", "median", "min", "max", "sum"]
        elif isinstance(add_ratios, list):
            self.ratio_base_cols = add_ratios
        else:
            self.ratio_base_cols = []

    def fit(self, X: pd.DataFrame, y=None):
        FeatureUtils.check_columns(X, self.numeric_cols + self.group_cols)
        
        self.stats_ = {}
        grouped = X.groupby(self.group_cols)
        
        for agg in self.aggs:
            self.stats_[agg] = grouped[self.numeric_cols].agg(agg).reset_index()
        
        # Сохраняем также глобальные статистики для нормализации
        self.global_stats_ = {
            'min': X[self.numeric_cols].min(),
            'max': X[self.numeric_cols].max(),
            'mean': X[self.numeric_cols].mean(),
            'std': X[self.numeric_cols].std()
        }
        
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame, meta_usage: bool = False):
        self._validate_is_fitted()
        df = X.copy()
        
        metadata_dict = {}
        feature_lineage_dict = {}
        
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
                    if c not in feature_lineage_dict:
                        feature_lineage_dict[c] = []
                    feature_lineage_dict[c].append(col_name)
        
        # Добавление всех типов отклонений
        for c in self.numeric_cols:
            # Получаем все доступные агрегации для этой колонки
            available_aggs = {}
            for agg in self.aggs:
                col_name = f"{agg}_{c}"
                if col_name in df.columns:
                    available_aggs[agg] = col_name
            
            # Разности (diff)
            if "diff" in self.deviation_types:
                for base_agg in self.deviation_base_cols:
                    if base_agg not in available_aggs:
                        continue
                    
                    base_col = available_aggs[base_agg]
                    diff_col = f"diff_{base_agg}_{c}"
                    df[diff_col] = df[c] - df[base_col]
                    
                    # Обработка бесконечностей
                    if self.handle_infinity == "clip":
                        df[diff_col] = df[diff_col].clip(-1e10, 1e10)
                    elif self.handle_infinity == "fillna":
                        df[diff_col] = df[diff_col].fillna(0)
                    
                    if meta_usage:
                        metadata_dict[diff_col] = {
                            'operation': 'difference',
                            'base_aggregation': base_agg,
                            'base_column': c,
                            'type': 'deviation',
                            'formula': f'{c} - {base_agg}({c})',
                            'transformer': 'GroupAggregationFeatures'
                        }
                        if c not in feature_lineage_dict:
                            feature_lineage_dict[c] = []
                        feature_lineage_dict[c].append(diff_col)
            
            # Отношения (ratio)
            if "ratio" in self.deviation_types:
                for base_agg in self.ratio_base_cols:
                    if base_agg not in available_aggs:
                        continue
                    
                    base_col = available_aggs[base_agg]
                    ratio_col = f"ratio_{base_agg}_{c}"
                    df[ratio_col] = FeatureUtils.safe_divide(df[c], df[base_col], self.eps)
                    
                    if self.handle_infinity == "clip":
                        df[ratio_col] = df[ratio_col].clip(0, 1e10)
                    elif self.handle_infinity == "fillna":
                        df[ratio_col] = df[ratio_col].fillna(1)
                    
                    if meta_usage:
                        metadata_dict[ratio_col] = {
                            'operation': 'ratio',
                            'base_aggregation': base_agg,
                            'base_column': c,
                            'type': 'deviation',
                            'formula': f'{c} / {base_agg}({c})',
                            'transformer': 'GroupAggregationFeatures'
                        }
                        if c not in feature_lineage_dict:
                            feature_lineage_dict[c] = []
                        feature_lineage_dict[c].append(ratio_col)
            
            # Z-score
            if "zscore" in self.deviation_types:
                if "mean" in available_aggs and "std" in available_aggs:
                    mean_col = available_aggs["mean"]
                    std_col = available_aggs["std"]
                    zscore_col = f"zscore_{c}"
                    df[zscore_col] = (df[c] - df[mean_col]) / df[std_col].clip(lower=self.eps)
                    
                    if self.handle_infinity == "clip":
                        df[zscore_col] = df[zscore_col].clip(-10, 10)
                    elif self.handle_infinity == "fillna":
                        df[zscore_col] = df[zscore_col].fillna(0)
                    
                    if meta_usage:
                        metadata_dict[zscore_col] = {
                            'operation': 'zscore',
                            'base_column': c,
                            'type': 'normalization',
                            'formula': f'({c} - mean({c})) / std({c})',
                            'transformer': 'GroupAggregationFeatures'
                        }
                        if c not in feature_lineage_dict:
                            feature_lineage_dict[c] = []
                        feature_lineage_dict[c].append(zscore_col)
            
            # Нормализованное значение в группе (0-1)
            if "normalized" in self.deviation_types:
                if "min" in available_aggs and "max" in available_aggs:
                    min_col = available_aggs["min"]
                    max_col = available_aggs["max"]
                    range_col = f"range_{c}"
                    normalized_col = f"normalized_{c}"
                    
                    # Сначала сохраняем размах
                    df[range_col] = df[max_col] - df[min_col]
                    # Нормализуем
                    df[normalized_col] = (df[c] - df[min_col]) / df[range_col].clip(lower=self.eps)
                    
                    # Ограничиваем [0, 1]
                    df[normalized_col] = df[normalized_col].clip(0, 1)
                    
                    if meta_usage:
                        metadata_dict[normalized_col] = {
                            'operation': 'normalized',
                            'base_column': c,
                            'type': 'normalization',
                            'formula': f'({c} - min({c})) / (max({c}) - min({c}))',
                            'transformer': 'GroupAggregationFeatures'
                        }
                        metadata_dict[range_col] = {
                            'operation': 'range',
                            'base_column': c,
                            'type': 'auxiliary',
                            'formula': f'max({c}) - min({c})',
                            'transformer': 'GroupAggregationFeatures'
                        }
                        if c not in feature_lineage_dict:
                            feature_lineage_dict[c] = []
                        feature_lineage_dict[c].extend([normalized_col, range_col])
            
            # Доля от суммы по группе
            if "proportion" in self.deviation_types:
                if "sum" in available_aggs:
                    sum_col = available_aggs["sum"]
                    proportion_col = f"proportion_{c}"
                    df[proportion_col] = FeatureUtils.safe_divide(df[c], df[sum_col], self.eps)
                    
                    df[proportion_col] = df[proportion_col].clip(0, 1)
                    
                    if meta_usage:
                        metadata_dict[proportion_col] = {
                            'operation': 'proportion',
                            'base_column': c,
                            'type': 'proportion',
                            'formula': f'{c} / sum({c})',
                            'transformer': 'GroupAggregationFeatures'
                        }
                        if c not in feature_lineage_dict:
                            feature_lineage_dict[c] = []
                        feature_lineage_dict[c].append(proportion_col)
            
            # Расстояние до границ (min и max)
            if "distance_to_boundary" in self.deviation_types:
                if "min" in available_aggs and "max" in available_aggs:
                    min_col = available_aggs["min"]
                    max_col = available_aggs["max"]
                    
                    # Расстояние до минимума
                    dist_to_min_col = f"dist_to_min_{c}"
                    df[dist_to_min_col] = df[c] - df[min_col]
                    
                    # Расстояние до максимума
                    dist_to_max_col = f"dist_to_max_{c}"
                    df[dist_to_max_col] = df[max_col] - df[c]
                    
                    range_col = df[max_col] - df[min_col]
                    nearest_boundary_col = f"nearest_boundary_{c}"
                    df[nearest_boundary_col] = np.minimum(
                        df[dist_to_min_col], 
                        df[dist_to_max_col]
                    ) / range_col.clip(lower=self.eps)
                    
                    if meta_usage:
                        metadata_dict[dist_to_min_col] = {
                            'operation': 'distance_to_min',
                            'base_column': c,
                            'type': 'distance',
                            'formula': f'{c} - min({c})',
                            'transformer': 'GroupAggregationFeatures'
                        }
                        metadata_dict[dist_to_max_col] = {
                            'operation': 'distance_to_max',
                            'base_column': c,
                            'type': 'distance',
                            'formula': f'max({c}) - {c}',
                            'transformer': 'GroupAggregationFeatures'
                        }
                        metadata_dict[nearest_boundary_col] = {
                            'operation': 'nearest_boundary',
                            'base_column': c,
                            'type': 'distance',
                            'formula': 'min(c - min, max - c) / (max - min)',
                            'transformer': 'GroupAggregationFeatures'
                        }
                        if c not in feature_lineage_dict:
                            feature_lineage_dict[c] = []
                        feature_lineage_dict[c].extend([dist_to_min_col, dist_to_max_col, nearest_boundary_col])
        
        # Добавление рангов
        if self.add_rank:
            for c in self.numeric_cols:
                rank_col = f"rank_{c}"
                df[rank_col] = df.groupby(self.group_cols)[c].rank(pct=False)
                
                # Процентный ранг
                pct_rank_col = f"pct_rank_{c}"
                df[pct_rank_col] = df.groupby(self.group_cols)[c].rank(pct=True)
                
                if meta_usage:
                    metadata_dict[rank_col] = {
                        'operation': 'rank',
                        'base_column': c,
                        'type': 'ranking',
                        'group_cols': self.group_cols,
                        'method': 'average',
                        'transformer': 'GroupAggregationFeatures'
                    }
                    metadata_dict[pct_rank_col] = {
                        'operation': 'percentile_rank',
                        'base_column': c,
                        'type': 'ranking',
                        'group_cols': self.group_cols,
                        'transformer': 'GroupAggregationFeatures'
                    }
                    if c not in feature_lineage_dict:
                        feature_lineage_dict[c] = []
                    feature_lineage_dict[c].extend([rank_col, pct_rank_col])
        
        # Добавление кумулятивных признаков
        # TODO: продумать другую логику / убрать
        if self.add_cumulative:
            for c in self.numeric_cols:
                df = df.sort_values(self.group_cols + [c])
                cumsum_col = f"cumsum_{c}"
                df[cumsum_col] = df.groupby(self.group_cols)[c].cumsum()
                
                cumsum_total = df.groupby(self.group_cols)[c].transform('sum')
                cumsum_pct_col = f"cumsum_pct_{c}"
                df[cumsum_pct_col] = FeatureUtils.safe_divide(df[cumsum_col], cumsum_total, self.eps)
                
                if meta_usage:
                    metadata_dict[cumsum_col] = {
                        'operation': 'cumulative_sum',
                        'base_column': c,
                        'type': 'cumulative',
                        'group_cols': self.group_cols,
                        'transformer': 'GroupAggregationFeatures'
                    }
                    metadata_dict[cumsum_pct_col] = {
                        'operation': 'cumulative_percentage',
                        'base_column': c,
                        'type': 'cumulative',
                        'group_cols': self.group_cols,
                        'transformer': 'GroupAggregationFeatures'
                    }
                    if c not in feature_lineage_dict:
                        feature_lineage_dict[c] = []
                    feature_lineage_dict[c].extend([cumsum_col, cumsum_pct_col])
        

        self.generated_features_ = [c for c in df.columns if c not in X.columns]
        
        if meta_usage:
            df.metadata = metadata_dict
            df.feature_lineage = feature_lineage_dict
            
            # Подсчет статистики по типам признаков
            type_counts = {
                'aggregation': 0,
                'deviation': 0,
                'normalization': 0,
                'proportion': 0,
                'distance': 0,
                'ranking': 0,
                'cumulative': 0,
                'auxiliary': 0
            }
            
            for v in metadata_dict.values():
                feat_type = v.get('type', 'unknown')
                if feat_type in type_counts:
                    type_counts[feat_type] += 1
                else:
                    type_counts['auxiliary'] += 1
            
            df.metadata_stats = {
                'total_features_generated': len(self.generated_features_),
                'type_distribution': type_counts,
                'total_base_aggregations': len(self.aggs) * len(self.numeric_cols),
                'deviation_features_generated': sum(1 for t in self.deviation_types if t != 'zscore' or self.add_zscore),
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
        

        metadata_dict = {}
        feature_lineage_dict = {}
        
        if meta_usage:
            for col in X_new.columns:
                metadata_dict[col] = self._get_feature_metadata(col, X_base)
                
                base_cols = self._get_base_columns(col)
                for base_col in base_cols:
                    if base_col not in feature_lineage_dict:
                        feature_lineage_dict[base_col] = []
                    feature_lineage_dict[base_col].append(col)
        
        selected_new = X_new[self.selected_]
        result = pd.concat([X_base, selected_new], axis=1)
        
        if meta_usage:

            object.__setattr__(result, 'metadata', metadata_dict)
            object.__setattr__(result, 'feature_lineage', feature_lineage_dict)
            
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


        for c in cols:
            x = X[c]

            if "log" in self.unary:
                X_new[f"log_{c}"] = FeatureUtils.safe_log(x)
            if "sqrt" in self.unary:
                X_new[f"sqrt_{c}"] = FeatureUtils.safe_sqrt(x)

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
        

        if self._is_unary_feature(feature_name):
            parts = feature_name.split('_', 1) 
            operation = parts[0]
            base_col = parts[1]
            metadata.update({
                'type': 'unary',
                'operation': operation,
                'base_column': base_col,
                'formula': f"{operation}({base_col})"
            })
        
        elif self._is_pairwise_feature(feature_name):
            # Find the operation (first part)
            parts = feature_name.split('_', 1)
            operation = parts[0]
            rest = parts[1]
            
           
            possible_splits = []
            for i in range(1, len(rest)):
                col1_candidate = rest[:i]
                col2_candidate = rest[i+1:]  
                
                if col1_candidate in self.cols_ and col2_candidate in self.cols_:
                    possible_splits.append((col1_candidate, col2_candidate))
            
            if possible_splits:
                col1, col2 = possible_splits[0]
                metadata.update({
                    'type': 'pairwise',
                    'operation': operation,
                    'columns': [col1, col2],
                    'formula': f"{col1} {self._get_operator_symbol(operation)} {col2}"
                })
            else:

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

