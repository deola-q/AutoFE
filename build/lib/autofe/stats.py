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


    def transform(self, X: pd.DataFrame):
        self._validate_is_fitted()
        df = X.copy()

        # merge-based
        for agg, stats in self.stats_.items():
            stats = stats.copy()
            stats.columns = self.group_cols + [f"{agg}_{c}" for c in self.numeric_cols]
            df = df.merge(stats, on=self.group_cols, how="left")

        if self.add_deviation:
            for c in self.numeric_cols:
                mean_col = f"mean_{c}"
                df[f"diff_{c}"] = df[c] - df[mean_col]
                df[f"ratio_{c}"] = df[c] / (df[mean_col].replace(0, np.nan))

        if self.add_rank:
            df = df.sort_values(self.group_cols)
            for c in self.numeric_cols:
                df[f"rank_{c}"] = df.groupby(self.group_cols)[c].rank()

        self.generated_features_ = [c for c in df.columns if c not in X.columns]
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


    def transform(self, X):
        self._validate_is_fitted()
        X_base = X.copy()
        X_new = self._make_features(X_base)
        selected_new = X_new[self.selected_]
        return pd.concat([X_base, selected_new], axis=1)


    def _make_features(self, X):
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
                    X_new[f"ratio_{a}_{b}"] = X[a] / (X[b].replace(0, np.nan))

                if "diff" in self.pairwise:
                    X_new[f"diff_{a}_{b}"] = X[a] - X[b]

        return X_new


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

