
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            raise ValueError(f"Missing required column: {c}")
        if not np.issubdtype(out[c].dtype, np.number):
            s = out[c].astype(str).str.strip()
            s = s.str.replace(",", "", regex=False)
            s = s.str.replace("%", "", regex=False)
            s = s.str.replace("\u00a0", "", regex=False)
            out[c] = pd.to_numeric(s, errors="coerce")
    return out

@dataclass
class MalariaOutbreakModel:
    outbreak_ratio: float = 1.5
    random_state: int = 7

    def __post_init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.feature_cols: Optional[List[str]] = None

    def _infer_feature_cols(self, df: pd.DataFrame, target_col: str = "cases") -> List[str]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            numeric_cols = [c for c in df.columns if c not in {target_col, "week"}]
        feature_cols = [c for c in numeric_cols if c != target_col]
        if not feature_cols:
            raise ValueError("No feature columns found. Please include numeric weather variables.")
        return feature_cols

    def fit(self, train_df: pd.DataFrame, target_col: str = "cases", feature_cols: Optional[List[str]] = None) -> "MalariaOutbreakModel":
        self.feature_cols = feature_cols if feature_cols is not None else self._infer_feature_cols(train_df, target_col=target_col)
        train_df = _coerce_numeric(train_df, self.feature_cols + [target_col])
        mask = train_df[self.feature_cols + [target_col]].notna().all(axis=1)
        cleaned = train_df.loc[mask].copy()
        if cleaned.empty:
            raise ValueError("After cleaning, no valid rows remained. Check your data types/values.")
        X = cleaned[self.feature_cols].values
        y = cleaned[target_col].astype(float).values

        pre = ColumnTransformer(
            transformers=[("num", StandardScaler(with_mean=True, with_std=True), list(range(X.shape[1])))],
            remainder="drop"
        )
        gbr = GradientBoostingRegressor(random_state=self.random_state)
        self.pipeline = Pipeline(steps=[("pre", pre), ("reg", gbr)])
        self.pipeline.fit(X, y)
        return self

    def predict(self, weeks_df: pd.DataFrame, week_col: Optional[str] = None) -> pd.DataFrame:
        if self.pipeline is None or self.feature_cols is None:
            raise RuntimeError("Model is not fitted. Call fit(...) first.")
        weeks_df = _coerce_numeric(weeks_df, self.feature_cols)
        missing = [c for c in self.feature_cols if c not in weeks_df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        mask = weeks_df[self.feature_cols].notna().all(axis=1)
        if not mask.all():
            weeks_df = weeks_df.loc[mask].copy()

        preds = self.pipeline.predict(weeks_df[self.feature_cols])
        if week_col and week_col in weeks_df.columns:
            result = weeks_df[[week_col]].copy()
        elif "week" in weeks_df.columns:
            result = weeks_df[["week"]].copy()
        else:
            result = pd.DataFrame({"week_index": np.arange(len(weeks_df))})
            week_col = "week_index"
        result["pred_cases"] = preds
        ratio = result["pred_cases"].shift(-1) / result["pred_cases"]
        result["outbreak_next_week_flag"] = ratio > self.outbreak_ratio
        return result

    def predict_and_flag_current(self, weeks_df: pd.DataFrame, week_col: Optional[str] = None) -> pd.DataFrame:
        if self.pipeline is None or self.feature_cols is None:
            raise RuntimeError("Model is not fitted. Call fit(...) first.")
        weeks_df = _coerce_numeric(weeks_df, self.feature_cols)
        missing = [c for c in self.feature_cols if c not in weeks_df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        mask = weeks_df[self.feature_cols].notna().all(axis=1)
        if not mask.all():
            weeks_df = weeks_df.loc[mask].copy()

        preds = self.pipeline.predict(weeks_df[self.feature_cols])
        if week_col and week_col in weeks_df.columns:
            res = weeks_df[[week_col]].copy()
        elif "week" in weeks_df.columns:
            res = weeks_df[["week"]].copy()
        else:
            res = pd.DataFrame({"week_index": np.arange(len(weeks_df))})
        res["pred_cases"] = preds
        res["outbreak_flag"] = res["pred_cases"] / res["pred_cases"].shift(1) > self.outbreak_ratio
        res["outbreak_flag"] = res["outbreak_flag"].fillna(False)
        return res
