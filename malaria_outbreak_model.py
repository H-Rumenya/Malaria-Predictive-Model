
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

@dataclass
class MalariaOutbreakModel:
    """
    Simple weekly malaria cases regression model + outbreak flagger.
    
    - Train on historical weekly weather + observed 'cases'.
    - Predict upcoming weekly cases from 2–6 weeks of weather.
    - Flag an outbreak when pred_week_t / pred_week_(t-1) > outbreak_ratio.
    
    Assumptions:
      * Input DataFrames are weekly granularity.
      * Column 'cases' exists in training data (int/float).
      * A 'week' column (date-like or string) is optional; used for nice output if present.
      * All other numeric columns are treated as weather features.
    """
    outbreak_ratio: float = 1.5
    random_state: int = 7
    
    def __post_init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.feature_cols: Optional[List[str]] = None

    def _infer_feature_cols(self, df: pd.DataFrame, target_col: str = "cases") -> List[str]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target_col]
        if not feature_cols:
            raise ValueError("No numeric feature columns found. Please include numeric weather variables.")
        return feature_cols

    def fit(self, train_df: pd.DataFrame, target_col: str = "cases") -> "MalariaOutbreakModel":
        """
        Fit the model on historical weekly data.
        """
        self.feature_cols = self._infer_feature_cols(train_df, target_col=target_col)
        X = train_df[self.feature_cols].copy()
        y = train_df[target_col].astype(float).values
        
        pre = ColumnTransformer(
            transformers=[("num", StandardScaler(with_mean=True, with_std=True), self.feature_cols)],
            remainder="drop"
        )
        gbr = GradientBoostingRegressor(random_state=self.random_state)
        self.pipeline = Pipeline(steps=[("pre", pre), ("reg", gbr)])
        self.pipeline.fit(X, y)
        return self

    def predict(self, weeks_df: pd.DataFrame, week_col: Optional[str] = None) -> pd.DataFrame:
        """
        Predict weekly cases for the provided weeks_df (2–6 rows typically).
        Returns a DataFrame with columns: [week(optional), pred_cases, outbreak_flag].
        """
        if self.pipeline is None or self.feature_cols is None:
            raise RuntimeError("Model is not fitted. Call fit(...) first.")
        
        # Keep only known feature columns; raise if any required are missing
        missing = [c for c in self.feature_cols if c not in weeks_df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        
        preds = self.pipeline.predict(weeks_df[self.feature_cols])
        result = weeks_df.copy()
        if week_col and week_col in weeks_df.columns:
            result = result[[week_col]].copy()
        else:
            if "week" in weeks_df.columns:
                week_col = "week"
                result = weeks_df[["week"]].copy()
            else:
                result = pd.DataFrame({"week_index": np.arange(len(weeks_df))})
                week_col = "week_index"
        
        result["pred_cases"] = preds
        
        # outbreak flag: pred_w2 / pred_w1 > threshold
        ratio = result["pred_cases"].shift(-1) / result["pred_cases"]
        result["outbreak_next_week_flag"] = ratio > self.outbreak_ratio
        # The flag indicates if the *following* week is an outbreak relative to the current week.
        # If you want the current week's status compared to previous, invert the shift.
        
        return result

    def predict_and_flag_current(self, weeks_df: pd.DataFrame, week_col: Optional[str] = None) -> pd.DataFrame:
        """
        Alternative: Flag outbreak for each week compared to the immediately previous predicted week.
        outbreak_flag_t = pred_t / pred_(t-1) > threshold
        The first week will have flag = False (no previous week to compare).
        """
        if self.pipeline is None or self.feature_cols is None:
            raise RuntimeError("Model is not fitted. Call fit(...) first.")
        
        missing = [c for c in self.feature_cols if c not in weeks_df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        
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
