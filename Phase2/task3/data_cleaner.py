"""
DataCleaner -- turns a raw, loaded DataFrame into a research-ready one.

Built against what the actual supplied CSVs contain, not a generic
textbook cleaner:

  * the `date` column mixes ISO ("2018-01-02") and day-first
    ("21-02-2018") formats within the *same* column -- pandas' own
    format='mixed'/dayfirst guessing is not reliable here (several of
    the day-first rows, e.g. "01-03-2019", are themselves ambiguous
    with an ISO reading), so dates are parsed deterministically by
    which token is 4 digits;
  * a number of rows are exact duplicates of an earlier date, some of
    them disguised by using the other date format for the same day --
    so de-duplication must run *after* date normalisation, never before;
  * the continuous band-signal columns (BB06, BB07) and VB05 are blank
    for their first few rows -- a genuine rolling-window warm-up, not a
    data error -- and must not be fabricated by filling with 0 or a
    forward-fill from nothing.
"""

import numpy as np
import pandas as pd

import config


class DataCleaner:
    def __init__(self, kind):
        """`kind` is 'price' or 'signals' -- it decides which columns are
        numeric, which are boolean, and what the warm-up columns are."""
        if kind not in ("price", "signals"):
            raise ValueError("kind must be 'price' or 'signals'")
        self.kind = kind
        self._report = {}

    # ---------------------------------------------------------- dates --
    @staticmethod
    def _parse_date_token(token):
        """Deterministic parse for a single date string that may be
        'YYYY-MM-DD' or 'DD-MM-YYYY'. The two never collide: an ISO date
        always starts with a 4-digit year, a day-first date never does,
        so the first token's length settles the format with no guessing
        (which is what makes rows like '01-03-2019' unambiguous here even
        though they would be ambiguous in isolation)."""
        s = str(token).strip()
        parts = s.split("-")
        if len(parts) != 3:
            return pd.NaT
        if len(parts[0]) == 4:
            y, m, d = parts
        else:
            d, m, y = parts
        try:
            return pd.Timestamp(year=int(y), month=int(m), day=int(d))
        except ValueError:
            return pd.NaT

    def _parse_dates(self, data):
        parsed = data["date"].map(self._parse_date_token)
        n_bad = parsed.isna().sum()
        self._report["unparseable_dates"] = int(n_bad)
        out = data.copy()
        out["date"] = parsed
        return out

    # -------------------------------------------------------- validate --
    def validate(self, data):
        """Non-mutating data-quality check. Returns a dict of issues found
        so callers can decide whether to proceed."""
        issues = {}
        df = self._parse_dates(data)
        issues["unparseable_dates"] = int(df["date"].isna().sum())
        issues["duplicate_dates"] = int(df["date"].duplicated().sum())
        issues["not_sorted"] = bool((df["date"].diff().dropna() < pd.Timedelta(0)).any())

        if self.kind == "price":
            for col in config.PRICE_COLUMNS:
                issues[f"missing_{col}"] = int(df[col].isna().sum())
            issues["negative_volume"] = int((df["volume"] < 0).sum())
            issues["high_below_low"] = int((df["high"] < df["low"]).sum())
            bad_ohlc = (
                (df["high"] < df[["open", "close"]].max(axis=1))
                | (df["low"] > df[["open", "close"]].min(axis=1))
            )
            issues["ohlc_inconsistent"] = int(bad_ohlc.sum())
        else:
            for col in config.BOOLEAN_SIGNALS:
                if col not in df.columns:
                    continue
                vals = set(pd.unique(df[col].dropna()))
                issues[f"{col}_non_binary_values"] = sorted(vals - {0, 1, 0.0, 1.0})
            for col in config.CONTINUOUS_SIGNALS:
                if col not in df.columns:
                    continue
                issues[f"{col}_missing"] = int(df[col].isna().sum())

        return issues

    # ------------------------------------------------------------ core --
    def sort_chronologically(self, data):
        out = data.sort_values("date", kind="mergesort").reset_index(drop=True)
        return out

    def remove_duplicates(self, data):
        """Drop rows sharing a date (post date-normalisation, so the two
        formats used for the same day are caught). Where duplicate rows
        disagree on values, that is logged, not silently averaged."""
        dupe_mask = data.duplicated(subset="date", keep=False)
        n_dupe_dates = int(data.loc[dupe_mask, "date"].nunique())

        value_cols = [c for c in data.columns if c != "date"]
        disagreeing = 0
        if n_dupe_dates:
            for _, group in data.loc[dupe_mask].groupby("date"):
                if group[value_cols].nunique().gt(1).any():
                    disagreeing += 1

        self._report["duplicate_dates_found"] = n_dupe_dates
        self._report["duplicate_dates_with_conflicting_values"] = disagreeing
        out = data.drop_duplicates(subset="date", keep="first").reset_index(drop=True)
        self._report["rows_dropped_as_duplicate"] = len(data) - len(out)
        return out

    def handle_missing(self, data):
        """Column-specific missing-value policy, applied and logged rather
        than blanket-filled:

          * PRICE columns: a genuinely missing OHLCV bar is dropped
            (fabricating a forward-filled candle would invent a trade that
            never happened, and none is expected in this dataset -- so any
            occurrence is worth surfacing, not silently patching).
          * SIGNAL boolean columns: also dropped if missing (none expected;
            a missing flag is not a safe default of 0).
          * SIGNAL continuous columns (BB06, BB07, VB05): left as NaN. This
            is warm-up, not damage -- rows before a rolling window fills
            are genuinely undefined, and a strategy reading these columns
            must treat NaN as "no signal yet" (flat), never as a fabricated
            0 or a leak of a later value backwards.
        """
        out = data.copy()
        if self.kind == "price":
            before = len(out)
            out = out.dropna(subset=config.PRICE_COLUMNS)
            self._report["price_rows_dropped_missing"] = before - len(out)
        else:
            present_bool = [c for c in config.BOOLEAN_SIGNALS if c in out.columns]
            before = len(out)
            out = out.dropna(subset=present_bool)
            self._report["signal_rows_dropped_missing_boolean"] = before - len(out)
            present_cont = [c for c in config.CONTINUOUS_SIGNALS if c in out.columns]
            self._report["continuous_warmup_nans"] = {
                c: int(out[c].isna().sum()) for c in present_cont
            }
        return out

    def clean(self, data):
        """Full pipeline: parse dates -> sort -> de-duplicate -> handle
        missing values -> final schema checks. Returns a research-ready
        frame indexed by `date`."""
        df = self._parse_dates(data)
        n_before = len(df)
        df = df.dropna(subset=["date"])
        self._report["rows_dropped_unparseable_date"] = n_before - len(df)

        df = self.sort_chronologically(df)
        df = self.remove_duplicates(df)
        df = self.handle_missing(df)

        if self.kind == "price":
            bad = (df["high"] < df["low"]) | (df["volume"] < 0)
            self._report["rows_dropped_bad_ohlc"] = int(bad.sum())
            df = df.loc[~bad]
        else:
            for col in config.BOOLEAN_SIGNALS:
                if col in df.columns:
                    df[col] = df[col].astype(float)

        df = df.set_index("date")
        self._report["final_row_count"] = len(df)
        self._report["date_range"] = (
            str(df.index.min().date()) if len(df) else None,
            str(df.index.max().date()) if len(df) else None,
        )
        return df

    def get_report(self):
        """Return the diagnostics accumulated by the last clean() call."""
        return dict(self._report)
