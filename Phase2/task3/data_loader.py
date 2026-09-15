"""
DataLoader -- reads the two supplied CSVs and checks they have the shape
the rest of the pipeline assumes, before any cleaning happens.
"""

import pandas as pd

import config


class DataLoader:
    """Loads a single raw CSV and validates its expected schema."""

    def load(self, source):
        """Read a CSV file into a DataFrame, exactly as supplied (no
        parsing/cleaning here -- that is DataCleaner's job, so a schema
        problem is caught before any transformation masks it)."""
        df = pd.read_csv(source)
        df.columns = df.columns.str.strip().str.lower()
        return df

    def validate_schema(self, data, kind):
        """Check that `data` has the columns a given `kind` ('price' or
        'signals') requires. Raises ValueError with a specific message
        rather than letting a KeyError surface three modules downstream."""
        if kind == "price":
            required = {"date", *config.PRICE_COLUMNS}
        elif kind == "signals":
            required = {"date", *[c.lower() for c in config.ALL_SIGNALS]}
        else:
            raise ValueError(f"Unknown schema kind: {kind!r}")

        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f"{kind} data is missing required columns: {sorted(missing)}"
            )
        if data.empty:
            raise ValueError(f"{kind} data loaded with zero rows")
        return True

    def load_price(self):
        df = self.load(config.PRICE_FILE)
        self.validate_schema(df, "price")
        return df

    def load_signals(self):
        df = self.load(config.SIGNALS_FILE)
        df.columns = [c.upper() if c.lower() != "date" else "date" for c in df.columns]
        self.validate_schema(df.rename(columns=str.lower), "signals")
        return df
