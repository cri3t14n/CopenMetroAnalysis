from __future__ import annotations

import pyarrow.parquet as pq
import pandas as pd


class MetroParquetInspector:
    VALID_SYSTEMS = {"m1m2", "m3m4"}

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.parquet_file = pq.ParquetFile(file_path)
        self.schema = self.parquet_file.schema
        self.columns = self.schema.names

    def print_schema(self) -> None:
        print("=" * 60)
        print("SCHEMA")
        print("=" * 60)
        print(self.schema)

    def print_columns(self) -> None:
        print("=" * 60)
        print("COLUMNS")
        print("=" * 60)
        for col in self.columns:
            print(f"- {col}")

    def preview(self, n: int = 10, columns: list[str] | None = None) -> pd.DataFrame:
        """
        Read only the first n rows from the parquet file, without loading all data.
        """
        batch = next(self.parquet_file.iter_batches(batch_size=n, columns=columns))
        return batch.to_pandas()

    def _validate_system(self, system: str) -> None:
        if system not in self.VALID_SYSTEMS:
            raise ValueError(
                f"Invalid system '{system}'. Expected one of: {sorted(self.VALID_SYSTEMS)}"
            )

    def preview_system(
        self,
        system: str,
        n: int = 10,
        columns: list[str] | None = None,
        batch_size: int = 5000,
    ) -> pd.DataFrame:
        """
        Scan batches until enough rows for the chosen system are found.
        This avoids loading the full dataset into memory.
        """
        self._validate_system(system)

        if columns is None:
            columns = self.columns

        collected = []
        total_found = 0

        for batch in self.parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            df = batch.to_pandas()
            df = df[df["system"] == system]

            if not df.empty:
                collected.append(df)
                total_found += len(df)

            if total_found >= n:
                break

        if not collected:
            return pd.DataFrame(columns=columns)

        result = pd.concat(collected, ignore_index=True).head(n)
        return result

    def unique_systems(self) -> list[str]:
        found = set()

        for batch in self.parquet_file.iter_batches(batch_size=5000, columns=["system"]):
            df = batch.to_pandas()
            found.update(df["system"].dropna().unique())

        return sorted(found)

    def stations_for_system(self, system: str, batch_size: int = 5000) -> list[str]:
        self._validate_system(system)

        found = set()

        for batch in self.parquet_file.iter_batches(
            batch_size=batch_size,
            columns=["system", "station"],
        ):
            df = batch.to_pandas()
            df = df[df["system"] == system]

            if not df.empty:
                found.update(df["station"].dropna().unique())

        return sorted(found)

    def station_ids_for_system(self, system: str, batch_size: int = 5000) -> pd.DataFrame:
        self._validate_system(system)

        collected = []

        for batch in self.parquet_file.iter_batches(
            batch_size=batch_size,
            columns=["system", "stationId", "station"],
        ):
            df = batch.to_pandas()
            df = df[df["system"] == system]

            if not df.empty:
                collected.append(df[["stationId", "station"]])

        if not collected:
            return pd.DataFrame(columns=["stationId", "station"])

        result = (
            pd.concat(collected, ignore_index=True)
            .drop_duplicates()
            .sort_values(["station", "stationId"])
            .reset_index(drop=True)
        )
        return result

    def date_range(self, system: str | None = None, batch_size: int = 5000) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        min_ts = None
        max_ts = None

        cols = ["timestamp"] if system is None else ["timestamp", "system"]
        if system is not None:
            self._validate_system(system)

        for batch in self.parquet_file.iter_batches(batch_size=batch_size, columns=cols):
            df = batch.to_pandas()

            if system is not None:
                df = df[df["system"] == system]

            if df.empty:
                continue

            local_min = df["timestamp"].min()
            local_max = df["timestamp"].max()

            if min_ts is None or local_min < min_ts:
                min_ts = local_min
            if max_ts is None or local_max > max_ts:
                max_ts = local_max

        return min_ts, max_ts

    def numeric_summary(
        self,
        system: str | None = None,
        batch_size: int = 5000,
    ) -> pd.DataFrame:
        """
        Returns summary stats for ins, outs, passengerNo.
        This does scan the file, but only the needed columns.
        """
        cols = ["ins", "outs", "passengerNo"]
        if system is not None:
            self._validate_system(system)
            cols = ["system"] + cols

        collected = []

        for batch in self.parquet_file.iter_batches(batch_size=batch_size, columns=cols):
            df = batch.to_pandas()

            if system is not None:
                df = df[df["system"] == system]
                df = df.drop(columns=["system"])

            if not df.empty:
                collected.append(df)

        if not collected:
            return pd.DataFrame()

        result = pd.concat(collected, ignore_index=True)
        return result.describe()

    def load_columns(
        self,
        columns: list[str],
        system: str | None = None,
        batch_size: int = 5000,
    ) -> pd.DataFrame:
        """
        Load only specific columns, optionally filtered by system.
        """
        missing = [col for col in columns if col not in self.columns]
        if missing:
            raise ValueError(f"Unknown columns: {missing}")

        read_columns = columns.copy()
        if system is not None:
            self._validate_system(system)
            if "system" not in read_columns:
                read_columns.append("system")

        collected = []

        for batch in self.parquet_file.iter_batches(batch_size=batch_size, columns=read_columns):
            df = batch.to_pandas()

            if system is not None:
                df = df[df["system"] == system]

            collected.append(df)

        if not collected:
            return pd.DataFrame(columns=columns)

        result = pd.concat(collected, ignore_index=True)

        if system is not None and "system" not in columns:
            result = result.drop(columns=["system"])

        return result
    
    def preview_station_id(
        self,
        station_id: int,
        n: int = 10,
        columns: list[str] | None = None,
        batch_size: int = 5000,
    ) -> pd.DataFrame:
        """
        Return up to the first n rows matching a specific stationId,
        without loading the entire parquet file.
        """
        if columns is None:
            columns = self.columns

        if "stationId" not in columns:
            columns = list(columns) + ["stationId"]

        collected = []
        total_found = 0

        for batch in self.parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            df = batch.to_pandas()
            df = df[df["stationId"] == station_id]

            if not df.empty:
                collected.append(df)
                total_found += len(df)

            if total_found >= n:
                break

        if not collected:
            return pd.DataFrame(columns=columns)

        result = pd.concat(collected, ignore_index=True).head(n)
        return result