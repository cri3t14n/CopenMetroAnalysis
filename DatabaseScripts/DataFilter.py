from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq


@dataclass(frozen=True)
class MetroFilterState:
    system: str | None = None
    station_id: int | None = None
    station_name: str | None = None
    start_time: pd.Timestamp | None = None
    end_time: pd.Timestamp | None = None
    selected_columns: list[str] | None = None


class MetroDataQuery:
    VALID_SYSTEMS = {"m1m2", "m3m4"}
    DEFAULT_BATCH_SIZE = 5000

    def __init__(
        self,
        file_path: str | Path,
        state: MetroFilterState | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.file_path = Path(file_path)
        self.parquet_file = pq.ParquetFile(str(self.file_path))
        self.columns = self.parquet_file.schema.names
        self.batch_size = batch_size
        self.state = state or MetroFilterState()

    def _clone(self, **updates) -> "MetroDataQuery":
        new_state = replace(self.state, **updates)
        return MetroDataQuery(
            file_path=self.file_path,
            state=new_state,
            batch_size=self.batch_size,
        )

    def _validate_system(self, system: str) -> None:
        if system not in self.VALID_SYSTEMS:
            raise ValueError(
                f"Invalid system '{system}'. Expected one of {sorted(self.VALID_SYSTEMS)}"
            )

    def _normalize_timestamp(self, value) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        return ts

    def filter_system(self, system: str) -> "MetroDataQuery":
        self._validate_system(system)
        return self._clone(system=system)

    def filter_station_id(self, station_id: int) -> "MetroDataQuery":
        return self._clone(station_id=station_id)

    def filter_station_name(self, station_name: str) -> "MetroDataQuery":
        return self._clone(station_name=station_name)

    def filter_date_range(self, start=None, end=None) -> "MetroDataQuery":
        start_ts = self.state.start_time if start is None else self._normalize_timestamp(start)
        end_ts = self.state.end_time if end is None else self._normalize_timestamp(end)

        if start_ts is not None and end_ts is not None and start_ts > end_ts:
            raise ValueError("start must be earlier than or equal to end")

        return self._clone(start_time=start_ts, end_time=end_ts)

    def select_columns(self, columns: list[str]) -> "MetroDataQuery":
        missing = [col for col in columns if col not in self.columns]
        if missing:
            raise ValueError(f"Unknown columns: {missing}")
        return self._clone(selected_columns=columns)

    def reset_filters(self) -> "MetroDataQuery":
        return MetroDataQuery(self.file_path, batch_size=self.batch_size)

    def _required_read_columns(self) -> list[str]:
        needed = set(self.state.selected_columns or self.columns)

        if self.state.system is not None:
            needed.add("system")
        if self.state.station_id is not None:
            needed.add("stationId")
        if self.state.station_name is not None:
            needed.add("station")
        if self.state.start_time is not None or self.state.end_time is not None:
            needed.add("timestamp")

        return [col for col in self.columns if col in needed]

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.state.system is not None:
            df = df[df["system"] == self.state.system]

        if self.state.station_id is not None:
            df = df[df["stationId"] == self.state.station_id]

        if self.state.station_name is not None:
            df = df[df["station"] == self.state.station_name]

        if self.state.start_time is not None:
            df = df[df["timestamp"] >= self.state.start_time]

        if self.state.end_time is not None:
            df = df[df["timestamp"] <= self.state.end_time]

        return df

    def _finalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        selected = self.state.selected_columns
        if selected is None:
            return df
        return df[selected]

    def iter_filtered_batches(self) -> Iterable[pd.DataFrame]:
        read_columns = self._required_read_columns()

        for batch in self.parquet_file.iter_batches(
            batch_size=self.batch_size,
            columns=read_columns,
        ):
            df = batch.to_pandas()
            df = self._apply_filters(df)

            if not df.empty:
                df = self._finalize_columns(df)
                yield df

    def preview(self, n: int = 10, sort_by_time: bool = False) -> pd.DataFrame:
        collected = []
        total = 0

        for df in self.iter_filtered_batches():
            collected.append(df)
            total += len(df)
            if total >= n:
                break

        if not collected:
            return pd.DataFrame(columns=self.state.selected_columns or self.columns)

        result = pd.concat(collected, ignore_index=True).head(n)

        if sort_by_time and "timestamp" in result.columns:
            result = result.sort_values("timestamp").reset_index(drop=True)

        return result

    def count_rows(self) -> int:
        total = 0
        for df in self.iter_filtered_batches():
            total += len(df)
        return total

    def to_pandas(self, sort_by_time: bool = True) -> pd.DataFrame:
        collected = list(self.iter_filtered_batches())

        if not collected:
            return pd.DataFrame(columns=self.state.selected_columns or self.columns)

        result = pd.concat(collected, ignore_index=True)

        if sort_by_time and "timestamp" in result.columns:
            result = result.sort_values("timestamp").reset_index(drop=True)

        return result

    def save_parquet(self, output_path: str | Path, sort_by_time: bool = True) -> None:
        output_path = Path(output_path)
        df = self.to_pandas(sort_by_time=sort_by_time)
        df.to_parquet(output_path, index=False)

    def save_csv(self, output_path: str | Path, sort_by_time: bool = True) -> None:
        output_path = Path(output_path)
        df = self.to_pandas(sort_by_time=sort_by_time)
        df.to_csv(output_path, index=False)

    def info(self) -> None:
        print("=" * 60)
        print("QUERY STATE")
        print("=" * 60)
        print(f"file_path      : {self.file_path}")
        print(f"system         : {self.state.system}")
        print(f"station_id     : {self.state.station_id}")
        print(f"station_name   : {self.state.station_name}")
        print(f"start_time     : {self.state.start_time}")
        print(f"end_time       : {self.state.end_time}")
        print(f"selected_cols  : {self.state.selected_columns}")
        print(f"batch_size     : {self.batch_size}")