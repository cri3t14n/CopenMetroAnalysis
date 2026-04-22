from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "DatabaseScripts"))

from DataFilter import MetroDataQuery

query_filtered = (
    MetroDataQuery("./metro_passengers.parquet")
    .filter_system("m1m2")
    .filter_station_id(8603328)
    .filter_date_range("2024-01-01", "2024-02-01")
)

print(query_filtered.info())
print(query_filtered.preview(20, sort_by_time=True))