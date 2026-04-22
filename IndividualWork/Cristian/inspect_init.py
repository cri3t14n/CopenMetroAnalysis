from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "DatabaseScripts"))

from ParquetInspector import MetroParquetInspector

inspector = MetroParquetInspector("./metro_passengers.parquet")

# print("\nPreview m1m2:")
# print(inspector.preview_system("m1m2", n=20))

# print("\nPreview m3m4 with selected columns:")
# print(inspector.preview_system("m3m4", n=10, columns=["timestamp", "station", "system", "passengerNo"]))

# print("\nStation ids in m1m2:")
# print(inspector.station_ids_for_system("m1m2"))


# print("\nNumeric summary for m3m4:")
# print(inspector.numeric_summary("m3m4"))

print(inspector.preview_station_id(8603328, n=20))