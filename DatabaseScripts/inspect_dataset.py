import pandas as pd

file_path = "./metro_passengers.parquet"

df = pd.read_parquet(file_path)

print("=" * 50)
print("BASIC INFO")
print("=" * 50)
print("Shape:", df.shape)

print("\nColumns:")
for col in df.columns:
    print(f"- {col}")

print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nFirst 10 rows:")
print(df.head(10))

print("\nSummary statistics:")
print(df.describe(include="all"))