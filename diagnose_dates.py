import pandas as pd
df = pd.read_csv("Nassau Candy Distributor.csv")
df.columns = df.columns.str.strip()
print(df[['Order Date', 'Ship Date']].head(10))
od = pd.to_datetime(df['Order Date'], format='%d-%m-%Y', errors='coerce')
sd = pd.to_datetime(df['Ship Date'], format='%d-%m-%Y', errors='coerce')
print("unparseable Order Date:", od.isna().sum(), "/", len(df))
print("unparseable Ship Date:", sd.isna().sum(), "/", len(df))
delay = (sd - od).dt.days
print(delay.describe())
print("negative delays:", (delay < 0).sum())
print("delays over 60 days:", (delay > 60).sum())
print("sample of large delays:")
mask = delay > 60
print(pd.DataFrame({'Order Date': df['Order Date'], 'Ship Date': df['Ship Date'], 'Delay': delay})[mask].head(10))
