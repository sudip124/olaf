import pandas as pd

# Symbols to remove
invalid_symbols = [
    'DUMMYTATAM',
    'DUMMYSKFIN', 
    'DUMMYDBRLT',
    'OLAELEC',
    'RITES',
    'RADICO',
    'RELINFRA',
    'SKFINDIA'
]

# Read the CSV file
df = pd.read_csv('symbols.csv')

# Remove invalid symbols
df = df[~df['Symbol'].isin(invalid_symbols)]

# Save back to CSV
df.to_csv('symbols.csv', index=False)

print(f"Removed {len(invalid_symbols)} invalid symbols from symbols.csv")