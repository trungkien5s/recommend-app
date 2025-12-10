# debug_countries.py - Kiểm tra tên quốc gia trong các file

import pandas as pd

print("=" * 60)
print("KIỂM TRA TÊN QUỐC GIA TRONG CÁC FILE DỮ LIỆU")
print("=" * 60)

# 1. Load Cost
file_id_cost = "1Deat1SWIY1f0cWWsxhAzDF1K5YIY83rT"
file_path_cost = f"https://drive.google.com/uc?export=download&id={file_id_cost}"

try:
    df_cost = pd.read_csv(file_path_cost)
    df_cost = df_cost.dropna(subset=['Country'])
    countries_cost = set(df_cost['Country'].unique())
    print(f"\n1. FILE COST: {len(countries_cost)} quốc gia")
    print("   Các quốc gia:", sorted(countries_cost)[:10], "...")
except Exception as e:
    print(f"\n1. FILE COST: LỖI - {e}")
    countries_cost = set()

# 2. Load Weather
file_id_weather = "12moZNfbEpVNO39HxQXnIPSoM1ItAR-sE"
file_path_weather = f"https://drive.google.com/uc?export=download&id={file_id_weather}"

try:
    df_weather = pd.read_csv(file_path_weather)
    countries_weather = set(df_weather['Country'].unique())
    print(f"\n2. FILE WEATHER: {len(countries_weather)} quốc gia")
    print("   Các quốc gia:", sorted(countries_weather)[:10], "...")
except Exception as e:
    print(f"\n2. FILE WEATHER: LỖI - {e}")
    countries_weather = set()

# 3. Load Review
file_id_review = "1tcsQodOIGlroMDDdfYowl1OHJUSHYrRB"
file_path_review = f"https://drive.google.com/uc?export=download&id={file_id_review}"

try:
    df_review = pd.read_csv(file_path_review)
    countries_review = set(df_review['Country'].unique())
    print(f"\n3. FILE REVIEW: {len(countries_review)} quốc gia")
    print("   Các quốc gia:", sorted(countries_review)[:10], "...")
except Exception as e:
    print(f"\n3. FILE REVIEW: LỖI - {e}")
    countries_review = set()

# Phân tích giao nhau và hiệu
print("\n" + "=" * 60)
print("PHÂN TÍCH GIAO NHAU")
print("=" * 60)

all_countries_intersection = countries_cost & countries_weather & countries_review
print(f"\nQuốc gia có trong CẢ 3 FILE: {len(all_countries_intersection)}")
print(sorted(all_countries_intersection))

# Kiểm tra Switzerland cụ thể
print("\n" + "=" * 60)
print("KIỂM TRA SWITZERLAND/THỤY SĨ")
print("=" * 60)

switzerland_variants = ['Switzerland', 'Thụy Sĩ', 'Swiss', 'switzerland', 'SWITZERLAND']

for variant in switzerland_variants:
    in_cost = variant in countries_cost
    in_weather = variant in countries_weather
    in_review = variant in countries_review
    
    if in_cost or in_weather or in_review:
        print(f"\n'{variant}':")
        print(f"  - Cost: {'✅' if in_cost else '❌'}")
        print(f"  - Weather: {'✅' if in_weather else '❌'}")
        print(f"  - Review: {'✅' if in_review else '❌'}")

# Tìm tên chứa "swit" hoặc "thụy"
print("\n" + "=" * 60)
print("TÌM KIẾM TÊN TƯƠNG TỰ")
print("=" * 60)

def find_similar(countries, keyword):
    return [c for c in countries if keyword.lower() in c.lower()]

print("\nTìm 'swit' trong Cost:", find_similar(countries_cost, 'swit'))
print("Tìm 'swit' trong Weather:", find_similar(countries_weather, 'swit'))
print("Tìm 'swit' trong Review:", find_similar(countries_review, 'swit'))

print("\nTìm 'thụy' trong Cost:", find_similar(countries_cost, 'thụy'))
print("Tìm 'thụy' trong Weather:", find_similar(countries_weather, 'thụy'))
print("Tìm 'thụy' trong Review:", find_similar(countries_review, 'thụy'))

# Quốc gia chỉ trong Weather
only_in_weather = countries_weather - countries_cost - countries_review
print(f"\n\nQuốc gia CHỈ có trong WEATHER ({len(only_in_weather)}):")
print(sorted(only_in_weather))

# Quốc gia chỉ trong Cost
only_in_cost = countries_cost - countries_weather - countries_review
print(f"\n\nQuốc gia CHỈ có trong COST ({len(only_in_cost)}):")
print(sorted(only_in_cost))

# Quốc gia chỉ trong Review
only_in_review = countries_review - countries_weather - countries_cost
print(f"\n\nQuốc gia CHỈ có trong REVIEW ({len(only_in_review)}):")
print(sorted(only_in_review))
