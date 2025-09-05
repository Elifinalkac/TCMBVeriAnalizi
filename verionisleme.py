import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. VERİYİ YÜKLE
df = pd.read_csv("tcmb_aylik_veriler.csv")

# print("İlk 5 Satır:\n", df.head())
# print("\nVeri Seti Bilgisi:\n")
# print(df.info())

# 2. TARİH SÜTUNUNU DÜZENLE
if "Tarih" in df.columns:
    df["Tarih"] = pd.to_datetime(df["Tarih"], errors="coerce")

# 3. EKSİK DEĞERLERİ DOLDUR
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# 4. YENİ ÖZELLİKLER EKLE (Feature Engineering)
# Aylık değişim oranları
for col in ["tuketici_fiyat_endeksi", "konut_fiyat_endeksi", "usd_try", "eur_try", "gbp_try"]:
    if col in df.columns:
        df[f"{col}_aylik_degisim"] = df[col].pct_change() * 100

# Yıllık (12 aylık) değişim
for col in ["tuketici_fiyat_endeksi", "konut_fiyat_endeksi"]:
    if col in df.columns:
        df[f"{col}_yillik_degisim"] = df[col].pct_change(12) * 100

# 5. ÖLÇEKLENDİRME (Scaling)
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

# 6. KORELASYON ANALİZİ
# plt.figure(figsize=(12, 8))
# sns.heatmap(df[numeric_cols].corr(), annot=False, cmap="coolwarm")
# plt.title("Ekonomik Göstergeler Korelasyon Matrisi")
# plt.show()

# 7. KÜMELEME (K-Means ile Segmentasyon)
X = df[["tuketici_fiyat_endeksi", "konut_fiyat_endeksi", "usd_try", "eur_try", "gbp_try"]].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df["Ekonomi_Küme"] = kmeans.fit_predict(X_scaled)

# plt.figure(figsize=(10, 6))
# plt.scatter(df["usd_try"], df["tuketici_fiyat_endeksi"], c=df["Ekonomi_Küme"], cmap="viridis")
# plt.title("K-Means Kümeleme: USD/TRY vs TÜFE")
# plt.xlabel("USD/TRY")
# plt.ylabel("TÜFE")
# plt.show()

# 8. TEMİZLENMİŞ VERİYİ KAYDET
df.to_csv("tcmb_aylik_veriler_clean.csv", index=False)
print("\nVeri ön işleme tamamlandı. Temizlenmiş ve özellik mühendisliği yapılmış veri kaydedildi.")
