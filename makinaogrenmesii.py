import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df=pd.read_csv('tcmb_aylik_veriler.csv')
df["Tarih"] = pd.to_datetime(df["Tarih"])

sutunlar = [
    'konut_fiyat_endeksi',
    'tuketici_fiyat_endeksi',
    'ito_gecinme_endeksi',
    'reel_kesim_guven_endeksi',
    'istihdam_orani',
    'issizlik_orani',
    'toplam_sifir_konut_satisi',
    'toplam_ikinciel_konut_satisi',
    'resmi_rezerv_varliklari',
    'toplam_binek_otomobil_uretimi'
]
# Bir döngü (loop) kullanarak her sütunu tek tek gez
for sutun in sutunlar:
    # Boş (NaN) değerleri sütunun ortalaması ile doldur
    # inplace=True parametresi, değişikliğin orijinal df'e uygulanmasını sağlar
    df[sutun] = df[sutun].fillna(df[sutun].mean())
# print(df.isnull().sum())

# ------------------------------------------------------------------

# LİNEAR REGRESYON
"""
X=df[["usd_try","tuketici_kredisi_faizi","1_aylik_mevduat_faizi","3_aylik_mevduat_faizi","gbp_try"]]
y=df["tuketici_fiyat_endeksi"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)

# y_pred=model.predict(X_test)

# mae=mean_absolute_error(y_test,y_pred)
# mse=mean_squared_error(y_test,y_pred)
# rmse=np.sqrt(mse)
# r2=r2_score(y_test,y_pred)

# print("Model Performansı")
# print(f"MAE (Ortalama Mutlak Hata)  :{mae:.2f}")
# print(f"MSE (Ortalama Kare Hata)    :{mse:.2f}")
# print(f"RMSE (Karekök Ortalama Hata):{rmse:.2f}")
# print(f"R2 (Doğruluk Skoru)         :{r2:.2f}")

# mean_y = y_test.mean()
# percent_mae = (mae / mean_y) * 100
# percent_rmse = (rmse / mean_y) * 100

# print(f"MAE yüzdesi: {percent_mae:.2f}%")
# print(f"RMSE yüzdesi: {percent_rmse:.2f}%")

# print("Yeni Değerler Giriniz:")
# usd_try=float(input("USD/TRY kuru:"))
# tuketici_kredisi_faizi=float(input("Tüketici Kredisi Faizi (%):"))
# mevduat_1ay=float(input("1 Aylık Mevduat Faizi (%):"))
# mevduat_3ay=float(input("3 Aylık Mevduat Faizi(%):"))
# gbp_try=float(input("GBP/TRY kuru:"))

# yeni_veri=pd.DataFrame([[usd_try,tuketici_kredisi_faizi,mevduat_1ay,mevduat_3ay,gbp_try]],
#                        columns=["usd_try","tuketici_kredisi_faizi","1_aylik_mevduat_faizi","3_aylik_mevduat_faizi","gbp_try"])

# tahmin=model.predict(yeni_veri)[0]

# print(f"\n Tahmin edilen Tğketici Fiyat Endeksi: {tahmin:.2f}")
"""
# --------------------------------------------------------------------

# KARAR AĞAÇLARI
# DecisionTreeRegressor
"""
X = df[[
    "usd_try","eur_try","gbp_try",
    "tuketici_kredisi_faizi","1_aylik_mevduat_faizi","3_aylik_mevduat_faizi",
    "usd_mevduat_faizi","eur_mevduat_faizi",
    "konut_fiyat_endeksi","istihdam_orani","issizlik_orani",
    "resmi_rezerv_varliklari","toplam_binek_otomobil_uretimi",
    "toplam_sifir_konut_satisi","toplam_ikinciel_konut_satisi"
]]
y = df["tuketici_fiyat_endeksi"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar Ağacı Regressor modeli
model = DecisionTreeRegressor(random_state=42, max_depth=5)  # max_depth ile aşırı öğrenmeyi azalt
model.fit(X_train, y_train)

# Tahminler
y_pred = model.predict(X_test)

# Hata metrikleri
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Performansı (Decision Tree)")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.2f}")

# Yüzdelik hata
mean_y = y_test.mean()
percent_mae = (mae / mean_y) * 100
percent_rmse = (rmse / mean_y) * 100
print(f"MAE yüzdesi : {percent_mae:.2f}%")
print(f"RMSE yüzdesi: {percent_rmse:.2f}%")

print("\nYeni değerler giriniz:")
usd_try = float(input("USD/TRY kuru: "))
eur_try = float(input("EUR/TRY kuru: "))
gbp_try = float(input("GBP/TRY kuru: "))
tuketici_kredisi_faizi = float(input("Tüketici Kredisi Faizi (%): "))
mevduat_1ay = float(input("1 Aylık Mevduat Faizi (%): "))
mevduat_3ay = float(input("3 Aylık Mevduat Faizi (%): "))
usd_mevduat_faizi = float(input("USD Mevduat Faizi (%): "))
eur_mevduat_faizi = float(input("EUR Mevduat Faizi (%): "))
konut_fiyat_endeksi = float(input("Konut Fiyat Endeksi: "))
istihdam_orani = float(input("İstihdam Oranı (%): "))
issizlik_orani = float(input("İşsizlik Oranı (%): "))
resmi_rezerv_varliklari = float(input("Resmi Rezerv Varlıkları: "))
toplam_binek_otomobil_uretimi = float(input("Toplam Binek Otomobil Üretimi: "))
toplam_sifir_konut_satisi = float(input("Toplam Sıfır Konut Satışı: "))
toplam_ikinciel_konut_satisi = float(input("Toplam İkinci El Konut Satışı: "))

yeni_veri = pd.DataFrame([[usd_try, eur_try, gbp_try,
                           tuketici_kredisi_faizi, mevduat_1ay, mevduat_3ay,
                           usd_mevduat_faizi, eur_mevduat_faizi,
                           konut_fiyat_endeksi, istihdam_orani, issizlik_orani,
                           resmi_rezerv_varliklari, toplam_binek_otomobil_uretimi,
                           toplam_sifir_konut_satisi, toplam_ikinciel_konut_satisi]],
                          columns=X.columns)

tahmin = model.predict(yeni_veri)[0]
print(f"\nTahmin edilen Tüketici Fiyat Endeksi: {tahmin:.2f}")
"""
# -------------------------------------------------------------------

# DecisionTreeClassifier
"""
# Örnek: TÜFE’yi 3 kategoriye ayır
# düşük, orta, yüksek
df["tuketici_fiyat_kategori"] = pd.qcut(df["tuketici_fiyat_endeksi"], q=3, labels=["Düşük", "Orta", "Yüksek"])
X = df[[
    "usd_try","eur_try","gbp_try",
    "tuketici_kredisi_faizi","1_aylik_mevduat_faizi","3_aylik_mevduat_faizi",
    "usd_mevduat_faizi","eur_mevduat_faizi",
    "konut_fiyat_endeksi","istihdam_orani","issizlik_orani",
    "resmi_rezerv_varliklari","toplam_binek_otomobil_uretimi",
    "toplam_sifir_konut_satisi","toplam_ikinciel_konut_satisi"
]]
y = df["tuketici_fiyat_kategori"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar Ağacı Sınıflandırıcı
clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(X_train, y_train)

# Tahmin
y_pred = clf.predict(X_test)

# Doğruluk ve rapor
acc = accuracy_score(y_test, y_pred)
print(f"Doğruluk Skoru: {acc:.2f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Kullanıcıdan veri alarak tahmin
print("\nYeni değerler giriniz:")
usd_try = float(input("USD/TRY kuru: "))
eur_try = float(input("EUR/TRY kuru: "))
gbp_try = float(input("GBP/TRY kuru: "))
tuketici_kredisi_faizi = float(input("Tüketici Kredisi Faizi (%): "))
mevduat_1ay = float(input("1 Aylık Mevduat Faizi (%): "))
mevduat_3ay = float(input("3 Aylık Mevduat Faizi (%): "))
usd_mevduat_faizi = float(input("USD Mevduat Faizi (%): "))
eur_mevduat_faizi = float(input("EUR Mevduat Faizi (%): "))
konut_fiyat_endeksi = float(input("Konut Fiyat Endeksi: "))
istihdam_orani = float(input("İstihdam Oranı (%): "))
issizlik_orani = float(input("İşsizlik Oranı (%): "))
resmi_rezerv_varliklari = float(input("Resmi Rezerv Varlıkları: "))
toplam_binek_otomobil_uretimi = float(input("Toplam Binek Otomobil Üretimi: "))
toplam_sifir_konut_satisi = float(input("Toplam Sıfır Konut Satışı: "))
toplam_ikinciel_konut_satisi = float(input("Toplam İkinci El Konut Satışı: "))

yeni_veri = pd.DataFrame([[usd_try, eur_try, gbp_try,
                           tuketici_kredisi_faizi, mevduat_1ay, mevduat_3ay,
                           usd_mevduat_faizi, eur_mevduat_faizi,
                           konut_fiyat_endeksi, istihdam_orani, issizlik_orani,
                           resmi_rezerv_varliklari, toplam_binek_otomobil_uretimi,
                           toplam_sifir_konut_satisi, toplam_ikinciel_konut_satisi]],
                          columns=X.columns)

tahmin_kategori = clf.predict(yeni_veri)[0]
print(f"\nTahmin edilen Tüketici Fiyat Endeksi Kategorisi: {tahmin_kategori}")
"""
# ---------------------------------------------------------------------

