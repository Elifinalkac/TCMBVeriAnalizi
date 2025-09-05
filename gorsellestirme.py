import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df=pd.read_csv('tcmb_aylik_veriler.csv')

df['Tarih']=pd.to_datetime(df['Tarih'])

# df.set_index('Tarih',inplace=True)
# df.plot(y='usd_try')
# plt.show()

# df['eur_try'].plot(title='Euro nun zaman içindeki değişmi',xlabel='Tarih',ylabel='Euro')
# plt.show()

# df.set_index('Tarih', inplace=True)
# yillik_ort_euro=df['eur_try'].resample('Y').mean()
# yillik_ort_euro.plot(kind='line', figsize=(10,6), marker='o')
# plt.title("Yıllara Göre Ortalama EUR/TRY")
# plt.xlabel("Yıl")
# plt.ylabel("Ortalama EUR/TRY")
# plt.grid(True)
# plt.show()

# df.set_index('Tarih', inplace=True)
# df[['usd_try','eur_try','gbp_try']].plot(figsize=(12,6))
# plt.title("USD, EUR ve GBP Zaman İçindeki Değişimi")
# plt.ylabel("Kur (TL)")
# plt.xlabel("Tarih")
# plt.grid(True)
# plt.show()

# Yıllık ortalama döviz kurları
# df.set_index('Tarih', inplace=True)
# yillik_ort=df[['usd_try','eur_try','gbp_try']].resample('Y').mean()
# yillik_ort.plot(kind='line',figsize=(10,6),marker='o')
# plt.title("Yıllık Ortalama Döviz Kurları")
# plt.ylabel("Kur (TL)")
# plt.xlabel("Yıl")
# plt.grid(True)
# plt.show()

# Faiz oranları karşılaştırması
# df.set_index('Tarih', inplace=True)
# df[['tuketici_kredisi_faizi','1_aylik_mevduat_faizi','3_aylik_mevduat_faizi']].plot(figsize=(12,6))
# plt.title("Kredi ve Mevzuat Oranları")
# plt.xlabel("Tarih")
# plt.ylabel("Faiz (%)")
# plt.grid(True)
# plt.show()


# df.set_index("Tarih", inplace=True)
# plt.figure(figsize=(10,6))
# plt.plot(df.index, df["tuketici_kredisi_faizi"], label="Tüketici Kredisi Faizi")
# plt.plot(df.index, df["1_aylik_mevduat_faizi"], label="1 Aylık Mevduat")
# plt.plot(df.index, df["3_aylik_mevduat_faizi"], label="3 Aylık Mevduat")
# plt.plot(df.index, df["tuketici_fiyat_endeksi"].pct_change(12)*100, label="Yıllık Enflasyon (TÜFE)", linestyle="--")
# plt.title("Faiz Oranları ve Yıllık Enflasyon")
# plt.ylabel("%")
# plt.legend()
# plt.grid(True)
# plt.show()

# df.set_index("Tarih", inplace=True)
# df["tuketici_kredisi_faizi"].plot(label="Tüketici kredisi faizi")
# df["1_aylik_mevduat_faizi"].plot(label="Tüfe")
# plt.legend()
# plt.title("Faiz Oranları ve Yıllık Enflasyon")
# plt.ylabel("%")
# plt.show()

# df.set_index("Tarih", inplace=True)
# fig, ax1 = plt.subplots(figsize=(10,6))
# ax1.plot(df.index, df["toplam_sifir_konut_satisi"], label="Sıfır Konut Satışı", color="blue")
# ax1.plot(df.index, df["toplam_ikinciel_konut_satisi"], label="İkinci El Konut Satışı", color="green")
# ax1.set_ylabel("Satış Adedi")
# ax2 = ax1.twinx()
# ax2.plot(df.index, df["tuketici_kredisi_faizi"], color="red", label="Kredi Faizi")
# ax2.set_ylabel("Faiz (%)")
# ax1.set_title("Konut Satışları ve Kredi Faizi")
# ax1.legend(loc="upper left")
# ax2.legend(loc="upper right")
# plt.grid(True)
# plt.show()

# df.set_index("Tarih", inplace=True)
# fig, ax1 = plt.subplots(figsize=(10,6))
# ax1.plot(df.index, df["resmi_rezerv_varliklari"], color="blue", label="Rezervler")
# ax1.set_ylabel("Milyar $")
# ax2 = ax1.twinx()
# ax2.plot(df.index, df["usd_try"], color="red", label="USD/TRY")
# ax2.set_ylabel("Kur")
# ax1.set_title("Rezervler ve USD/TRY")
# ax1.legend(loc="upper left")
# ax2.legend(loc="upper right")
# plt.grid(True)
# plt.show()

# df.set_index("Tarih", inplace=True)
# plt.figure(figsize=(10,6))
# plt.plot(df.index, df["istihdam_orani"], label="İstihdam Oranı")
# Burada x eksenini (df.index) ve y eksenini (df["..."]) sen manuel veriyorsun
# plt.plot(df.index, df["issizlik_orani"], label="İşsizlik Oranı")
# plt.title("İstihdam ve İşsizlik Oranları")
# plt.ylabel("%")
# plt.legend()
# plt.grid(True)
# plt.show()

# df.set_index("Tarih", inplace=True)
# plt.figure(figsize=(10,6))
# df["istihdam_orani"].plot(label="İstihdam Oranı")
# # Bu kez pandas’ın kendi plot() metodunu kullanıyorsun.
# # Pandas otomatik olarak df.index’i x ekseni yapıyor, Series değerlerini de y ekseni.
# # O yüzden sen df.index yazmak zorunda kalmıyorsun.
# df["issizlik_orani"].plot(label="issizlik_orani")
# plt.ylabel("%")
# plt.legend()
# plt.grid(True)
# plt.show()

# df.set_index("Tarih", inplace=True)
# corr_cols = ["tuketici_kredisi_faizi","1_aylik_mevduat_faizi","usd_try","eur_try","tuketici_fiyat_endeksi","konut_fiyat_endeksi","resmi_rezerv_varliklari"]
# corr = df[corr_cols].corr()
# plt.figure(figsize=(8,6))
# sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Seçilmiş Göstergeler Arası Korelasyon")
# plt.show()