import numpy as np
import pandas as pd

df=pd.read_csv('tcmb_aylik_veriler.csv')
# print(df.head())
# print(df.isnull())
# print(df.isnull().sum())
# print(df.describe())

# print(df[['usd_try','eur_try']])

# df=df.drop('usd_try',axis=1)
# print(df)

# usd ortalama fiyat
# ortama_usd=df['usd_try'].mean()
# print(f"Ortalama Fiyat: {ortama_usd}TL")


# usd nin en yüksek fiyatı
# usd_en_yuksek=df['usd_try'].max()
# print(usd_en_yuksek)

# usd nin en yüksek fiyatı bütün satırı al
# usd_en_yuksek=df.loc[df['usd_try'].idxmax()]
# print(usd_en_yuksek)

# usd nin en yüksek fiyatı bütün satırı al
# usd_en_yuksek=df.loc[df['usd_try'].idxmax()]
# usd_en_yuksek_tarih=usd_en_yuksek['Tarih']
# print(usd_en_yuksek_tarih)

# usd'nin en yüksek olduğu satırı bul ve 'tarih' sütunundaki değeri al
# en_yuksek_kurun_tarihi = df.loc[df['usd_try'].idxmax(), 'Tarih']
# Sonucu ekrana yazdır
# print(en_yuksek_kurun_tarihi)

# usd fiyatı 20Tl yi geçtiğindeki tablo hali
# usd_20tl_gecti=df[df['usd_try']>20]
# print(usd_20tl_gecti)

# df['usd_try']=df['usd_try'].astype('float')
# Boş olan yerleri ortalama ile dolduralım
# df['konut_fiyat_endeksi']=df['konut_fiyat_endeksi'].mean()

# df['tuketici_fiyat_endeksi']=df['tuketici_fiyat_endeksi'].mean()

# df['ito_gecinme_endeksi']=df['ito_gecinme_endeksi'].mean()

# df['reel_kesim_guven_endeksi']=df['ito_gecinme_endeksi'].mean()

# df['istihdam_orani']=df['istihdam_orani'].mean()

# df['issizlik_orani']=df['issizlik_orani'].mean()

# df['toplam_sifir_konut_satisi']=df['toplam_sifir_konut_satisi'].mean()

# df['toplam_ikinciel_konut_satisi']=df['toplam_ikinciel_konut_satisi'].mean()

# df['resmi_rezerv_varliklari']=df['resmi_rezerv_varliklari'].mean()

# df['toplam_binek_otomobil_uretimi']=df['toplam_binek_otomobil_uretimi'].mean()
# print(df.isnull().sum())

# Fonksiyon ile kısa yolu
# Ortalamayla doldurulacak sütunların listesini oluştur
# sutunlar = [
#     'konut_fiyat_endeksi',
#     'tuketici_fiyat_endeksi',
#     'ito_gecinme_endeksi',
#     'reel_kesim_guven_endeksi',
#     'istihdam_orani',
#     'issizlik_orani',
#     'toplam_sifir_konut_satisi',
#     'toplam_ikinciel_konut_satisi',
#     'resmi_rezerv_varliklari',
#     'toplam_binek_otomobil_uretimi'
# ]
# # Bir döngü (loop) kullanarak her sütunu tek tek gez
# for sutun in sutunlar:
#     # Boş (NaN) değerleri sütunun ortalaması ile doldur
#     # inplace=True parametresi, değişikliğin orijinal df'e uygulanmasını sağlar
#     df[sutun] = df[sutun].fillna(df[sutun].mean())
# print(df.isnull().sum())

# Değer bulunmayan tüm satırları silmek isteseydik
# temiz_df=df.dropna()
# diyerek boş değerlerin olduğu satırlar silinirdi

# veya "değer yok" ile doldurmak isteyseydik
# doldurulmus_df=df.fillna("değer yok")
# print(doldurulmus_df)

# Yeni sütun ekleme
# fark=df['eur_try']-df['usd_try']
# df.insert(19,'Yeni Sütun',fark)
# print(df.head())

# eur sıralama büyükten küçüğe
# df_yuksek_eur=df.sort_values(by='eur_try',ascending=False)
# print(df_yuksek_eur.head())

# usd belirli kısımları alma
# df_yuksek_usd=df.sort_values(by='eur_try',ascending=False)
# df_yuksek=df_yuksek_usd[df_yuksek_usd['eur_try']>45]
# print(df_yuksek.head())

# df_filtreli=df[(df['eur_try']>40)& (df['gbp_try']>52)]
# print(df_filtreli)

# df_secili=df.loc[:,['eur_try','gbp_try']]
# # Tüm satırları ve seçili sütunları getirir
# print(df_secili)

# yıllık ortalama usd değeri
# df['Tarih'] = pd.to_datetime(df['Tarih'])
# df_yillik_ortalama = df.groupby(df['Tarih'].dt.year)['usd_try'].mean()
# print(df_yillik_ortalama)

# df['Tarih'] = pd.to_datetime(df['Tarih'])
# toplama_ve_ortalama=df.groupby(df['Tarih'].dt.year).agg(
#     {
#         'usd_try':'mean',
#         'eur_try': 'mean'
#     }
# )
# print(toplama_ve_ortalama)

# usd nin en yüksek olduğu tarihi bulma
# max_usd_tarih = df.loc[df['usd_try'].idxmax(), 'Tarih']
# idxmax() usd_try sütununda en büyük değerin indexini verir.
# .loc[ index , 'Tarih'] o satırdaki Tarih bilgisini alır.
# print(max_usd_tarih)


