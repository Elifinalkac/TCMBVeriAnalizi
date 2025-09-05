import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. VERİYİ YÜKLE
df = pd.read_csv("tcmb_aylik_veriler.csv")
df["Tarih"] = pd.to_datetime(df["Tarih"], format="%Y-%m")
df = df.fillna(method="ffill")

# Hedef değişken (sadece TÜFE)
data = df["tuketici_fiyat_endeksi"].values.reshape(-1, 1)

# 2. ÖLÇEKLENDİR
scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(data)

# 3. ZAMAN SERİSİ SETİ OLUŞTUR
def create_sequences(dataset, time_step=12):
    X, y = [], []
    for i in range(len(dataset)-time_step):
        X.append(dataset[i:(i+time_step), 0])  # geçmiş 12 ay
        y.append(dataset[i+time_step, 0])      # 13. ay (çıktı)
    return np.array(X), np.array(y)

time_step = 12  # 12 aylık geçmişe bakarak tahmin
X, y = create_sequences(data_scaled, time_step)

# LSTM girişi için 3D reshape (örnek, zaman adımı, özellik sayısı)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Eğitim / Test ayırımı (%80 eğitim)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 4. LSTM MODELİ
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(time_step,1)),
    Dropout(0.2),
    LSTM(32),
    Dense(1)  # tek değer tahmin
])

model.compile(loss="mse", optimizer="adam")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    verbose=1
)

# 5. TAHMİN
y_pred = model.predict(X_test)

# Orijinal ölçeğe döndür
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
y_pred_inv = scaler.inverse_transform(y_pred)

# 6. HATA ÖLÇÜMÜ
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# 7. GELECEK AY TAHMİNİ
last_sequence = data_scaled[-time_step:].reshape(1, time_step, 1)
future_pred = model.predict(last_sequence)
future_pred_inv = scaler.inverse_transform(future_pred)
print(f"Gelecek Ay TÜFE Tahmini (LSTM): {future_pred_inv[0][0]:.2f}")

# 8. GRAFİK
plt.plot(y_test_inv, label="Gerçek")
plt.plot(y_pred_inv, label="Tahmin")
plt.legend()
plt.show()
