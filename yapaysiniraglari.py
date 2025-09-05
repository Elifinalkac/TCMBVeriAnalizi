import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. VERİYİ YÜKLE
df = pd.read_csv("tcmb_aylik_veriler.csv")
df["Tarih"] = pd.to_datetime(df["Tarih"], format="%Y-%m")
df = df.fillna(method="ffill")  # eksik veri varsa doldur

# Hedef değişken
y = df["tuketici_fiyat_endeksi"].values.reshape(-1, 1)

# Özellikler
X = df.drop(columns=["Tarih", "tuketici_fiyat_endeksi"]).values

# 2. EĞİTİM / TEST AYIRIMI
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 3. ÖLÇEKLENDİR
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# 4. MODEL
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dense(1, activation="linear")
])

optimizer = Adam(learning_rate=0.0005)
model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])

# 5. EĞİTİM
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_test_scaled, y_test_scaled),
    epochs=200,
    batch_size=16,
    verbose=1
)

# 6. TAHMİN
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# 7. GELECEK AY
last_data = df.drop(columns=["Tarih", "tuketici_fiyat_endeksi"]).iloc[-1].values.reshape(1, -1)
last_data_scaled = scaler_X.transform(last_data)
future_pred_scaled = model.predict(last_data_scaled)
future_pred = scaler_y.inverse_transform(future_pred_scaled)
print(f"Gelecek Ay TÜFE Tahmini: {future_pred[0][0]:.2f}")
