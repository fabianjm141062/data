import pandas as pd

# Contoh data
data = {
    'Jenis_Tanah_Lapis_1': ['Lempung', 'Pasir', 'Lempung'],
    'Kepadatan_Lapis_1': [1.5, 1.6, 1.4],
    'Jenis_Tanah_Lapis_2': ['Pasir', 'Lempung', 'Pasir'],
    'Kepadatan_Lapis_2': [1.7, 1.5, 1.8],
    'Jenis_Tanah_Lapis_3': ['Lempung', 'Pasir', 'Lempung'],
    'Kepadatan_Lapis_3': [1.6, 1.8, 1.7],
    'Diameter_Tiang': [0.5, 0.6, 0.55],
    'Panjang_Tiang': [10, 15, 12],
    'Daya_Dukung': [500, 700, 600]
}

df = pd.DataFrame(data)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Encode categorical data
df_encoded = pd.get_dummies(df, columns=['Jenis_Tanah_Lapis_1', 'Jenis_Tanah_Lapis_2', 'Jenis_Tanah_Lapis_3'])

# Pisahkan fitur dan target
X = df_encoded.drop('Daya_Dukung', axis=1)
y = df_encoded['Daya_Dukung']

# Bagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model
model = LinearRegression()
model.fit(X_train, y_train)
import streamlit as st

# Input dari pengguna
jenis_tanah_lapis_1 = st.selectbox('Jenis Tanah Lapis 1', ['Lempung', 'Pasir'])
kepadatan_lapis_1 = st.number_input('Kepadatan Lapis 1', min_value=0.0, max_value=2.0, step=0.1)
jenis_tanah_lapis_2 = st.selectbox('Jenis Tanah Lapis 2', ['Lempung', 'Pasir'])
kepadatan_lapis_2 = st.number_input('Kepadatan Lapis 2', min_value=0.0, max_value=2.0, step=0.1)
jenis_tanah_lapis_3 = st.selectbox('Jenis Tanah Lapis 3', ['Lempung', 'Pasir'])
kepadatan_lapis_3 = st.number_input('Kepadatan Lapis 3', min_value=0.0, max_value=2.0, step=0.1)
diameter_tiang = st.number_input('Diameter Tiang (m)', min_value=0.0, max_value=1.0, step=0.01)
panjang_tiang = st.number_input('Panjang Tiang (m)', min_value=0, max_value=30, step=1)

# Buat data input model
data_input = {
    'Kepadatan_Lapis_1': [kepadatan_lapis_1],
    'Kepadatan_Lapis_2': [kepadatan_lapis_2],
    'Kepadatan_Lapis_3': [kepadatan_lapis_3],
    'Diameter_Tiang': [diameter_tiang],
    'Panjang_Tiang': [panjang_tiang],
    f'Jenis_Tanah_Lapis_1_{jenis_tanah_lapis_1}': [1],
    f'Jenis_Tanah_Lapis_2_{jenis_tanah_lapis_2}': [1],
    f'Jenis_Tanah_Lapis_3_{jenis_tanah_lapis_3}': [1]
}

df_input = pd.DataFrame(data_input).reindex(columns=X.columns, fill_value=0)

# Prediksi daya dukung
prediksi = model.predict(df_input)[0]

# Tampilkan hasil
st.write(f"Daya Dukung yang Diprediksi: {prediksi:.2f} kN")
