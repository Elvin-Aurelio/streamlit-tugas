import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Config Halaman
st.set_page_config(
    page_title="Prediksi Biaya Asuransi - Regresi Linear",
    layout="wide"
)

st.title("Analisis & Prediksi Biaya Asuransi (Linear Regression)")
st.write("Aplikasi interaktif untuk melatih model Linear Regression dan memprediksi biaya asuransi berdasarkan variabel input.")

# 1. Load Data
@st.cache_data
def load_data():
    # Pastikan file insurance.csv berada di direktori yang sama
    df = pd.read_csv('insurance.csv')
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Gagal memuat dataset 'insurance.csv'. Pastikan file berada di folder yang sama. Error: {e}")
    st.stop()

# Sidebar - Dataset Info & Filter
st.sidebar.header("Pengaturan Model & Data")

if st.sidebar.checkbox("Tampilkan Preview Data"):
    st.subheader("Preview Dataset")
    st.dataframe(df.head(10))

# 2. Preprocessing Data
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Fitur dan Target
X = df_encoded.drop(columns=['charges'])
y = df_encoded['charges']

# Test Split Slider
test_size = st.sidebar.slider("Ukuran Data Uji (Test Size %)", min_value=10, max_value=40, value=20, step=5) / 100.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# 3. Model Training
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Metrik Evaluasi
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display Metrik
st.subheader("Evaluasi Performa Model")
col1, col2, col3, col4 = st.columns(4)
col1.metric("R² Score", f"{r2:.4f}")
col2.metric("RMSE", f"${rmse:,.2f}")
col3.metric("MAE", f"${mae:,.2f}")
col4.metric("MSE", f"{mse:,.0f}")

st.divider()

# 4. Visualisasi Interaktif
st.subheader("Visualisasi Interaktif")

tab1, tab2, tab3 = st.tabs(["Nilai Aktual vs Prediksi", "Koefisien Fitur", "Eksplorasi Variabel (Scatter Plot)"])

with tab1:
    st.write("### Grafik Aktual vs Prediksi Biaya Asuransi")
    fig_pred = px.scatter(
        x=y_test, 
        y=y_pred, 
        labels={'x': 'Nilai Aktual (Actual Charges)', 'y': 'Nilai Prediksi (Predicted Charges)'},
        title="Perbandingan Nilai Aktual vs Prediksi",
        opacity=0.7,
        trendline="ols"
    )
    # Garis Ideal
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig_pred.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Garis Ideal (Perfect Fit)', line=dict(color='red', dash='dash')))
    st.plotly_chart(fig_pred, use_container_width=True)

with tab2:
    st.write("### Pengaruh / Bobot Masing-Masing Fitur (Koefisien Regresi)")
    coef_df = pd.DataFrame({
        'Fitur': X.columns,
        'Koefisien': model.coef_
    }).sort_values(by='Koefisien', ascending=True)
    
    fig_coef = px.bar(
        coef_df, 
        x='Koefisien', 
        y='Fitur', 
        orientation='h',
        title="Koefisien Model Regresi Linear",
        color='Koefisien',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_coef, use_container_width=True)

with tab3:
    st.write("### Hubungan Antar Variabel Dataset")
    col_x = st.selectbox("Pilih Variabel X:", options=['age', 'bmi', 'children'], index=0)
    col_color = st.selectbox("Pilih Kategori (Warna):", options=['smoker', 'sex', 'region'], index=0)
    
    fig_scatter = px.scatter(
        df, 
        x=col_x, 
        y='charges', 
        color=col_color,
        title=f"Hubungan Antara {col_x.capitalize()} dan Charges berdasarkan {col_color.capitalize()}",
        trendline="ols"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# 5. Form Prediksi Biaya Asuransi Interaktif
st.subheader("Simulator Prediksi Biaya Asuransi")
st.write("Masukkan data di bawah untuk memprediksi perkiraan biaya asuransi:")

col_in1, col_in2, col_in3 = st.columns(3)

with col_in1:
    input_age = st.number_input("Umur (Age)", min_value=18, max_value=100, value=30)
    input_sex = st.selectbox("Jenis Kelamin", options=['male', 'female'])

with col_in2:
    input_bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    input_children = st.number_input("Jumlah Anak", min_value=0, max_value=10, value=0)

with col_in3:
    input_smoker = st.selectbox("Perokok?", options=['no', 'yes'])
    input_region = st.selectbox("Wilayah (Region)", options=['southwest', 'southeast', 'northwest', 'northeast'])

# Konversi Input Pengguna ke Format yang Dibutuhkan Model
input_data = pd.DataFrame([{
    'age': input_age,
    'bmi': input_bmi,
    'children': input_children,
    'sex_male': 1 if input_sex == 'male' else 0,
    'smoker_yes': 1 if input_smoker == 'yes' else 0,
    'region_northwest': 1 if input_region == 'northwest' else 0,
    'region_southeast': 1 if input_region == 'southeast' else 0,
    'region_southwest': 1 if input_region == 'southwest' else 0
}])

# Pastikan urutan kolom sesuai saat training
input_data = input_data.reindex(columns=X.columns, fill_value=0)

if st.button("Hitung Prediksi Biaya", type="primary"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimasi Biaya Asuransi: ${max(0, prediction):,.2f}")