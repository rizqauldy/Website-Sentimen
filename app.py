import streamlit as st
import joblib
import re

# 1. Memuat (Load) Model dan Vectorizer yang sudah disimpan
model = joblib.load('model_sentimen.pkl')
vectorizer = joblib.load('vectorizer_sentimen.pkl')

# 2. Fungsi Pembersih Teks (Sama persis seperti di Kaggle)
def bersihkan_teks(teks):
    teks = teks.lower()
    teks = re.sub(r'<.*?>', ' ', teks)
    teks = re.sub(r'[^a-z0-9\s]', '', teks)
    return teks

# 3. Mendesain Tampilan Website dengan Streamlit
st.title("🎬 Aplikasi Pendeteksi Sentimen Film")
st.write("Masukkan ulasan film (dalam bahasa Inggris) di bawah ini untuk mengetahui apakah sentimennya Positif atau Negatif.")

# Membuat kotak input teks
input_user = st.text_area("Ketik ulasan di sini:", height=150)

# Membuat tombol untuk memulai prediksi
if st.button("Deteksi Sentimen"):
    if input_user:
        # Proses di balik layar saat tombol diklik
        teks_bersih = bersihkan_teks(input_user)
        teks_vektor = vectorizer.transform([teks_bersih])
        hasil_prediksi = model.predict(teks_vektor)[0]
        
        # Menampilkan hasil ke layar
        if hasil_prediksi == 'positive':
            st.success("✨ Hasil: Sentimen POSITIF")
        else:
            st.error("😡 Hasil: Sentimen NEGATIF")
    else:
        st.warning("Teks ulasan tidak boleh kosong!")