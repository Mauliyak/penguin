import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from sklearn.naive_bayes import GaussianNB

st.write("""
# Penguin Prediction App
This app predicts the **Palmer Penguin** species!
Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")

img = Image.open('lter_penguins.png')
img = img.resize((700, 418))
st.image(img, use_column_width=False)

img2 = Image.open('culmen_depth.png')
img2 = img2.resize((700, 451))
st.image(img2, use_column_width=False)

st.sidebar.header('Parameter Inputan')

# Upload File CSV untuk parameter inputan
uplouad_file = st.sidebar.file_uploader("Upload file CSV Anda", type=["csv"])
if uplouad_file is not None:
    inputan = pd.read_csv(uplouad_file)
else:
    def input_user():
        pulau = st.sidebar.selectbox('Pulau', ('Biscoe', 'Dream', 'Torgersen'))
        gender = st.sidebar.selectbox('Gender',('Pria', 'Wanita'))
        panjang_paruh = st.sidebar.selectbox('Panjang Paruh (mm)', 32.1,59.6,43.9)
        kedalaman_paruh = st.sidebar.selectbox('Kedalaman Paruh (mm)',13.1,21.5,17.2)
        panjang_sirip = st.sidebar.selectbox('Panjang Sirip (mm)',172,0,231.0,201.0)
        masa_tubuh = st.sidebar.selectbox('Masa Tubuh (g)',2700.0,6300.0,4207.0)
        data = {'pulau' : pulau,
                'panjang_paruh_mm' : panjang_paruh,
                'kedalaman_paruh_mm' : kedalaman_paruh,
                'panjang_sirip_mm' : panjang_sirip,
                'masa_tubuh_g' : masa_tubuh,
                'gender' : gender}
        fitur = pd.DataFrame(data, index=[0])
        return fitur
    inputan = input_user()

# Menggabungkan inputan data dataset penguin
penguin_raw = pd.read_csv('penguins_example.csv')
penguins = penguin_raw.drop(columns=['jenis'])
df = pd.concat([inputan, penguins], axis=0)

# Encode untuk fitur ordinal
encode = ['gender', 'pulau']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1] #ambil barisan pertama (input dan user)

# Menampilkan parameter hasil inputan
st.subheader('parameter Inputan')

if uplouad_file is not None:
    st.write(df)
else:
    st.write('Menunggu file csv untuk di upload. saat ini memakai sampel inputan(seperti tampilan di bawah).')
    st.write(df)

#
load_model = pickle.load(open('penguins_clf.pkl','rb'))

#Terapan model 
prediksi = load_model.predict(df)
prediksi_proba = load_model.predict_proba(df)

st.subheader('Keterangan Label Kelas')
jenis_penguin = np.array(['Adelie','Chinstrap','Gento'])
st.write(jenis_penguin)

st.subheader('Hasil perediksi (Klasifikasi penguin)')
st.write(jenis_penguin[prediksi])

st.subheader('Probalitas Hasil Prediksi (kalsifikasi penguin)')
st.write(prediksi_proba)
