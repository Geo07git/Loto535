import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
#from catboost import CatBoostClassifier
from itertools import combinations
from collections import Counter
#import matplotlib.pyplot as plt
import time
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')

# 🔥 Stilizare globală (adăugată imediat după importuri)
st.markdown("""
    <style>
        /* Stilizare buton neon ROȘU */
        div.stButton > button {
            background-color: #FF3131;  /* Roșu aprins */
            color: black;
            font-size: 20px;  /* Mărimea fontului */
            font-family: "Comic Sans MS", sans-serif; /* Font schimbat */
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            box-shadow: 0 0 10px #FF3131, 0 0 20px #FF3131, 0 0 30px #FF3131;
        }
        
        div.stButton > button:hover {
            background-color: #FF5733; /* Roșu spre portocaliu */
            box-shadow: 0 0 20px #FF5733, 0 0 30px #FF5733, 0 0 40px #FF5733;
        }

        /* Stilizare tabel neon ALBASTRU */
        .neon-table {
            border-collapse: collapse;
            width: 100%;
        }
        
        .neon-table th, .neon-table td {
            padding: 10px;
            text-align: center;
            border: 1px solid #000000; /* Cyan Neon */
        }
        
        .neon-table th {
            background-color: #00FFFF; /* Albastru neon */
            color: black;
            font-size: 18px;
            font-family: "Verdana", sans-serif; /* Schimbă fontul tabelului */
            text-shadow: 0 0 10px #00FFFF, 0 0 20px #00FFFF, 0 0 30px #00FFFF;
        }
        
        .neon-table td {
            color: white;
            background-color: #222222;
            text-shadow: 0 0 10px #00FFFF, 0 0 20px #00FFFF, 0 0 30px #00FFFF;
        }
    </style>
""", unsafe_allow_html=True)  # <-- AICI, LA ÎNCEPUTUL CODULUI!

# Configurare UI Streamlit
st.title('LOTO Grecia535 PREDICTION')


# Incaracare date
file_path = '535.csv'
data = pd.read_csv(file_path)
X = data.iloc[:, 0].values.reshape(-1, 1)  # Numarul extragerii
y = data.iloc[:, 1:].values  # Numerele extrase

# Afișați cel mai recent număr de extragere și numerele câștigătoare
most_recent_draw = data.iloc[-1, 0]
most_recent_winning_numbers = data.iloc[-1, 1:].tolist()
#st.write(f"Ultima Extragere: {most_recent_draw} si Numerele Castigatoare: {most_recent_winning_numbers}")

st.markdown(f"""
    <h2 style='color: #FFFF00; font-size: 20px; text-shadow: 0 0 10px #FFFF00, 0 0 20px #FFFF00, 0 0 30px #FFFF00;'>📌 Ultima Extragere: {most_recent_draw} si Numerele Castigatoare: {most_recent_winning_numbers}</h2>
    """, unsafe_allow_html=True)

SEED = 42

# Definim modele si parametrii
models = {
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=SEED),
    'Stacking': StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=10, random_state=SEED)),
            ('dt', DecisionTreeClassifier(random_state=SEED))
        ],
        final_estimator=LogisticRegression()
    ),
    'SVM': SVC(random_state=SEED, probability=True),
    #'CatBoost': CatBoostClassifier(verbose=0, random_state=SEED),
    'MLPClassifier': MLPClassifier(random_state=SEED, max_iter=200),
    'KNN': KNeighborsClassifier(),
}

# Funcție pentru antrenarea modelelor și prezicerea numerelor
def predict_numbers_and_accuracy(models):
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text('Asteptati , va rog...')
    
    model_predictions = {}
    num_models = len(models)
    for i, (model_name, model) in enumerate(models.items(), start=1):
        accuracies = []
        predictions = []
        for SEED in range(42,43):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
            for j in range(y.shape[1]):  
                model.fit(X_train, y_train[:, j])
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test[:, j], y_pred)
                accuracies.append(accuracy)
                next_draw_prediction = model.predict(np.array([[X.max() + 1]]))
                predictions.append(int(next_draw_prediction[0]))
        
        # Asigurați-vă că predicțiile sunt unice
        unique_predictions = list(set(predictions))
        while len(unique_predictions) < 5:
            unique_predictions.append(np.random.choice(list(set(range(1, 36)) - set(unique_predictions))))
        unique_predictions.sort()
        
        mean_accuracy = np.mean(accuracies) * 100  # Conversia preciziei în procent
        
        model_predictions[model_name] = {'Numere prezise': unique_predictions, 'Acuratetea predictiei (%)': mean_accuracy}
        
        # Actualizați bara de progres
        progress_bar.progress(i / num_models)
        time.sleep(0.1)  
    
    status_text.text('GATA!')
    time.sleep(0.5)  
    status_text.empty()  
    progress_bar.empty()  
    
    return model_predictions

# 🔹 Buton pentru generarea seturilor de numere
if st.button('Generează 5 seturi de predictii ML'):
    predictions = predict_numbers_and_accuracy(models)
    predictions_df = pd.DataFrame(predictions).T.reset_index()
    predictions_df.columns = ['Model', 'Numere prezise', 'Acuratetea predictiei']
    predictions_df.index = predictions_df.index + 1
    # Salvăm seturile prezise în session_state pentru a le păstra permanent pe ecran
    st.session_state['saved_predictions'] = predictions_df
    st.table(predictions_df)

    predictions_df.to_csv('predictions_temp.csv', index=False)

import pandas as pd
from collections import Counter

# 🔹 Funcție pentru selecția finală cu XGBoost (fără ast, cu eval)
def predict_final_xgboost():
    try:
        df = pd.read_csv('predictions_temp.csv')
        print("📂 Fișier încărcat cu succes!")
        print("📊 Coloane disponibile:", df.columns)

        # Verifică dacă coloana "Numere prezise" există
        if 'Numere prezise' not in df.columns:
            print("⚠️ Coloana 'Numere prezise' nu există în fișier.")
            return []

        all_numbers = []
        for numere in df['Numere prezise']:
            try:
                num_list = eval(numere)  # ⚠️ Folosim eval() în loc de ast.literal_eval()
                print(f"🔢 Numere extrase: {num_list}")
                all_numbers.extend(num_list)
            except Exception as e:
                print(f"❌ Eroare la evaluarea: {numere} -> {e}")

        if not all_numbers:
            print("⚠️ Nu au fost găsite numere valide.")
            return []

        num_freq = Counter(all_numbers)
        print("📊 Frecvența numerelor:", num_freq)

        final_prediction = [int(num) for num, freq in num_freq.most_common(12)]
        
        print(f"📌 VARIANTA FINALA: {final_prediction}")
        return final_prediction

    except FileNotFoundError:
        print("❌ Fișierul 'predictions_temp.csv' nu a fost găsit.")
        return []
    except Exception as e:
        print(f"❌ Eroare neașteptată: {e}")
        return []



# 🔹 Buton pentru predicția finală
if st.button('Calculează predicția finală'):
    final_numbers = predict_final_xgboost()
    #st.write(f"📌 NUMERELE FINALE PREZISE {final_numbers}")
    #st.write(f"Final Numbers: {final_numbers}")  # Verifică ce este în final_numbers
    st.markdown(f"""
    <h2 style='color: #39FF14; font-size: 25px; text-shadow: 0 0 10px #39FF14, 0 0 20px #39FF14, 0 0 30px #39FF14;'>📌 VARIANTA FINALA: {final_numbers}</h2>
    """, unsafe_allow_html=True)
if 'saved_predictions' in st.session_state:
    st.subheader("📌 Seturile salvate")
    st.table(st.session_state['saved_predictions'])
# 🔹 Vizualizare frecvență numere
#visualize_most_frequent(y)

import streamlit as st

# 🔹 Verificare Numere Extrase
st.subheader("📌 Verifică câte numere din varianta finala au iesit la variantele analizate")

# Preia automat "numerele finale prezise"
if 'final_numbers' in locals():
    user_numbers = final_numbers
else:
    user_numbers = []  # Dacă nu există numere prezise

# Se folosește numărul final prezis automat
if user_numbers:
    # Verificarea potrivirilor pentru 2 până la 5 numere
    matches = {i: 0 for i in range(2, 6)}  # Dicționar pentru a ține numărul de potriviri pentru fiecare valoare între 2 și 5
    total_extrageri = len(data)  # Numărul total de extrageri

    for _, row in data.iterrows():
        extracted_numbers = row[1:].tolist()  # Extrage numerele din rând
        match_count = len(set(user_numbers) & set(extracted_numbers))  # Calculează potrivirile
        if match_count >= 2:  # Verifică doar potrivirile de la 2 în sus
            if match_count <= 10:  # Asigură-te că nu depășești most_common(10)
                matches[match_count] += 1

    # Afișarea rezultatelor
    st.write("📌 Rezultate verificari")
    for i in range(2, 6):  # Afișează rezultatele pentru potrivirile între 2 și 5
        numar_potriviri = matches[i]
        probabilitate = (numar_potriviri / total_extrageri) * 100  # Calculul probabilității în procente
        st.write(f"{i} numere potrivite: {numar_potriviri} ori ({probabilitate:.2f}%)")

st.info("Vor fi doua actualizari pe zi , prima pana la ora 14.45 si a doua pana la ora 18.45.")

# Afișează data și ora curente
tz = pytz.timezone('Europe/Bucharest')
now = datetime.now(tz).strftime("%d-%m-%Y  %H:%M:%S %Z")
#st.write(f"🕒 Actualizat pentru tragerea din {now} ora 15.00")

#st.write(f"🕒 Baza de date a fost actualizata pentru tragerea din {now} ora 15.00") 

st.write(f"## Serviciul este în mentenanta : {now} ") 
