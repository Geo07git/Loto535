import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
import random
from datetime import datetime
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pytz
from datetime import datetime
import time
#from xgboost import XGBRegressor

# -----------------------------------
# Scraper from Loto49.ro history
# -----------------------------------
st.set_page_config(page_title="Lotto Romania (Hybrid AI)", layout="wide")

urls = {
    "Loto6/49": "https://www.loto49.ro/arhiva-loto49.php",
    "SuperLoto": "https://www.loto49.ro/arhiva-superloto.php",
    "Joker": "https://www.loto49.ro/arhiva-joker.php"
}
selected_loto = st.selectbox("Select loterry:", list(urls.keys()))

@st.cache_data(ttl=86400)
def fetch_loto49_history(url):
    #url = "https://www.loto49.ro/arhiva-loto49.php"
    #url = "https://www.loto49.ro/arhiva-superloto.php"
    #url = "https://www.loto49.ro/arhiva-joker.php"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Find the history table
        table = soup.find("table")
        rows = table.find_all("tr")[1:]  # skip header row

        results = []
        for row in rows:
            cols = [c.get_text(strip=True) for c in row.find_all("td")]
            if len(cols) >= 2:
                date_str = cols[0]
                nums_str = cols[1]
                draw_date = datetime.strptime(date_str, "%Y-%m-%d")
                numbers = [int(c) for c in cols[1:7] if c.isdigit()]
                results.append({
            "date": draw_date.strftime("%Y-%m-%d"),
            "numbers": numbers
        })
        return results

    except Exception as e:
        st.error(f"Failed to fetch data from manalotto.com: {e}")
        return []

# -----------------------------
# Analysis functions
# -----------------------------
def number_frequencies(results):
    cnt = Counter()
    for entry in results:
        cnt.update(entry["numbers"])
    return cnt

def co_occurrence(results):
    co = defaultdict(Counter)
    for entry in results:
        nums = entry["numbers"]
        for a in nums:
            for b in nums:
                if a != b:
                    co[a][b] += 1
    return co

def calculate_intervals(results):
    """Calculate average gap between draws for each number."""
    intervals = {}
    appearances = defaultdict(list)

    for i, entry in enumerate(sorted(results, key=lambda x: x["date"])):
        for num in entry["numbers"]:
            appearances[num].append(i)

    for num, idxs in appearances.items():
        gaps = [j - i for i, j in zip(idxs, idxs[1:])]
        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            last_seen_gap = (len(results) - 1) - idxs[-1]
            intervals[num] = {"avg_gap": avg_gap, "last_gap": last_seen_gap}
    return intervals

# Define top_n pentru fiecare loto
top_n_dict = {
    "Loto6/49": 6,
    "SuperLoto": 5,
    "Joker": 5
}
# Folosești selecția utilizatorului
top_n = top_n_dict[selected_loto]
# -----------------------------
# Heuristic Prediction
# -----------------------------
def predict_next_numbers(results, top_n=top_n):
    freq = number_frequencies(results)
    most_common = [n for n, _ in freq.most_common(10)]
    return sorted(random.sample(most_common, top_n)) if len(most_common) >= top_n else sorted(most_common)

# -----------------------------
# AI / Agent Prediction
# -----------------------------
def ai_predict_next_numbers(results, co, top_n=top_n, co_threshold=1):
    freq = number_frequencies(results)
    intervals = calculate_intervals(results)

    # 1. Pick "due soon" numbers: last_gap > avg_gap
    due_numbers = [num for num, stats in intervals.items() if stats["last_gap"] >= stats["avg_gap"]]
    due_numbers = sorted(due_numbers, key=lambda n: freq[n], reverse=True)[:2]  # top 2 by frequency

    # 2. Add co-occurring partners
    partner_candidates = []
    for num in due_numbers:
        partner_candidates.extend([n for n, count in co[num].most_common() if count >= co_threshold])
    partner_candidates = [n for n in partner_candidates if n not in due_numbers]  # eliminăm due_numbers duplicate
    partner_candidates = sorted(partner_candidates, key=lambda n: freq[n], reverse=True)  # ordonare după frecvență

    # 3. Fill with top frequency numbers
    freq_candidates = [n for n, _ in freq.most_common(top_n)]

    # Merge into a set
    final_set = list(due_numbers)

    for n in partner_candidates:
        if n not in final_set and len(final_set) < top_n:
            final_set.append(n)
    
    for n in freq_candidates:
        if n not in final_set and len(final_set) < top_n:
            final_set.append(n)

    return sorted(list(final_set))[:top_n]

# -----------------------------
# Machine Learning Part
# -----------------------------
def prepare_dataset(history):
    loto_classes = {
        "Loto6/49": list(range(1, 50)),
        "SuperLoto": list(range(1, 41)),
        "Joker": list(range(1, 46))
    }
    mlb = MultiLabelBinarizer(classes=loto_classes[selected_loto])
    X = []
    y = []
    for i in range(len(history) - 1):
        X.append(history[i]["numbers"])
        y.append(history[i + 1]["numbers"])
    X_enc = mlb.fit_transform(X)
    y_enc = mlb.fit_transform(y)
    return X_enc, y_enc, mlb

def train_model(history):
    X, y, mlb = prepare_dataset(history)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="micro")

    joblib.dump((model, mlb, acc, f1), "lotto_model.pkl")
    return model, mlb, (acc, f1)

def load_model():
    try:
        model, mlb, acc, f1 = joblib.load("lotto_model.pkl")
        return model, mlb, (acc, f1)
    except:
        return None, None, (None, None)

def ml_prediction(history, top_n=top_n):
    model, mlb, _ = load_model()
    if model is None:
        model, mlb, _ = train_model(history)

    last_draw = history[-1]["numbers"]
    X_last = mlb.transform([last_draw])
    probs = model.predict_proba(X_last)

    # Extract probability of "1" for each number
    avg_probs = []
    for p in probs:
        if p.shape[1] == 2:  # ambele clase 0 și 1
            avg_probs.append(p[0, 1])
        else:  # doar o singură coloană (există doar clasa 0)
            avg_probs.append(0.0)   
    avg_probs = np.array(avg_probs)
    top_indices = np.argsort(avg_probs)[-top_n:]
    prediction = mlb.classes_[top_indices]
    return sorted(prediction.tolist())

# -----------------------------
# Streamlit Application
# -----------------------------
st.title("🎰 Lotto Romania Hybrid AI Predictor")

# Fetch data
with st.spinner("Fetching draw history from Loto49.ro..."):
    data = fetch_loto49_history(urls[selected_loto])
    df = pd.DataFrame(data)

if df.empty:
    st.error("No draw history available.")
    st.stop()

df["date"] = pd.to_datetime(df["date"])

# Date range filter
default_start = df["date"].min().date().replace(day=1)  # First day of month
default_end = df["date"].max().date()
start_date, end_date = st.date_input("Select date range", [default_start, default_end])
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)
df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

# Show history
st.subheader("📅 Draw History")
df_display = df.copy()
df_display["numbers_str"] = df_display["numbers"].apply(lambda nums: " - ".join(f"{n:02d}" for n in nums))
st.dataframe(df_display[["date", "numbers_str"]].sort_values("date", ascending=False))

# Frequency chart
st.subheader("📊 Number Frequencies")
freq = number_frequencies(df.to_dict("records"))
freq_df = pd.DataFrame(sorted(freq.items()), columns=["Number", "Frequency"]).set_index("Number")
st.bar_chart(freq_df)

st.subheader("🔗 Numbers that co-occur with each other")
co = co_occurrence(df.to_dict("records"))

selected = st.number_input("Select a number", min_value=1, max_value=49, value=22)

# Buton pentru afișare tabel
if st.button("Show Co-occurrence Table"):
    if selected in co:
        co_df = pd.DataFrame(co[selected].items(), columns=["Partner", "Times"]).sort_values("Times", ascending=False)
        st.table(co_df)
    else:
        st.write("No data for that number in selected range.")

# 1️⃣ Heuristic Predictions
st.subheader("✨ Heuristic Predictions")
heuristic_preds = []
for i in range(1):
    pred = predict_next_numbers(df.to_dict("records"), top_n=top_n)
    heuristic_preds.append(pred)
    st.success(f"Heuristic Prediction {i+1}: {pred}")

# 2️⃣ AI/Agent Predictions
st.subheader("🤖 AI/Agent Predictions")
ai_preds = []
for i in range(1):
    pred = ai_predict_next_numbers(df.to_dict("records"), co, top_n=top_n)
    ai_preds.append(pred)
    st.info(f"AI Agent Prediction {i+1}: {pred}")

# 3️⃣ ML Model Predictions
st.subheader("🧠 ML Model Predictions")
ml_preds = []
for i in range(1):
    pred = ml_prediction(df.to_dict("records"), top_n=top_n)
    ml_preds.append(pred)
    st.warning(f"ML Prediction {i+1}: {pred}")

# Combine All Numbers into a Final Variant

all_numbers = set()

# Flatten Heuristic
for variant in heuristic_preds:
    if isinstance(variant, list):
        all_numbers.update(variant)
    else:
        all_numbers.add(variant)

# Flatten AI/Agent
for variant in ai_preds:
    if isinstance(variant, list):
        all_numbers.update(variant)
    else:
        all_numbers.add(variant)

# Flatten ML
for variant in ml_preds:
    if isinstance(variant, list):
        all_numbers.update(variant)
    else:
        all_numbers.add(variant)

# Sort and display final variant
final_variant = sorted(all_numbers)
st.subheader("🎯 Final Combined Prediction")
st.success(f"Numbers: {final_variant}")

if st.button("Train / Retrain Model"):
    with st.spinner("Training AI model..."):
        model, mlb, (acc, f1) = train_model(df.to_dict("records"))
    st.success(f"Model retrained successfully! ✅ Accuracy: {acc:.2f}, F1: {f1:.2f}")

# Afișează data și ora curente
tz = pytz.timezone('Europe/Bucharest')
now = datetime.now(tz).strftime("%d-%m-%Y")#  %H:%M:%S %Z")
#st.write(f"🕒 Actualizat pentru tragerea din {now} ora 15.00")

st.subheader(f"🕒 Baza de date a fost actualizata pentru tragerile din {now}") 

#st.write(f"🛠️ Serviciul este în mentenanta : {now} ") 



#streamlit==1.38.0
#pandas==2.2.2
#requests==2.32.3
#beautifulsoup4==4.12.3
#scikit-learn==1.5.2
#joblib==1.4.2
#numpy==1.26.4



