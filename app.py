import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Used Car Price Prediction", page_icon="", layout="wide")

@st.cache_resource
def load_model():
    if not os.path.exists("car_price_model.pkl"):
        raise FileNotFoundError("car_price_model.pkl not found. Run train_model.py first.")
    if not os.path.exists("label_encoders.pkl"):
        raise FileNotFoundError("label_encoders.pkl not found. Run train_model.py first.")
    with open("car_price_model.pkl","rb") as f:
        model = pickle.load(f)
    with open("label_encoders.pkl","rb") as f:
        encoders = pickle.load(f)
    return model, encoders

@st.cache_data
def load_data():
    if os.path.exists("used_cars.csv"):
        return pd.read_csv("used_cars.csv")
    return pd.DataFrame()

if "history" not in st.session_state:
    st.session_state.history=[]

st.title(" Used Car Price Prediction")

try:
    model, encoders = load_model()
except Exception as e:
    st.error(str(e))
    st.stop()

df = load_data()

st.sidebar.header("Car Details")

brand = st.sidebar.selectbox("Brand", list(encoders["Brand"].classes_))
year = st.sidebar.number_input("Year", 2000, 2035, 2020)
km = st.sidebar.number_input("KM Driven", 0, 500000, 30000)
fuel = st.sidebar.selectbox("Fuel", list(encoders["Fuel"].classes_))
trans = st.sidebar.selectbox("Transmission", list(encoders["Transmission"].classes_))
owner = st.sidebar.selectbox("Owner", list(encoders["Owner"].classes_))
mileage = st.sidebar.number_input("Mileage", 5.0, 40.0, 20.0)
engine = st.sidebar.number_input("Engine (CC)", 800, 5000, 1200)

if st.sidebar.button("Predict Price"):
    try:
        row = pd.DataFrame({
            "Brand":[encoders["Brand"].transform([brand])[0]],
            "Year":[year],
            "KM_Driven":[km],
            "Fuel":[encoders["Fuel"].transform([fuel])[0]],
            "Transmission":[encoders["Transmission"].transform([trans])[0]],
            "Owner":[encoders["Owner"].transform([owner])[0]],
            "Mileage":[mileage],
            "Engine":[engine]
        })
        pred = float(model.predict(row)[0])
        st.success(f"Estimated Price:  {pred:,.0f}")
        st.session_state.history.append({
            "Brand":brand,
            "Year":year,
            "Predicted Price":round(pred,2)
        })
    except Exception as ex:
        st.exception(ex)

tab1, tab2, tab3 = st.tabs(["Prediction History","Dataset","Batch Prediction"])

with tab1:
    if st.session_state.history:
        hist = pd.DataFrame(st.session_state.history)
        st.dataframe(hist, use_container_width=True)
        st.download_button(
            "Download History CSV",
            hist.to_csv(index=False).encode(),
            "prediction_history.csv",
            "text/csv"
        )
    else:
        st.info("No predictions yet.")

with tab2:
    if not df.empty:
        st.dataframe(df.head(), use_container_width=True)
        if "Price" in df.columns:
            st.bar_chart(df.groupby("Brand")["Price"].mean())
    else:
        st.warning("used_cars.csv not found.")

with tab3:
    uploaded = st.file_uploader("Upload CSV")
    if uploaded:
        try:
            batch = pd.read_csv(uploaded)
            req = ["Brand","Year","KM_Driven","Fuel","Transmission","Owner","Mileage","Engine"]
            missing=[c for c in req if c not in batch.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                for c in ["Brand","Fuel","Transmission","Owner"]:
                    batch[c]=encoders[c].transform(batch[c])
                batch["Predicted Price"]=model.predict(batch[req])
                st.dataframe(batch, use_container_width=True)
                st.download_button(
                    "Download Predictions",
                    batch.to_csv(index=False).encode(),
                    "predictions.csv",
                    "text/csv"
                )
        except Exception as e:
            st.exception(e)