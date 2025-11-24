import streamlit as st
import joblib
import pandas as pd
import time

st.markdown("""
    <style>

    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&family=Montserrat:wght@500;700&display=swap');

    /* Background animated gradient */
    .stApp {
        background: linear-gradient(135deg, #dbeafe, #fde2e4, #e0f7fa);
        background-size: 400% 400%;
        animation: gradientMove 12s ease infinite;
        font-family: 'Poppins', sans-serif;
    }

    @keyframes gradientMove {
        0%{background-position:0% 50%}
        50%{background-position:100% 50%}
        100%{background-position:0% 50%}
    }

    /* Title Styling */
    .title {
        font-size: 45px;
        text-align: center;
        font-weight: 800;
        font-family: 'Montserrat', sans-serif;
        color: #0f172a;
        text-shadow: 2px 2px 6px #cbd5e1;
        letter-spacing: 1px;
    }

    /* Header Colors */
    .section-header {
        font-size: 26px;
        font-weight: 700;
        font-family: 'Montserrat';
        color: #1e3a8a;
        padding: 8px;
        border-left: 6px solid #1e3a8a;
        background: #e0e7ff;
        border-radius: 6px;
        transition: 0.3s;
    }
    .section-header:hover {
        background: #c7d2fe;
        transform: scale(1.02);
    }
    

    /* Processing Animation */
    .processing-box {
        padding: 15px;
        background: #fff3cd;
        border-left: 8px solid #ffcc00;
        animation: fadeBlink 1s infinite;
        border-radius: 10px;
        font-weight: 600;
        color: #664d03;
    }
    @keyframes fadeBlink {
        0% {opacity: 0.4;}
        100% {opacity: 1;}
    }

    /* Bulk Spam / Ham Box */
    .spam-bulk {
        padding: 15px;
        background: #ffe2e2;
        border-left: 8px solid #ff3b3b;
        color: #b30000;
        border-radius: 10px;
        font-weight: 600;
        animation: slideIn 0.5s ease-out;
        font-family: 'Poppins';
    }
    .ham-bulk {
        padding: 15px;
        background: #e1ffe7;
        border-left: 8px solid #2ecc71;
        color: #145a32;
        border-radius: 10px;
        font-weight: 600;
        animation: slideIn 0.5s ease-out;
        font-family: 'Poppins';
    }
    @keyframes slideIn {
        from {transform: translateX(-20px); opacity: 0;}
        to {transform: translateX(0); opacity: 1;}
    }

    </style>
""", unsafe_allow_html=True)
model = joblib.load("spam_sgd_clf.pkl")

st.markdown("<h1 class='title'>📩 Spam Classifier Project</h1>", unsafe_allow_html=True)

with st.expander("ℹ️ About This Project"):
    st.markdown("""
    ### 📌 SMS Spam Detection System  
    This project predicts if a message is **Spam** or **Ham** using:
    - SGD Classifier  
    - CountVectorizer / TF-IDF
    - Streamlit / Pandas / Joblib  

    ### 👨‍💻 Developer  
    **SM Organization**

    📞 Contact Us- 9899XXXXX
    
    📧 Email: your-email@example.com
               
    📍 Located at Noida sec-15  
    """)

col1, col2 = st.columns([2, 3], gap="large")

with col1:

    st.markdown("<div class='section-header'>🔍 Single Msg Prediction</div>", unsafe_allow_html=True)

    text = st.text_input("Enter MSG", placeholder="Type your message here...")

    if st.button("Predict"):
        result = model.predict([text])

        if result == "spam":
            st.markdown("<div class='spam-bulk'>🚨 SPAM DETECTED — Irrelevant ❌</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='ham-bulk'>✔️ SAFE MESSAGE — HAM 😊</div>", unsafe_allow_html=True)

with col2:

    st.markdown("<div class='section-header'>📂 Bulk Msg Prediction</div>", unsafe_allow_html=True)

    file = st.file_uploader("Select file containing bulk msgs", type=['txt', 'csv'])

    if file is not None:

        loading_box = st.empty()

        loading_box.markdown(
            "<div class='processing-box'>📄 Processing your file… Please wait...</div>",
            unsafe_allow_html=True
        )
        time.sleep(1)

        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file, header=None, names=["Msg"])
            else:
                content = file.read().decode("utf-8").splitlines()
                df = pd.DataFrame(content, columns=["Msg"])
        except:
            loading_box.empty()
            st.error("❌ Error reading file. Upload a valid TXT or CSV.")
            st.stop()

        loading_box.empty()

        place = st.empty()
        place.dataframe(df)

        if st.button("Predict", key="b2"):

            with st.spinner("🔍 Predicting messages..."):
                time.sleep(2)

            df['result'] = model.predict(df.Msg)

            for i in range(len(df)):
                msg = df['Msg'][i]
                res = df['result'][i]

                if res == "spam":
                    st.markdown(f"<div class='spam-bulk'>🚨 SPAM: {msg}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='ham-bulk'>✔️ HAM: {msg}</div>", unsafe_allow_html=True)

            place.dataframe(df)

st.markdown(
    "<p style='text-align:center; font-weight:700; font-family:Montserrat;'>Developed ❤️ by <b>SM Organization</b></p>",
    unsafe_allow_html=True
)
