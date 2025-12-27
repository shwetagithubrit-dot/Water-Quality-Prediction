import streamlit as st
import pandas as pd
import pickle
import numpy as np
import base64
import plotly.express as px 
st.set_page_config(
    page_title="Streamlined Water Potability Predictor",
    layout="wide",
    initial_sidebar_state="collapsed" 
)
IMAGE_URLS = {
    "importance": "https://th.bing.com/th/id/OIP.4e6qbH5l993ugLtUf5QrDQHaE7?o=7rm=3&rs=1&pid=ImgDetMain&o=7&rm=3", 
    "model": "https://www.dasca.org/content/images/main/the-predictive-modeling-process.jpg", 
    "applications": "https://tse4.mm.bing.net/th/id/OIP.OHltWu_D1uAlmfIb8AK72wHaE8?rs=1&pid=ImgDetMain&o=7&rm=3" 
}
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        /* Base Streamlit App Styling */
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        /* Apply semi-transparent white background to content area for best readability */
        .main, .css-18e3th9, .css-1d391kg {{
            background-color: rgba(255, 255, 255, 0.95);
            padding: 25px; 
            border-radius: 12px;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.2); 
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 24px; /* Space out the tabs */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_background("https://www.elegantthemes.com/blog/wp-content/uploads/2013/09/bg-9-full.jpg")
#Load the Model
@st.cache_resource 
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        st.sidebar.success("Machine Learning Model Loaded Successfully!")
        return model
    except FileNotFoundError:
        st.sidebar.error("Error: 'model.pkl' not found. Please ensure it's in the same directory.")
        return None
model = load_model()

FEATURE_NAMES = [
    'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
]

INPUT_DETAILS = {
    'ph': ('pH value (0-14)', 7.0, 0.0, 14.0, 'Measures acidity/alkalinity.'),
    'Hardness': ('Hardness (mg/L)', 196.0, 0.0, 500.0, 'Concentration of dissolved minerals.'),
    'Solids': ('Total Solids (ppm)', 25000.0, 0.0, 65000.0, 'Total dissolved solids (TDS).'),
    'Chloramines': ('Chloramines (ppm)', 7.0, 0.0, 20.0, 'Disinfectant added to water.'),
    'Sulfate': ('Sulfate (mg/L)', 333.0, 0.0, 500.0, 'Naturally occurring chemical.'),
    'Conductivity': ('Conductivity ($\mu$S/cm)', 430.0, 0.0, 1000.0, 'Ability of water to conduct electricity.'),
    'Organic_carbon': ('Organic Carbon (ppm)', 14.0, 0.0, 30.0, 'Total organic carbon (TOC) measurement.'),
    'Trihalomethanes': ('Trihalomethanes ($\mu$g/L)', 66.0, 0.0, 150.0, 'Byproduct of chlorine disinfection.'),
    'Turbidity': ('Turbidity (NTU)', 4.0, 0.0, 10.0, 'Cloudiness or haziness of water.'),
}
if 'latest_prediction' not in st.session_state:
    st.session_state['latest_prediction'] = None

def project_introduction_page():
    st.title("💧 𝓦𝓪𝓽𝓮𝓻 𝓠𝓾𝓪𝓵𝓲𝓽𝔂 𝓟𝓻𝓮𝓭𝓲𝓬𝓽𝓲𝓸𝓷")

    st.write("") 
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<h2 style='text-align:center;'>Importance</h2>", unsafe_allow_html=True)
        st.image(
            "https://sinay.ai/wp-content/uploads/2022/08/shutterstock_1785736046-1024x622.jpg",
            use_container_width=True,
            caption="Water Treatment & Environmental Health"
        )
        st.markdown("""
        <p style='text-align:justify;'>
        <b>Water quality prediction</b> is crucial because it provides early warnings of contamination, shifting management from reactive fixes to proactive prevention. This technology protects <b>public health, optimizes treatment costs, and safeguards environmental ecosystems</b> by ensuring a sustainable supply of safe water.
        </p>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("<h2 style='text-align:center;'>overview</h2>", unsafe_allow_html=True)
        st.image(
            "https://sinay.ai/wp-content/uploads/2021/05/Sensor-water-1-1024x576.jpg",
            use_container_width=True,
            caption="Water Quality Prediction Overview"
        )
        st.markdown("""
        <p style='text-align:justify;'>
        <b>Water quality prediction</b> involves using scientific data such as <b>pH, turbidity, dissolved oxygen, temperature, and other chemical indicators</b> to forecast the future condition of water. Modern prediction systems use machine learning models that analyze data to determine whether water will be safe or polluted.This approach provides a faster, more accurate, and cost-effective way to monitor water compared to manual testing alone..
        </p>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("<h2 style='text-align:center;'>Applications</h2>", unsafe_allow_html=True)
        st.image(
            "https://as1.ftcdn.net/v2/jpg/09/38/94/66/1000_F_938946640_wP3vkh8QZUmGUgvsfAy9S7JUpselk2Se.jpg",
            use_container_width=True,
            caption="Application and Real-Time Use"
        )
        st.markdown("""
        <p style='text-align:justify;'>
        <b>Water quality prediction</b> is used in many real-world areas, such as <b>monitoring drinking water sources, managing wastewater treatment plants, and predicting pollution in natural water bodies like rivers, ponds, and lakes</b>. It helps in agriculture by ensuring safe irrigation water and supports industries in meeting environmental standards.
        </p>
        """, unsafe_allow_html=True)
def prediction_page():
    st.header("🧪 Prediction Input")
    st.markdown("Enter the 9 key quality parameters below to analyze potability.")

    if model is None:
        st.warning("Cannot run predictions because the model failed to load.")
        return
    with st.form(key='prediction_form'):
        cols = st.columns(3)
        user_inputs = {}
        
        for i, (feature, (label, default, min_val, max_val, desc)) in enumerate(INPUT_DETAILS.items()):
            
            with cols[i % 3]:
                key_name = f"input_val_{feature}"

                if key_name not in st.session_state:
                    st.session_state[key_name] = default
                
                if feature == 'ph':
                    user_inputs[feature] = st.slider(
                        label,
                        min_value=min_val,
                        max_value=max_val,
                        value=st.session_state[key_name],
                        step=0.01,
                        format="%.2f",
                        key=key_name
                    )
                else:
                    user_inputs[feature] = st.number_input(
                        label,
                        min_value=min_val,
                        max_value=max_val,
                        value=st.session_state[key_name],
                        step=0.1,
                        format="%.2f",
                        key=key_name
                    )
                
                with st.expander("Details"):
                    st.caption(f"**Range:** {min_val:.2f} to {max_val:.2f}")
                    st.caption(desc)
                
        st.markdown("---")
        
        col_pred, col_clear = st.columns([1, 1])
        
        predict_button = col_pred.form_submit_button("Analyze Water Potability", type="primary", use_container_width=True)
        def clear_form_values():
            for feature, (_, default, _, _, _) in INPUT_DETAILS.items():
                st.session_state[f"input_val_{feature}"] = default
            
        col_clear.form_submit_button("Clear Form", on_click=clear_form_values, use_container_width=True)
    if predict_button:
        # 1. Create a DataFrame from the inputs
        current_inputs = {f: user_inputs[f] for f in FEATURE_NAMES}
        input_data = pd.DataFrame([current_inputs])
        input_data = input_data[FEATURE_NAMES] # Re-order columns

        try:
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            proba_potable = prediction_proba[1]
            proba_not_potable = prediction_proba[0]
            
            st.session_state['latest_prediction'] = {
                'prediction': prediction,
                'proba_potable': proba_potable,
                'proba_not_potable': proba_not_potable,
                'inputs': current_inputs
            }
            
            st.toast("Prediction complete! Go to the 'Result Analysis' tab to view.", icon='✅')

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.exception(e)
            st.session_state['latest_prediction'] = None # Clear result on failure


def result_page():
    st.header("📊 Result Analysis")
    result = st.session_state['latest_prediction']
    if result is None:
        st.warning("Please navigate to the **Prediction Input** tab and run an analysis first.")
        return
    
    prediction = result['prediction']
    proba_potable = result['proba_potable']
    proba_not_potable = result['proba_not_potable']
   
    st.subheader("Final Potability Determination")
    if prediction == 1:
        st.balloons()
        st.success(f"## 🎉 Prediction: Potable (Safe to Drink)", icon="✅")
    else:
        st.error(f"## 🛑 Prediction: Not Potable (Unsafe to Drink)", icon="❌")

    st.markdown("---")

    col_conf, col_chart = st.columns([2, 3])
    
    with col_conf:
        st.subheader("Model Confidence")
        st.metric(label="Potable Confidence", value=f"{proba_potable * 100:.2f}%")
        st.progress(proba_potable, text=f"Potability Confidence: {proba_potable * 100:.2f}%")

    with col_chart:
        df_proba = pd.DataFrame({
            'Outcome': ['Potable', 'Not Potable'],
            'Confidence': [proba_potable, proba_not_potable]
        })
        fig_proba = px.bar(df_proba, x='Outcome', y='Confidence', 
                            color='Outcome', 
                            color_discrete_map={'Potable': 'green', 'Not Potable': 'red'},
                            title="Confidence Breakdown",
                            height=250)
        fig_proba.update_layout(showlegend=False, margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig_proba, use_container_width=True)

    st.markdown("---")
    st.subheader("Input Values Used for Analysis")
    input_data_display = {
        INPUT_DETAILS[k][0]: f"{v:.2f}" for k, v in result['inputs'].items()
    }
    df_inputs = pd.DataFrame(input_data_display.items(), columns=['Feature', 'Value'])
    st.table(df_inputs.set_index('Feature'))

def main():
    tab_intro, tab_predict, tab_result = st.tabs([
        "💧 Project Introduction", 
        "🧪 Prediction Input", 
        "📊 Result Analysis"
    ])

    with tab_intro:
        project_introduction_page()
    with tab_predict:
        prediction_page()
    with tab_result:
        result_page()

if __name__ == "__main__":
    for feature, (_, default, _, _, _) in INPUT_DETAILS.items():
        if f"input_val_{feature}" not in st.session_state:
            st.session_state[f"input_val_{feature}"] = default
            
    main()