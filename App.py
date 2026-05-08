# -*- coding: utf-8 -*-
# ============================================================
# Fluorosis Risk Zone Classifier -- Streamlit Web App
# Student : K G Bojamma | SRN : PES1PG25CA093
# SDG 3 (Good Health) | SDG 6 (Clean Water)
# PES University | MCA AIML Project 2025
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title = "Fluorosis Risk Zone Classifier",
    page_icon  = "water",
    layout     = "centered"
)

# ── Load Model and Encoders ──────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load('best_model.pkl')
    le_state = joblib.load('le_state.pkl')
    le_zone  = joblib.load('le_zone.pkl')
    return model, le_state, le_zone

model, le_state, le_zone = load_artifacts()

# ── Load District Reference Data ─────────────────────────────
@st.cache_data
def load_district_data():
    df = pd.read_csv('district_features.csv')
    return df

district_df = load_district_data()

# ── Helper: Risk Zone Display ─────────────────────────────────
def display_risk(zone_label):
    if zone_label == 'Safe':
        st.success('SAFE ZONE')
        st.markdown("""
        **Fluoride contamination is LOW in this district.**

        - Affected villages are below the safe threshold
        - Groundwater is relatively safe for drinking
        - Routine monitoring is still recommended
        """)
    elif zone_label == 'Borderline':
        st.warning('BORDERLINE ZONE')
        st.markdown("""
        **Fluoride contamination is MODERATE in this district.**

        - This district is in the warning range
        - Regular water testing is strongly advised
        - Defluoridation measures should be considered
        """)
    else:
        st.error('HIGH RISK ZONE')
        st.markdown("""
        **Fluoride contamination is HIGH in this district.**

        - Immediate intervention is recommended
        - Jal Shakti Ministry / CGWB should be notified
        - Defluoridation plants should be prioritised here
        - Children are especially vulnerable to fluorosis
        """)

# ── App Header ────────────────────────────────────────────────
st.title('Fluorosis Risk Zone Classifier')
st.markdown("""
Predicts whether an Indian district is at **Safe**, **Borderline**, or **High Risk**
for groundwater fluoride contamination using Machine Learning.

**SDG 3** -- Good Health and Well-Being  |  **SDG 6** -- Clean Water and Sanitation
""")
st.divider()

# ── Two Tabs ──────────────────────────────────────────────────
tab1, tab2 = st.tabs(['Predict by District', 'Predict by Manual Input'])

# ────────────────────────────────────────────────────────────
# TAB 1 -- Select a known State + District from the dataset
# ────────────────────────────────────────────────────────────
with tab1:
    st.subheader('Select a State and District')
    st.caption('These are the 307 districts from the India Water Quality dataset.')

    # State dropdown
    states = sorted(district_df['State Name'].unique())
    selected_state = st.selectbox('Select State', states)

    # District dropdown filtered by state
    districts = sorted(
        district_df[district_df['State Name'] == selected_state]['District Name'].unique()
    )
    selected_district = st.selectbox('Select District', districts)

    if st.button('Predict Risk Zone', key='btn_district'):

        row = district_df[
            (district_df['State Name']    == selected_state) &
            (district_df['District Name'] == selected_district)
        ].iloc[0]

        features = pd.DataFrame([{
            'affected_villages'   : row['affected_villages'],
            'affected_blocks'     : row['affected_blocks'],
            'affected_habitations': row['affected_habitations'],
            'affected_panchayats' : row['affected_panchayats'],
            'coverage_ratio'      : row['coverage_ratio'],
            'state_encoded'       : row['state_encoded']
        }])

        pred_encoded = model.predict(features)[0]
        pred_label   = le_zone.inverse_transform([pred_encoded])[0]

        st.divider()
        st.markdown(f'### Results for **{selected_district}**, {selected_state}')

        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Affected Villages',    int(row['affected_villages']))
        col2.metric('Affected Blocks',      int(row['affected_blocks']))
        col3.metric('Affected Habitations', int(row['affected_habitations']))
        col4.metric('Coverage Ratio',       f"{row['coverage_ratio']:.3f}")

        st.divider()
        display_risk(pred_label)

# ────────────────────────────────────────────────────────────
# TAB 2 -- Enter custom values manually
# ────────────────────────────────────────────────────────────
with tab2:
    st.subheader('Enter District Features Manually')
    st.caption('Use this to predict for any district by entering known values.')

    col1, col2 = st.columns(2)

    with col1:
        aff_villages    = st.number_input('Affected Villages',    min_value=1,   value=50)
        aff_blocks      = st.number_input('Affected Blocks',      min_value=1,   value=5)
        aff_habitations = st.number_input('Affected Habitations', min_value=1,   value=60)

    with col2:
        aff_panchayats = st.number_input('Affected Panchayats',  min_value=1,   value=20)
        coverage       = st.number_input('Coverage Ratio',       min_value=0.0,
                                         max_value=1.0, value=0.10, step=0.01,
                                         format='%.3f')
        state_manual   = st.selectbox('State', sorted(le_state.classes_), key='manual_state')

    if st.button('Predict Risk Zone', key='btn_manual'):

        state_enc = le_state.transform([state_manual])[0]

        features = pd.DataFrame([{
            'affected_villages'   : aff_villages,
            'affected_blocks'     : aff_blocks,
            'affected_habitations': aff_habitations,
            'affected_panchayats' : aff_panchayats,
            'coverage_ratio'      : coverage,
            'state_encoded'       : state_enc
        }])

        pred_encoded = model.predict(features)[0]
        pred_label   = le_zone.inverse_transform([pred_encoded])[0]

        st.divider()
        display_risk(pred_label)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    'Fluorosis Risk Zone Classifier | PES University MCA AIML Project 2025 | '
    'K G Bojamma (PES1PG25CA093) | Dataset: India Water Quality Data (data.gov.in)'
)
