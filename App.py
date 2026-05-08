# -*- coding: utf-8 -*-
# ============================================================
# Fluorosis Risk Zone Classifier -- Streamlit Web App
# Student : K G Bojamma | SRN : PES1PG25CA093
# SDG 3 (Good Health) | SDG 6 (Clean Water)
# PES University | MCA AIML Project 2025
# ============================================================

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
import matplotlib.pyplot as plt
import joblib

# ── Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title = "Fluorosis Risk Zone Classifier",
    page_icon  = "water",
    layout     = "wide"
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

# ── Color Map ─────────────────────────────────────────────────
COLORS = {
    'Safe'      : '#2ecc71',
    'Borderline': '#f39c12',
    'High Risk' : '#e74c3c'
}

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

# ── Chart 1: Risk Zone Pie Chart ──────────────────────────────
def plot_risk_pie():
    zone_counts = district_df['risk_zone'].value_counts()
    colors = [COLORS[z] for z in zone_counts.index]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        zone_counts.values,
        labels    = zone_counts.index,
        autopct   = '%1.1f%%',
        colors    = colors,
        startangle= 140,
        textprops = {'fontsize': 11}
    )
    ax.set_title('Risk Zone Distribution\n(307 Districts)', fontsize=12, pad=10)
    plt.tight_layout()
    return fig

# ── Chart 2: District Comparison Bar Chart ───────────────────
def plot_district_comparison(selected_state, selected_district, selected_value):
    state_data = district_df[district_df['State Name'] == selected_state].copy()
    state_data = state_data.sort_values('affected_villages', ascending=True)

    fig, ax = plt.subplots(figsize=(7, max(4, len(state_data) * 0.35)))

    bar_colors = []
    for _, row in state_data.iterrows():
        if row['District Name'] == selected_district:
            bar_colors.append('#2c3e50')
        else:
            bar_colors.append(COLORS.get(row['risk_zone'], '#bdc3c7'))

    ax.barh(state_data['District Name'], state_data['affected_villages'],
            color=bar_colors, edgecolor='white', linewidth=0.5)

    ax.set_title(f'Districts in {selected_state}\n(Dark bar = selected district)',
                 fontsize=11, pad=10)
    ax.set_xlabel('Number of Affected Villages')
    plt.tight_layout()
    return fig

# ── Chart 3: State-wise High Risk Count ──────────────────────
def plot_state_risk():
    high_risk = (
        district_df[district_df['risk_zone'] == 'High Risk']
        .groupby('State Name')
        .size()
        .reset_index(name='High Risk Districts')
        .sort_values('High Risk Districts', ascending=True)
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(high_risk['State Name'], high_risk['High Risk Districts'],
            color='#e74c3c', edgecolor='white', linewidth=0.5)

    for i, val in enumerate(high_risk['High Risk Districts']):
        ax.text(val + 0.05, i, str(val), va='center', fontsize=9)

    ax.set_title('High Risk Districts per State', fontsize=12, pad=10)
    ax.set_xlabel('Number of High Risk Districts')
    plt.tight_layout()
    return fig

# ── Chart 4: Affected Villages Distribution ───────────────────
def plot_village_distribution(selected_value):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(district_df['affected_villages'], bins=30,
            color='#3498db', edgecolor='white', linewidth=0.5, alpha=0.8)
    ax.axvline(selected_value, color='#e74c3c', linewidth=2,
               linestyle='--', label=f'Selected: {selected_value}')
    ax.set_title('Distribution of Affected Villages\nacross All Districts', fontsize=11)
    ax.set_xlabel('Affected Villages')
    ax.set_ylabel('Number of Districts')
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig

# ════════════════════════════════════════════════════════════
#  APP LAYOUT
# ════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────
st.title('Fluorosis Risk Zone Classifier')
st.markdown("""
Predicts whether an Indian district is at **Safe**, **Borderline**, or **High Risk**
for groundwater fluoride contamination using Machine Learning.

**SDG 3** -- Good Health and Well-Being  |  **SDG 6** -- Clean Water and Sanitation  |
**Dataset:** 1,01,041 fluoride records | 307 districts | 21 states
""")
st.divider()

# ── Overview Section ──────────────────────────────────────────
st.subheader('Dataset Overview')

col1, col2, col3, col4 = st.columns(4)
col1.metric('Total Districts',  '307')
col2.metric('States Covered',   '21')
col3.metric('Affected Villages','1,01,041')
col4.metric('Model Accuracy',   '100%')

st.markdown(' ')

# Overview charts side by side
ov_col1, ov_col2, ov_col3 = st.columns(3)

with ov_col1:
    st.pyplot(plot_risk_pie())

with ov_col2:
    st.pyplot(plot_state_risk())

with ov_col3:
    # Placeholder until user selects district
    st.markdown('##### Affected Villages Distribution')
    st.caption('Select a district below to see where it falls in the distribution.')
    placeholder_fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(district_df['affected_villages'], bins=30,
            color='#3498db', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Affected Villages')
    ax.set_ylabel('Number of Districts')
    ax.set_title('All 307 Districts')
    plt.tight_layout()
    st.pyplot(placeholder_fig)
    plt.close()

st.divider()

# ── Prediction Section ────────────────────────────────────────
st.subheader('Predict Risk Zone')

tab1, tab2 = st.tabs(['Predict by District', 'Predict by Manual Input'])

# ── TAB 1 ─────────────────────────────────────────────────────
with tab1:
    st.caption('Select a state and district from the 307 districts in the dataset.')

    t1_col1, t1_col2 = st.columns(2)
    with t1_col1:
        states = sorted(district_df['State Name'].unique())
        selected_state = st.selectbox('Select State', states)
    with t1_col2:
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

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Affected Villages',    int(row['affected_villages']))
        m2.metric('Affected Blocks',      int(row['affected_blocks']))
        m3.metric('Affected Habitations', int(row['affected_habitations']))
        m4.metric('Coverage Ratio',       f"{row['coverage_ratio']:.3f}")

        st.divider()

        # Risk result
        res_col, chart_col = st.columns([1, 1])
        with res_col:
            display_risk(pred_label)
        with chart_col:
            st.pyplot(plot_village_distribution(int(row['affected_villages'])))
            plt.close()

        # District comparison chart
        st.markdown(f'#### How {selected_district} compares to other districts in {selected_state}')
        st.pyplot(plot_district_comparison(selected_state, selected_district,
                                           int(row['affected_villages'])))
        plt.close()

# ── TAB 2 ─────────────────────────────────────────────────────
with tab2:
    st.caption('Enter custom values to predict for any district.')

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

        res_col, chart_col = st.columns([1, 1])
        with res_col:
            display_risk(pred_label)
        with chart_col:
            st.pyplot(plot_village_distribution(aff_villages))
            plt.close()

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    'Fluorosis Risk Zone Classifier | PES University MCA AIML Project 2025 | '
    'K G Bojamma (PES1PG25CA093) | Dataset: India Water Quality Data (data.gov.in)'
)
