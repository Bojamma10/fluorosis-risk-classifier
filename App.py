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
    layout     = "centered"
)

# ── Custom CSS for clean look ─────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .section-title {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 10px;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

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
    return pd.read_csv('district_features.csv')

district_df = load_district_data()

# ── Color Map ─────────────────────────────────────────────────
COLORS = {
    'Safe'      : '#2ecc71',
    'Borderline': '#f39c12',
    'High Risk' : '#e74c3c'
}

# ── Risk Zone Result Display ──────────────────────────────────
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

# ════════════════════════════════════════════════════════════
#  PAGE HEADER
# ════════════════════════════════════════════════════════════
st.title('Fluorosis Risk Zone Classifier')
st.markdown("""
> Predicts whether an Indian district is at **Safe**, **Borderline**, or **High Risk**
> for groundwater fluoride contamination using Machine Learning.

**SDG 3** -- Good Health and Well-Being &nbsp;&nbsp;|&nbsp;&nbsp;
**SDG 6** -- Clean Water and Sanitation
""")
st.divider()

# ════════════════════════════════════════════════════════════
#  SECTION 1 -- DATASET OVERVIEW (4 metrics)
# ════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">Dataset Overview</p>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric('Total Districts',   '307')
c2.metric('States Covered',    '21')
c3.metric('Fluoride Records',  '1,01,041')
c4.metric('Model Accuracy',    '100%')

st.markdown(' ')

# ════════════════════════════════════════════════════════════
#  SECTION 2 -- TWO OVERVIEW CHARTS (side by side, full width)
# ════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">Risk Zone Analysis</p>', unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2)

# Chart 1 -- Pie chart
with chart_col1:
    zone_counts = district_df['risk_zone'].value_counts()
    colors_pie  = [COLORS[z] for z in zone_counts.index]

    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.pie(
        zone_counts.values,
        labels     = zone_counts.index,
        autopct    = '%1.1f%%',
        colors     = colors_pie,
        startangle = 140,
        textprops  = {'fontsize': 12}
    )
    ax1.set_title('Risk Zone Distribution\nacross 307 Districts', fontsize=13, pad=12)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

# Chart 2 -- State-wise High Risk bar chart
with chart_col2:
    high_risk = (
        district_df[district_df['risk_zone'] == 'High Risk']
        .groupby('State Name').size()
        .reset_index(name='Count')
        .sort_values('Count', ascending=True)
    )

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    bars = ax2.barh(high_risk['State Name'], high_risk['Count'],
                    color='#e74c3c', edgecolor='white', linewidth=0.5)
    for bar in bars:
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                 str(int(bar.get_width())), va='center', fontsize=9)
    ax2.set_title('High Risk Districts\nper State', fontsize=13, pad=12)
    ax2.set_xlabel('Number of Districts')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

st.divider()

# ════════════════════════════════════════════════════════════
#  SECTION 3 -- PREDICTION
# ════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">Predict Risk Zone</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(['Predict by District', 'Predict by Manual Input'])

# ── TAB 1 ─────────────────────────────────────────────────────
with tab1:
    st.caption('Select a state and district from the 307 districts in the dataset.')
    st.markdown(' ')

    col1, col2 = st.columns(2)
    with col1:
        states         = sorted(district_df['State Name'].unique())
        selected_state = st.selectbox('Select State', states)
    with col2:
        districts = sorted(
            district_df[district_df['State Name'] == selected_state]['District Name'].unique()
        )
        selected_district = st.selectbox('Select District', districts)

    st.markdown(' ')
    predict_btn = st.button('Predict Risk Zone', key='btn_district', use_container_width=True)

    if predict_btn:
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
        st.markdown(f'#### Results for {selected_district}, {selected_state}')
        st.markdown(' ')

        # 4 metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Affected Villages',    int(row['affected_villages']))
        m2.metric('Affected Blocks',      int(row['affected_blocks']))
        m3.metric('Affected Habitations', int(row['affected_habitations']))
        m4.metric('Coverage Ratio',       f"{row['coverage_ratio']:.3f}")

        st.markdown(' ')

        # Risk zone result -- full width
        display_risk(pred_label)

        st.markdown(' ')

        # Chart -- where does this district sit?
        fig3, ax3 = plt.subplots(figsize=(8, 3))
        ax3.hist(district_df['affected_villages'], bins=30,
                 color='#3498db', edgecolor='white', alpha=0.8, label='All Districts')
        ax3.axvline(int(row['affected_villages']), color='#e74c3c',
                    linewidth=2.5, linestyle='--',
                    label=f'{selected_district}: {int(row["affected_villages"])} villages')
        ax3.set_title('Where does this district sit among all 307 districts?', fontsize=12)
        ax3.set_xlabel('Number of Affected Villages')
        ax3.set_ylabel('Number of Districts')
        ax3.legend(fontsize=10)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        st.markdown(' ')

        # Chart -- comparison within state
        state_data = district_df[
            district_df['State Name'] == selected_state
        ].sort_values('affected_villages', ascending=True)

        bar_colors = [
            '#2c3e50' if d == selected_district else COLORS.get(z, '#bdc3c7')
            for d, z in zip(state_data['District Name'], state_data['risk_zone'])
        ]

        fig4, ax4 = plt.subplots(figsize=(8, max(3, len(state_data) * 0.4)))
        ax4.barh(state_data['District Name'], state_data['affected_villages'],
                 color=bar_colors, edgecolor='white', linewidth=0.5)
        ax4.set_title(
            f'All Districts in {selected_state}\n'
            f'(Dark = {selected_district}, Green = Safe, Orange = Borderline, Red = High Risk)',
            fontsize=11
        )
        ax4.set_xlabel('Number of Affected Villages')
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

# ── TAB 2 ─────────────────────────────────────────────────────
with tab2:
    st.caption('Enter custom values to predict for any district.')
    st.markdown(' ')

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

    st.markdown(' ')
    manual_btn = st.button('Predict Risk Zone', key='btn_manual', use_container_width=True)

    if manual_btn:
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

        st.markdown(' ')

        fig5, ax5 = plt.subplots(figsize=(8, 3))
        ax5.hist(district_df['affected_villages'], bins=30,
                 color='#3498db', edgecolor='white', alpha=0.8, label='All Districts')
        ax5.axvline(aff_villages, color='#e74c3c', linewidth=2.5, linestyle='--',
                    label=f'Your Input: {aff_villages} villages')
        ax5.set_title('Where does your input sit among all 307 districts?', fontsize=12)
        ax5.set_xlabel('Number of Affected Villages')
        ax5.set_ylabel('Number of Districts')
        ax5.legend(fontsize=10)
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    'Fluorosis Risk Zone Classifier | PES University MCA AIML Project 2025 | '
    'K G Bojamma (PES1PG25CA093) | Dataset: India Water Quality Data (data.gov.in)'
)
