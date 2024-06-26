import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from streamlit_option_menu import option_menu

# Function to upload file
def upload_file():
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            return data
        except Exception as e:
            st.error(f"Error: {e}")
    return None

# Function for exploratory data analysis
def page_exploratory_data_analysis(data):
    st.write("Uploaded Data:")
    st.write(data.head())
    
    st.write("Descriptive Statistics:")
    st.write(data.describe())
    
    st.write("Scatterplot:")
    scatter_cols = st.multiselect("Select columns for Scatterplot", data.columns)
    if len(scatter_cols) == 2:
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x=scatter_cols[0], y=scatter_cols[1], ax=ax)
        st.pyplot(fig)
    
    st.write("Boxplot:")
    boxplot_col = st.selectbox("Select column for Boxplot", data.columns)
    if boxplot_col:
        fig, ax = plt.subplots()
        sns.boxplot(data=data[boxplot_col], ax=ax)
        st.pyplot(fig)
    
    st.write("Pie Chart:")
    pie_col = st.selectbox("Select column for Pie Chart", data.columns)
    if pie_col:
        fig, ax = plt.subplots()
        data[pie_col].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
        st.pyplot(fig)
    
    st.write("Feature Correlation:")
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    corr_data = data[numeric_cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_data, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Function for survival analysis
def page_survival_analysis(data):
    st.write("Define columns for survival analysis:")
    duration_col = st.selectbox("Duration Column", data.columns, key="duration_col")
    event_col = st.selectbox("Event Column (Censored/Not)", data.columns, key="event_col")
    predictor_cols = st.multiselect("Select predictor columns", data.columns, key="predictor_cols")
    
    if duration_col and event_col and predictor_cols:
        try:
            data[duration_col] = pd.to_numeric(data[duration_col], errors='coerce')
            data[event_col] = pd.to_numeric(data[event_col], errors='coerce')
            data = data.dropna(subset=[duration_col, event_col])
            for col in predictor_cols:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            data = data.dropna(subset=predictor_cols)
            data = data.sort_values(by=duration_col)
            
            kmf = KaplanMeierFitter()
            kmf.fit(data[duration_col], event_observed=data[event_col])
            
            st.write("Kaplan-Meier Curve:")
            fig, ax = plt.subplots()
            kmf.plot_survival_function(ax=ax)
            st.pyplot(fig)
            
            st.write("Proportional Hazard Test:")
            cph = CoxPHFitter()
            data_cph = data[[duration_col, event_col] + predictor_cols].rename(columns={duration_col: 'duration', event_col: 'event'})
            cph.fit(data_cph, duration_col='duration', event_col='event')
            st.write(cph.summary)
            
            # Check proportional hazards assumption
            results = proportional_hazard_test(cph, data_cph, time_transform='rank')
            st.write("Proportional Hazards Assumption:")
            st.write(results.summary)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Function for assumption check
def page_assumption_check(data):
    st.write("Assumption check will be implemented here...")

# Main function for sidebar navigation
def main():
    df = upload_file()
    if df is not None:
        with st.sidebar:
            selected = option_menu(
                menu_title='Survival Analysis',
                options=[
                    'Data Exploration',
                    'Survival Analysis',
                ],
                icons=['image', 'bar-chart', 'check2-circle'],
                menu_icon='clock',  # Icon for the main menu
                default_index=0
            )

        if selected == 'Data Exploration':
            page_exploratory_data_analysis(df)
        elif selected == 'Survival Analysis':
            page_survival_analysis(df)
    else:
        st.write("Please upload a data file to proceed.")

if __name__ == "__main__":
    main()
