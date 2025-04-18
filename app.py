import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Wind Power Forecasting", layout="wide")

st.title("🌬️ Wind Power Generation Forecasting")

st.sidebar.header("📂 Upload CSV Files (4 Locations)")
uploaded_files = []
for i in range(1, 5):
    uploaded_file = st.sidebar.file_uploader(f"Upload Location{i}.csv", type="csv", key=f"file_{i}")
    if uploaded_file:
        uploaded_files.append((uploaded_file, f"Location{i}"))

if len(uploaded_files) == 4:
    dfs = []
    for file, name in uploaded_files:
        df = pd.read_csv(file)
        df['Location'] = name
        dfs.append(df)

    merged_data = pd.concat(dfs, ignore_index=True)
    st.success("✅ All 4 files uploaded and merged successfully!")

    st.subheader("📊 Preview of Merged Dataset")
    st.dataframe(merged_data.head())

    with st.expander("📈 Dataset Summary"):
        st.write("**Info:**")
        buffer = []
        merged_data.info(buf=buffer.append)
        st.text('\n'.join(buffer))

        st.write("**Statistical Summary:**")
        st.dataframe(merged_data.describe().T)

        st.write("**Missing Values:**")
        st.write(merged_data.isnull().sum())

        st.write("**Duplicate Rows:**")
        st.write(merged_data.duplicated().sum())

    st.subheader("🔁 Encoding 'Location' Column")
    encoded_data = pd.get_dummies(merged_data, columns=['Location'], drop_first=True)
    st.dataframe(encoded_data.head())

    st.download_button("⬇️ Download Cleaned Dataset", data=encoded_data.to_csv(index=False), file_name="cleaned_data.csv")

    st.subheader("📊 Data Visualization")

    numeric_cols = encoded_data.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns available for plotting.")
    else:
        # Correlation Heatmap
        with st.expander("🔗 Correlation Heatmap"):
            st.write("Correlation Matrix:")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(encoded_data[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

        # Boxplot for outlier detection
        with st.expander("📦 Boxplot (Outlier Detection)"):
            selected_col = st.selectbox("Select a numeric column for boxplot", numeric_cols, key="boxplot")
            fig, ax = plt.subplots()
            sns.boxplot(data=encoded_data, y=selected_col, ax=ax)
            st.pyplot(fig)

        # Distribution Plot
        with st.expander("📈 Distribution Plot"):
            selected_col = st.selectbox("Select a column for distribution", numeric_cols, key="distplot")
            fig, ax = plt.subplots()
            sns.histplot(encoded_data[selected_col], kde=True, ax=ax)
            st.pyplot(fig)

        # Line plot over time (if datetime column exists)
        date_col = None
        for col in encoded_data.columns:
            if pd.api.types.is_datetime64_any_dtype(encoded_data[col]):
                date_col = col
                break

        if date_col:
            with st.expander("🕒 Line Plot Over Time"):
                target_col = st.selectbox("Select numeric column to plot over time", numeric_cols, key="lineplot")
                fig, ax = plt.subplots()
                encoded_data = encoded_data.sort_values(by=date_col)
                ax.plot(encoded_data[date_col], encoded_data[target_col])
                ax.set_xlabel("Time")
                ax.set_ylabel(target_col)
                st.pyplot(fig)

        # Custom plot selector
        with st.expander("🎛️ Custom Plot"):
            plot_type = st.selectbox("Choose plot type", ["Histogram", "Line", "Scatter"])
            x_col = st.selectbox("X-axis", encoded_data.columns)
            y_col = st.selectbox("Y-axis", encoded_data.columns)

            fig, ax = plt.subplots()
            if plot_type == "Histogram":
                sns.histplot(data=encoded_data, x=x_col, ax=ax)
            elif plot_type == "Line":
                ax.plot(encoded_data[x_col], encoded_data[y_col])
            elif plot_type == "Scatter":
                sns.scatterplot(data=encoded_data, x=x_col, y=y_col, ax=ax)
            st.pyplot(fig)

else:
    st.warning("Please upload all 4 CSV files to continue.")
