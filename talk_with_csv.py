# Import necessary libraries
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cohere
import numpy as np
import streamlit as st

# Set your Cohere API key
co = cohere.Client("k4zNlpCx6RXdgvIOfEk2l66fnNiwAEfK1aGJuhwo")

    
# Define the function to ask Cohere
def ask_cohere(question, context):
    response = co.generate(
        model="command-xlarge-nightly",  # Use a valid model ID
        prompt=f"Question: {question}\nContext: {context}\nAnswer:",
        max_tokens=150,
        temperature=0.5,
    )
    return response.generations[0].text.strip()


# Load CSV Data
@st.cache_data  # Use st.cache_data for caching loaded data
def load_data(filename):
    try:
        df = pd.read_csv(filename)
        return df
    except pd.errors.ParserError as e:
        st.error(f"Error parsing CSV file: {e}")
        return pd.DataFrame()


# Statistical Analysis Function (Handles data type mismatches)
def calculate_statistics(df, columns):
    numeric_df = df[columns].select_dtypes(include=[np.number])
    stats = {
        "Mean": numeric_df.mean(),
        "Median": numeric_df.median(),
        "Mode": numeric_df.mode().iloc[0],
        "Standard Deviation": numeric_df.std(),
        "Correlation": numeric_df.corr(),
    }
    return stats


# Plot Generation Function (Handles non-numeric data)
def generate_plot(plot_type, df, columns, fig=None):
    if fig is None:
        fig, ax = plt.subplots()

    if plot_type == "histogram":
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].plot(kind="hist", title=f"Histogram of {col}", ax=ax)
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
            else:
                st.warning(f"Histogram is not suitable for non-numeric column: {col}")

    elif plot_type == "scatter" and len(columns) == 2:
        if all(pd.api.types.is_numeric_dtype(df[col]) for col in columns):
            ax.scatter(df[columns[0]], df[columns[1]])
            ax.set_xlabel(columns[0])
            ax.set_ylabel(columns[1])
            ax.set_title(f"Scatter Plot of {columns[0]} vs {columns[1]}")
            st.pyplot(fig)
        else:
            st.warning("Scatter plot requires both columns to be numeric")

    elif plot_type == "line":
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].plot(kind="line", title=f"Line Plot of {col}", ax=ax)
                ax.set_xlabel("Index")
                ax.set_ylabel(col)
                st.pyplot(fig)
            else:
                st.warning(f"Line plot is not suitable for non-numeric column: {col}")

    elif plot_type == "distribution":
        for col in columns:
            df[col].value_counts().plot(
                kind="bar", title=f"Distribution of {col}", ax=ax
            )
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            st.pyplot(fig)

    else:
        st.warning(f"Plot type '{plot_type}' is not yet supported.")


# Main User Interface (Streamlit)
st.set_page_config(page_title="👨‍💻 Talk with your CSV 👨‍💻")
st.title("👨‍💻 Talk with your CSV 👨‍💻")

# Upload CSV with unique key
uploaded_file = st.file_uploader("Upload a CSV file:", type=".csv", key="data_upload")
df = pd.DataFrame()  # Create an empty DataFrame initially

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write(df)  # Display the loaded DataFrame
    st.success("File uploaded successfully!")  # Add a success message

# User input for statistical analysis
columns_for_stats = st.multiselect("Select columns for statistics:", df.columns)
if columns_for_stats:
    stats = calculate_statistics(df, columns_for_stats)
    st.write("*Statistical Summary*")
    for col_name, stat_dict in stats.items():
        st.write(f"{col_name}:")
        for stat_name, stat_value in stat_dict.items():
            st.write(f"\t- {stat_name}: {stat_value}")

# User input for generating plots
plot_type = st.selectbox(
    "Choose plot type:", ["histogram", "scatter", "line", "distribution"]
)
columns_for_plot = st.multiselect("Select columns for plotting:", df.columns)
if plot_type and columns_for_plot:
    generate_plot(plot_type, df, columns_for_plot)

# User queries using Cohere (Prompt)
user_query = st.text_input("Ask a question about the data:", "")
if user_query and uploaded_file is not None:
    # Preprocess user query (extract keywords)
    keywords = re.findall(r"\w+", user_query)  # Extract alphanumeric tokens

    # Structure the context based on the CSV data
    num_columns = df.select_dtypes(include=[np.number]).columns
    context = f"""
    The dataset contains the following numerical columns: {', '.join(num_columns)}.
    Here are some statistics:
    - Number of rows: {len(df)}
    - Mean values: {df[num_columns].mean().to_dict()}
    - Median values: {df[num_columns].median().to_dict()}
    - Mode values: {df[num_columns].mode().iloc[0].to_dict()}
    - Standard deviation: {df[num_columns].std().to_dict()}
    """

    answer = ask_cohere(user_query, context)
    st.write("**Response:**")
    st.write(answer)
