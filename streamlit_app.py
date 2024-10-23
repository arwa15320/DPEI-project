import streamlit as st
import pandas as pd
import pickle
import gdown
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set(style="whitegrid")

# Function to download and load the model using gdown
def load_model_from_drive(file_id):
    output = 'vehicle_price_model.pkl'
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output, quiet=False)
        with open(output, 'rb') as file:
            model = pickle.load(file)
        if isinstance(model, RandomForestRegressor):
            return model
        else:
            st.error("Loaded model is not a RandomForestRegressor.")
            return None
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

# Preprocess the input data
def preprocess_input(data, model):
    input_df = pd.DataFrame(data, index=[0])  # Create DataFrame with an index
    # One-Hot Encoding for categorical features based on the training model's features
    input_df_encoded = pd.get_dummies(input_df, drop_first=True)

    # Reindex to ensure it matches the model's expected input
    model_features = model.feature_names_in_  # Get the features used during training
    input_df_encoded = input_df_encoded.reindex(columns=model_features, fill_value=0)  # Fill missing columns with 0
    return input_df_encoded

# Visualize price distribution
def visualize_price_distribution(df):
    st.subheader("📊 Price Distribution in Dataset")
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Price'], bins=30, kde=True, color='#61a5c2')
    plt.title("Vehicle Price Distribution", fontsize=16)
    plt.xlabel("Price ($)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True)
    st.pyplot(plt.gcf())

# Visualize the relationship between Year and Price
def visualize_year_vs_price(df):
    st.subheader("📉 Year vs. Price in Dataset")
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x='Year', y='Price', data=df, color='#ff6f61', edgecolor='black')
    plt.title("Vehicle Year vs. Price", fontsize=16)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Price ($)", fontsize=14)
    plt.grid(True)
    st.pyplot(plt.gcf())

# Sidebar for styling and better layout
def render_sidebar():
    st.sidebar.markdown("### 🔧 App Controls")
    st.sidebar.write("Adjust the input values to predict the vehicle price.")
    st.sidebar.markdown("---")
    st.sidebar.write("Visualizations will help you explore the dataset trends.")
    st.sidebar.markdown("---")

# Main Streamlit app
def main():
    st.set_page_config(page_title="Vehicle Price Prediction", layout="wide", page_icon="🚗")

    st.title("🚗 Vehicle Price Prediction App")
    st.write("Enter the vehicle details below to predict its price and explore related visualizations.")
    
    # Add Sidebar
    render_sidebar()

    # Upload CSV file
    uploaded_file = st.file_uploader("📂 Upload the Australian Vehicle Prices CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)  # Load the uploaded CSV file

        # Display dataset visualizations
        with st.beta_expander("Explore Dataset Visualizations"):
            visualize_price_distribution(df)
            visualize_year_vs_price(df)

        # Create input fields for all required features
        st.markdown("### Vehicle Input Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            year = st.number_input("Year", min_value=1900, max_value=2024, value=2020)
            engine = st.number_input("Engine Size (L)", min_value=0.0, value=2.0)
            drive_type = st.selectbox("Drive Type", ["FWD", "RWD", "AWD"])
        
        with col2:
            used_or_new = st.selectbox("Used or New", ["Used", "New"])
            transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])
        
        with col3:
            fuel_consumption = st.number_input("Fuel Consumption (L/100km)", min_value=0.0, value=8.0)
            kilometres = st.number_input("Kilometres", min_value=0, value=50000)
            cylinders_in_engine = st.number_input("Cylinders in Engine", min_value=1, value=4)
            body_type = st.selectbox("Body Type", ["Sedan", "SUV", "Hatchback", "Coupe", "Convertible"])
            doors = st.selectbox("Number of Doors", [2, 3, 4, 5])

        # Button for prediction
        if st.button("🚀 Predict Price"):
            file_id = '11btPBNR74na_NjjnjrrYT8RSf8ffiumo'  # Google Drive file ID
            model = load_model_from_drive(file_id)

            if model is not None:
                # Preprocess the user input
                input_data = {
                    'Year': year,
                    'UsedOrNew': used_or_new,
                    'Transmission': transmission,
                    'Engine': engine,
                    'DriveType': drive_type,
                    'FuelType': fuel_type,
                    'FuelConsumption': fuel_consumption,
                    'Kilometres': kilometres,
                    'CylindersinEngine': cylinders_in_engine,
                    'BodyType': body_type,
                    'Doors': doors
                }
                input_df = preprocess_input(input_data, model)

                try:
                    # Make the prediction
                    prediction = model.predict(input_df)

                    # Display the result
                    st.subheader("💵 Predicted Price:")
                    st.write(f"**${prediction[0]:,.2f}**")

                    # Visualize the predicted price compared to the dataset
                    st.subheader("📊 Predicted Price vs Dataset Prices")
                    fig, ax = plt.subplots()
                    sns.histplot(df['Price'], bins=30, kde=True, color='#61a5c2', label='Dataset Prices', ax=ax)
                    ax.axvline(x=prediction[0], color='red', linestyle='--', label='Predicted Price')
                    ax.set_title('Predicted Price Compared to Dataset Prices', fontsize=16)
                    ax.set_xlabel('Price ($)', fontsize=14)
                    ax.set_ylabel('Frequency', fontsize=14)
                    ax.legend()
                    plt.grid(True)
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
            else:
                st.error("Failed to load the model.")
    else:
        st.info("Please upload the dataset to start.")

if __name__ == "__main__":
    main()
