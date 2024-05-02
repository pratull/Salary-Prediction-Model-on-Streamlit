import streamlit as st
from data_preprocessing import preprocess_data
from model import build_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Add this import for plotting

def main():
    st.title("Pay Predictor")
    st.markdown("---")

    page = st.sidebar.radio("Navigation", ["Home", "Prediction", "Visualization"])

    if page == "Home":
        st.markdown(
            """
            ## Welcome to Pay Predictor - Salary Prediction App

            This app helps you predict salary based on education, experience, location, job title, age, and gender.

            Use the navigation on the left to explore.
            """
        )

    elif page == "Prediction":
        st.markdown("## Salary Prediction")
        st.write("Fill in the details to predict salary.")
        st.markdown("---")

        file_path = "C:\\Users\\ASUS\\Downloads\\2348445_CIA 3 PROJECT\\salary_prediction_data.csv"
        X, y = preprocess_data(file_path)

        education = st.selectbox("Education", ["Bachelors", "Masters", "PhD"])
        experience = st.slider("Experience (years)", min_value=int(X[:, 1].min()), max_value=int(X[:, 1].max()))
        location = st.selectbox("Location", ["New York", "California", "Texas"])
        job_title = st.selectbox("Job Title", ["Data Scientist", "Software Engineer", "Product Manager"])
        age = st.slider("Age", min_value=int(X[:, 4].min()), max_value=int(X[:, 4].max()))
        gender = st.selectbox("Gender", ["Male", "Female"])

        st.markdown("---")

        if st.button("Predict Salary"):
            education_map = {"Bachelors": 0, "Masters": 1, "PhD": 2}
            location_map = {"New York": 0, "California": 1, "Texas": 2}
            job_title_map = {"Data Scientist": 0, "Software Engineer": 1, "Product Manager": 2}
            gender_map = {"Male": 0, "Female": 1}

            education_encoded = education_map[education]
            location_encoded = location_map[location]
            job_title_encoded = job_title_map[job_title]
            gender_encoded = gender_map[gender]

            user_input = np.array([[education_encoded, experience, location_encoded, job_title_encoded, age, gender_encoded]])

            model = build_model(X, y)
            predicted_salary = model.predict(user_input)

            st.success(f"Predicted Salary: ${predicted_salary[0]:,.2f}")

    elif page == "Visualization":
        st.markdown("## Visualizations")
        st.write("Select a visualization to explore.")
        st.markdown("---")

        # Load and preprocess the data
        file_path = "C:\\Users\\ASUS\\Downloads\\2348445_CIA 3 PROJECT\\salary_prediction_data.csv"
        X, y = preprocess_data(file_path)
        data = pd.read_csv(file_path)  # Load raw data

        # Visualization options
        visualization_option = st.radio("", ["Bar Chart", "Pie Chart", "Scatter Plot"])

        if visualization_option == "Bar Chart":
            st.markdown("### Bar Chart")
            # Bar chart of Education Level counts
            education_counts = data['Education'].value_counts()
            st.bar_chart(education_counts, use_container_width=True)

        elif visualization_option == "Pie Chart":
            st.markdown("### Pie Chart")
            # Pie chart of Gender counts
            gender_counts = data['Gender'].value_counts()
            labels = gender_counts.index
            values = gender_counts.values
            fig, ax = plt.subplots()
            ax.pie(values, labels=labels, autopct='%1.1f%%')
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig)

        elif visualization_option == "Scatter Plot":
            st.markdown("### Scatter Plot")
            # Scatter plot of Salary vs Age
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(data['Age'], data['Salary'], color='blue')
            ax.set_xlabel("Age", fontsize=14)
            ax.set_ylabel("Salary", fontsize=14)
            ax.set_title("Salary vs Age", fontsize=16)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
