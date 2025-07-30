# Tool Wear Prediction Using Deep Learning

This project implements a deep learning model to predict tool wear and the current machining process stage based on real-time sensor data from a CNC machine. It serves as a predictive maintenance tool to anticipate tool failure and optimize machining operations.

## Key Features:

*   **Deep Learning Model:** A 1D Convolutional Neural Network (CNN) built with TensorFlow/Keras for time-series data classification.
*   **Web Interface:** A user-friendly web application built with Flask that allows users to upload sensor data in a CSV file and receive a prediction for the current machining process stage.
*   **Data Preprocessing:** Includes scripts for label encoding and feature scaling (`StandardScaler`) to prepare the data for the model.
*   **End-to-End Workflow:** From data loading and preprocessing (`train_model.py`) to model training and deployment in a web app (`app.py`).

## Technology Stack:

*   **Backend:** Python, Flask
*   **Machine Learning:** TensorFlow, Keras, Scikit-learn
*   **Data Handling:** Pandas, NumPy
*   **Frontend:** HTML, CSS

## How It Works:

1.  **Training:** The `train_model.py` script loads the `experiment.csv` dataset, preprocesses the features and labels, builds the CNN model, and trains it. The trained model, scaler, and label encoder are saved in the `model/` directory.
2.  **Prediction:** The Flask application (`app.py`) loads the saved model and preprocessing objects. It provides a web form where a user can upload a CSV file with new sensor data. The application then preprocesses the uploaded data, feeds it to the model, and displays the predicted machining process stage.


