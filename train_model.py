import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.utils import to_categorical
import os

# Load dataset
df = pd.read_csv('experiment.csv')

# Feature and label separation
X = df.drop('Machining_Process', axis=1).values
y = df['Machining_Process']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for CNN
X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)

# Build model
model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_reshaped.shape[1], 1)))
model.add(Conv1D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(y_categorical.shape[1], activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save model & preprocessing tools
os.makedirs('model', exist_ok=True)
model.save('model/cnn_model.h5')
np.save('model/scaler.npy', scaler.mean_)
np.save('model/scaler_std.npy', scaler.scale_)
np.save('model/label_classes.npy', label_encoder.classes_)
