Speech Emotion Recognition – Sound Classification (LSTM)

Dataset used: Toronto Emotional Speech Set (TESS)
Link: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

Project Overview

This project focuses on Speech Emotion Recognition (SER) by classifying audio signals into emotional categories using a Long Short-Term Memory (LSTM) deep learning model. The system processes raw audio files, extracts meaningful acoustic features, and learns temporal dependencies to identify emotions accurately.

Objectives

Build a sound-classification pipeline for emotion recognition

Extract relevant audio features such as MFCC, Chroma, and Mel Spectrograms

Train an LSTM-based neural network model

Evaluate the model using accuracy, loss curves, and confusion matrix

Model Architecture (LSTM)

The model uses an LSTM architecture designed to capture sequential and temporal patterns in audio signals.

Key Components:

Feature extraction using MFCCs

LSTM layers for sequence modeling

Dense layers for final classification

Softmax activation for multi-class outputs

Example Architecture:

Input → LSTM(128) → Dropout → LSTM(64) → Dense(32) → Output(Softmax)

Technologies and Libraries Used
Python Libraries

NumPy – numerical computation

Pandas – dataset handling

Librosa – audio loading and feature extraction

Matplotlib / Seaborn – visualization

Scikit-learn – preprocessing and evaluation

TensorFlow / Keras – LSTM model building and training

Development Tools

Jupyter Notebook or Google Colab

GPU acceleration (optional)

Workflow

Load audio files from the TESS dataset

Extract audio features using Librosa (MFCC, Chroma, Mel Spectrogram)

Preprocess data:

Label encoding

Train-test split

Reshaping for LSTM input

Train the LSTM model

Evaluate model performance using metrics and confusion matrix

Results

(Add your actual model performance here once training is complete.)

Accuracy: TBD

Loss curves: TBD

Repository Structure
|-- data/                 # TESS dataset (not included)
|-- notebooks/            # Jupyter notebooks
|-- models/               # Saved trained models
|-- src/
      ├── preprocessing.py
      ├── feature_extraction.py
      ├── model.py
|-- README.md

Future Improvements

Combine CNN and LSTM for hybrid feature learning

Deploy model using Flask or Streamlit

Enable real-time microphone-based emotion detection

Acknowledgment

Dataset: Toronto Emotional Speech Set (TESS), available via Kaggle.
