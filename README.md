# Speech Emotion Recognition — Sound Classification (LSTM)

Dataset (research source)
https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

Overview
--------
This project implements Speech Emotion Recognition (SER) — classifying emotional states from short speech recordings — using a recurrent neural network (LSTM). The work focuses on extracting robust audio features (e.g., MFCCs, chroma, mel spectrogram features) and feeding them to an LSTM-based classifier to model temporal patterns in speech for emotion recognition.

Why LSTM?
---------
Speech signals are time-series data with temporal dependencies. Long Short-Term Memory (LSTM) networks capture long-range dependencies and dynamics in sequential data, making them well-suited for modeling prosody and temporal patterns relevant to emotion.

Key Ideas / Pipeline
--------------------
1. Data: Use the TESS (Toronto Emotional Speech Set) dataset (link above) as the primary dataset for training and evaluation.
2. Preprocessing:
   - Standardize sampling rate (e.g., 16 kHz or 22.05 kHz).
   - Trim/silence removal and optional voice activity detection.
   - Feature extraction per audio frame/segment:
     - MFCC (typ. 13–40 coefficients) ± deltas
     - Chroma
     - Mel spectrogram / Mel-frequency energies
     - Spectral contrast (optional)
     - Tonnetz (optional)
   - Aggregate features into fixed-length sequences (padding or windowing) or use variable-length sequences with masking.
   - Normalize (z-score) features per coefficient or per utterance.
3. Model:
   - Stacked LSTM layers (1–3 layers), with dropout/ recurrent dropout for regularization.
   - Fully connected (Dense) output head with softmax for multi-class emotion classification.
   - Loss: Categorical cross-entropy (or focal loss if class imbalance).
   - Optimizer: Adam (with learning-rate scheduling).
4. Training:
   - Train/validation/test split (e.g., 70/15/15) or speaker-independent split.
   - Early stopping with model checkpointing.
   - Data augmentation (optional): additive noise, time-shifting, pitch/time stretching to improve robustness.
5. Evaluation:
   - Metrics: Accuracy, precision/recall/F1 per class, confusion matrix.
   - Consider cross-validation or speaker-independent evaluation for realistic performance.

Primary Libraries Used
----------------------
- Audio and feature extraction:
  - librosa — audio loading and signal processing
  - soundfile (pysoundfile) — audio I/O
- Data handling:
  - numpy, pandas
- ML / Deep learning:
  - TensorFlow / Keras (preferred) or PyTorch equivalents
- Scikit-learn:
  - preprocessing, metrics, train_test_split
- Visualization:
  - matplotlib, seaborn
- Optional:
  - tqdm for progress bars, joblib for caching, natsort for deterministic file ordering

Typical Model Architecture (example)
------------------------------------
- Input: sequence of feature vectors (timesteps × feature_dim)
- LSTM (128 units, return_sequences=True)
- Dropout (0.3)
- LSTM (64 units, return_sequences=False)
- Dropout (0.3)
- Dense (64, ReLU)
- BatchNormalization (optional)
- Dense (num_classes, softmax)

Training Recipe (high level)
----------------------------
- Batch size: 16–64 (depending on GPU/memory)
- Epochs: 30–150 with early stopping
- Learning rate: 1e-3 with ReduceLROnPlateau or custom scheduler
- Regularization: Dropout 0.2–0.5, L2 weight decay if needed
- Use stratified or balanced sampling if classes are imbalanced

Reproducibility & Tips
----------------------
- Fix random seeds (numpy, tensorflow, python) for reproducible runs.
- Normalize features after the train split to avoid leakage.
- Use a speaker-independent split for realistic evaluation: ensure speakers in test set are unseen during training.
- Monitor per-class metrics and confusion matrix to detect frequently confused emotion pairs.
- If data is limited, consider transfer learning on large-scale speech/audio models or pretraining an encoder.

Expected Outputs
----------------
- Trained LSTM model (.h5 or saved-model)
- Evaluation report: accuracy, F1 scores, confusion matrix
- Visualizations: training curves (loss/accuracy), confusion matrix, representative spectrograms

Recommended File Structure (example)
------------------------------------
- data/                — raw/downloaded dataset
- notebooks/           — experiments, EDA, plots
- src/
  - features.py        — audio feature extraction
  - dataset.py         — dataset loader and batching
  - model.py           — LSTM model definition
  - train.py           — training loop and checkpoints
  - evaluate.py        — evaluation scripts and metrics
- checkpoints/         — saved models
- README.md

How to run (example)
--------------------
1. Create environment and install deps:
   pip install -r requirements.txt
   (requirements typically include: numpy pandas librosa soundfile tensorflow scikit-learn matplotlib seaborn)
2. Download the dataset (TESS) and place under data/tess/
3. Extract features and prepare datasets:
   python src/features.py --data_dir data/tess --out_dir data/features
4. Train:
   python src/train.py --features data/features --model_out checkpoints/ser_lstm.h5
5. Evaluate:
   python src/evaluate.py --model checkpoints/ser_lstm.h5 --features data/features

References
----------
- TESS Dataset (Toronto Emotional Speech Set): https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
- Librosa documentation — https://librosa.org/
- Chollet, F. — "Deep Learning with Python" (for Keras/TensorFlow examples)
- Papers / resources on speech emotion recognition and feature engineering (MFCCs, prosodic features)

License & Attribution
---------------------
This repository and code are provided for educational/research purposes. Check the TESS dataset license and cite relevant sources if used in publications.

Contact
-------
For improvements, experiment results, or questions, open an issue or submit a PR.

