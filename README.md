# 📱 SMS Spam Detection Using Deep Learning and NLP

> **Objective:** Develop a deep-learning system that automatically classifies SMS messages as **spam** or **legitimate** using NLP and an LSTM neural network.

---

## 🗂️ Project Structure

```
spam-message-detector/
│
├── data/
│   └── SMSSpamCollection       ← Download from UCI dataset (see below)
│
├── notebooks/
│   ├── spam_detection.py       ← Full training pipeline
│   └── plots/                  ← Auto-generated charts
│
├── models/
│   └── spam_model.keras        ← Saved model (after training)
│
├── app/
│   └── app.py                  ← Streamlit web interface
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Technology Stack

| Component | Library |
|-----------|---------|
| Data handling | Pandas, NumPy |
| Deep learning | TensorFlow / Keras |
| NLP | TextVectorization, Embedding |
| Model | LSTM + Dropout + Sigmoid |
| Evaluation | Scikit-learn |
| Visualisation | Matplotlib, Seaborn |
| Web interface | Streamlit |

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

Download the **SMS Spam Collection** dataset from UCI:  
🔗 https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

Place the file `SMSSpamCollection` (tab-separated, no extension) inside the `data/` folder.

### 3. Train the model

```bash
python notebooks/spam_detection.py
```

This will:
- Load and explore the data
- Build and train the LSTM model
- Save the best model to `models/spam_model.keras`
- Generate evaluation plots in `notebooks/plots/`

### 4. Launch the web app

```bash
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

---

## 🧠 Model Architecture

```
SMS Message (raw string)
        ↓
TextVectorization  (vocab = 10,000, seq_len = 100)
        ↓
Embedding          (64-dim, mask_zero=True)
        ↓
LSTM               (64 units)
        ↓
Dropout            (0.5)
        ↓
Dense              (32, ReLU)
        ↓
Dropout            (0.3)
        ↓
Dense              (1, Sigmoid)
        ↓
   SPAM / NOT SPAM
```

---

## 📊 Training Phases

| Phase | What happens |
|-------|-------------|
| 1 | Dataset loading & exploratory analysis |
| 2 | Label encoding + train/test split (80/20) |
| 3 | Text vectorization (word → integer sequences) |
| 4 | LSTM model training (early stopping) |
| 5 | Evaluation — accuracy, F1, confusion matrix |
| 6 | Streamlit interactive web interface |

---

## 📈 Expected Results

After training on the full SMS Spam Collection dataset (~5,574 messages):

| Metric | Approx. value |
|--------|--------------|
| Test Accuracy | ~97–99% |
| Spam Precision | ~95–98% |
| Spam Recall | ~90–96% |
| F1-score | ~93–97% |

---

## 💡 Key Concepts

- **Embedding** — converts word IDs into dense 64-dimensional vectors that capture semantic meaning.
- **LSTM** — a recurrent network that reads words sequentially, remembering context across the message.
- **Dropout** — randomly disables neurons during training to prevent overfitting.
- **Binary Crossentropy** — the loss function for 2-class classification (spam vs. ham).
- **EarlyStopping** — halts training when validation accuracy stops improving, saving the best weights.

---

*SMS Spam Detection · Deep Learning Project · LSTM + Keras*
