# 🦠 Dengue Fever Prediction Model

A machine learning project that predicts **future dengue fever cases** using historical data such as temperature, humidity, rainfall, and previous case counts.
The model is built using **Python**, **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**, and **TensorFlow (Keras LSTM)**.

---

## 📘 Project Overview

This project uses an **LSTM (Long Short-Term Memory)** neural network to model temporal dependencies in dengue incidence data.
It can be used to forecast **the number of dengue cases** based on past trends and environmental features.

The system includes:

* Data preprocessing and scaling
* Exploratory Data Analysis (EDA) with Seaborn & Matplotlib
* LSTM-based deep learning model with dropout and bidirectional layers
* Early stopping and model checkpointing
* Prediction of future dengue cases (configurable horizon)

---

## 📂 Folder Structure

```
📁 dengue-prediction
│
├── dengue_prediction.py         # Main training & prediction script
├── dataset.csv                  # Input dataset (example format)
├── README.md                    # Project documentation
├── dengue_model.h5              # Trained model (output)
├── dengue_best.h5               # Best model checkpoint
├── scaler.pkl                   # Scaler used for normalization
└── README_dengue_model.txt      # Auto-generated training summary
```

---

## ⚙️ Features

✅ **Automatic feature detection** — identifies date & target (cases) columns
✅ **Univariate or multivariate support** — uses weather & other features if present
✅ **Configurable** — choose past days, future prediction days, epochs, etc.
✅ **Visualization** — training loss, predicted vs. actual values
✅ **Scalable** — easily adapted for other time series forecasting problems

---

## 🧠 Model Architecture

```text
Input (PAST_DAYS × features)
        │
  LSTM(64, return_sequences=True)
        │
     Dropout(0.2)
        │
  Bidirectional LSTM(32)
        │
     Dropout(0.2)
        │
      Dense(16, ReLU)
        │
      Dense(1, Linear)
        ↓
 Predicted Dengue Cases
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/dengue-prediction.git
cd dengue-prediction
```

### 2️⃣ Install Dependencies

Create and activate a virtual environment, then install the required libraries:

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` yet, use this:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
```

Expected columns (example):

| Date       | Temperature | Rainfall | Humidity | Cases |
| ---------- | ----------- | -------- | -------- | ----- |
| 2020-01-01 | 30.5        | 12.3     | 85       | 25    |
| 2020-01-02 | 31.2        | 8.7      | 83       | 30    |

The script automatically detects your date and target columns.

### 4️⃣ Train the Model

Run the training script:

```bash
python dengue_prediction.py
```

This will:

* Preprocess and scale the data
* Train an LSTM model
* Save model and scaler files to `/mnt/data/`

---

## 📊 Outputs

| File                      | Description                        |
| ------------------------- | ---------------------------------- |
| `dengue_model.h5`         | Final trained model                |
| `dengue_best.h5`          | Best checkpoint during training    |
| `scaler.pkl`              | Data scaler for future predictions |
| `README_dengue_model.txt` | Summary of training run            |

---

## 🔍 Example Results

After training, the script plots:

* **Training vs Validation Loss**
* **Actual vs Predicted Dengue Cases**

Example output (for the last 60 samples):



---<img width="1000" height="600" alt="outcome_distribution" src="https://github.com/user-attachments/assets/c348cadb-1853-452b-af25-61c4526f4c4f" />


## 📈 How to Predict Future Cases

After training, you can modify the script to load the trained model and make new predictions.

```python
import pickle, tensorflow as tf
import numpy as np, pandas as pd
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("dengue_model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Prepare new input (last PAST_DAYS rows)
data = pd.read_csv("dataset.csv")
# ... preprocess same as training ...
# model.predict(...) to forecast next day's cases
```

---

## 📚 Future Improvements

* [ ] Add multi-step forecasting (predict multiple future days)
* [ ] Integrate with weather APIs for real-time prediction
* [ ] Deploy via Flask/Django web app
* [ ] Visualization dashboard (Plotly/Dash)
* [ ] AutoML tuning with Optuna or KerasTuner

---

## 🤝 Contributing

Contributions are welcome!
To contribute:

1. Fork the repo
2. Create a new branch
3. Commit your changes
4. Submit a pull request

---

## 🧾 License

This project is licensed under the **MIT License**.
See [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**Sazzad Hossain**
📧 sazzadhossain74274@gmail.com
🌐 https://www.linkedin.com/in/sazzadhossain1461/


