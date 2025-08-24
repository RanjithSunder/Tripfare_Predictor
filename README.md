# 🚗 TripFare Predictor

TripFare Predictor is a **Streamlit web application** and **ML pipeline** to predict urban taxi fares using a combination of **rule-based logic** and **machine learning models**.

---

## 📂 Project Structure
```
tripfare-predictor/
│── streamlit_app.py     # Streamlit app for fare prediction
│── tripfare_ml.py       # ML pipeline: data preprocessing, training, evaluation
│── requirements.txt     # Project dependencies
│── README.md            # Documentation
│── .gitignore           # Ignore unnecessary files
│── models/              # Trained ML models
│── data/                # Datasets
```

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tripfare-predictor.git
   cd tripfare-predictor
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Mac/Linux
   venv\Scripts\activate      # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Application

### Run Streamlit app
```bash
streamlit run streamlit_app.py
```

### Train & Evaluate Model
```bash
python tripfare_ml.py
```

The trained model will be saved in the `models/` directory.

---

## 📊 Features
- 📍 Haversine distance calculation between pickup & dropoff  
- ⏰ Time-based features (rush hours, weekends, night trips)  
- 💰 Rule-based & ML-based fare prediction  
- 📊 EDA visualizations (matplotlib + seaborn)  
- 🗺️ Interactive trip route map (Plotly)  
- 💵 Fare breakdown chart  

---

## 📌 Notes
- Default dataset path in `tripfare_ml.py` should be updated with your dataset.  
- ML model gets saved in `models/best_tripfare_model.pkl`.  
- If model isn’t available, the app falls back to rule-based predictions.

---
## ⚠️ Note on Large Files
This project uses [Git LFS](https://git-lfs.github.com/) to manage large files (datasets & models).
Make sure to install Git LFS before cloning:
```bash
git lfs install
git clone https://github.com/RanjithSunder/Tripfare_Predictor.git
```
---

## 📜 License

MIT License
