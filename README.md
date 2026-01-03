# IPL Win Predictor

A machine learning application that predicts the probability of a batting team winning an IPL (Indian Premier League) cricket match based on real-time match conditions.

## Features

- **Real-time Win Probability Prediction**: Input live match data and get instant win probability for the batting team
- **Interactive Streamlit UI**: User-friendly web interface for predictions
- **Trained RandomForest Model**: 99.93% test accuracy on historical IPL data
- **Match Statistics**: Automatic calculation of Current Run Rate (CRR), Required Run Rate (RRR), and remaining balls/wickets
- **Data-Driven**: Built on cleaned and preprocessed IPL match data

## Project Structure

```
IPL Win Predictor/
├── main.py                    # Streamlit web UI (production app)
├── cleaning_data.py           # Data preprocessing pipeline
├── train.py                   # Model training script
├── pipe.pkl                   # Trained RandomForest pipeline (binary)
├── requirements.txt           # Python dependencies with pinned versions
├── Datasets/
│   ├── FIRST DATASET.csv      # Raw IPL matches data
│   ├── SECOND DATASET.csv     # Raw IPL deliveries/runs data
│   └── FINAL DATASET.csv      # Processed dataset for training
├── background.jpg             # Background image for Streamlit UI
└── README.md                  # This file
```

## Installation

### Prerequisites
- Python 3.11 or higher
- pip (Python package manager)

### Setup

1. **Clone/Navigate to project directory:**
   ```bash
   cd /Users/swastikshukla/Development/IPL\ Win\ Predictor
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv .venv
   ```

3. **Activate virtual environment:**
   ```bash
   source ./.venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Workflow

The project follows a three-step workflow:

#### 1. Data Cleaning (Optional - already done)
```bash
./.venv/bin/python cleaning_data.py
```
- Reads raw datasets from `Datasets/FIRST DATASET.csv` and `Datasets/SECOND DATASET.csv`
- Cleans team names (old names → current franchises)
- Merges match and delivery data
- Outputs `Datasets/FINAL DATASET.csv` for training

#### 2. Model Training
```bash
./.venv/bin/python train.py
```
- Loads processed data from `Datasets/FINAL DATASET.csv`
- Builds scikit-learn Pipeline: ColumnTransformer (OneHotEncoder) + RandomForestClassifier
- Trains on 75% of data, tests on 25%
- Saves trained model as `pipe.pkl`
- Prints test accuracy

#### 3. Run Prediction App
```bash
./.venv/bin/python -m streamlit run main.py
```
- Launches interactive Streamlit web interface
- Input: batting team, bowling team, venue, current runs, wickets down, overs bowled
- Output: Win probability (%) + pie chart visualization
- Opens automatically at `http://localhost:8501`

## File Descriptions

| File | Purpose |
|------|---------|
| `main.py` | **Streamlit UI** - Interactive web app for predictions. Loads `pipe.pkl` model and renders input forms, calculations, and results. |
| `cleaning_data.py` | **Data Pipeline** - Preprocesses raw IPL datasets (merges, cleans team names, filters columns), outputs `FINAL DATASET.csv`. |
| `train.py` | **Model Training** - Trains RandomForest pipeline on cleaned data with class weight balancing, saves `pipe.pkl`. |
| `pipe.pkl` | **Trained Model** - Binary serialized scikit-learn Pipeline (ColumnTransformer + RandomForestClassifier). ~99.93% accuracy. |
| `requirements.txt` | **Dependencies** - Pinned Python package versions for reproducible environment. |
| `Datasets/` | **Data Folder** - Contains raw and processed IPL match/delivery data. |
| `background.jpg` | **UI Asset** - Background image displayed in Streamlit app. |

## Model Details

### Architecture
- **Preprocessing**: ColumnTransformer with OneHotEncoder (encodes categorical: batting_team, bowling_team, venue)
- **Estimator**: RandomForestClassifier
  - n_estimators: 200 trees
  - class_weight: 'balanced' (handles class imbalance)
  - random_state: 100 (reproducibility)
- **Pipeline**: sklearn.pipeline.Pipeline

### Training Data
- **Source**: `Datasets/FINAL DATASET.csv` (processed from raw IPL data)
- **Train/Test Split**: 75% train / 25% test (random_state=100)
- **Target**: Binary classification (0 = batting team loses, 1 = batting team wins)
- **Test Accuracy**: 99.93%

### Features Used
- `batting_team` (categorical): Team batting second
- `bowling_team` (categorical): Team bowling
- `venue` (categorical): Match location
- `runs` (numerical): Current runs scored
- `wickets` (numerical): Wickets lost
- `overs` (numerical): Overs bowled
- Derived: CRR, RRR, balls_left, target

## Dependencies

All dependencies are listed in `requirements.txt` with pinned versions:
- **scikit-learn==1.3.2**: Machine learning library
- **pandas==2.3.3**: Data manipulation
- **streamlit==1.52.2**: Web UI framework
- **matplotlib==3.10.8**: Data visualization
- **joblib**: Model serialization

**Important**: scikit-learn version 1.3.2 must be used to load `pipe.pkl` (model compatibility).

## Commands Reference

```bash
# Activate virtual environment
source ./.venv/bin/activate

# Run data cleaning
./.venv/bin/python cleaning_data.py

# Train model
./.venv/bin/python train.py

# Run Streamlit app
./.venv/bin/python -m streamlit run main.py

# Install packages
pip install -r requirements.txt
```

## Troubleshooting

### Model unpickling errors
- Ensure scikit-learn version is 1.3.2: `pip show scikit-learn`
- Reinstall if needed: `pip install scikit-learn==1.3.2`

### Background image not loading
- Verify `background.jpg` exists in project root
- Check file permissions

### ModuleNotFoundError
- Activate virtual environment: `source ./.venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

## Future Enhancements

- Add probability calibration (Platt/Isotonic scaling)
- Hyperparameter tuning with GridSearchCV
- Add player performance features
- Deploy as web service (Flask/FastAPI)
- Add match history and statistics dashboard

## Notes

- The app requires live match input (current runs, wickets, overs).
- Predictions are based on historical IPL data patterns.
- Model uses teams and venues from modern IPL (8 teams).
- Virtual environment (.venv) ensures dependency isolation and reproducibility.

---