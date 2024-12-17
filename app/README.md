# How to run
Upload a file (only csv) for whatever you want to train/predict whit ML models.

### Prerequiste
run `pip install -r requirements.txt`

### What are we supporting?
- K-Neighbors Classifier
- Decision Tree Classifier
- Random Forest Classifier
- Logistic Regression
- XGB Classifier
- Bagging Classifier
- Gradient Boosting Classifier

### Getting Started

#### FastAPI

```
uvicorn main:app --reload`
```

#### Streamlit
```
streamlit run app.py --ip localhost --port 8000
```