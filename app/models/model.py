import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# split the dataset
from sklearn.model_selection import train_test_split

# Models
from sklearn.linear_model import LogisticRegression , SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import AdaBoostClassifier , BaggingClassifier , GradientBoostingClassifier , RandomForestClassifier , VotingClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb

# Metrix 
from sklearn.metrics import accuracy_score , classification_report , precision_score , recall_score , confusion_matrix

# Tuning
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV

# Roc
from sklearn.metrics import roc_curve, auc

# load model
import pickle

# PDF
from fpdf import FPDF

# preprossing
from imblearn.over_sampling import SMOTE
from .utils.utils import preprocess

NAMES = {
    'ovo' : 'One Vs One Classifier (estimator=RandomForestClassifier)',
    'knn' : 'K-Neighbors Classifier',
    'SGD': 'SGD Classifier',
    'Lgr': 'Logistic Regression',
    'xgb': 'XGB Classifier',
    'dt' : 'Decision Tree Classifier',
    'rf' : 'Random Forest Classifier',
    'bag': 'Bagging Classifier',
    'gdb': 'Gradient Boosting Classifier',
    'SVC' : 'SVC',
}
rNAMES = {v : k for k, v in NAMES.items()}

models = {
    # 'ovo' : OneVsOneClassifier(estimator=RandomForestClassifier()),
    'knn' : KNeighborsClassifier(n_neighbors=5),
    # 'SGD': SGDClassifier(),
    'Lgr': LogisticRegression(),
    'xgb': xgb.XGBClassifier(eval_metric='logloss'),
    'dt' : DecisionTreeClassifier(),
    'rf' : RandomForestClassifier(),
    'bag': BaggingClassifier(),
    'gdb': GradientBoostingClassifier(),
    # 'SVC' : SVC(),
    # 'baf_clf' : BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,max_samples=100, bootstrap=True)
          }

available_models = [NAMES[k] for k, _ in models.items()]
no_cv_socres = ['xgb']

def predict(df, model_name: str = 'K-Neighbors Classifier'):

    ''' preprocess
    '''
    x, y = preprocess(df) 
    '''
    '''

    print(f'[info] prediction({model_name})')
    print('_' * 90)

    base_dir = os.path.dirname(__file__)
    with open(os.path.join(base_dir, f'params/model-{rNAMES[model_name]}.pkl'), 'rb') as f:
        model_dict = pickle.load(f)
    model = model_dict['model']
    
    if rNAMES[model_name] not in no_cv_socres:
        result = {
            'ratio' : np.mean(model.predict_proba(x)[:, 1]),
            'prob' : np.mean(model.predict(x)),
            'total' : df.shape
        }
        print(result)
    else:
        result = {
            'ratio' : np.mean(model.predict_proba(x)),
            'prob' : np.mean(model.predict(x)),
            'total' : df.shape
        }

    return result

def fit(df, y_col:str='Churn'):

    ''' preprocess
    '''
    x, y = preprocess(df) 
    '''
    '''
    smote = SMOTE()
    x,y = smote.fit_resample(x,y)
    x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.35 , random_state=42 , stratify=y ,shuffle=True)

    # report
    report = {}
    for key, model in models.items():
        print(f'For {key} Model : ')
        print('_' * 90)

        # k-fold'
        if key not in no_cv_socres:
            cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
        else:
            cv_scores = np.array([0] * 5)

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y_probs = model.predict_proba(x_test)[:, 1]

        train_accuracy = accuracy_score(y_train, model.predict(x_train))
        test_accuracy = accuracy_score(y_test, y_pred)
        
        report[f'{key}'] = {
            'CV Scores' : f"{np.round(cv_scores * 100, 2)}",
            'Train score' : f"{train_accuracy * 100:.2f} %",
            'Test score' : f"{test_accuracy* 100:.2f} %",
            'Classification result' : classification_report(y_test, y_pred),
        }

        # save params
        base_dir = os.path.dirname(__file__)
        with open(os.path.join(base_dir, f'params/model-{key}.pkl'), 'wb') as f:
            print(f'[INFO] save parameters: {os.path.join(base_dir, f"params/model-{key}.pkl")}')
            pickle.dump(
                {'model' : model}, f
            )
            
        # confusion matrix
        plt.figure(figsize=(5,5))
        confusion_mat = confusion_matrix(y_test, y_pred)
        sns.heatmap(confusion_mat, fmt='g', annot=True, cbar=False, vmin=0, cmap='Blues')
        plt.title(f'Confusion Matrix for {key} Model')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(base_dir, f'./results/cm-heatmap-{key}.png'), dpi=300, bbox_inches='tight')
        plt.close()

                
        # ROC
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        roc_image_path = os.path.join(base_dir, f"./results/roc_curve-{key}.png")
        plt.savefig(roc_image_path)
        plt.close()

    # 4. PDF로 결과물 저장
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # 제목
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(200, 10, txt="Binary Classification Model Results", ln=True, align='C')


    for idx, (k, v) in enumerate(report.items()):
        # 모델 성능 지표 추가
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 1, txt=f"[{idx+1}] Report ({NAMES[k]}):\n")
        for m, s in v.items():
            pdf.multi_cell(0, 10, txt=f'{m}:\n\t' + s)
            
        cm_image_path = os.path.join(base_dir, f'./results/cm-heatmap-{k}.png')
        roc_image_path = os.path.join(base_dir, f'./results/roc_curve-{k}.png')
        
        # Confusion Matrix 이미지 추가
        pdf.ln(10)
        pdf.cell(200, 10, txt="Confusion Matrix:", ln=True)
        pdf.image(cm_image_path, x=10, y=pdf.get_y(), w=80)

        # ROC Curve 이미지 추가
        pdf.ln(90)  # 이미지 아래에 여백 추가
        pdf.cell(200, 10, txt="ROC Curve:", ln=True)
        pdf.image(roc_image_path, x=10, y=pdf.get_y(), w=80)
        
        pdf.add_page()

    # PDF 저장
    pdf.output(os.path.join(base_dir, "./results/classification_report.pdf"))

    # 생성된 이미지는 필요 없으므로 삭제
    # import os
    # os.remove(roc_image_path)
    # os.remove(cm_image_path)

    print("[INFO] PDF 파일이 생성되었습니다.")



if __name__ == "__main__":
    ...