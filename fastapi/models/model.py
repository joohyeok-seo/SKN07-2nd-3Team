import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# split the dataset
from sklearn.model_selection import train_test_split

# preprossing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler , LabelEncoder , OrdinalEncoder
from sklearn.pipeline import Pipeline , FeatureUnion
from sklearn_features.transformers import DataFrameSelector
from datasist.structdata import detect_outliers
from sklearn.model_selection import cross_val_score, cross_val_predict
from imblearn.over_sampling import SMOTE

# Models
from sklearn.linear_model import LogisticRegression , SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import AdaBoostClassifier , BaggingClassifier , GradientBoostingClassifier , RandomForestClassifier , VotingClassifier
from sklearn.multiclass import OneVsOneClassifier
import xgboost as xgb

# Metrix 
from sklearn.metrics import accuracy_score , classification_report , precision_score , recall_score , confusion_matrix

# Tuning
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV

# Roc
from sklearn.metrics import roc_curve, auc



models = {
    # 'ovo' : OneVsOneClassifier(estimator=RandomForestClassifier()),
    # 'knn' : KNeighborsClassifier(),
    # 'SGD': SGDClassifier(),
    'Lgr': LogisticRegression(),
    # 'xgb': xgb.XGBClassifier(),
    'dt' : DecisionTreeClassifier(),
    'RF' : RandomForestClassifier(),
    # 'bag': BaggingClassifier(),
    # 'gdb': GradientBoostingClassifier(),
    # 'SVC' : SVC(),
    # 'baf_clf' : BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,max_samples=100, bootstrap=True)
          }

def fit(df, y_col:str='Churn'):

    ''' preprocess
    
    '''
    df.drop(['Area code',] , axis = 1 , inplace = True)
    df['International plan'] = df['International plan'].map({'No' : False , 'Yes' : True})
    df['Voice mail plan'] = df['Voice mail plan'].map({'No' : False , 'Yes' : True})
    df.drop(['State'], axis=1, inplace=True)

    '''
    '''

    x =df.drop(y_col , axis =1)
    y= df[y_col]
    smote = SMOTE()
    x,y = smote.fit_resample(x,y)
    x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.35 , random_state=42 , stratify=y ,shuffle=True)
    print('x_train shape -- ', x_train.shape)
    print('y_train shape -- ', y_train.shape)
    print('x_test shape -- ', x_test.shape)
    print('y_test shape -- ', y_test.shape)

    sc = StandardScaler()
    x_train_final = sc.fit_transform(x_train)
    x_test_final = sc.transform(x_test)

    # report
    report = {}

    for key, model in models.items():
        print(f'For {key} Model : ')
        print('_' * 90)
        
        cv_scores = cross_val_score(model, x_train_final, y_train, cv=5, scoring='accuracy')
        print(f"CV Scores: {cv_scores}")
        
        model.fit(x_train_final, y_train)
        
        y_pred = model.predict(x_test_final)
        y_probs = model.predict_proba(x_test_final)[:, 1]
        
        train_accuracy = accuracy_score(y_train, model.predict(x_train_final))
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Train Score is {train_accuracy}")
        print(f"Test Score is {test_accuracy}")
        
        print(classification_report(y_test, y_pred))
        report[f'{key}'] = classification_report(y_test, y_pred)

        # confusion matrix
        plt.figure(figsize=(5,5))

        confusion_mat = confusion_matrix(y_test, y_pred)

        sns.heatmap(confusion_mat, fmt='g', annot=True, cbar=False, vmin=0, cmap='Blues')
        plt.title(f'Confusion Matrix for {key} Model')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'./data/heatmap-{key}.png', dpi=300, bbox_inches='tight')
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
        roc_image_path = f"./data/roc_curve-{key}.png"
        plt.savefig(roc_image_path)
        plt.close()

    # PDF
    from fpdf import FPDF

    # 4. PDF로 결과물 저장
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # 제목
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(200, 10, txt="Binary Classification Model Results", ln=True, align='C')


    for k, v in report.items():
        # 모델 성능 지표 추가
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=f"Classification Report ({k}):\n" + v)
        cm_image_path = f'./data/heatmap-{k}.png'
        roc_image_path = f'./data/roc_curve-{k}.png'
        
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
    pdf.output("./data/classification_report.pdf")

    # 생성된 이미지는 필요 없으므로 삭제
    # import os
    # os.remove(roc_image_path)
    # os.remove(cm_image_path)

    print("PDF 파일이 생성되었습니다.")

