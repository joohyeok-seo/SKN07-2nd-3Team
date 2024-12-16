import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from fpdf import FPDF
import numpy as np


'''
pip install matplotlib scikit-learn fpdf

'''

# 1. 이진 분류 모델 학습 예시 (로지스틱 회귀)
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

# 2. 모델 성능 평가 (Confusion Matrix, Classification Report, ROC Curve)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# ROC Curve 계산
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# 3. Matplotlib로 시각화

# ROC Curve 그리기
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
roc_image_path = "roc_curve.png"
plt.savefig(roc_image_path)
plt.close()

# Confusion Matrix 시각화
plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks([0, 1], ['Class 0', 'Class 1'])
plt.yticks([0, 1], ['Class 0', 'Class 1'])
cm_image_path = "confusion_matrix.png"
plt.savefig(cm_image_path)
plt.close()

# 4. PDF로 결과물 저장
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# 제목
pdf.set_font("Arial", size=16, style='B')
pdf.cell(200, 10, txt="Binary Classification Model Results", ln=True, align='C')

# 모델 성능 지표 추가
pdf.ln(10)
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, txt="Classification Report:\n" + report)

# Confusion Matrix 이미지 추가
pdf.ln(10)
pdf.cell(200, 10, txt="Confusion Matrix:", ln=True)
pdf.image(cm_image_path, x=10, y=pdf.get_y(), w=180)

# ROC Curve 이미지 추가
pdf.ln(90)  # 이미지 아래에 여백 추가
pdf.cell(200, 10, txt="ROC Curve:", ln=True)
pdf.image(roc_image_path, x=10, y=pdf.get_y(), w=180)

# PDF 저장
pdf.output("classification_report.pdf")

# 생성된 이미지는 필요 없으므로 삭제
import os
os.remove(roc_image_path)
os.remove(cm_image_path)

print("PDF 파일이 생성되었습니다.")
