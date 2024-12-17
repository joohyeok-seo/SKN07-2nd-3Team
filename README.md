# 부리부리방범대

## 팀 소개
<table>
  <tr>
    <th>김나예</th>
    <th>김서진</th>
    <th>나성호</th>
    <th>서주혁</th>
    <th>신동익</th>
  </tr>
  <tr>
    <td><img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/%EC%9D%B8%ED%98%95.png" width="175" height="175"></td>
    <td><img src="https://i.namu.wiki/i/yHMdZs8LhKDP0D0XmvNkWe4NplRU5BDyXiZNDk5BTOd9ON5KtykFiDO_Q7SDpQLA-q9Q4fyFKfzM3apcZnPGtg.webp" width="175" height="175"></td>
    <td><img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/%EC%8A%A4%EB%85%B8%EC%9A%B0%EB%A7%A8.png" width="175" height="175"></td>
    <td><img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/baby.jpg" width="175" height="175"></td>
    <td><img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/%EB%8F%99%EC%9D%B5.jpg" width="175" height="175"></td>
  </tr>
  <tr>
    <th>KNN Model</th>
    <th>Decision Tree Model</th>
    <th>XGBoost Model</th>
    <th>RNN Model</th>
    <th>Random Forest Model</th>
  </tr>
</table>

---

## 프로젝트 소개<BR>
고객 이탈률 예측은 고객 경험을 개선하고 경쟁력을 높이는 데 중요한 역할을 합니다. 예측 모델을 통해 고객의 불만이나 불편을 사전에 파악하고 이를 해결함으로써 고객 만족도를 향상시킬 수 있습니다. 또한, 이탈률을 예측함으로써 경쟁사보다 더 나은 서비스를 제공하고, 고객 충성도를 유지하며, 장기적인 성장을 이끌어낼 수 있습니다.<BR>
우리는 구독 기반 서비스 기업, 전자상거래 기업, 금융 서비스 기업 등 서비스 제공 기업을 위한 **고객 이탈률 예측 모델**을 구축했습니다. 이를 통해 기업은 이탈 대응 전략을 수립하고, 서비스 개선안을 도출하여 더 효율적인 고객 관리와 경쟁력 있는 시장 전략을 마련할 수 있습니다.

## (프로젝트명)

### 개요
#### 가입 고객 이탈 예측 모델 설계 및 구축
<br>

**1. 훈련·테스트 데이터 선정**<br>
&nbsp;&nbsp;&nbsp;&nbsp;- 결측치가 적고 불필요한 feature가 적은 데이터를 선정해 모델 학습<br>

**2. ML 성능 비교**<br>
&nbsp;&nbsp;&nbsp;&nbsp;- ML 알고리즘 5개를 선정해 동일한 데이터로 성능 비교 후, 가장 성능이 좋은 알고리즘을 채택 <br>
&nbsp;&nbsp;&nbsp;&nbsp;- Kaggle 머신러닝 웹 커뮤니티의 **Telecom Churn Dataset** 채택<br>

**3. 예측 모델 설계·구축**<br>
&nbsp;&nbsp;&nbsp;&nbsp;- 채택한 알고리즘으로 ML 모델 설계 및 구축하여 성능 검증<br>

### 목표
여러 머신러닝 모델의 비교, 분석을 통해 고객 이탈 예측에 가장 적합한 고성능 모델을 구축합니다.

---

## EDA
### Correlation Matrix
![download](https://github.com/user-attachments/assets/6f8277cb-c581-465d-b751-56b82aa42e2b)

### 낮통화 이용률
![download](https://github.com/user-attachments/assets/0506510c-8f6c-4107-ad54-341797303ecf)

### 저녁통화 이용률
![download](https://github.com/user-attachments/assets/f3801354-610a-4863-8835-173f94369d2d)

### 야간통화 이용률
![download](https://github.com/user-attachments/assets/986f494c-42d6-4a46-9389-824cc258261a)

### 고객센터 상담 비율
![download](https://github.com/user-attachments/assets/1dac0981-380d-46f8-b8cc-7872f46c555c)

### 국제전화 가입비율
![download](https://github.com/user-attachments/assets/1db52c47-3d11-4d62-b159-37f264a24ee5)

### 음성사사함 가입비율
![download](https://github.com/user-attachments/assets/aa216f78-ad8d-4a0b-83d1-7b5d57e456bf)

### 이탈률
![download](https://github.com/user-attachments/assets/aa611630-d6a6-473f-8955-8428e40611d2)

### 고개센터 통화량에 따른 이탈률 & 이상치
![churn_vs_customer_service_calls](https://github.com/user-attachments/assets/e491b51b-ce10-41ea-b632-9ded739fc8e7)

### 통화량이 높은 고객들 대상 이탈률
![churn_vs_total_day_minutes](https://github.com/user-attachments/assets/9618780c-ebd7-4a49-8a94-511026fcb1cf)

### 데이터 전처리
<pre>
<code>
columns_to_drop = ['State', 'Area code']
train_data = train_data.drop(columns=columns_to_drop, axis=1)
test_data = test_data.drop(columns=columns_to_drop, axis=1)
</code>
</pre>

---

## Machine Learning
1. KNN Model
2. Decision Tree Model
3. XGBoost Model
4. Random Forest Model

### Evaluation Metrics by Model
<table>
  <tr>
    <th>Model</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
    <th>ROC AUC</th>
  </tr>
  <tr>
    <th>KNN</th>
    <th>0.885</th>
    <th>0.828</th>
    <th>0.304</th>
    <th>0.444</th>
    <th>0.646</th>
  </tr>
  <tr>
    <th>Disition Tree</th>
    <th>0.913</th>
    <th>0.680</th>
    <th>0.737</th>
    <th>0.707</th>
    <th>0.840</th>
  </tr>
  <tr>
    <th>XGBoost</th>
    <th>0.957</th>
    <th>0.934</th>
    <th>0.747</th>
    <th>0.830</th>
    <th>0.911</th>
  </tr>
  <tr>
    <th>Random Forest</th>
    <th>0.958</th>
    <th>0.972</th>
    <th>0.737</th>
    <th>0.838</th>
    <th>0.924</th>
  </tr>
</table>

### 최종 모델 선정 과정
이번 프로젝트에서는 XGBoost와 Random Forest 모델이 가장 높은 성능을 보였습니다. 각 모델의 주요 성능 지표를 비교한 결과 다음과 같은 결론을 도출했습니다.

- Random Forest는 Precision에서 가장 높은 성능을 보였으나, Recall이 상대적으로 낮아 이탈 고객을 놓칠 가능성이 있었습니다.

- XGBoost는 Recall과 F1 Score에서 더 균형 잡힌 성능을 보였습니다. 특히 Recall이 Random Forest보다 높아 이탈 고객을 더 많이 예측하는 강점을 보였습니다.

### 최종 모델 선정 (XGBoost 선택)
1. 균형 잡힌 성능
- Recall이 Random Forest보다 높아 이탈 고객을 더 많이 예측할 수 있습니다.
- F1 Score도 높은 수준을 유지하며 정확도와 재현율 간 균형을 이룹니다.
2. 과적합 방지
- Random Forest는 Precision은 높았지만 과적합 경향이 보였습니다.
- XGBoost는 안정적이면서도 일반화 성능이 뛰어났습니다.
3. 해석 가능성
- XGBoost를 통해 Feature Importance를 분석하면 이탈 요인을 명확히 파악할 수 있습니다.

---

## Deep Learning
### RNN Model
1. RNN Model Create
   <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/rnn_model_create.png" alt="rnn_model_create" width="900px">

2. Used to Prevent Overfitting
![ prevent_overfitting ](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/early%20stopping.png)

3. Training Result
![ training_result ](https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/training_result.png)
  - Training during 34 epoch

4. Loss Curve
   <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/RNN_model_loss.png" alt="loss_curve" width="1100px">

5. Learning Curve
   <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/RNN_model_learning_curve.png" alt="learning_curve" width="1100px">

6. Evaluation
   <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/RNN_model_evaluation.png" alt="rnn_model_evaluation" width="1100px">

### RNN 모델의 한계
1. 성능 부족
- 정확도(Accuracy)가 88.9%로 나쁘지 않지만, Recall이 43.2%로 매우 낮습니다.
- 이는 이탈 고객을 제대로 예측하지 못하는 한계가 있음을 의미합니다.
2. F1 Score
- F1 Score가 0.526으로 낮아 Precision과 Recall의 균형이 부족합니다.
3. ROC-AUC
- ROC-AUC 점수가 0.698로, 다른 머신러닝 모델(XGBoost, Random Forest)에 비해 식별 능력이 떨어집니다.

### XGBoost와의 비교
- XGBoost는 모든 성능 지표에서 RNN보다 우수한 결과를 보였습니다.
- 특히 Recall과 ROC-AUC에서 큰 차이를 보이며, 이탈 고객 예측에 더 효과적입니다.

RNN은 딥러닝 모델로 적용되었지만, 성능과 효율성 면에서 부족했습니다.
따라서 XGBoost가 최종 모델로 선정되었습니다.

---

## 기술 스택

---

## 한 줄 회고
- 김나예:
- 김서진:
- 나성호:
- 서주혁:
- 신동익:
