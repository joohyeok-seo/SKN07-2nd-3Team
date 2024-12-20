# Booribooribang Team

## Team Introduction
<table>
  <tr>
    <th>Naye</th>
    <th>Seojin</th>
    <th>Seongho</th>
    <th>Joohyeok</th>
    <th>Dongik</th>
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

## Project Overview
Customer churn prediction plays a crucial role in improving customer experience and enhancing competitiveness. Predictive models can identify customer dissatisfaction or concerns in advance, enabling companies to address these issues proactively. This approach not only increases customer satisfaction but also provides better services than competitors, maintaining customer loyalty and ensuring long-term business growth.
Our project focuses on building a customer churn prediction model for subscription-based businesses, e-commerce platforms, financial services companies, and other service providers. Through this model, companies can devise strategies to reduce churn, improve customer management efficiency, and establish competitive market strategies.

### Project Title
**Customer Churn Defense Strategy!!!**

### Overview
#### Design and Construction of a Customer Churn Prediction Model

**1. Selection of Training and Test Data**
- Selected data with minimal noise and fewer unnecessary features to train the model effectively.

**2. Comparison of ML Performance**
- Compared the performance of five machine learning algorithms using a consistent dataset, selecting the algorithm with the best performance.
- Used the Telecom Churn Dataset from the Kaggle machine learning community.

**3. Design and Construction of the Prediction Model**
- Constructed and validated the model using the selected algorithm for maximum performance.

### Objective
- To build an optimized and high-performing customer churn prediction model through the comparison and analysis of various machine learning models.

---

## Key Data Description
<table> <tr> <th>Number</th> <th>Column</th> <th>Meaning</th> <th>Example</th> </tr> <tr> <th>1</th> <th>State</th> <th>The state where the customer resides</th> <th>CA" (California), "NY" (New York)</th> </tr> <tr> <th>2</th> <th>International plan</th> <th>Whether the customer has subscribed to an international calling plan</th> <th>"Yes" (Subscribed), "No" (Not subscribed)</th> </tr> <tr> <th>3</th> <th>Voice mail plan</th> <th>Whether the customer has subscribed to a voicemail plan</th> <th>"Yes" (Subscribed), "No" (Not subscribed)</th> </tr> <tr> <th>4</th> <th>Total day minutes</th> <th>Total call duration during daytime (minutes)</th> <th>265.1 → 265.1 minutes</th> </tr> <tr> <th>5</th> <th>Total eve minutes</th> <th>Total call duration during evening hours (minutes)</th> <th>197.4 → 197.4 minutes</th> </tr> <tr> <th>6</th> <th>Total intl minutes</th> <th>Total call duration for international calls (minutes)</th> <th>10.0 → 10 minutes</th> </tr> <tr> <th>7</th> <th>Customer service calls</th> <th>Number of calls made to customer service</th> <th>1 → 1 call</th> </tr> <tr> <th>8</th> <th>Churn</th> <th>Whether the customer has churned</th> <th>True (Churned), False (Retained)</th> </tr> </table>

---

## EDA
### Correlation Matrix
![download](https://github.com/user-attachments/assets/6f8277cb-c581-465d-b751-56b82aa42e2b)
- A strong correlation is observed between features like Total Day Minutes and Customer Service Calls with Churn. It is confirmed that longer weekly call durations are associated with higher.
  
### 주간 통화 이용률
![download](https://github.com/user-attachments/assets/0506510c-8f6c-4107-ad54-341797303ecf)
- Customers with higher call durations during the day (Total Day Minutes) show relatively higher churn rates. This suggests that higher call volumes may increase the likelihood of dissatisfaction.
  
### 야간 통화 이용률
![download](https://github.com/user-attachments/assets/f3801354-610a-4863-8835-173f94369d2d)
- Call volume during the evening (Total Eve Minutes) shows a relatively low correlation with churn rates.

### 심야 통화 이용률
![download](https://github.com/user-attachments/assets/986f494c-42d6-4a46-9389-824cc258261a)
- Call volume during the night (Total Night Minutes) shows minimal correlation with customer churn.

### 고객센터 상담 비율
<table>
  <tr>
    <td align="center"><b>Distribution of Customer Service Calls</b></td>
    <td align="center"><b>Customer Service Calls별 Churn 비율</b></td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/1dac0981-380d-46f8-b8cc-7872f46c555c" alt="Distribution Graph" width="600">
    </td>
    <td>
      <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/curstomer_service_calls.png" alt="Churn Rate Graph" width="400">
    </td>
  </tr>
</table>

- As the number of Customer Service Calls increases, the probability of churn rises sharply, reflecting customer dissatisfaction and challenges in issue resolution.

### 국제전화 가입비율
<table>
  <tr>
    <td align="center"><b>국제전화 분포</b></td>
    <td align="center"><b>국제전화 가입별 Churn 비율</b></td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/1db52c47-3d11-4d62-b159-37f264a24ee5" alt="Distribution Graph" width="500">
    </td>
    <td>
      <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/Subscribed.png" alt="Churn Rate Graph" width="500">
    </td>
  </tr>
</table>

- Customers subscribed to international call plans exhibit higher churn rates, highlighting the need for strategies to improve service quality or address customer dissatisfaction.

### 음성사사함 가입비율
<table>
  <tr>
    <td align="center"><b>음성사서함 분포</b></td>
    <td align="center"><b>음성사서함 가입별 Churn 비율</b></td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/aa216f78-ad8d-4a0b-83d1-7b5d57e456bf" alt="Distribution Graph" width="500">
    </td>
    <td>
      <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/voice_mail.png" alt="Churn Rate Graph" width="500">
    </td>
  </tr>
</table>

- Customers without a Voice Mail Plan show relatively higher churn rates, suggesting the potential of voice mail plans to contribute to customer retention.

### 이탈률
![download](https://github.com/user-attachments/assets/aa611630-d6a6-473f-8955-8428e40611d2)
- Although the overall churn rate among customers is low, churn occurs more frequently under specific conditions, such as high call volumes and increased customer service usage.

### 고객센터 통화량에 따른 이탈률 & 이상치
![churn_vs_customer_service_calls](https://github.com/user-attachments/assets/e491b51b-ce10-41ea-b632-9ded739fc8e7)
- Customers who made more than five calls to customer service show significantly higher churn rates, indicating that service dissatisfaction is a major contributing factor.

### 통화량이 높은 고객들 대상 이탈률
![churn_vs_total_day_minutes](https://github.com/user-attachments/assets/9618780c-ebd7-4a49-8a94-511026fcb1cf)
- Customers with abnormally high call volumes experience a sharp increase in churn rates, indicating the need for targeted management for these individuals.

### Data Preprocessing
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
    <th>Decision Tree</th>
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

### Final Model Selection Process
In this project, the XGBoost and Random Forest models demonstrated the highest performance. By comparing the key performance metrics of each model, the following conclusions were drawn:

- Random Forest: Achieved the highest Precision, but its relatively low Recall indicated a higher likelihood of missing churn customers.

- XGBoost: Delivered more balanced performance in terms of Recall and F1 Score. Its higher Recall compared to Random Forest was particularly advantageous in predicting more churn customers.

### Final Model Selection (XGBoost Chosen)
1. Balanced Performance
- Recall is higher than Random Forest, allowing more churn customers to be predicted.
- Maintains a high F1 Score, ensuring a balance between Precision and Recall.
2. Overfitting Prevention
- Random Forest showed high Precision but exhibited a tendency to overfit.
- XGBoost demonstrated stable and superior generalization performance.
3. Interpretability
- XGBoost enables the analysis of Feature Importance, providing clear insights into the factors influencing customer churn.

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

### Limitations of the RNN Model
1. Performance Deficiency
- While the Accuracy of 88.9% is decent, the Recall is extremely low at 43.2%.
- This indicates a significant limitation in predicting churn customers effectively.
2. F1 Score
- The F1 Score is 0.526, highlighting a lack of balance between Precision and Recall.
3. ROC-AUC
- The ROC-AUC score is 0.698, showing inferior discriminative ability compared to other machine learning models such as XGBoost and Random Forest.

### Comparison with XGBoost
- XGBoost outperformed RNN across all performance metrics.
- The most notable differences were in Recall and ROC-AUC, where XGBoost proved to be far more effective in predicting churn customers.

Although RNN was applied as a deep learning model, it fell short in terms of performance and efficiency. 
Therefore, XGBoost was selected as the final model.

---

## Churn Prediction Model Implementation
Users can select various machine learning models and upload a CSV file to check the churn prediction results.

### Project Execution Process
1. CSV File Upload
- Upload the dataset to be used for prediction.
2. Model Selection
- Users can choose from various models such as Decision Tree, Random Forest, or XGBoost.
3. Review Prediction Results
- Users can check the predicted churn rate and the number of customers likely to churn.

<table> 
  <tr> <td align="center"><b>CSV File Upload</b></td> <td align="center"><b>Model Selection</b></td> <td align="center"><b>Churn Prediction Results</b></td> </tr> <tr> <td> <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/streamlit%2001.png" alt="Model Selection" width="300"> </td> <td> <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/streamlit%2002.png" alt="CSV File Upload" width="300"> </td> <td> <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/streamlit%2003.png" alt="Churn Prediction Results" width="300"> </td> </tr> 
</table>

--- 

## 한 줄 회고
- 김나예:
- 김서진:
- 나성호:
- 서주혁:
- 신동익:
