import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import mode

# --- 1. 데이터 준비 및 전처리 ---
print("--- 1. Data Preparation and Preprocessing ---")
try:
    df = pd.read_csv('creditcard.csv')
except FileNotFoundError:
    print("Error: 'creditcard.csv' not found. Please check the file path.")
    exit()

# RobustScaler를 사용하여 'Time'과 'Amount' 피처 스케일링
rob_scaler = RobustScaler()
df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', df.pop('scaled_amount'))
df.insert(1, 'scaled_time', df.pop('scaled_time'))
print("Data preprocessing complete.")
print("-" * 60)

# --- 2. 최종 평가를 위한 데이터 분리 ---
print("--- 2. Splitting Data for Final Evaluation ---")
X = df.drop('Class', axis=1)
y = df['Class']

# 전체 데이터를 사용하여 최종 테스트 세트를 만듭니다.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
train_df = pd.concat([X_train, y_train], axis=1)
print(f"Train set size: {len(train_df)}")
print(f"Test set size: {len(X_test)}")
print("-" * 60)


# --- 3. 여러 개의 언더샘플링된 데이터셋 생성 ---
n_datasets = 50
undersampled_datasets = []
print(f"--- 3. Creating {n_datasets} Undersampled Datasets ---")

fraud_df = train_df[train_df['Class'] == 1]
non_fraud_df = train_df[train_df['Class'] == 0]

for i in range(n_datasets):
    non_fraud_sample = non_fraud_df.sample(n=len(fraud_df), random_state=42 + i)
    balanced_df = pd.concat([fraud_df, non_fraud_sample])
    undersampled_datasets.append(balanced_df)

print(f"All {n_datasets} datasets created. Each with ~{len(undersampled_datasets[0])} samples.")
print("-" * 60)


# --- 4. 모델 학습 및 평가 ---
start_time = time.time()
final_predictions = {}

# 사용할 모델 정의
log_reg_model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)

# a) 단일 언더샘플링 데이터셋으로 모델 학습
print("--- Training Logistic Regression on a Single Undersampled Dataset ---")
single_train_df = undersampled_datasets[0]
X_train_single = single_train_df.drop('Class', axis=1)
y_train_single = single_train_df['Class']

log_reg_model.fit(X_train_single, y_train_single)
y_pred_single = log_reg_model.predict(X_test)
final_predictions['LogisticRegression_Single_Dataset'] = y_pred_single


# b) 여러 데이터셋에 대한 투표 앙상블 (배깅)
print(f"--- Training Voting Ensemble on {n_datasets} Datasets ---")
predictions_list = []
for i, dataset in enumerate(undersampled_datasets):
    X_train_sample = dataset.drop('Class', axis=1)
    y_train_sample = dataset['Class']
    log_reg_model.fit(X_train_sample, y_train_sample)
    y_pred = log_reg_model.predict(X_test)
    predictions_list.append(y_pred)

predictions_array = np.array(predictions_list)
majority_vote_preds, _ = mode(predictions_array, axis=0, keepdims=False)
final_predictions['LogisticRegression_Voting_Ensemble'] = majority_vote_preds

end_time = time.time()
print(f"\nTotal training and prediction time: {end_time - start_time:.2f} seconds")
print("-" * 60)


# --- 5. 최종 모델 성능 비교 ---
print("\n--- Overall Model Performance Comparison ---")
for name, preds in final_predictions.items():
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, pos_label=1, zero_division=0)
    recall = recall_score(y_test, preds, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)
    
    print(f"\n>> Model: {name}")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - Precision (for Fraud class): {precision:.4f}")
    print(f"  - Recall (for Fraud class): {recall:.4f}")
    print(f"  - F1-Score (for Fraud class): {f1:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, preds, zero_division=0))
    print("-" * 40) 