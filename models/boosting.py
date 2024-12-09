from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False
)



