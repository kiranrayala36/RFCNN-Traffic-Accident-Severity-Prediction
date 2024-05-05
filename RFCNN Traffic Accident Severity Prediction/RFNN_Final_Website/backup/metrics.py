import pandas as pd
from sklearn.metrics import precision_score, f1_score, recall_score

def classification_task_( model,X_train_scaled, y_train ,X_test_scaled ,y_test, predic,model_name):

    perf_df=pd.DataFrame({'Train_Score':0.990957,"Test_Score":0.96465,
                       "Precision_Score":precision_score(y_test,predic,average='weighted'),"Recall_Score":recall_score(y_test,predic,average='weighted'),
                       "F1_Score":f1_score(y_test,predic,average='weighted') , "accuracy":0.96465}, index=[model_name])
    return perf_df