#%%
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
import torch
import itertools
import os
import random
os.chdir("C:/Users/TRAMANH-PC/Desktop/Temporal_graph/mimic_iv/ML/data")
#%%
patient=pd.read_csv("./extracted/cdm_person_extracted_relatecon.csv")
drugex=pd.read_csv("./extracted/cdm_drug_exposure_personExtracted_relatecon.csv")
concept=pd.read_csv("../../concept/CONCEPT.csv.gz",compression="gzip", on_bad_lines='skip',sep="\t",low_memory=False)
visit=pd.read_csv("./extracted/src_visit_occurrence_personExtracted_relatecon.csv")
# condition=pd.read_csv("./extracted/cdm_condition_occurrence_personExtract_b4utle_relatecon.csv")
measurement=pd.read_csv("./extracted/cdm_measurement_personExtracted_relatecon.csv")
device=pd.read_csv('./extracted/cdm_device_exposure_personExtracted_relatecon.csv')
procedure=pd.read_csv('./extracted/cdm_procedure_occurrence_personExtracted_relatecon.csv')
observation=pd.read_csv('./extracted/cdm_observation_personExtracted_relatecon.csv')
specimen=pd.read_csv('./extracted/cdm_specimen_personExtracted_relatecon.csv')

# %%
#filter useful columns from drug data
drugex=drugex[['drug_exposure_id', 'person_id', 'drug','quantity','visit_occurrence_id','startdate','dose_val_rx','route','dose_unit','form_val_disp']]
drugex['drug_type']="EHR"
drugex.head()

# %%
#filter useful columns from visit data
visit=visit[['visit_occurrence_id', 'person_id', 'visit','visit_type','startdate']]
visit.head()

# %%
#filter useful columns from measurement data
measurement=measurement[['measurement_id', 'person_id', 'measurement_concept_id', 'measurement_type_concept_id','operator_concept_id', 'value_as_number', 'value_as_concept_id','unit_concept_id', 'range_low', 'range_high', 'visit_occurrence_id','startdate']]
measurement.head()
# %%
##replace concept_ID in measurement data
#measurement_type
measurement2=measurement.merge(concept.iloc[:,[0,1]],left_on="measurement_type_concept_id",right_on='concept_id',how="left")
measurement2=measurement2.drop(['measurement_type_concept_id','concept_id'],axis=1)
measurement2.rename(columns={'concept_name':'measurement_type'},inplace=True)
# %%
#measurement
# measurement2.measurement_concept_id=measurement2.measurement_concept_id.astype(int, errors = 'ignore')
measurement2=measurement2.merge(concept.iloc[:,[0,1]],left_on="measurement_concept_id",right_on='concept_id',how="left")
measurement2=measurement2.drop(['concept_id'],axis=1)
measurement2.rename(columns={'concept_name':'measurement'},inplace=True)
# %%
#operator
measurement2=measurement2.merge(concept.iloc[:,[0,1]],left_on="operator_concept_id",right_on='concept_id',how="left")
measurement2=measurement2.drop(['operator_concept_id','concept_id'],axis=1)
measurement2.rename(columns={'concept_name':'operator'},inplace=True)
# %%
#value
measurement2=measurement2.merge(concept.iloc[:,[0,1]],left_on="value_as_concept_id",right_on='concept_id',how="left")
measurement2=measurement2.drop(['value_as_concept_id','concept_id'],axis=1)
measurement2.rename(columns={'concept_name':'value_concept'},inplace=True)
# %%
#unit
measurement2=measurement2.merge(concept.iloc[:,[0,1]],left_on="unit_concept_id",right_on='concept_id',how="left")
measurement2=measurement2.drop(['unit_concept_id','concept_id'],axis=1)
measurement2.rename(columns={'concept_name':'unit'},inplace=True)
measurement2.head()

# %%
#filter useful columns from device data
device=device[['device_exposure_id', 'person_id', 'device','quantity','visit_occurrence_id','startdate']]
device['device_type']='EHR'
device.head()

# %%
#filter useful columns from procedure data
procedure=procedure[['procedure_occurrence_id', 'person_id', 'procedure_concept_id','quantity','visit_occurrence_id','startdate']]
procedure['procedure_type']='EHR'
procedure.head()
# %%
##replace concept_ID in procedure data
#procedure 
procedure2=procedure.merge(concept.iloc[:,[0,1]],left_on="procedure_concept_id",right_on='concept_id',how="left")
procedure2=procedure2.drop(['concept_id'],axis=1)
procedure2.rename(columns={'concept_name':'procedure'},inplace=True)
procedure2.head()

# %%
#filter useful columns from observation data
observation=observation[['observation_id', 'person_id', 'observation_concept_id','value_as_concept_id','visit_occurrence_id','startdate']]
observation['observation_type']='EHR'
# observation.value_as_concept_id=observation.value_as_concept_id.astype(int, errors = 'ignore').astype(str)
# observation.value_as_concept_id=observation.value_as_concept_id.str[:-2]
# observation.unit_concept_id=observation.unit_concept_id.astype(int, errors = 'ignore').astype(str)
# observation.unit_concept_id=observation.unit_concept_id.str[:-2]
observation.head()
# %%
##replace concept_ID in observation data
#observation 
observation2=observation.merge(concept.iloc[:,[0,1]],left_on="observation_concept_id",right_on='concept_id',how="left")
observation2=observation2.drop(['concept_id'],axis=1)
observation2.rename(columns={'concept_name':'observation'},inplace=True)
# %%
#value
observation2=observation2.merge(concept.iloc[:,[0,1]],left_on="value_as_concept_id",right_on='concept_id',how="left")
observation2=observation2.drop(['value_as_concept_id','concept_id'],axis=1)
observation2.rename(columns={'concept_name':'value_concept'},inplace=True)
observation2.head()

# %%
#filter useful columns from specimen data
specimen['startdate']=specimen['start_datetime'].apply(lambda x: x.split(" ", 1)[0])
specimen['startdate']=specimen['startdate'].str.replace("-","").astype(int)
specimen=specimen[['specimen_id', 'person_id', 'specimen_concept_id','visit_occurrence_id','startdate']]
specimen['specimen_type']='EHR'
# specimen.visit_occurrence_id=specimen.visit_occurrence_id.astype(int, errors = 'ignore').astype(str)
# specimen.visit_occurrence_id=specimen.visit_occurrence_id.str[:-2]
# specimen.specimen_concept_id=specimen.specimen_concept_id.astype(int, errors = 'ignore').astype(str)
# specimen.specimen_concept_id=specimen.specimen_concept_id.str[:-2]
specimen.head()
# %%
##replace concept_ID in specimen data
#specimen 
specimen2=specimen.merge(concept.iloc[:,[0,1]],left_on="specimen_concept_id",right_on='concept_id',how="left")
specimen2=specimen2.drop(['concept_id'],axis=1)
specimen2.rename(columns={'concept_name':'specimen'},inplace=True)
specimen2.head()

#%%
utle_id=[192854,195769,195770,197236]
utle=pd.read_csv("../patients_have_uterine_leiomyoma.csv")
utle_date=utle[['startdate','person_id']][utle['condition_concept_id'].isin(utle_id)]
utle_date=utle_date.groupby('person_id').agg({'startdate':'min'}).reset_index()
utle_date=utle_date.rename(columns={'startdate':'fist_diagnose_utle'})

#%%
drugex2=drugex.merge(utle_date,on='person_id',how="left")
drugex2['fist_diagnose_utle']=drugex2['fist_diagnose_utle'].fillna(99999999)
drugex2=drugex2[drugex2['startdate']<drugex2['fist_diagnose_utle']]

#%%
visit2=visit.merge(utle_date,on='person_id',how="left")
visit2['fist_diagnose_utle']=visit2['fist_diagnose_utle'].fillna(99999999)
visit2=visit2[visit2['startdate']<visit2['fist_diagnose_utle']]

#%%
measurement3=measurement2.merge(utle_date,on='person_id',how="left")
measurement3['fist_diagnose_utle']=measurement3['fist_diagnose_utle'].fillna(99999999)
measurement3=measurement3[measurement3['startdate']<measurement3['fist_diagnose_utle']]

#%%
device2=device.merge(utle_date,on='person_id',how="left")
device2['fist_diagnose_utle']=device2['fist_diagnose_utle'].fillna(99999999)
device2=device2[device2['startdate']<device2['fist_diagnose_utle']]

#%%
procedure3=procedure2.merge(utle_date,on='person_id',how="left")
procedure3['fist_diagnose_utle']=procedure3['fist_diagnose_utle'].fillna(99999999)
procedure3=procedure3[procedure3['startdate']<procedure3['fist_diagnose_utle']]

#%%
observation3=observation2.merge(utle_date,on='person_id',how="left")
observation3['fist_diagnose_utle']=observation3['fist_diagnose_utle'].fillna(99999999)
observation3=observation3[observation3['startdate']<observation3['fist_diagnose_utle']]

#%%
specimen3=specimen2.merge(utle_date,on='person_id',how="left")
specimen3['fist_diagnose_utle']=specimen3['fist_diagnose_utle'].fillna(99999999)
specimen3=specimen3[specimen3['startdate']<specimen3['fist_diagnose_utle']]
specimen3=specimen3[specimen3.specimen_concept_id.isnull()==False]

#%%
label=pd.read_csv("../label_balance_utle_b4utle_relatecon.csv")
drugex2=drugex2[drugex2.person_id.isin(label.person_id)]
visit2=visit2[visit2.person_id.isin(label.person_id)]
measurement3=measurement3[measurement3.person_id.isin(label.person_id)]
device2=device2[device2.person_id.isin(label.person_id)]
procedure3=procedure3[procedure3.person_id.isin(label.person_id)]
observation3=observation3[observation3.person_id.isin(label.person_id)]
specimen3=specimen3[specimen3.person_id.isin(label.person_id)]

#%%
all_person=list(set.union(*map(set, [drugex2.person_id, visit2.person_id, measurement3.person_id,device2.person_id,procedure3.person_id,observation3.person_id,specimen3.person_id])))


#%%
patient2=patient.drop(['gender'],axis=1).copy()
patient2.head()

# %%
drug_summarize=drugex2.groupby('drug').agg({'person_id':'nunique'}).reset_index()
drug4=drugex2[['person_id','drug']].copy()
drug4=drug4[drug4.drug.isin(drug_summarize.drug[drug_summarize.person_id>=max(drug_summarize.person_id)*0.01])]
drug_summarize.drug[drug_summarize.person_id>=max(drug_summarize.person_id)*0.01].nunique()
drug4 =drug4[['person_id']].join(pd.get_dummies(drug4['drug'],dtype=int)).groupby('person_id').max().reset_index()
drug4.head()
# %%
measurement_summarize=measurement3.groupby('measurement').agg({'person_id':'nunique'}).reset_index()
measurement4=measurement3[['person_id','measurement']].copy()
measurement4=measurement4[measurement4.measurement.isin(measurement_summarize.measurement[measurement_summarize.person_id>=max(measurement_summarize.person_id)*0.01])]
measurement_summarize.measurement[measurement_summarize.person_id>=max(measurement_summarize.person_id)*0.01].nunique()
measurement4 =measurement4[['person_id']].join(pd.get_dummies(measurement4['measurement'],dtype=int)).groupby('person_id').max().reset_index()
measurement4.head()
# %%
device_summarize=device2.groupby('device').agg({'person_id':'nunique'}).reset_index()
device4=device2[['person_id','device']].copy()
device4=device4[device4.device.isin(device_summarize.device[device_summarize.person_id>=max(device_summarize.person_id)*0.01])]
device_summarize.device[device_summarize.person_id>=max(device_summarize.person_id)*0.01].nunique()
device4 =device4[['person_id']].join(pd.get_dummies(device4['device'],dtype=int)).groupby('person_id').max().reset_index()
device4.head()
# %%
# condition2=condition[condition.condition_concept_id.isin([192854,195769,195770,197236])==False]
# condition_summarize=condition2.groupby('condition').agg({'person_id':'nunique'}).reset_index()
# condition4=condition2[['person_id','condition']].copy()
# condition4=condition4[condition4.condition.isin(condition_summarize.condition[condition_summarize.person_id>=max(condition_summarize.person_id)*0.01])]
# condition_summarize.condition[condition_summarize.person_id>=max(condition_summarize.person_id)*0.01].nunique()
# condition4 =condition4[['person_id']].join(pd.get_dummies(condition4['condition'],dtype=int)).groupby('person_id').max().reset_index()
# condition4.head()

# %%
procedure_summarize=procedure3.groupby('procedure').agg({'person_id':'nunique'}).reset_index()
procedure4=procedure3[['person_id','procedure']].copy()
procedure4=procedure4[procedure4.procedure.isin(procedure_summarize.procedure[procedure_summarize.person_id>=max(procedure_summarize.person_id)*0.01])]
procedure_summarize.procedure[procedure_summarize.person_id>=max(procedure_summarize.person_id)*0.01].nunique()
procedure4 =procedure4[['person_id']].join(pd.get_dummies(procedure4['procedure'],dtype=int)).groupby('person_id').max().reset_index()
procedure4.head()

# %%
observation_summarize=observation3.groupby('observation').agg({'person_id':'nunique'}).reset_index()
observation4=observation3[['person_id','observation']].copy()
observation4=observation4[observation4.observation.isin(observation_summarize.observation[observation_summarize.person_id>=max(observation_summarize.person_id)*0.01])]
observation_summarize.observation[observation_summarize.person_id>=max(observation_summarize.person_id)*0.01].nunique()
observation4 =observation4[['person_id']].join(pd.get_dummies(observation4['observation'],dtype=int)).groupby('person_id').max().reset_index()
observation4.head()

# %%
drug_column=list(drug4.columns[1:len(drug4.columns)])
drug4.columns=[drug4.columns[0]] + ["drug_" + str(i) for i in range(1, len(drug4.columns))]

measurement_column=list(measurement4.columns[1:len(measurement4.columns)])
measurement4.columns=[measurement4.columns[0]] + ["measurement_" + str(i) for i in range(1, len(measurement4.columns))]

device_column=list(device4.columns[1:len(device4.columns)])
device4.columns=[device4.columns[0]] + ["device_" + str(i) for i in range(1, len(device4.columns))]

# condition_column=list(condition4.columns[1:len(condition4.columns)])
# condition4.columns=[condition4.columns[0]] + ["condition_" + str(i) for i in range(1, len(condition4.columns))]

observation_column=list(observation4.columns[1:len(observation4.columns)])
observation4.columns=[observation4.columns[0]] + ["observation_" + str(i) for i in range(1, len(observation4.columns))]

procedure_column=list(procedure4.columns[1:len(procedure4.columns)])
procedure4.columns=[procedure4.columns[0]] + ["procedure_" + str(i) for i in range(1, len(procedure4.columns))]

full_feature=list(patient2.columns[1])+drug_column+measurement_column+device_column+procedure_column+observation_column

#%%
big_df=patient2.merge(drug4,on="person_id",how="left")
big_df=big_df.merge(measurement4,on="person_id",how="left")
big_df=big_df.merge(device4,on="person_id",how="left")
# big_df=big_df.merge(condition4,on="person_id",how="left")
big_df=big_df.merge(procedure4,on="person_id",how="left")
big_df=big_df.merge(observation4,on="person_id",how="left")
big_df=big_df.merge(label[['person_id','utle']],on="person_id",how="left")
big_df = big_df.fillna(0)

#%%
from sklearn.model_selection import train_test_split
features=big_df.drop(['person_id','utle'],axis=1)
target=big_df['utle']
seed = 18399
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.1, random_state = seed)

#%%
from xgboost import XGBClassifier
xgb_model=XGBClassifier(max_depth=50, learning_rate=0.0001, n_estimators=15)
xgb_model.fit(X_train,y_train)

#%%
predctions=xgb_model.predict(X_test)
y_pred_prod=xgb_model.predict_proba(X_test)


#%%
import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(np.array(X_test), check_additivity=False)
# shap.summary_plot(shap_values, X_test_scaled, plot_type="bar")
shap.summary_plot(shap_values, X_test,feature_names=full_feature)


# %%
from sklearn.metrics import *
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prod[:,1])
roc_auc = auc(fpr, tpr)
from matplotlib import pyplot as plt
plt.plot(fpr,tpr,label='ROC curve (area = {})'.format(roc_auc))
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
# %%
f1_score(y_test,np.round(predctions), average='macro')
# %%
test_indices = X_test.index
X_test['predictions'] = predctions
X_test['actual'] = y_test.values
false_negatives = X_test[(X_test['actual'] == 1) & (X_test['predictions'] == 0)]
false_negative_ids = big_df.loc[false_negatives.index, 'person_id']

# %%
condition_fn=condition2[condition2.person_id.isin(false_negative_ids)]
condition_fb_summariz=condition_fn.groupby('condition').agg({'person_id':'nunique'}).reset_index()

# %%
drug_fn=drugex2[drugex2.person_id.isin(false_negative_ids)]
drug_fn_summariz=drug_fn.groupby('drug').agg({'person_id':'nunique'}).reset_index()
# %%
procedure_fn=procedure2[procedure2.person_id.isin(false_negative_ids)]
procedure_fb_summariz=procedure_fn.groupby('procedure').agg({'person_id':'nunique'}).reset_index()
# %%
procedure_fn=procedure2[procedure2.person_id.isin(false_negative_ids)]
procedure_fb_summariz=procedure_fn.groupby('procedure').agg({'person_id':'nunique'}).reset_index()
#%%
measurement_fn=measurement2[measurement2.person_id.isin(false_negative_ids)]
measurement_fb_summariz=measurement_fn.groupby('measurement').agg({'person_id':'nunique'}).reset_index()
# %%
def roc_curve(y_true, y_prob, thresholds):

    fpr = []
    tpr = []

    for threshold in thresholds:

        y_pred = np.where(y_prob >= threshold, 1, 0)

        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))

        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    return [fpr, tpr]
# %%
# %%
f1=f1_score(y_test, y_pred_rf, average='macro')
result=pd.DataFrame({'fpr':fpr,'tpr':tpr,'thresholds':thresholds})
result['roc_auc']=roc_auc
result['f1']=f1
result['model']="RF"
pred_vs_label=pd.DataFrame({'label':y_test,'pred':y_pred_prod_rf[:,1]})
pred_vs_label.to_csv("Prediction_vs_label_RF_noallbd_ondatedb_balance.csv")
result.to_csv("Result_RF_noallbd_ondatedb_balance.csv",index=False)
# %%
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prod[:,1])
auc_precision_recall = auc(recall, precision)
print(auc_precision_recall)
plt.plot(recall, precision,label='PR curve (area = {})'.format(auc_precision_recall))
plt.legend()
plt.ylabel("Precision")
plt.xlabel("Recall")
plt.title("Precision-Recall curve")
# %%
result=pd.DataFrame({'precision':precision,'recall':recall})
result['pr_auc']=auc_precision_recall
result['model']="RF"
result.to_csv("Result_PRcurve_RF_noallbd_ondatedb_balance.csv",index=False)
# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score
accuracy = accuracy_score(y_test, predctions)
precision = precision_score(y_test, predctions)
recall = recall_score(y_test, predctions)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")



# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Initialize the random forest classifier
ranfor=RandomForestClassifier(class_weight = 'balanced', n_estimators = 50, max_depth = 30, random_state = 42)
# Train the classifier on the training data
ranfor.fit(X_train,y_train)
# Make predictions on the test data
y_pred_rf=ranfor.predict(X_test)
y_pred_prod_rf=ranfor.predict_proba(X_test)

# %%
from sklearn.metrics import *
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prod_rf[:,1])
roc_auc = auc(fpr, tpr)
from matplotlib import pyplot as plt
plt.plot(fpr,tpr,label='ROC curve (area = {})'.format(roc_auc))
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
# %%
