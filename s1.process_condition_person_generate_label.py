#%%
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
import torch
import itertools
import os
import random
os.chdir("C:/Users/TRAMANH-PC/Desktop/Temporal_graph/mimic_iv/ML")
# %%
#Load data
patient=pd.read_csv("./data/src_patient.csv.gz", compression="gzip")
concept=pd.read_csv("../concept/CONCEPT.csv.gz",compression="gzip", on_bad_lines='skip',sep="\t",low_memory=False)
condition=pd.read_csv("./data/cdm_condition_occurrence.csv.gz", compression="gzip")

# %%
#filter useful columns from condition data
condition=condition[['condition_occurrence_id', 'person_id', 'condition_concept_id','condition_type_concept_id','visit_occurrence_id','startdate']]
condition.head()
# %%
##replace concept_ID in condition data
#condition_type
condition2=condition.merge(concept.iloc[:,[0,1]],left_on="condition_type_concept_id",right_on='concept_id',how="left")
condition2=condition2.drop(['condition_type_concept_id','concept_id'],axis=1)
condition2.rename(columns={'concept_name':'condition_type'},inplace=True)
# %%
#condition
condition2=condition2.merge(concept.iloc[:,[0,1]],left_on="condition_concept_id",right_on='concept_id',how="left")
condition2=condition2.drop(['concept_id'],axis=1)
condition2.rename(columns={'concept_name':'condition'},inplace=True)
condition2.head()

#%%
utle_id=[192854,195769,195770,197236]
utle=condition2[condition2.condition_concept_id.isin(utle_id)]
# utle.to_csv("patients_have_uterine_leiomyoma.csv",index=False)
patient_utle=utle.person_id.unique()

#%%
utle_date=utle[['startdate','person_id']][utle['condition_concept_id'].isin(utle_id)]
utle_date=utle_date.groupby('person_id').agg({'startdate':'min'}).reset_index()
utle_date=utle_date.rename(columns={'startdate':'fist_diagnose_utle'})

#%%
condition2=condition2.merge(utle_date,on='person_id',how="left")
condition2['fist_diagnose_utle']=condition2['fist_diagnose_utle'].fillna(99999999)
condition3=condition2[condition2['startdate']<=condition2['fist_diagnose_utle']]

#%%
patient2=patient[patient.person_id.isin(condition3.person_id)]
patient2=patient2[patient2.gender=="F"]
label=patient2.copy()
# label.person_id.nunique() #95479
label['utle']=np.where(label['person_id'].isin(patient_utle), 1, 0)
# label.utle.sum() #2864
random.seed(514)
exclude_person=random.sample(list(label.person_id[label.utle==0]), (95479-2864)-2864)
label=label[label.person_id.isin(exclude_person)==False]
label=label[['person_id','utle']]
label.to_csv('label_balance_utle_b4utle_514.csv',index=False)
label_person=label['person_id']
label_person.to_csv('person_id_utle_exact_514.csv',index=False)
#%%
condition4=condition3[condition3.person_id.isin(label.person_id)]
condition4.to_csv("./data/extracted/cdm_condition_occurrence_personExtract_b4utle_514.csv",index=False)
patient3=patient2[patient2.person_id.isin(label.person_id)]
patient3.to_csv("./data/extracted/cdm_person_extracted_514.csv",index=False)


# %%
