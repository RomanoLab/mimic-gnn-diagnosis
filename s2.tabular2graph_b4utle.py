 #%%
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
import torch
import itertools
import os
import random
os.chdir("C:/Users/TRAMANH-PC/Desktop/Temporal_graph/mimic_iv/ML/data")
# %%
#Load data
patient=pd.read_csv("./extracted/cdm_person_extracted_relatecon.csv")
drugex=pd.read_csv("./extracted/cdm_drug_exposure_personExtracted_relatecon.csv")
concept=pd.read_csv("../../concept/CONCEPT.csv.gz",compression="gzip", on_bad_lines='skip',sep="\t",low_memory=False)
visit=pd.read_csv("./extracted/src_visit_occurrence_personExtracted_relatecon.csv")
condition=pd.read_csv("./extracted/cdm_condition_occurrence_personExtract_b4utle_relatecon.csv")
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
# measurement.operator_concept_id=measurement.operator_concept_id.astype(int, errors = 'ignore').astype(str)
# measurement.operator_concept_id=measurement.operator_concept_id.str[:-2]
# measurement.value_as_concept_id=measurement.value_as_concept_id.astype(int, errors = 'ignore').astype(str)
# measurement.value_as_concept_id=measurement.value_as_concept_id.str[:-2]
# measurement.unit_concept_id=measurement.unit_concept_id.astype(int, errors = 'ignore').astype(str)
# measurement.unit_concept_id=measurement.unit_concept_id.str[:-2]
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
# visit2=visit.merge(utle_date,on='person_id',how="left")
# visit2['fist_diagnose_utle']=visit2['fist_diagnose_utle'].fillna(99999999)
# visit2=visit2[visit2['startdate']<visit2['fist_diagnose_utle']]

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

#%%
label=pd.read_csv("../label_balance_utle_b4utle_relatecon.csv")
drugex2=drugex2[drugex2.person_id.isin(label.person_id)]
measurement3=measurement3[measurement3.person_id.isin(label.person_id)]
device2=device2[device2.person_id.isin(label.person_id)]
procedure3=procedure3[procedure3.person_id.isin(label.person_id)]
observation3=observation3[observation3.person_id.isin(label.person_id)]
specimen3=specimen3[specimen3.person_id.isin(label.person_id)]

#%%
all_person=list(set.union(*map(set, [drugex2.person_id, measurement3.person_id,device2.person_id,procedure3.person_id,observation3.person_id,specimen3.person_id,condition.person_id])))
procedure3.visit_occurrence_id=pd.to_numeric(procedure3.visit_occurrence_id,errors="coerce")
procedure3.visit_occurrence_id=procedure3.visit_occurrence_id.astype(int,errors="ignore")
all_visit=np.unique(np.concatenate([drugex2.visit_occurrence_id, measurement3.visit_occurrence_id,device2.visit_occurrence_id,procedure3.visit_occurrence_id,observation3.visit_occurrence_id,specimen3.visit_occurrence_id,condition.visit_occurrence_id]))
visit2=visit[visit.person_id.isin(label.person_id)]
visit2=visit[visit.visit_occurrence_id.isin(all_visit)]
# %%
###PATIENT NODES
#Generate patient ID mapping
patient_sorted_df = patient.sort_values(by="person_id").drop_duplicates().set_index("person_id")
patient_sorted_df = patient_sorted_df.reset_index(drop=False)
patient_id_mapping = patient_sorted_df["person_id"].drop_duplicates().reset_index(drop=False)
patient_id_mapping = patient_sorted_df["person_id"].unique()
patient_id_mapping = pd.DataFrame(data = {
    'patientID': patient_id_mapping,
    'mappedID': pd.RangeIndex(len(patient_id_mapping))
})
# %%
#sort features by id_mapping
patient_node_map = patient_sorted_df[["person_id","birthyear"]].drop_duplicates().reset_index(drop=False)
patient_node_map = patient_node_map.set_index('person_id')
patient_node_map = patient_node_map.reindex(index=patient_id_mapping['patientID'])
patient_node_map = patient_node_map.reset_index(drop=False)
patient_node_map =patient_node_map[["birthyear"]]

#%%
#generate nodes features for patient
patient_node_features = patient_node_map


#%%
###DRUG NODES
#Generate drug ID mapping
drug_id=drugex2['drug'].sort_values().drop_duplicates().reset_index(drop=True).to_frame()
drug_id['drug_id']=list(range(1, drug_id.shape[0]+1))
drugex3=drugex2.merge(drug_id)
#%%
drug_sorted_df = drugex3.sort_values(by="drug_id").drop_duplicates().set_index("drug_id")
drug_sorted_df = drug_sorted_df.reset_index(drop=False)
drug_id_mapping = drug_sorted_df["drug_id"].unique()
drug_id_mapping = pd.DataFrame(data = {
    'drugID': drug_id_mapping,
    'mappedID': pd.RangeIndex(len(drug_id_mapping))
})
# %%
#sort features by id_mapping
drug_node_map = drug_sorted_df[["drug_id","drug_type"]].drop_duplicates().reset_index(drop=False)
drug_node_map = drug_node_map.set_index('drug_id')
drug_node_map = drug_node_map.reindex(index=drug_id_mapping['drugID'])
drug_node_map = drug_node_map.reset_index(drop=False)
drug_node_map = drug_node_map[['drug_type']]
#%%
#generate nodes features for drug
drug_node_features = pd.concat([drug_node_map, pd.get_dummies(drug_node_map["drug_type"], dtype=int)], axis=1, join='inner')
drug_node_features.drop(["drug_type"], axis=1, inplace=True)
drug_node_features.head()

#%%
drug_id_name=drug_id_mapping.merge(drug_sorted_df[["drug_id",'drug']],left_on='drugID',right_on='drug_id').drop_duplicates().drop('drug_id',axis=1).reset_index(drop=True)
drug_id_name.to_csv('Drug_nodeID_name_relatecon_b4utle.csv')

# %%
###VISIT NODES
#Generate visit ID mapping
visit_sorted_df = visit2.sort_values(by="visit_occurrence_id").drop_duplicates().set_index("visit_occurrence_id")
visit_sorted_df = visit_sorted_df.reset_index(drop=False)
visit_id_mapping = visit_sorted_df["visit_occurrence_id"].unique()
visit_id_mapping = pd.DataFrame(data = {
    'visitID': visit_id_mapping,
    'mappedID': pd.RangeIndex(len(visit_id_mapping))
})
# %%
#sort features by id_mapping
visit_node_map = visit_sorted_df[["visit_occurrence_id",'visit_type','visit']].drop_duplicates().reset_index(drop=False)
visit_node_map = visit_node_map.set_index('visit_occurrence_id')
visit_node_map = visit_node_map.reindex(index=visit_id_mapping['visitID'])
visit_node_map = visit_node_map.reset_index(drop=False)
visit_node_map =visit_node_map[['visit_type','visit']]
#%%
#generate nodes features for visit
visit_node_features = pd.concat([visit_node_map, pd.get_dummies(visit_node_map["visit_type"], dtype=int)], axis=1, join='inner')
visit_node_features = pd.concat([visit_node_features, pd.get_dummies(visit_node_features["visit"], dtype=int)], axis=1, join='inner')
visit_node_features.drop(["visit_type",'visit'], axis=1, inplace=True)
visit_node_features.head()

# %%
###CONDITION NODES: only have node label
#Generate condition ID mapping
condition_sorted_df = condition.sort_values(by="condition_concept_id").drop_duplicates().set_index("condition_concept_id")
condition_sorted_df = condition_sorted_df.reset_index(drop=False)
condition_id_mapping = condition_sorted_df["condition_concept_id"].unique()
condition_id_mapping = pd.DataFrame(data = {
    'conditionID': condition_id_mapping,
    'mappedID': pd.RangeIndex(len(condition_id_mapping))
})
condition_id_name=condition_id_mapping.merge(condition_sorted_df[["condition_concept_id",'condition']],left_on='conditionID',right_on='condition_concept_id').drop_duplicates().drop('conditionID',axis=1).reset_index(drop=True)
condition_id_name.to_csv('condition_nodeID_name_relatecon_b4utle.csv')

# %%
###MEASUREMENT NODES
#Generate measurement ID mapping
measurement3.measurement_type="EHR"
measurement_sorted_df = measurement3.sort_values(by="measurement_concept_id").drop_duplicates().set_index("measurement_concept_id")
measurement_sorted_df = measurement_sorted_df.reset_index(drop=False)
measurement_id_mapping = measurement_sorted_df["measurement_concept_id"].unique()
measurement_id_mapping = pd.DataFrame(data = {
    'measurementID': measurement_id_mapping,
    'mappedID': pd.RangeIndex(len(measurement_id_mapping))
})
# %%
#sort features by id_mapping
measurement_node_map = measurement_sorted_df[["measurement_concept_id",'measurement_type']].drop_duplicates().reset_index(drop=False)
measurement_node_map = measurement_node_map.set_index('measurement_concept_id')
measurement_node_map = measurement_node_map.reindex(index=measurement_id_mapping['measurementID'])
measurement_node_map = measurement_node_map.reset_index(drop=False)
measurement_node_map =measurement_node_map[['measurement_type']]
#%%
#generate nodes features for measurement
measurement_node_features = pd.concat([measurement_node_map, pd.get_dummies(measurement_node_map["measurement_type"], dtype=int)], axis=1, join='inner')
measurement_node_features.drop(["measurement_type"], axis=1, inplace=True)
measurement_node_features.head()

#%%
measurement_id_name=measurement_id_mapping.merge(measurement_sorted_df[["measurement_concept_id",'measurement']],left_on='measurementID',right_on='measurement_concept_id').drop_duplicates().drop('measurementID',axis=1).reset_index(drop=True)
measurement_id_name.to_csv('measurement_nodeID_name_relatecon_b4utle.csv')

#%%
###DEVICE NODES
#Generate device ID mapping
device_id=device2['device'].sort_values().drop_duplicates().reset_index(drop=True).to_frame()
device_id['device_id']=list(range(1, device_id.shape[0]+1))
device3=device2.merge(device_id)
#%%
device_sorted_df = device3.sort_values(by="device_id").drop_duplicates().set_index("device_id")
device_sorted_df = device_sorted_df.reset_index(drop=False)
device_id_mapping = device_sorted_df["device_id"].unique()
device_id_mapping = pd.DataFrame(data = {
    'deviceID': device_id_mapping,
    'mappedID': pd.RangeIndex(len(device_id_mapping))
})
# %%
#sort features by id_mapping
device_node_map = device_sorted_df[["device_id","device_type"]].drop_duplicates().reset_index(drop=False)
device_node_map = device_node_map.set_index('device_id')
device_node_map = device_node_map.reindex(index=device_id_mapping['deviceID'])
device_node_map = device_node_map.reset_index(drop=False)
device_node_map = device_node_map[['device_type']]
#%%
#generate nodes features for device
device_node_features = pd.concat([device_node_map, pd.get_dummies(device_node_map["device_type"], dtype=int)], axis=1, join='inner')
device_node_features.drop(["device_type"], axis=1, inplace=True)
device_node_features.head()

#%%
device_id_name=device_id_mapping.merge(device_sorted_df[["device_id",'device']],left_on='deviceID',right_on='device_id').drop_duplicates().drop('device_id',axis=1).reset_index(drop=True)
device_id_name.to_csv('device_nodeID_name_relatecon_b4utle.csv')

#%%
###PROCEDURE NODES
#Generate drug ID mapping
procedure_sorted_df = procedure3.sort_values(by="procedure_concept_id").drop_duplicates().set_index("procedure_concept_id")
procedure_sorted_df = procedure_sorted_df.reset_index(drop=False)
procedure_id_mapping = procedure_sorted_df["procedure_concept_id"].unique()
procedure_id_mapping = pd.DataFrame(data = {
    'procedureID': procedure_id_mapping,
    'mappedID': pd.RangeIndex(len(procedure_id_mapping))
})

#%%
procedure_id_name=procedure_id_mapping.merge(procedure_sorted_df[["procedure_concept_id",'procedure']],left_on='procedureID',right_on='procedure_concept_id').drop_duplicates().drop('procedureID',axis=1).reset_index(drop=True)
procedure_id_name.to_csv('procedure_nodeID_name_relatecon_b4utle.csv')

#%%
###OBSERVATION NODES
#Generate observation ID mapping
observation_sorted_df = observation3.sort_values(by="observation_concept_id").drop_duplicates().set_index("observation_concept_id")
observation_sorted_df = observation_sorted_df.reset_index(drop=False)
observation_id_mapping = observation_sorted_df["observation_concept_id"].unique()
observation_id_mapping = pd.DataFrame(data = {
    'observationID': observation_id_mapping,
    'mappedID': pd.RangeIndex(len(observation_id_mapping))
})
# %%
#sort features by id_mapping
observation_node_map = observation_sorted_df[["observation_concept_id","observation_type"]].drop_duplicates().reset_index(drop=False)
observation_node_map = observation_node_map.set_index('observation_concept_id')
observation_node_map = observation_node_map.reindex(index=observation_id_mapping['observationID'])
observation_node_map = observation_node_map.reset_index(drop=False)
observation_node_map = observation_node_map[['observation_type']]
#%%
#generate nodes features for observation
observation_node_features = pd.concat([observation_node_map, pd.get_dummies(observation_node_map["observation_type"], dtype=int)], axis=1, join='inner')
observation_node_features.drop(["observation_type"], axis=1, inplace=True)
observation_node_features.head()

#%%
observation_id_name=observation_id_mapping.merge(observation_sorted_df[["observation_concept_id",'observation']],left_on='observationID',right_on='observation_concept_id').drop_duplicates().drop('observationID',axis=1).reset_index(drop=True)
observation_id_name.to_csv('observation_nodeID_name_relatecon_b4utle.csv')

#%%
###specimen NODES
#Generate specimen ID mapping
specimen3=specimen3[specimen3.specimen_concept_id.isnull()==False]
specimen_sorted_df = specimen3.sort_values(by="specimen_concept_id").drop_duplicates().set_index("specimen_concept_id")
specimen_sorted_df = specimen_sorted_df.reset_index(drop=False)
specimen_id_mapping = specimen_sorted_df["specimen_concept_id"].unique()
specimen_id_mapping = pd.DataFrame(data = {
    'specimenID': specimen_id_mapping,
    'mappedID': pd.RangeIndex(len(specimen_id_mapping))
})
# %%
#sort features by id_mapping
specimen_node_map = specimen_sorted_df[["specimen_concept_id","specimen_type"]].drop_duplicates().reset_index(drop=False)
specimen_node_map = specimen_node_map.set_index('specimen_concept_id')
specimen_node_map = specimen_node_map.reindex(index=specimen_id_mapping['specimenID'])
specimen_node_map = specimen_node_map.reset_index(drop=False)
specimen_node_map = specimen_node_map[['specimen_type']]

#%%
#generate nodes features for specimen
specimen_node_features = pd.concat([specimen_node_map, pd.get_dummies(specimen_node_map["specimen_type"], dtype=int)], axis=1, join='inner')
specimen_node_features.drop(["specimen_type"], axis=1, inplace=True)
specimen_node_features.head()

#%%
specimen_id_name=specimen_id_mapping.merge(specimen_sorted_df[["specimen_concept_id",'specimen']],left_on='specimenID',right_on='specimen_concept_id').drop_duplicates().drop('specimenID',axis=1).reset_index(drop=True)
specimen_id_name.to_csv('specimen_nodeID_name_relatecon_b4utle.csv')

# %%
###PATIENT -> DRUG edge: patient exposed to drug
#filter some useful features for edge
p2d=drug_sorted_df[['drug_id','person_id','quantity','dose_val_rx','route','dose_unit','form_val_disp','drug_exposure_id']]
p2d = pd.concat([p2d, pd.get_dummies(p2d["dose_unit"], dtype=int)], axis=1, join='inner')
p2d = pd.concat([p2d, pd.get_dummies(p2d["route"], dtype=int)], axis=1, join='inner')
p2d.drop(['dose_unit','route'],inplace=True,axis=1)
# %%
#edge index
p2d_patient_id = pd.merge(p2d['person_id'],patient_id_mapping,left_on='person_id',right_on='patientID',how='left')
p2d_patient_id = torch.from_numpy(p2d_patient_id['mappedID'].values)

p2d_drug_id = pd.merge(p2d['drug_id'],drug_id_mapping,left_on='drug_id',right_on='drugID',how='left')
p2d_drug_id = torch.from_numpy(p2d_drug_id['mappedID'].values)

p2d_edge_index = torch.stack([p2d_patient_id, p2d_drug_id], dim=0)
print(p2d_edge_index)
p2d_edge_index.dtype
# %%
#edge features
# p2d_edge_features = p2d.drop(["drug_concept_id", "person_id"],axis=1) #temporal graph will be touched later
p2d_edge_features = p2d.drop(["drug_id", "person_id",'drug_exposure_id'],axis=1)
p2d_edge_features.head()

# %%
###PATIENT -> VISIT edge: patient has medical visit
#edge index
p2v = visit_sorted_df[["person_id","visit_occurrence_id"]]

p2v_patient_id = pd.merge(p2v['person_id'],patient_id_mapping,left_on='person_id',right_on='patientID',how='left')
p2v_patient_id = torch.from_numpy(p2v_patient_id['mappedID'].values)

p2v_visit_id = pd.merge(p2v['visit_occurrence_id'],visit_id_mapping,left_on='visit_occurrence_id',right_on='visitID',how='left')
p2v_visit_id = torch.from_numpy(p2v_visit_id['mappedID'].values)

p2v_edge_index = torch.stack([p2v_patient_id, p2v_visit_id], dim=0)
print(p2v_edge_index)
p2v_edge_index.dtype
# %%
###PATIENT -> CONDITION edge: patient has condition
#filter some useful features for edge
# p2c=condition_sorted_df[['condition_concept_id','condition_occurrence_id','person_id','condition_type']]
# p2c = pd.concat([p2c, pd.get_dummies(p2c["condition_type"], dtype=int)], axis=1, join='inner')
# # %%
# #edge index
# p2c_patient_id = pd.merge(p2c['person_id'],patient_id_mapping,left_on='person_id',right_on='patientID',how='left')
# p2c_patient_id = torch.from_numpy(p2c_patient_id['mappedID'].values)

# p2c_condition_id = pd.merge(p2c['condition_concept_id'],condition_id_mapping,left_on='condition_concept_id',right_on='conditionID',how='left')
# p2c_condition_id = torch.from_numpy(p2c_condition_id['mappedID'].values)

# p2c_edge_index = torch.stack([p2c_patient_id, p2c_condition_id], dim=0)
# print(p2c_edge_index)
# p2c_edge_index.dtype

# %%
###PATIENT -> MEASUREMENT edge: patient has lab measurement
#filter some useful features for edge
p2m=measurement_sorted_df[['measurement_concept_id','measurement_id','person_id','value_as_number','range_low','range_high']]
# %%
#edge index
p2m_patient_id = pd.merge(p2m['person_id'],patient_id_mapping,left_on='person_id',right_on='patientID',how='left').dropna()
p2m_patient_id = torch.from_numpy(p2m_patient_id['mappedID'].values)

p2m_measurement_id = pd.merge(p2m['measurement_concept_id'],measurement_id_mapping,left_on='measurement_concept_id',right_on='measurementID',how='left').dropna()
p2m_measurement_id = torch.from_numpy(p2m_measurement_id['mappedID'].values)

p2m_edge_index = torch.stack([p2m_patient_id, p2m_measurement_id], dim=0)
print(p2m_edge_index)
p2m_edge_index.dtype

# %%
#edge features
p2m_edge_features = p2m.drop(["measurement_concept_id", "person_id",'measurement_id'],axis=1) #temporal can be touched later!
p2m_edge_features.head()

# %%
###PATIENT -> DEVICE edge: patient exposed to device
#filter some useful features for edge
p2de=device_sorted_df[['device_id','device_exposure_id','person_id','quantity']]
# %%
#edge index
p2de_patient_id = pd.merge(p2de['person_id'],patient_id_mapping,left_on='person_id',right_on='patientID',how='left')
p2de_patient_id = torch.from_numpy(p2de_patient_id['mappedID'].values)

p2de_device_id = pd.merge(p2de['device_id'],device_id_mapping,left_on='device_id',right_on='deviceID',how='left')
p2de_device_id = torch.from_numpy(p2de_device_id['mappedID'].values)

p2de_edge_index = torch.stack([p2de_patient_id, p2de_device_id], dim=0)
print(p2de_edge_index)
p2de_edge_index.dtype
# %%
#edge features
p2de_edge_features = p2de.drop(["device_id", "person_id",'device_exposure_id'],axis=1) #temporal can be touched later!
p2de_edge_features.head()

# %%
###PATIENT -> PROCEDURE edge: patient has procedure
#filter some useful features for edge
p2pr=procedure_sorted_df[['procedure_concept_id','procedure_occurrence_id','person_id','quantity']].dropna()
# %%
#edge index
p2pr_patient_id = pd.merge(p2pr['person_id'],patient_id_mapping,left_on='person_id',right_on='patientID',how='left').dropna()
p2pr_patient_id = torch.from_numpy(p2pr_patient_id['mappedID'].values)

p2pr_procedure_id = pd.merge(p2pr['procedure_concept_id'],procedure_id_mapping,left_on='procedure_concept_id',right_on='procedureID',how='left').dropna()
p2pr_procedure_id = torch.from_numpy(p2pr_procedure_id['mappedID'].values)

p2pr_edge_index = torch.stack([p2pr_patient_id, p2pr_procedure_id], dim=0)
print(p2pr_edge_index)
p2pr_edge_index.dtype
# %%
#edge features
p2pr_edge_features = p2pr.drop(["procedure_concept_id", "person_id",'procedure_occurrence_id'],axis=1)
p2pr_edge_features.head()

# %%
###PATIENT -> OBSERVATION edge: patient has lab observation
#filter some useful features for edge
p2o=observation_sorted_df[['observation_concept_id','observation_id','person_id']].dropna()
# %%
#edge index
p2o_patient_id = pd.merge(p2o['person_id'],patient_id_mapping,left_on='person_id',right_on='patientID',how='left').dropna()
p2o_patient_id = torch.from_numpy(p2o_patient_id['mappedID'].values)

p2o_observation_id = pd.merge(p2o['observation_concept_id'],observation_id_mapping,left_on='observation_concept_id',right_on='observationID',how='left').dropna()
p2o_observation_id = torch.from_numpy(p2o_observation_id['mappedID'].values)

p2o_edge_index = torch.stack([p2o_patient_id, p2o_observation_id], dim=0)
print(p2o_edge_index)
p2o_edge_index.dtype


# %%
###PATIENT -> specimen edge: patient has specimen
#filter some useful features for edge
p2s=specimen_sorted_df[['specimen_concept_id','specimen_id','person_id']].dropna()
# %%
#edge index
p2s_patient_id = pd.merge(p2s['person_id'],patient_id_mapping,left_on='person_id',right_on='patientID',how='left').dropna()
p2s_patient_id = torch.from_numpy(p2s_patient_id['mappedID'].values)

p2s_specimen_id = pd.merge(p2s['specimen_concept_id'],specimen_id_mapping,left_on='specimen_concept_id',right_on='specimenID',how='left').dropna()
p2s_specimen_id = torch.from_numpy(p2s_specimen_id['mappedID'].values)

p2s_edge_index = torch.stack([p2s_patient_id, p2s_specimen_id], dim=0)
print(p2s_edge_index)
p2s_edge_index.dtype


# %%
###VISIT -> DRUG edge:
#edge index
v2d=drug_sorted_df[["drug_id","visit_occurrence_id"]].dropna()
v2d.visit_occurrence_id=v2d.visit_occurrence_id.astype(int)

v2d_visit_id = pd.merge(v2d['visit_occurrence_id'],visit_id_mapping,left_on='visit_occurrence_id',right_on='visitID',how='left').dropna()
v2d_visit_id = torch.from_numpy(v2d_visit_id['mappedID'].values)

v2d_drug_id = pd.merge(v2d['drug_id'],drug_id_mapping,left_on='drug_id',right_on='drugID',how='left').dropna()
v2d_drug_id = torch.from_numpy(v2d_drug_id['mappedID'].values)

v2d_edge_index = torch.stack([v2d_visit_id, v2d_drug_id], dim=0)
v2d_edge_index=v2d_edge_index.to(torch.int64)
print(v2d_edge_index)
v2d_edge_index.dtype
# %%
###VISIT -> CONDITION edge:
#edge index
# v2c=condition_sorted_df[["condition_concept_id","visit_occurrence_id"]].dropna()
# v2c.visit_occurrence_id=v2c.visit_occurrence_id.astype(int)

# v2c_visit_id = pd.merge(v2c['visit_occurrence_id'],visit_id_mapping,left_on='visit_occurrence_id',right_on='visitID',how='left').dropna()
# v2c_visit_id = torch.from_numpy(v2c_visit_id['mappedID'].values)

# v2c_condition_id = pd.merge(v2c['condition_concept_id'],condition_id_mapping,left_on='condition_concept_id',right_on='conditionID',how='left').dropna()
# v2c_condition_id = torch.from_numpy(v2c_condition_id['mappedID'].values)

# v2c_edge_index = torch.stack([v2c_visit_id, v2c_condition_id], dim=0)
# v2c_edge_index=v2c_edge_index.to(torch.int64)
# print(v2c_edge_index)
# v2c_edge_index.dtype
# %%
###VISIT -> MEASUREMENT edge:
#edge index
v2m=measurement_sorted_df[["measurement_concept_id","visit_occurrence_id"]].dropna()
v2m.visit_occurrence_id=v2m.visit_occurrence_id.astype(int)

v2m_visit_id = pd.merge(v2m['visit_occurrence_id'],visit_id_mapping,left_on='visit_occurrence_id',right_on='visitID',how='left').dropna()
v2m_visit_id = torch.from_numpy(v2m_visit_id['mappedID'].values)

v2m_measurement_id = pd.merge(v2m['measurement_concept_id'],measurement_id_mapping,left_on='measurement_concept_id',right_on='measurementID',how='left').dropna()
v2m_measurement_id = torch.from_numpy(v2m_measurement_id['mappedID'].values)

v2m_edge_index = torch.stack([v2m_visit_id, v2m_measurement_id], dim=0)
v2m_edge_index=v2m_edge_index.to(torch.int64)
print(v2m_edge_index)
v2m_edge_index.dtype
# %%
###VISIT -> DEVICE edge:
#edge index
v2de=device_sorted_df[["device_id","visit_occurrence_id"]].dropna()
v2de.visit_occurrence_id=v2de.visit_occurrence_id.astype(int)

v2de_visit_id = pd.merge(v2de['visit_occurrence_id'],visit_id_mapping,left_on='visit_occurrence_id',right_on='visitID',how='left').dropna()
v2de_visit_id = torch.from_numpy(v2de_visit_id['mappedID'].values)

v2de_device_id = pd.merge(v2de['device_id'],device_id_mapping,left_on='device_id',right_on='deviceID',how='left').dropna()
v2de_device_id = torch.from_numpy(v2de_device_id['mappedID'].values)

v2de_edge_index = torch.stack([v2de_visit_id, v2de_device_id], dim=0)
v2de_edge_index=v2de_edge_index.to(torch.int64)
print(v2de_edge_index)
v2de_edge_index.dtype
# %%
###VISIT -> PROCEDURE edge:
#edge index
procedure_sorted_df.visit_occurrence_id= procedure_sorted_df.visit_occurrence_id.replace('n', np.nan)
v2pr=procedure_sorted_df[["procedure_concept_id","visit_occurrence_id"]].dropna()
v2pr.visit_occurrence_id=v2pr.visit_occurrence_id.astype(int)

v2pr_visit_id = pd.merge(v2pr['visit_occurrence_id'],visit_id_mapping,left_on='visit_occurrence_id',right_on='visitID',how='left').dropna()
v2pr_visit_id = torch.from_numpy(v2pr_visit_id['mappedID'].values)

v2pr_procedure_id = pd.merge(v2pr['procedure_concept_id'],procedure_id_mapping,left_on='procedure_concept_id',right_on='procedureID',how='left').dropna()
v2pr_procedure_id = torch.from_numpy(v2pr_procedure_id['mappedID'].values)

v2pr_edge_index = torch.stack([v2pr_visit_id, v2pr_procedure_id], dim=0)
v2pr_edge_index=v2pr_edge_index.to(torch.int64)
print(v2pr_edge_index)
v2pr_edge_index.dtype
# %%
###VISIT -> OBSERVATION edge:
#edge index
v2o=observation_sorted_df[["observation_concept_id","visit_occurrence_id"]].dropna()
v2o.visit_occurrence_id=v2o.visit_occurrence_id.astype(int)

v2o_visit_id = pd.merge(v2o['visit_occurrence_id'],visit_id_mapping,left_on='visit_occurrence_id',right_on='visitID',how='left').dropna()
v2o_visit_id = torch.from_numpy(v2o_visit_id['mappedID'].values)

v2o_observation_id = pd.merge(v2o['observation_concept_id'],observation_id_mapping,left_on='observation_concept_id',right_on='observationID',how='left').dropna()
v2o_observation_id = torch.from_numpy(v2o_observation_id['mappedID'].values)

v2o_edge_index = torch.stack([v2o_visit_id, v2o_observation_id], dim=0)
v2o_edge_index=v2o_edge_index.to(torch.int64)
print(v2o_edge_index)
v2o_edge_index.dtype
# %%
###VISIT -> specimen edge:
#edge index
specimen_sorted_df.visit_occurrence_id= specimen_sorted_df.visit_occurrence_id.replace('n', np.nan)
v2s=specimen_sorted_df[["specimen_concept_id","visit_occurrence_id"]].dropna()
v2s.visit_occurrence_id=v2s.visit_occurrence_id.astype(int)

v2s_visit_id = pd.merge(v2s['visit_occurrence_id'],visit_id_mapping,left_on='visit_occurrence_id',right_on='visitID',how='left').dropna()
v2s_visit_id = torch.from_numpy(v2s_visit_id['mappedID'].values)

v2s_specimen_id = pd.merge(v2s['specimen_concept_id'],specimen_id_mapping,left_on='specimen_concept_id',right_on='specimenID',how='left').dropna()
v2s_specimen_id = torch.from_numpy(v2s_specimen_id['mappedID'].values)

v2s_edge_index = torch.stack([v2s_visit_id, v2s_specimen_id], dim=0)
v2s_edge_index=v2s_edge_index.to(torch.int64)
print(v2s_edge_index)
v2s_edge_index.dtype
#%%
p2d_edge_features['dose_val_rx']=pd.to_numeric(p2d_edge_features['dose_val_rx'], errors='coerce')
p2d_edge_features['form_val_disp']=pd.to_numeric(p2d_edge_features['form_val_disp'], errors='coerce')

# %%
drug_node_features_mat=drug_node_features.to_numpy()
patient_node_features_mat=patient_node_features.to_numpy()
visit_node_features_mat=visit_node_features.to_numpy()
measurement_node_features_mat=measurement_node_features.to_numpy()
device_node_features_mat=device_node_features.to_numpy()
observation_node_features_mat=observation_node_features.to_numpy()
specimen_node_features_mat=specimen_node_features.to_numpy()

p2d_edge_features_mat=p2d_edge_features.to_numpy()
p2pr_edge_features_mat=p2pr_edge_features.to_numpy()
p2m_edge_features_mat=p2m_edge_features.to_numpy()
p2de_edge_features_mat=p2de_edge_features.to_numpy()

# %%
data = HeteroData()
data['drug'].x = torch.from_numpy(drug_node_features_mat)
data['drug'].node_id=drug_id_mapping['drugID'].to_numpy()
data['patient'].x = torch.from_numpy(patient_node_features_mat)
data['patient'].node_id=patient_id_mapping['patientID'].to_numpy()
data['visit_occurrence'].x=torch.from_numpy(visit_node_features_mat)
data['visit_occurrence'].node_id=visit_id_mapping['visitID'].to_numpy()
# data['condition'].node_id=condition_id_mapping['conditionID'].to_numpy()
data['measurement'].x=torch.from_numpy(measurement_node_features_mat)
data['measurement'].node_id=measurement_id_mapping['measurementID'].to_numpy()
data['device'].x = torch.from_numpy(device_node_features_mat)
data['device'].node_id=device_id_mapping['deviceID'].to_numpy()
data['procedure'].node_id=procedure_id_mapping['procedureID'].to_numpy()
data['observation'].x = torch.from_numpy(observation_node_features_mat)
data['observation'].node_id=observation_id_mapping['observationID'].to_numpy()
data['specimen'].x = torch.from_numpy(specimen_node_features_mat)
data['specimen'].node_id=specimen_id_mapping['specimenID'].to_numpy()

data['patient', 'drug_expose', 'drug'].edge_index = p2d_edge_index
# data['patient', 'drug_expose', 'drug'].edge_attr = torch.from_numpy(p2d_edge_features_mat)
data['patient', 'has', 'visit_occurrence'].edge_index = p2v_edge_index
# data['patient', 'condition_occurrence', 'condition'].edge_index = p2c_edge_index
data['patient', 'has', 'measurement'].edge_index = p2m_edge_index
# data['patient', 'has', 'measurement'].edge_attr = torch.from_numpy(p2m_edge_features_mat)
data['patient', 'device_expose', 'device'].edge_index = p2de_edge_index
# data['patient', 'device_expose', 'device'].edge_attr = torch.from_numpy(p2de_edge_features_mat)
data['patient', 'procedure_occurrence', 'procedure'].edge_index = p2pr_edge_index
# data['patient', 'procedure_occurrence', 'procedure'].edge_attr = torch.from_numpy(p2pr_edge_features_mat)
data['patient', 'has', 'observation'].edge_index = p2o_edge_index
data['patient', 'has', 'specimen'].edge_index = p2s_edge_index

data['visit_occurrence', 'link', 'drug'].edge_index = v2d_edge_index
# data['visit_occurrence', 'link', 'condition'].edge_index = v2c_edge_index
data['visit_occurrence', 'link', 'measurement'].edge_index = v2m_edge_index
data['visit_occurrence', 'link', 'device'].edge_index = v2de_edge_index
data['visit_occurrence', 'link', 'procedure'].edge_index = v2pr_edge_index
data['visit_occurrence', 'link', 'observation'].edge_index = v2o_edge_index
data['visit_occurrence', 'link', 'specimen'].edge_index = v2s_edge_index

#%%
torch.save(data,'Graph3_b4utle_balance_nodate_MIMICIV_noedgefts_nocon.pth')


#############THIS PART IS USED TO WRITE LIST LINK################
# %%
p2d_id = pd.merge(p2d[['person_id','drug_concept_id']],patient_id_mapping,left_on='person_id',right_on='patientID',how='left').drop(columns = ['person_id'])
p2d_id = pd.merge(p2d_id,drug_id_mapping,left_on='drug_concept_id',right_on='drugID',how='left',suffixes=('_source','_target')).drop(columns = ['drug_concept_id'])
p2d_id.head()
#%%
p2d_id['patientID'] = 'patient_' + p2d_id['patientID'].astype(str)
p2d_id['drugID'] = 'drug_' + p2d_id['drugID'].astype(str)
p2d_id=p2d_id.rename(columns={"patientID": "source", "drugID": "target",})
p2d_id['type_link']='patient->drug'
p2d_id.head()

# %%
p2m_id = pd.merge(p2m[['person_id','measurement_concept_id']],patient_id_mapping,left_on='person_id',right_on='patientID',how='left').drop(columns = ['person_id'])
p2m_id = pd.merge(p2m_id,measurement_id_mapping,left_on='measurement_concept_id',right_on='measurementID',how='left',suffixes=('_source','_target')).drop(columns = ['measurement_concept_id'])
p2m_id.head()
#%%
p2m_id['patientID'] = 'patient_' + p2m_id['patientID'].astype(str)
p2m_id['measurementID'] = 'measurement_' + p2m_id['measurementID'].astype(str)
p2m_id=p2m_id.rename(columns={"patientID": "source", "measurementID": "target",})
p2m_id['type_link']='patient->measurement'
p2m_id.head()

# %%
p2de_id = pd.merge(p2de[['person_id','device_concept_id']],patient_id_mapping,left_on='person_id',right_on='patientID',how='left').drop(columns = ['person_id'])
p2de_id = pd.merge(p2de_id,device_id_mapping,left_on='device_concept_id',right_on='deviceID',how='left',suffixes=('_source','_target')).drop(columns = ['device_concept_id'])
p2de_id.head()
#%%
p2de_id['patientID'] = 'patient_' + p2de_id['patientID'].astype(str)
p2de_id['deviceID'] = 'device_' + p2de_id['deviceID'].astype(str)
p2de_id=p2de_id.rename(columns={"patientID": "source", "deviceID": "target",})
p2de_id['type_link']='patient->device'
p2de_id.head()

# %%
p2c_id = pd.merge(p2c[['person_id','condition_concept_id']],patient_id_mapping,left_on='person_id',right_on='patientID',how='left').drop(columns = ['person_id'])
p2c_id = pd.merge(p2c_id,condition_id_mapping,left_on='condition_concept_id',right_on='conditionID',how='left',suffixes=('_source','_target')).drop(columns = ['condition_concept_id'])
p2c_id.head()
#%%
p2c_id['patientID'] = 'patient_' + p2c_id['patientID'].astype(str)
p2c_id['conditionID'] = 'condition_' + p2c_id['conditionID'].astype(str)
p2c_id=p2c_id.rename(columns={"patientID": "source", "conditionID": "target",})
p2c_id['type_link']='patient->condition'
p2c_id.head()

# %%
link_patient=pd.concat([p2d_id, p2m_id,p2de_id,p2c_id], axis=0)
link_patient.to_csv("link_list_frompatient_nodate.csv",sep="\t",index=False)


# # %%
v2d_id = pd.merge(v2d[['visit_occurrence_id','drug_concept_id']],visit_id_mapping,left_on='visit_occurrence_id',right_on='visitID',how='left').drop(columns = ['visit_occurrence_id'])
v2d_id = pd.merge(v2d_id,drug_id_mapping,left_on='drug_concept_id',right_on='drugID',how='left',suffixes=('_source','_target')).drop(columns = ['drug_concept_id'])
v2d_id.head()
#%%
v2d_id['visitID'] = 'visit_' + v2d_id['visitID'].astype(str)
v2d_id['drugID'] = 'drug_' + v2d_id['drugID'].astype(str)
v2d_id=v2d_id.rename(columns={"visitID": "source", "drugID": "target",})
v2d_id['type_link']='visit->drug'
v2d_id.head()

# %%
v2m_id = pd.merge(v2m[['visit_occurrence_id','measurement_concept_id']],visit_id_mapping,left_on='visit_occurrence_id',right_on='visitID',how='left').drop(columns = ['visit_occurrence_id'])
v2m_id = pd.merge(v2m_id,measurement_id_mapping,left_on='measurement_concept_id',right_on='measurementID',how='left',suffixes=('_source','_target')).drop(columns = ['measurement_concept_id'])
v2m_id.head()
#%%
v2m_id['visitID'] = 'visit_' + v2m_id['visitID'].astype(str)
v2m_id['measurementID'] = 'measurement_' + v2m_id['measurementID'].astype(str)
v2m_id=v2m_id.rename(columns={"visitID": "source", "measurementID": "target",})
v2m_id['type_link']='visit->measurement'
v2m_id.head()

# %%
v2de_id = pd.merge(v2de[['visit_occurrence_id','device_concept_id']],visit_id_mapping,left_on='visit_occurrence_id',right_on='visitID',how='left').drop(columns = ['visit_occurrence_id'])
v2de_id = pd.merge(v2de_id,device_id_mapping,left_on='device_concept_id',right_on='deviceID',how='left',suffixes=('_source','_target')).drop(columns = ['device_concept_id'])
v2de_id.head()
#%%
v2de_id['visitID'] = 'visit_' + v2de_id['visitID'].astype(str)
v2de_id['deviceID'] = 'device_' + v2de_id['deviceID'].astype(str)
v2de_id=v2de_id.rename(columns={"visitID": "source", "deviceID": "target",})
v2de_id['type_link']='visit->device'
v2de_id.head()

# %%
v2c_id = pd.merge(v2c[['visit_occurrence_id','condition_concept_id']],visit_id_mapping,left_on='visit_occurrence_id',right_on='visitID',how='left').drop(columns = ['visit_occurrence_id'])
v2c_id = pd.merge(v2c_id,condition_id_mapping,left_on='condition_concept_id',right_on='conditionID',how='left',suffixes=('_source','_target')).drop(columns = ['condition_concept_id'])
v2c_id.head()
#%%
v2c_id['visitID'] = 'visit_' + v2c_id['visitID'].astype(str)
v2c_id['conditionID'] = 'condition_' + v2c_id['conditionID'].astype(str)
v2c_id=v2c_id.rename(columns={"visitID": "source", "conditionID": "target",})
v2c_id['type_link']='visit->condition'
v2c_id.head()

# %%
link_visit=pd.concat([v2d_id,v2m_id,v2de_id,v2c_id], axis=0)
link_visit.to_csv("link_list_fromvisit_nodate.csv",sep="\t",index=False)
# %%
