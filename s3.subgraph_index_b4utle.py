#%%
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
import torch
import itertools
import torch_geometric.transforms as T
import os
os.chdir("C:/Users/TRAMANH-PC/Desktop/Temporal_graph/mimic_iv/ML/data")

# %% Load full graph
data=torch.load("Graph3_b4utle_balance_nodate_MIMICIV_noedgefts_relatecon.pth")
# %% Load most condition df: type 2 diabetes
condition_interest_id=[192854,195769,195770,197236]
label=pd.read_csv("../label_balance_utle_b4utle_relatecon.csv")
# %%
condition_interest_index=np.where(np.isin(data['condition'].node_id,condition_interest_id))[0]
condition_nointerest_index=np.where(np.isin(data['condition'].node_id,condition_interest_id)==False)[0]
# %%
lpc1=np.where(np.isin(data['patient', 'condition_occurrence', 'condition'].edge_index[1],condition_interest_index))[0].tolist()
# %%
lvc1=np.where(np.isin(data['visit_occurrence', 'link', 'condition'].edge_index[1],condition_interest_index)==False)[0].tolist()

# %%
data_sub = data.clone()
data_sub['condition'].node_id=data_sub['condition']['node_id'][condition_nointerest_index]
#%%
def reindex_target(data_link,link_index):
    edge_index=pd.DataFrame(np.transpose(data_link.edge_index[:,link_index]), columns=['source', 'target'])
    condition_reindex=pd.DataFrame(edge_index['target'].drop_duplicates().sort_values().reset_index(drop=True))
    condition_reindex['reindex']=range(0,condition_reindex.shape[0])
    edge_index=pd.merge(edge_index,condition_reindex,on='target',how='left')
    return torch.from_numpy(np.transpose(edge_index[['source','reindex']].values))
#%%
data_sub['patient', 'condition_occurrence', 'condition'].edge_index =reindex_target(data['patient', 'condition_occurrence', 'condition'],lpc1) 
data_sub['visit_occurrence', 'link', 'condition'].edge_index =reindex_target(data['visit_occurrence', 'link', 'condition'],lvc1)

# %%
patient_utle_index=np.where(np.isin(data['patient'].node_id,(label['person_id'][label.utle==1])))[0].tolist()
label=np.repeat(0,len(data['patient'].node_id))
label[patient_utle_index]=1
data_sub['patient'].y=torch.from_numpy(label)
#%%
torch.save(data_sub,'sub_graph_noutle_label_balance_b4utle_noedgefts_relatecon.pth')


# %%
