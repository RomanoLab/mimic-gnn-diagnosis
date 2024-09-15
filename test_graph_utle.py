#%%
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
import torch
import itertools
import torch_geometric.transforms as T
import random
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
from torch_geometric.nn import HGTConv, Linear
torch.cuda.empty_cache()
from torch_geometric.loader import HGTLoader, NeighborLoader
from tqdm import tqdm
import gc
import os
gc.collect()
os.chdir("C:/Users/TRAMANH-PC/Desktop/Temporal_graph/mimic_iv/ML/data")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# %%
data_full=torch.load('sub_graph_noutle_label_balance_b4utle_noedgefts.pth')
data_full
#%%
data_full=T.Constant(node_types='condition')(data_full)
data_full=T.Constant(node_types='procedure')(data_full)

#%%
def normalize(x):
    x_normed = x / x.max(0, keepdim=True)[0]
    return x_normed

data_full['patient'].x=normalize(data_full['patient'].x)
# %%
random.seed(18399)
patient_index= range(0,len(data_full['patient'].node_id))
train_patient = sorted(random.sample(patient_index, int(len(patient_index) * 0.85)))
test_val_patient=list(set(patient_index) - set(train_patient))
val_patient=sorted(random.sample(test_val_patient, int(len(test_val_patient) * 0.5)))
test_patient=sorted(list(set(test_val_patient) - set(val_patient)))
# %%
train_mask=np.repeat(True,len(data_full['patient'].node_id))
train_mask[test_val_patient]=False
val_mask=np.repeat(False,len(data_full['patient'].node_id))
val_mask[val_patient]=True
test_mask=np.repeat(False,len(data_full['patient'].node_id))
test_mask[test_patient]=True
# %%
data_full['patient'].train_mask=train_mask
data_full['patient'].val_mask=val_mask
data_full['patient'].test_mask=test_mask

#%%
data_full['patient', 'drug_expose', 'drug'].edge_index = data_full['patient', 'drug_expose', 'drug'].edge_index.to(torch.long)
data_full['patient', 'has', 'visit_occurrence'].edge_index = data_full['patient', 'has', 'visit_occurrence'].edge_index.to(torch.long)
data_full['patient', 'condition_occurrence', 'condition'].edge_index = data_full['patient', 'condition_occurrence', 'condition'].edge_index.to(torch.long)
data_full['patient', 'has', 'measurement'].edge_index = data_full['patient', 'has', 'measurement'].edge_index.to(torch.long)
data_full['patient', 'device_expose', 'device'].edge_index = data_full['patient', 'device_expose', 'device'].edge_index.to(torch.long)
data_full['patient', 'procedure_occurrence', 'procedure'].edge_index = data_full['patient', 'procedure_occurrence', 'procedure'].edge_index.to(torch.long)
data_full['patient', 'has', 'observation'].edge_index = data_full['patient', 'has', 'observation'].edge_index.to(torch.long)
data_full['patient', 'has', 'specimen'].edge_index = data_full['patient', 'has', 'specimen'].edge_index.to(torch.long)

data_full['visit_occurrence', 'link', 'drug'].edge_index = data_full['visit_occurrence', 'link', 'drug'].edge_index.to(torch.long)
data_full['visit_occurrence', 'link', 'condition'].edge_index = data_full['visit_occurrence', 'link', 'condition'].edge_index.to(torch.long)
data_full['visit_occurrence', 'link', 'measurement'].edge_index = data_full['visit_occurrence', 'link', 'measurement'].edge_index.to(torch.long)
data_full['visit_occurrence', 'link', 'device'].edge_index = data_full['visit_occurrence', 'link', 'device'].edge_index.to(torch.long)
data_full['visit_occurrence', 'link', 'procedure'].edge_index = data_full['visit_occurrence', 'link', 'procedure'].edge_index.to(torch.long)
data_full['visit_occurrence', 'link', 'observation'].edge_index = data_full['visit_occurrence', 'link', 'observation'].edge_index.to(torch.long)
data_full['visit_occurrence', 'link', 'specimen'].edge_index = data_full['visit_occurrence', 'link', 'specimen'].edge_index.to(torch.long)
# %%
data=data_full.clone()
data=T.ToUndirected(merge=True)(data)
#%%
data['patient'].x=data['patient'].x.to(torch.float32)
data['patient'].y=data['patient'].y.to(torch.float32)
data['drug'].x=data['drug'].x.to(torch.float32)
data['visit_occurrence'].x=data['visit_occurrence'].x.to(torch.float32)
data['condition'].x=data['condition'].x.to(torch.float32)
data['measurement'].x=data['measurement'].x.to(torch.float32)
data['device'].x=data['device'].x.to(torch.float32)
data['procedure'].x=data['procedure'].x.to(torch.float32)
data['observation'].x=data['observation'].x.to(torch.float32)
data['specimen'].x=data['specimen'].x.to(torch.float32)

#%%
data['patient'].node_id=data['patient'].node_id.astype(np.int32)
data['drug'].node_id=data['drug'].node_id.astype(np.int32)
data['visit_occurrence'].node_id=data['visit_occurrence'].node_id.astype(np.int32)
data['device'].node_id=data['device'].node_id.astype(np.int32)
data['condition'].node_id=data['condition'].node_id.astype(np.int32)
data['measurement'].node_id=data['measurement'].node_id.astype(np.int32)
data['procedure'].node_id=data['procedure'].node_id.astype(np.int32)
data['observation'].node_id=data['observation'].node_id.astype(np.int32)
data['specimen'].node_id=data['specimen'].node_id.astype(np.int32)


# %%
class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        return self.lin(x_dict['patient'])
#%%
model=torch.load("Model_b4utle_basic_graph_Conv_epoch1_seed1443_relatecon.pth")
model.eval()
device = torch.device('cuda')
model.to(device)
data.to(device)
# %%
@torch.no_grad()
def test():
    model.eval()
    pred = torch.sigmoid(model(data.x_dict, data.edge_index_dict).squeeze())

    # accs = []
    # for split in ['train_mask','val_mask', 'test_mask']:
    #     mask = data['patient'][split]
    #     acc = (pred[mask] == data['patient'].y[mask]).sum() / mask.sum()
    #     accs.append(float(acc))
    return pred
# %%
pred_all=test()
pred_test=pred_all[data['patient']['test_mask']].cpu().numpy()
#%%
from sklearn.metrics import *
test_label=data['patient'].y[data['patient']['test_mask']].cpu().numpy()
fpr,tpr,thresholds = roc_curve(test_label,pred_test,drop_intermediate=False)
roc_auc = auc(fpr,tpr)
print(roc_auc)
#%%
from matplotlib import pyplot as plt
plt.plot(fpr,tpr,label='ROC curve (area = {})'.format(roc_auc))
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)

# %%
precision, recall, thresholds = precision_recall_curve(test_label,pred_test)
auc_precision_recall = auc(recall, precision)
print(auc_precision_recall)
plt.plot(recall, precision,label='PR curve (area = {})'.format(auc_precision_recall))
plt.legend()
plt.ylabel("Precision")
plt.xlabel("Recall")
plt.title("Precision-Recall curve");
# %%
f1_score(test_label,np.round(pred_test), average='macro')

# %%
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import CaptumExplainer
import captum 
# %%
explainer = Explainer(
    model=model,
    algorithm=CaptumExplainer('IntegratedGradients'),
    explanation_type="model",
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='binary_classification',
        task_level='node',
        return_type='probs',
    ),
)
# %%
# node_index = 1,3 # which node index to explain
hetero_explanation2 = explainer(
    data.x_dict,
    data.edge_index_dict,
    # index=torch.tensor([1, 3]),
)
# Print the node importance scores
print(hetero_explanation2.node_mask_dict)
#%%
df_node_important = []
for node_type, importance_tensor in hetero_explanation2.node_mask_dict.items():
    if node_type == 'visit_occurrence':
        continue
    # Get the importance scores as a flat list and corresponding node indices
    importance_scores = importance_tensor.squeeze().tolist()
    node_indices = data[node_type].node_id

    # Append data for each node in this node type
    for node_idx, score in zip(node_indices, importance_scores):
        df_node_important.append({
            'node_type': node_type,
            'importance_score': score,
            'node_id': node_idx
        })
df_node_important = pd.DataFrame(df_node_important) 
#%%
drug_name=pd.read_csv("Drug_nodeID_name_relatecon_b4utle.csv")
measurement_name=pd.read_csv("measurement_nodeID_name_relatecon_b4utle.csv")
procedure_name=pd.read_csv("procedure_nodeID_name_relatecon_b4utle.csv")
observation_name=pd.read_csv("observation_nodeID_name_relatecon_b4utle.csv")
specimen_name=pd.read_csv("specimen_nodeID_name_relatecon_b4utle.csv")
device_name=pd.read_csv("device_nodeID_name_relatecon_b4utle.csv")
condition_name=pd.read_csv('condition_nodeID_name_relatecon_b4utle.csv')

drug_name=drug_name[['drugID','drug']].rename({'drug':'name','drugID':'node_id'},axis=1)
drug_name['nodetype_ID']='drug_'+drug_name.node_id.astype(str)
measurement_name=measurement_name[['measurement_concept_id','measurement']].rename({'measurement':'name','measurement_concept_id':'node_id'},axis=1)
measurement_name['nodetype_ID']='measurement_'+measurement_name.node_id.astype(int).astype(str)
procedure_name=procedure_name[['procedure_concept_id','procedure']].rename({'procedure':'name','procedure_concept_id':'node_id'},axis=1)
procedure_name['node_id'] = procedure_name['node_id'].fillna(-1)
procedure_name['nodetype_ID']='procedure_'+procedure_name.node_id.astype(int, errors='ignore').astype(str)
observation_name=observation_name[['observation_concept_id','observation']].rename({'observation':'name','observation_concept_id':'node_id'},axis=1)
observation_name['nodetype_ID']='observation_'+observation_name.node_id.astype(int).astype(str)
specimen_name=specimen_name[['specimen_concept_id','specimen']].rename({'specimen':'name','specimen_concept_id':'node_id'},axis=1)
specimen_name['nodetype_ID']='specimen_'+specimen_name.node_id.astype(int).astype(str)
device_name=device_name[['deviceID','device']].rename({'device':'name','deviceID':'node_id'},axis=1)
device_name['nodetype_ID']='device_'+device_name.node_id.astype(str)
condition_name=condition_name[['condition_concept_id','condition']].rename({'condition':'name','condition_concept_id':'node_id'},axis=1)
condition_name['nodetype_ID']='condition_'+condition_name.node_id.astype(int).astype(str)

big_name_df=pd.concat([drug_name,measurement_name,procedure_name,observation_name,specimen_name,device_name,condition_name],axis=0)
df_node_important=df_node_important.dropna()
df_node_important['nodetype_ID']=df_node_important['node_type']+'_'+df_node_important['node_id'].astype(int).astype(str)
df_node_important=df_node_important.merge(big_name_df[['nodetype_ID','name']],on='nodetype_ID',how='left')
df_node_important['importance_score_abs']=abs(df_node_important['importance_score'])
# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score
accuracy = accuracy_score(test_label,np.round(pred_test,0))
precision = precision_score(test_label,np.round(pred_test,0))
recall = recall_score(test_label,np.round(pred_test,0))
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
# %%
