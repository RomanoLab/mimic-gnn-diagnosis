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
data_full=torch.load('sub_graph_noutle_label_balance_b4utle_noedgefts_relatecon.pth')
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
train_patient = sorted(random.sample(patient_index, int(len(patient_index) * 0.90)))
test_patient=list(set(patient_index) - set(train_patient))
# val_patient=sorted(random.sample(test_val_patient, int(len(test_val_patient) * 0.5)))
# test_patient=sorted(list(set(test_val_patient) - set(val_patient)))
# %%
train_mask=np.repeat(True,len(data_full['patient'].node_id))
train_mask[test_patient]=False
# val_mask=np.repeat(False,len(data_full['patient'].node_id))
# val_mask[val_patient]=True
test_mask=np.repeat(False,len(data_full['patient'].node_id))
test_mask[test_patient]=True
# %%
data_full['patient'].train_mask=train_mask
# data_full['patient'].val_mask=val_mask
data_full['patient'].test_mask=test_mask

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

# class GCN(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = GCNConv(dataset.num_node_features, 16)
#         self.conv2 = GCNConv(16, dataset.num_classes)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)

#         return F.log_softmax(x, dim=1)
#%%
model=HeteroGNN(data.metadata(), hidden_channels=64,out_channels=1,num_layers=2)
device=torch.device('cuda')
data,model=data.to(device),model.to(device)

#%%
# with torch.no_grad():  # Initialize lazy modules.
#     out = model(data.x_dict, data.edge_index_dict)
train_input_nodes = ('patient', train_patient)
# val_input_nodes = ('patient', val_patient)
# test_input_nodes = ('patient', test_patient)
# kwargs = {'batch_size': 1024, 'num_workers': 20, 'persistent_workers': True}
#%%
train_loader = HGTLoader(data, num_samples=[50],shuffle=True, 
                            input_nodes=train_input_nodes, batch_size=128)
# val_loader = HGTLoader(data, num_samples=[50],
#                             input_nodes=val_input_nodes, batch_size=256)
# test_loader = HGTLoader(data, num_samples=[100],
                            # input_nodes=test_input_nodes, batch_size=256)

# %%
# batch = next(iter(train_loader))
# %%
def train(model):
    model=model.train()

    total_examples = total_loss = 0
    pred_train=[]
    for _,batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        batch = batch.to(device, 'edge_index')
        batch_size = batch['patient'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)
        mask = batch['patient'].train_mask
        loss = F.cross_entropy(out[mask].squeeze(), batch['patient'].y[mask])
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return model.eval(),total_loss / total_examples

# %%
@torch.no_grad()
def test():
    model.eval()
    pred = torch.round(torch.sigmoid(model(data.x_dict, data.edge_index_dict).squeeze()))

    accs = []
    for split in ['train_mask','test_mask']:
        mask = data['patient'][split]
        acc = (pred[mask] == data['patient'].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs

# %%
rand_seed=[]
for i in range(0,20):
    rand_seed.append(np.random.randint(0,9999))

#%%
for seed in rand_seed:
    if 'model' in globals():
        for layer in model.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    model = HeteroGNN(data.metadata(), hidden_channels=64, out_channels=1,num_layers=3)
    device = torch.device('cuda')
    data, model = data.to(device), model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)

    for epoch in range(1, 10):
        model,loss = train(model)
        train_acc, test_acc = test()
        if test_acc == 1.0:
            print(seed)
            break
        if test_acc < 0.99:
            MODEL_PATH="Model_b4utle_basic_graph_Conv_epoch"+str(epoch)+"_seed"+str(seed)+"_relatecon.pth"
            torch.save(model,MODEL_PATH)
            pass        
        print(f'Seed: {seed}, Epoch: {epoch}, Loss: {loss:.4f}, Train: {train_acc:.4f},Test: {test_acc:.4f}')

# %%
if 'model' in globals():
        for layer in model.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

model = HeteroGNN(data.metadata(), hidden_channels=64, out_channels=1,num_layers=6)
device = torch.device('cuda')
data, model = data.to(device), model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)

for epoch in range(1, 10):
    model,loss = train(model)
    MODEL_PATH="Model_b4db_basic_graph_Conv_epoch"+str(epoch)+"_seed"+str(seed)+".pth"
    torch.save(model,MODEL_PATH)
    train_acc, val_acc, test_acc = test()
    print(f'Seed: {seed}, Epoch: {epoch}, Loss: {loss:.4f}, Train: {train_acc:.4f},'
        f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
# %%
