from Models import *
from DataProcessing import random_iterator, ToTensor
import pickle as pk
import time
import matplotlib.pyplot as plt

Net = ConvNet()
gpu0 = 'cuda:0'
gpu1 = 'cuda:1'

with open('nn_data_boston', 'rb') as f:
    Data = pk.load(f)
    Data = ToTensor(Data, type='image')
with open('nn_labels_boston', 'rb') as f:
    Labels = pk.load(f)
    Labels = ToTensor(Labels, type='label')


Labels = Labels.type(torch.LongTensor).to(gpu0)
Data = Data.to(gpu0)
Net = Net.to(gpu0)
Net = nn.DataParallel(Net, output_device=gpu0)
torch.backends.cudnn.benchmark = True


L_weight = torch.Tensor([0.75, 0.25]).to(gpu0)
Loss_fn = nn.CrossEntropyLoss(weight=L_weight)
Optimizer = torch.optim.Adam(Net.parameters(), lr=0.00008)


s = time.time()
j = 0
x, y = [], []
for i in random_iterator(cycle=8000, iteration=4, high=1200):
    Optimizer.zero_grad()
    loss = Loss_fn(Net(Data[i:i+24]), Labels[i:i+24])
    loss.backward()
    Optimizer.step()
    j += 1
    if j % 4 == 0:
        z = float(loss)
        print(j//4, 'Batch completed, Loss =', z)
        x.append(j//4)
        y.append(z)
print('Training completed, using: ', time.time()-s, 's')
plt.plot(x, y)
plt.show()


with open('CovNet_param_', 'wb') as f:
    Net = Net.module
    torch.save(Net.state_dict(), f)