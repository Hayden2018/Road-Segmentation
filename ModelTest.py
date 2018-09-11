import pickle as pk

from Models import *
from DataProcessing import ToArray, show, ToTensor

gpu0 = 'cuda:0'
gpu1 = 'cuda:1'

Net = ConvNet().to(gpu0)

with open('CovNet_param', 'rb') as f:
    state = torch.load(f, map_location=gpu0)
    Net.load_state_dict(state)
with open('nn_data_test', 'rb') as f:
    Data = pk.load(f)
    Data = ToTensor(Data, type='image')
with open('nn_labels_test', 'rb') as f:
    Labels = pk.load(f)


def get_accuracy(x, y):
    x = x.flatten()
    y = y.flatten()
    length = len(x)
    c = 0
    for i in range(length):
        if x[i] == y[i]:
            c += 1
    print('Accuracy:', c/length)


original = Data[0:10].to(gpu0)
label = Labels[0:10]
result = Net(original)
result = ToArray(result, type='label')
original = ToArray(original, type='image')

get_accuracy(result, label)

# show(label[8])
show(original[8])
show(result[8])





