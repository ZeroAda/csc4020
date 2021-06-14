import numpy as np

with open("index.txt","r") as handle:
    lines = handle.readlines()
    train_index = []
    for line in lines:
        train_index.append(int(line))
    handle.close()
train_index = np.array(train_index)
print(train_index)
