import torch
import numpy as np
probs = torch.FloatTensor([[0.05, 0.1, 0.85], [0.05, 0.05, 0.1]])

dist = torch.distributions.Categorical(probs)
print(dist)
for i in range(20):
    index = dist.sample()
    print(index)

ben = "a10"
for i in range(10):
    print(ben+"-"+str(i))