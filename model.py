from operator import mod
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
# actual linear QNet 
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        # check if the model folder path exists
        if os.path.exists(model_folder_path):
            pass
        else:
            # make the path if it doesn't exist
            os.makedirs(model_folder_path)
        # make whole filename
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
# for training
