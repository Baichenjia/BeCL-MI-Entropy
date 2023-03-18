import torch
import torch.nn as nn
import numpy as np
from dist_train.utils.helpers import create_nn
from base.modules.normalization import Normalizer
from base.modules.intrinsic_motivation import IntrinsicMotivationModule
import torch.nn.functional as F

class Discriminator(nn.Module, IntrinsicMotivationModule):
    def __init__(self, n, state_size, hidden_size, num_layers=4, normalize_inputs=False,
                 input_key='next_state', input_size=None, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.n = n
        self.state_size = int(state_size)  if input_size is None else int(input_size)
        self.input_key = str(input_key)
        assert num_layers >= 2
        self.num_layers = int(num_layers)
        self.count = 0 
        input_normalizer = Normalizer(self.state_size) if normalize_inputs else nn.Sequential()
        self.layers = create_nn(input_size=self.state_size, output_size=self.n, hidden_size=hidden_size,
                                num_layers=self.num_layers, input_normalizer=input_normalizer)

        self.loss = self.compute_info_nce_loss
        
    def forward(self, batch):
        x = batch[self.input_key]
        for layer in self.layers:
            x = layer(x)
        # cos = torch.sum(x * s, -1,keepdim=True)

        # for layer in self.layers:
        #     cos = layer(cos)
        return self.loss(x, batch['skill']).mean()
    
    def surprisal(self, batch):
        x = batch[self.input_key]
        for layer in self.layers:
            x = layer(x)
        # s = batch['state']
        # cos = torch.sum(x * s, -1,keepdim=True)
        # for layer in self.layers:
        #     cos = layer(cos)
        # args = APTArgs()
        # self.count += 1
        # reward = compute_apt_reward(x, x, args)
        # print(reward.mean())
        # return reward
        return torch.exp(-self.compute_info_nce_loss(x, batch['skill'])).squeeze()
    
    # def skill_assignment(self, batch):
    #     return torch.argmax(self.layers(batch[self.input_key]), dim=1)
    
    def compute_info_nce_loss(self, features, labels):
        # label positives samples

        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).long() # (b, b)
        features = F.normalize(features, dim=1) # (b, c)
        similarity_matrix = torch.matmul(features, features.T) # (b, b)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool) # (b, b - 1)
        labels = labels[~mask].view(labels.shape[0], -1) # (b, b - 1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1) # (b, b - 1)

        similarity_matrix = similarity_matrix / self.temperature
        similarity_matrix -= torch.max(similarity_matrix, 1)[0][:, None]
        similarity_matrix = torch.exp(similarity_matrix)

        # update not only when all negative sample existed
        # assert labels[pick_one_positive_sample_idx]
        # pick_one_positive_sample_idx = torch.argmax(labels, dim=-1, keepdim=True)
        # pick_one_positive_sample_idx = torch.zeros_like(labels).scatter_(-1, pick_one_positive_sample_idx, 1)
        
        # one positive pair 
        # other_pos = torch.zeros(similarity_matrix.shape[0],1) # (b, 1)
        # for i,value in enumerate(labels):
        #     try:
        #         idx = value.nonzero().flatten()
        #         other_pos[i] = similarity_matrix[i][np.random.choice(idx)]
        #     except:
        #         print("if no extra positive pair in batch, default 0 loss")
        #         other_pos[i] = torch.as_tensor(1, dtype=torch.float32)

        other_positive = features.view(-1,50,self.n)[:,49,:].view(-1,self.n)
        feature_dim = features.shape[-1]
        other_positive = other_positive.flatten()   # (feature_dim * goal_num)
        other_positive = other_positive.expand(50, -1)        #(50, feature_dim * goal_num)
        other_positive = other_positive.reshape(other_positive.shape[0], -1, feature_dim)        # (50, goal_num, feature_dim)
        other_positive = other_positive.permute(1,0,2)      # (goal_num, 50, feature_dim)
        other_positive = other_positive.reshape(-1,feature_dim)    #(50 * goal_num (b), feature_dim)
        
        # positives = torch.sum(other_pos, dim=-1, keepdim=True)
        positives = torch.sum(torch.exp(features * other_positive), dim=-1, keepdim=True)
        # positives = torch.as_tensor([1])
        negatives = torch.sum(similarity_matrix * (~labels.bool()).float(), dim=-1, keepdim=True)
        # negatives = torch.sum(similarity_matrix, dim=-1, keepdim=True)
        eps = torch.as_tensor(1e-6)
        loss = -torch.log(positives / (negatives + eps) + eps)
        return loss
