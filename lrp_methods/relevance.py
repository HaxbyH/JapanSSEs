import torch
import numpy as np
from lrp_methods.lrp import LRPModel

def get_relevance(X, model):
    X = X.float().to("cuda")
    # X = torch.cuda.FloatTensor(X)
    X = torch.unsqueeze(X, dim=0)
    lrp_model = LRPModel(model).to("cuda")
    r = lrp_model.forward(X)
    r = np.array(r)
    return r

def _relevance_channel_sum(relevance):
    channel_sum = np.sum(abs(relevance), axis=1)
    top_channel_arg = np.argsort(channel_sum)[::-1]
    return top_channel_arg

def is_relevance_position_found(relevance, position):
    top_channel_arg = _relevance_channel_sum(relevance)
    found_section = (top_channel_arg[0] == position)
    return found_section

def calculate_relevance_signal_position(relevance, position):
    top_channel_arg = _relevance_channel_sum(relevance)
    position_of_signal = np.where(top_channel_arg == position)[0]
    return position_of_signal