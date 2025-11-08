import torch
import torch.nn as nn

def wrapper(intervener):
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)
    return wrapped

class Collector():
    collect_state = True
    collect_action = False  
    def __init__(self, multiplier, head):
        self.head = head
        self.states = []
        self.actions = []
    def reset(self):
        self.states = []
        self.actions = []
    def __call__(self, b, s): 
        if self.head == -1:
            self.states.append(b[0, -1].detach().clone())  # (batch_size, seq_len, #key_value_heads x D_head)
        else:
            self.states.append(b[0, -1].reshape(32, -1)[self.head].detach().clone())
        return b
    
class ITI_Intervener():
    collect_state = True
    collect_action = True
    attr_idx = -1
    def __init__(self, direction, multiplier):
        if not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction)
        # keep dtype/device flexible; move to cuda/half when called
        self.direction = direction
        self.multiplier = multiplier
        self.states = []
        self.actions = []
    def reset(self):
        self.states = []
        self.actions = []
    def __call__(self, b, s): 
        # b: (batch_size, seq_len, #head x D_head)
        self.states.append(b[0, -1].detach().clone())
        action = self.direction.to(b.device)
        # ensure matching dtype
        if b.dtype == torch.float16:
            action = action.half()
        elif b.dtype == torch.bfloat16:
            action = action.to(torch.bfloat16)
        self.actions.append(action.detach().clone())
        b[0, -1] = b[0, -1] + action * self.multiplier
        return b