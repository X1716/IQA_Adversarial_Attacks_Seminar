import torch
import torch.nn.functional as F
METRIC_MAX_VAL = 1.0
def flow_loss(flows, padding_mode='constant', epsilon=1e-8, device='cpu'):
    paddings = (1,1,1,1)
    padded_flows = F.pad(flows,paddings,mode=padding_mode,value=0)
    shifted_flows = [
    padded_flows[:, :, 2:, 2:],  # bottom right (+1,+1)
    padded_flows[:, :, 2:, :-2],  # bottom left (+1,-1)
    padded_flows[:, :, :-2, 2:],  # top right (-1,+1)
    padded_flows[:, :, :-2, :-2]  # top left (-1,-1)
    ]
    #||\Delta u^{(p)} - \Delta u^{(q)}||_2^2 + # ||\Delta v^{(p)} - \Delta v^{(q)}||_2^2 
    loss=0
    for shifted_flow in shifted_flows:
        loss += torch.sum(torch.square(flows[:, 1] - shifted_flow[:, 1]) + torch.square(flows[:, 0] - shifted_flow[:, 0]) + epsilon).to(device)
    return loss.type(torch.float32)

def metric_calc(model, x0, x1=None, ref_img=None, maxval=METRIC_MAX_VAL):
    #d0 = model.forward(x0)
    d0 = model.forward(x0)
    if x1 is None:
        return d0, maxval * 1.5 # metric-dependent const
    #d1 = model.forward(x1)
    d1 = model.forward(x1)
    return d0, d1

def rank_loss(s_adv, s_other):
    #print('iter')
    return (s_other/(s_adv+s_other)) #.float().clone()