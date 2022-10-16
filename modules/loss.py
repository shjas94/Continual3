import torch

def _calculate_energy_partition(energy, task_class_set=None, coreset_mode=False):
    if not coreset_mode:
        return torch.logsumexp(-1 * energy[:, list(task_class_set)[-2:]], dim=1, keepdim=True).clone()
    else:
        return torch.logsumexp(-1 * energy[:, list(task_class_set)[:-2]], dim=1, keepdim=True).clone()

def _calculate_energy_positive(energy, y_ans_idx):

    return energy.gather(dim=1, index=y_ans_idx)

def _calculate_energy_negative(energy, y_ans_idx, device):
    total_class_len = len(y_ans_idx)
    y_neg_idx = torch.zeros_like(y_ans_idx)
    counter = 0
    while counter < total_class_len:
        temp_idx = torch.randint(0, energy.size(1), (1, 1)).to(device)
        if temp_idx == y_ans_idx[counter][0]:
            continue
        else:
            y_neg_idx[counter] = temp_idx
            counter+=1
    return energy.gather(dim=1, index=y_neg_idx.to(device))
            
    
def energy_nll_loss(energy, y_ans_idx, task_class_set=None, coreset_mode=False):
    energy_pos = _calculate_energy_positive(energy, y_ans_idx)
    energy_partition = _calculate_energy_partition(energy, task_class_set, coreset_mode)
    return (energy_pos + energy_partition).mean()

def contrastive_divergence(energy, y_ans_idx, device):
    energy_pos = _calculate_energy_positive(energy, y_ans_idx)
    energy_neg = _calculate_energy_negative(energy, y_ans_idx, device)
    cdiv_loss = (energy_pos-energy_neg).mean()
    # reg_loss = 1e-4*(energy_pos**2 + energy_neg**2).mean()
    loss = cdiv_loss.mean()
    return loss


def get_criterion(loss):
    if loss == 'nll_energy':
        return energy_nll_loss
    elif loss == 'contrastive_divergence':
        return contrastive_divergence
    elif loss == 'cross_entropy':
        return torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError