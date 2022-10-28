import torch

def _calculate_energy_positive(energy, y_ans_idx):

    return energy.gather(dim=1, index=y_ans_idx)

def _calculate_energy_partition(energy, task_class_set=None, classes_per_task=2, coreset_mode=False):
    if not coreset_mode:
        return torch.logsumexp(-1 * energy[:, list(task_class_set)[-1*classes_per_task:]], dim=1, keepdim=True).clone()
    else:
        return torch.logsumexp(-1 * energy[:, list(task_class_set)[:-1*classes_per_task]], dim=1, keepdim=True).clone()

def energy_nll_loss(energy, y_ans_idx, classes_per_task, task_class_set=None, coreset_mode=False):
    energy_pos = _calculate_energy_positive(energy, y_ans_idx)
    energy_partition = _calculate_energy_partition(energy, task_class_set, classes_per_task, coreset_mode)
    return (energy_pos + energy_partition).mean()

def _calculate_energy_negative(energy, y_ans_idx, device, class_per_task, coreset_mode):
    total_class_len = len(y_ans_idx)
    y_neg_idx = torch.zeros_like(y_ans_idx)
    counter = 0
    while counter < total_class_len:
        if coreset_mode:
            temp_idx = torch.randint(0, energy.size(1) - class_per_task, (1, 1)).to(device)
        else:           
            temp_idx = torch.randint(energy.size(1) - class_per_task, energy.size(1), (1, 1)).to(device)
        if temp_idx == y_ans_idx[counter]:
            continue
        else:
            y_neg_idx[counter] = temp_idx
            counter+=1
    return energy.gather(dim=1, index=y_neg_idx.to(device))

def _calculate_energy_negative_with_confused_energy(energy, y_ans_idx, num_classes, device):
    mask = torch.nn.functional.one_hot(y_ans_idx, num_classes)
    mask = ~mask.bool().to(device)
    negative_full_energies = torch.masked_select(energy, mask).reshape(energy.size(0), -1)
    negative_max = torch.max(negative_full_energies, dim=1)[0]
    return negative_max

def contrastive_divergence(energy, y_ans_idx, device, class_per_task, coreset_mode):
    energy_pos = _calculate_energy_positive(energy, y_ans_idx)
    energy_neg = _calculate_energy_negative(energy, y_ans_idx, device, class_per_task, coreset_mode)
    # energy_partition = -1*torch.cat((energy_pos, energy_neg), dim=1)
    # cdiv_loss = energy_pos+torch.logsumexp(energy_partition, dim=1)
    cdiv_loss = energy_pos - energy_neg
    loss = cdiv_loss.mean()
    return loss

def contrastive_divergence_with_confused_energy(energy, y_ans_idx, device, num_classes):
    energy_pos = _calculate_energy_positive(energy, y_ans_idx)
    energy_neg = _calculate_energy_negative_with_confused_energy(energy, y_ans_idx, num_classes, device).view(-1,1)

    energy_partition = -1*torch.cat((energy_pos, energy_neg), dim=1)
    cdiv_loss = energy_pos+torch.logsumexp(energy_partition, dim=1)
    loss = cdiv_loss.mean()
    return loss

def get_criterion(loss, use_memory):
    if loss == 'nll_energy' and use_memory:
        return energy_nll_loss
    elif loss == 'contrastive_divergence':
        return contrastive_divergence
    elif loss == 'contrastive_divergence_confused':
        return contrastive_divergence_with_confused_energy
    elif loss == 'cross_entropy':
        return torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError