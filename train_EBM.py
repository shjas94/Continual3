import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import wandb
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from modules.models import get_model
from modules.dataset import prepare_data, Coreset_Dataset
from modules.loss import get_criterion
from modules.coreset import Coreset_Manager
from utils.utils import seed_everything, get_optimizer, calculate_answer, get_target
from utils.drawer import draw_confusion, draw_tsne_proj
from utils.intermediate_infos import IntermediateInfos
import matplotlib.colors as mcolors

def trainer(args,
            device,  
            train_loader, 
            test_loaders,
            task_class_set, 
            total_class_set,
            model, 
            optimizer,
            criterion, 
            task_num,
            acc_matrix=None,
            memory_over_tasks_dataset=None):

    model.train()
    for p in model.parameters():
        p.require_grad = True
    if task_num != 1:
        temp_task_class_set = task_class_set[0]
        for i in range(1, len(task_class_set)):
            temp_task_class_set = temp_task_class_set.union(task_class_set[i])
        task_class_set = [temp_task_class_set]
    for e in range(args.epoch):
        total_class_set, train_accs, _, memory_in_epoch = train_one_epoch(args=args,
                                                         epoch=e,
                                                         device=device,
                                                         model=model,
                                                         loader=train_loader,
                                                         optimizer=optimizer,
                                                         criterion=criterion,
                                                         task_class_set =task_class_set[-1], 
                                                         total_class_set=total_class_set,
                                                         task_num=task_num,
                                                         memory_over_tasks_dataset=memory_over_tasks_dataset)
        task_infos = test_total(args=args,
                                device=device,
                                model=model, 
                                loaders=test_loaders, 
                                total_class_set=total_class_set,
                                task_num=task_num,
                                epoch=e,
                                acc_matrix=acc_matrix)
        
  
    return train_accs, task_infos, total_class_set, model, memory_in_epoch
        
def train_one_epoch(args,
                    epoch,
                    device, 
                    model, 
                    loader, 
                    optimizer, 
                    criterion, 
                    task_class_set, 
                    total_class_set,
                    task_num,
                    logger=None,
                    memory_over_tasks_dataset=None):
    
    pbar = tqdm(loader, total=loader.__len__(), position=0, leave=True)
    train_loss_list = []
    train_answers, total_len = 0, 0
    model.num_class = len(task_class_set)+len(total_class_set)
    
    memory_x, memory_y, memory_energy, memory_rep_in_epoch = torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
    
    for sample in pbar:
        optimizer.zero_grad()
        if args.dataset == "splitted_mnist" or args.dataset == "cifar10" or args.dataset == "cifar100":
            x, y = sample[0].to(device), sample[1].to(device)
        elif args.dataset == "oxford_pet":
            _, x, y = sample[0], sample[1].to(device), sample[2].to(device)
        elif args.dataset == "permuted_mnist":
            x = sample[1].to(device)
            if task_num == 1:
                y = sample[2].to(device)
            else:
                y = sample[2]+(10*(task_num-1))
                y = y.to(device)
        
        if args.use_memory and memory_over_tasks_dataset:
            memory_loader = DataLoader(dataset=memory_over_tasks_dataset,
                                       shuffle=True,
                                       batch_size=args.batch_size)
            mem_sample = next(iter(memory_loader))
            mem_x = mem_sample[0].to(device)
            mem_y = mem_sample[1].to(device)
            x = torch.cat((x, mem_x), dim=0)
            y = torch.cat((y, mem_y), dim=0)
                
        joint_targets = get_target(task_class_set, y).to(device)
        y_ans_idx     = torch.tensor([list(task_class_set).index(ans) for ans in y]).long()
        y_ans_idx     = y_ans_idx.view(len(y), 1).to(device)        
        energy        = torch.empty(0).to(device)
        for cl in range(joint_targets.shape[1]):
            en, _  = model(x, joint_targets[:,cl])
            energy = torch.cat((energy, en.reshape(-1,1)), dim = 1)
            
        if args.criterion   == 'nll_energy':
            loss = criterion(energy=energy, y_ans_idx=y_ans_idx)
        elif args.criterion == 'contrastive_divergence':
            loss = criterion(energy=energy, y_ans_idx=y_ans_idx, device=device)
            
            
        if args.use_memory:
            memory_x = torch.cat((memory_x, x.detach().cpu()))
            memory_y = torch.cat((memory_y, y.detach().cpu()))
            if args.memory_option == 'confused_pred':
                true_index = (y - min(y)).type(torch.LongTensor).view(-1, 1).detach().cpu()
                energy_cpu = energy.detach().cpu()
                true_energy = torch.gather(energy_cpu, dim=1, index=true_index)
                other_energy = energy_cpu[energy_cpu != true_energy].view(energy_cpu.shape[0], -1)
                neg_energy = torch.min(other_energy, dim=1, keepdim=True)[0]
                memory_energy = torch.cat((memory_energy, true_energy-neg_energy))
            else:
                memory_energy = torch.cat((memory_energy, model(x, y)[0].detach().cpu()))
        loss.backward()
        optimizer.step()
        
        #### Calculate Accs, Losses ####
        train_answers += calculate_answer(energy, y_ans_idx)
        total_len     += y.shape[0]        
        train_loss_list.append(loss.item())
        ################################
        
        desc = f"Train Epoch : {epoch+1}, Loss : {np.mean(train_loss_list):.3f}, Accuracy : {train_answers / total_len:.3f}"
        pbar.set_description(desc)
    
    memory_in_epoch = [memory_x.detach().cpu(), memory_y.detach().cpu(), memory_energy.detach().cpu()]
        
    total_class_set = total_class_set.union(task_class_set)
    return total_class_set, train_answers / total_len, train_loss_list, memory_in_epoch


def test_total(args,
               device,
               model, 
               loaders, 
               total_class_set, 
               task_num,
               epoch,
               acc_matrix):
    
    task_infos = IntermediateInfos()
    for i,loader in enumerate(loaders):
        task_acc, pred_classes, ys, reps, pred_reps, lowest_img_infos = \
            test_by_task(args, device, model, loader, total_class_set, i)
        task_infos.add_infos(task_acc, 'task_acc')
        task_infos.add_infos(pred_classes, 'pred_classes')
        task_infos.add_infos(ys, 'ys')
        task_infos.add_infos(reps, 'reps')
        task_infos.add_infos(pred_reps, 'pred_reps')
        task_infos.add_infos(lowest_img_infos, 'lowest_img_infos') 
    if args.wandb:
        if task_infos.__len__('task_acc') < args.num_tasks:
            for _ in range(args.num_tasks-task_infos.__len__('task_acc')):
                task_infos.add_infos(0, 'task_acc')
    return task_infos

@torch.no_grad()
def test_by_task(args,
                 device,
                 model,
                 loader,
                 total_class_set,
                 task_num,
                 logger=None):
    model.eval()
    pbar = tqdm(loader, total=loader.__len__(), position=0, leave=True)
    class_set = torch.tensor(list(total_class_set))
    pred_classes, ys = None, None
    reps, pred_reps = torch.empty(0), torch.empty(0)
    pred_energies, pred_indices= torch.empty(0), torch.empty(0)
    ids = []
    for i, sample in enumerate(pbar):
        if args.dataset == "splitted_mnist" or args.dataset == "cifar10" or args.dataset == "cifar100":
            x, y = sample[0].to(device), sample[1].to(device)
            
        elif args.dataset == "oxford_pet":
            id, x, y = sample[0], sample[1].to(device), sample[2].to(device)
            ids.extend(id)

        elif args.dataset == "permuted_mnist":
            x = sample[0].to(device)
            y = sample[1]+(10*(task_num))
            y = y.to(device)
        
        joint_targets = get_target(total_class_set, y).to(device)
        energy = torch.empty(0).to(device)
        rep = torch.empty(0)
        for cl in range(len(class_set)):
            en, r = model(x, joint_targets[:,cl])
            energy = torch.cat((energy, en.detach()), dim=-1) # |bxc|
            rep = torch.cat((rep, torch.unsqueeze(r.detach().cpu(), 0)), dim=0) # |bx1024xc|
        
        y = y.detach().cpu()
        energy = energy.detach().cpu() 
        pred_energy, pred_index = torch.min(energy, dim=1) # lowest energy among classes, prediction
        pred_class = class_set[pred_index] 
        
        ###### get prediction & intermediate representation
        gt_rep = rep[y, np.arange(len(y)), :]
        pred_rep = rep[pred_class,np.arange(len(pred_class)),:]
        reps = torch.cat((reps, gt_rep), dim=0)
        pred_reps = torch.cat((pred_reps, pred_rep))
        if i == 0:
            pred_energies = pred_energy
            pred_indices = pred_index.detach().cpu()
            pred_classes = pred_class
            ys = y
        elif i != 0:
            pred_energies = torch.cat((pred_energies, pred_energy), dim=0)
            pred_indices = torch.cat((pred_indices, pred_index), dim=0)
            pred_classes = torch.cat((pred_classes, pred_class), dim=0)
            ys = torch.cat((ys, y), dim=0)
        ######################################
        
        answer_sheet = pred_classes == ys
        accumulated_acc = np.sum(answer_sheet.numpy()) / len(ys.numpy())
        
        desc = f"Task {task_num+1} Test Acc : {accumulated_acc:.3f}"
        pbar.set_description(desc)
        
    pred_min_energies, pred_min_indices = torch.topk(pred_energies, 5, dim=-1, largest=False, sorted=False)
    pred_min_classes = pred_classes[pred_min_indices]
    y_min_classes = ys[pred_min_indices]
    pred_ids = [ids[i] for j in range(len(pred_min_indices)) for i in range(len(ids)) \
         if i == pred_min_indices[j]]

    lowest_img_infos = (pred_min_energies.numpy(), pred_min_classes.numpy(), pred_ids, y_min_classes.numpy(), ids)
    
    return accumulated_acc, pred_classes, ys, reps.numpy(), pred_reps.numpy(), lowest_img_infos
        

def main(args):
    print(args)
    seed_everything(args.seed)
    model = get_model(args.model)(args)
    
    if args.cuda:
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    model.to(device)
    train_loaders, test_loaders, task_class_sets = prepare_data(args)
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    optimizer = get_optimizer(args.optimizer, lr=args.lr, parameters=model.parameters())
    criterion = get_criterion(args.criterion)
    total_class_set = set()
    
    memory_over_tasks_dataset = Coreset_Dataset(torch.empty(0), torch.empty(0))
    
    if args.wandb:
            wandb.init(project='EBM-Continual',
                       group=args.model,
                       name=args.run_name,
                       config=args)
            wandb.watch(model)
    for task_num in range(len(train_loaders)):
        train_loader = train_loaders[task_num]
        print(f"=================Start Training Task{task_num+1}=================")
        _, task_infos, total_class_set, model, memory_in_epoch = trainer(args=args,
                                  device=device,
                                  train_loader=train_loader, 
                                  test_loaders=test_loaders[:task_num+1],
                                  task_class_set=task_class_sets[:task_num+1], 
                                  total_class_set=total_class_set, 
                                  model=model, 
                                  optimizer=optimizer,
                                  criterion=criterion, 
                                  task_num=task_num+1,
                                  acc_matrix=acc_matrix,
                                  memory_over_tasks_dataset=memory_over_tasks_dataset)
        if args.use_memory:
            coreset_manager = Coreset_Manager(model, args, memory_in_epoch, args.num_classes//args.num_tasks, memory_over_tasks_dataset, device)
            memory_over_tasks_dataset = coreset_manager.get_memory()
            wasserstein_dist_df = coreset_manager.get_wasserstein_dist_df()
            energy_dist_df = coreset_manager.get_energy_dist_df()
            
        if args.save_matrices:
            for i in range(task_infos.__len__('pred_classes')):
                np.save(os.path.join(args.data_root, f'task{task_num+1}_test{i+1}_pred'),
                        task_infos.get_infos('pred_classes')[i])
                np.save(os.path.join(args.data_root, f'task{task_num+1}_test{i+1}_ys'),
                        task_infos.get_infos('ys')[i])
                np.save(os.path.join(args.data_root, f'task{task_num+1}_test{i+1}_reps'),
                        task_infos.get_infos('reps')[i])
        
        if args.save_confusion_fig:
            total_pred = np.concatenate(task_infos.get_infos('pred_classes'), axis=0)
            total_y = np.concatenate(task_infos.get_infos('ys'), axis=0)
            draw_confusion(root=args.fig_root,
                               pred=total_pred, 
                               y=total_y, 
                               task_num=task_num+1,
                               dataset=args.dataset) 
        if args.save_gt_fig:
            total_rep = np.concatenate(task_infos.get_infos('reps'), axis=0)
            total_y = np.concatenate(task_infos.get_infos('ys'), axis=0)
            draw_tsne_proj(args.fig_root, total_rep, total_y, args.num_classes // args.num_tasks, task_num+1, args.seed, \
                task_infos.get_infos('lowest_img_infos'), dataset=args.dataset)
        if args.save_pred_tsne_fig:
            total_pred_rep = np.concatenate(task_infos.get_infos('pred_reps'), axis=0)
            total_pred = np.concatenate(task_infos.get_infos('pred_classes'), axis=0)
            draw_tsne_proj(args.fig_root, total_pred_rep, total_pred, args.num_classes // args.num_tasks, task_num+1, args.seed, \
                task_infos.get_infos('lowest_img_infos'), mcolors.CSS4_COLORS, 'pred', dataset=args.dataset)
        if args.save_pred_tsne_with_gt_label_fig:
            total_pred_rep = np.concatenate(task_infos.get_infos('pred_reps'), axis=0)
            total_y = np.concatenate(task_infos.get_infos('ys'), axis=0)
            draw_tsne_proj(args.fig_root, total_pred_rep, total_y, args.num_classes // args.num_tasks, task_num+1, args.seed, \
                task_infos.get_infos('lowest_img_infos'), mcolors.CSS4_COLORS, 'pred_with_gt_labels', dataset=args.dataset)
        if args.wandb:
            task_acc = task_infos.get_infos('task_acc')
            for i in range(task_infos.__len__('task_acc')):
                wandb.log({
                f"Task {i+1} Test Accuracy" : task_acc[i]
                })
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--fig_root', type=str, default='./asset')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=('splitted_mnist', 'permuted_mnist', 'cifar10','oxford_pet', 'cifar100', 'tiny_imagenet'))
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_tasks', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--cuda', default=True, action='store_true')
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--model', type=str, default='resnet_34', choices=('beginning', 'middle', 'end', 'ebm_mnist', 'end_oxford', 'resnet_18', 'resnet_34', 'resnet_50', 'resnet_101', 'resnet_152'))
    parser.add_argument('--optimizer', type=str, default='adam', choices=('adam', 'adamw', 'sgd'))
    parser.add_argument('--lr', type=float, default=1e-04)
    parser.add_argument('--criterion', type=str, default='nll_energy', choices=('nll_energy', 'contrastive_divergence'))
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--use_memory', action='store_true', default=True, help='use memory buffer for CIL if True')
    parser.add_argument('--memory_option', type=str, default='low_energy', choices=('random_sample', 'low_energy', \
                                                                                       'high_energy', 'min_score', 'max_score',\
                                                                                       'confused_pred', 'representation', 'bin_based'))
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--memory_size', type=int, default=20)
    parser.add_argument('--save_confusion_fig', default=True, action='store_true')
    parser.add_argument('--save_matrices', default=True, action='store_true')
    parser.add_argument('--save_gt_fig', default=True, action='store_true')
    parser.add_argument('--save_pred_tsne_fig', default=True, action='store_true')
    parser.add_argument('--save_pred_tsne_with_gt_label_fig', default=True, action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--run_name', type=str, default='')
    args = parser.parse_args()
    main(args)