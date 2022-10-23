import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import wandb
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from modules.models import get_model
from modules.dataset import prepare_data, get_augmentation, Coreset_Dataset
from modules.loss import get_criterion
from modules.coreset import Offline_Coreset_Manager, accumulate_candidate, online_bin_based_sampling, add_online_coreset
from utils.utils import seed_everything, get_optimizer, calculate_answer, get_target, get_ans_idx
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
            coresets  =None):
    model.train()
    if args.learning_mode == 'online':
        candidates_x = [torch.empty(0) for _ in range(args.num_classes // args.num_tasks)]
        candidates_y = [torch.empty(0) for _ in range(args.num_classes // args.num_tasks)]
        candidates = (candidates_x, candidates_y)
    
    for p in model.parameters():
        p.require_grad = True
    if task_num != 1:
        temp_task_class_set = task_class_set[0]
        for i in range(1, len(task_class_set)):
            temp_task_class_set = temp_task_class_set.union(task_class_set[i])
        task_class_set = [temp_task_class_set]
    for e in range(args.epoch):
        total_class_set, train_accs, _, memory_in_epoch = train_one_epoch(args=args,
                                                                          epoch=e+1,
                                                                          device=device,
                                                                          model=model,
                                                                          loader=train_loader,
                                                                          optimizer=optimizer,
                                                                          criterion=criterion,
                                                                          task_class_set =task_class_set[-1], 
                                                                          total_class_set=total_class_set,
                                                                          task_num=task_num,
                                                                          coresets=coresets,
                                                                          candidates=candidates \
                                                                              if args.learning_mode == 'online' else None)
        task_infos = test_total(args=args,
                                device=device,
                                model=model, 
                                loaders=test_loaders, 
                                total_class_set=total_class_set,
                                task_num=task_num,
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
                    coresets=None,
                    candidates=None):
    pbar = tqdm(loader, total=loader.__len__(), position=0, leave=True)
    train_loss_list = []
    train_answers, total_len = 0, 0
    model.num_class = len(task_class_set)+len(total_class_set)
    memory_x, memory_y, memory_energy, memory_rep_in_epoch = torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
    memory_in_epoch = None
    if args.use_memory and task_num != 1:
        memory_sampler = RandomSampler(data_source=coresets, 
                                       replacement=False)
        memory_loader = DataLoader(dataset=coresets,
                                sampler=memory_sampler,
                                batch_size=32,
                                drop_last=False)
    for it, sample in enumerate(pbar):
        optimizer.zero_grad()
        if args.dataset == "splitted_mnist" or args.dataset == "cifar10" or args.dataset == "cifar100" or args.dataset == "tiny_imagenet":
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
    
        
        joint_targets = get_target(task_class_set, y).to(device).long()
        # print(task_class_set)
        # task_class_set_tensor = torch.tensor(list(task_class_set))
        # joint_targets = task_class_set_tensor.view(1, -1).expand(len(y), len(task_class_set_tensor)).to(device).long()
        y_ans_idx     = get_ans_idx(task_class_set, y).to(device)
        # print('outside the model: ', joint_targets.shape)
        # print(f"epoch: {epoch}, joint_targets : {joint_targets.shape}")
        energy = model(x, joint_targets)
        if args.criterion   == 'nll_energy':
            if args.use_memory:
                loss = criterion(energy=energy,
                                 y_ans_idx=y_ans_idx,
                                 classes_per_task=args.num_classes//args.num_tasks,
                                 task_class_set=task_class_set,
                                 coreset_mode=False)
            else:
                loss = criterion(energy=energy, 
                                 y_ans_idx=y_ans_idx)          
        elif args.criterion == 'contrastive_divergence':
            loss = criterion(energy=energy, 
                             y_ans_idx=y_ans_idx, 
                             device=device)
        train_answers += calculate_answer(energy, y_ans_idx)
        total_len     += y.shape[0]
            
        if args.use_memory and len(coresets) != 0 and task_num != 1:
            mem_sample = next(iter(memory_loader))
            mem_x = mem_sample[0].to(device)
            mem_y = mem_sample[1].to(device)
            mem_joint_targets = get_target(task_class_set, mem_y).to(device) 
            mem_y_ans_idx     = get_ans_idx(task_class_set, mem_y).to(device)
            mem_energy = model(mem_x, mem_joint_targets)    
            mem_loss = criterion(energy=mem_energy, 
                                 y_ans_idx=mem_y_ans_idx,
                                 task_class_set=task_class_set,
                                 classes_per_task=args.num_classes // args.num_tasks, 
                                 coreset_mode=True)
            loss          += args.lam * mem_loss
            train_answers += calculate_answer(mem_energy, mem_y_ans_idx)
            total_len     += mem_y.shape[0]            
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        train_loss_list.append(loss.item())
        desc = f"Train Epoch : {epoch}, Loss : {np.mean(train_loss_list):.3f}, Accuracy : {train_answers / total_len:.3f}"
        pbar.set_description(desc)

        if args.use_memory and epoch == args.epoch and args.learning_mode == 'offline':
            memory_in_epoch = accumulate_candidate(args.memory_option, model, energy, memory_x, memory_y, memory_energy, joint_targets, y_ans_idx, x, y)
        elif args.use_memory and args.learning_mode == 'online':
            for i in range(args.num_classes // args.num_tasks): # class per task만큼 순회
                cl = torch.tensor(list(task_class_set)[-1*(args.num_classes // args.num_tasks):])[i] # i번째에 해당하는 class (i : task class set의 index, cl : index에 해당하는 class)
                idx = (y == cl).nonzero(as_tuple=True) # 현재 task에 해당하는 class들만 slicing후 indexing해서 class 선택
                x_cur = x[idx]
                y_cur = y[idx]
                if args.memory_size - candidates[1][i].size(0) >= y_cur.size(0): # memory에 남아 있는 공간이 batch 내의 해당 class에 속하는 data의 크기보다 크다면
                    candidates[0][i] = torch.cat((candidates[0][i], x_cur.detach().cpu()))
                    candidates[1][i] = torch.cat((candidates[1][i], y_cur.detach().cpu()))
                else: # 아니라면 -> online bin sampling 수행
                    new_candidates_i, new_memory_energy = online_bin_based_sampling(model=model, 
                                                                                    memory_size=args.memory_size, 
                                                                                    class_per_tasks=args.num_classes//args.num_tasks,
                                                                                    candidates=(candidates[0][i], candidates[1][i]),
                                                                                    task_class_set=task_class_set, 
                                                                                    x=x_cur, 
                                                                                    y=y_cur,
                                                                                    device=device)
                    candidates[0][i] = new_candidates_i[0]
                    candidates[1][i] = new_candidates_i[1]
                    if not os.path.exists('asset'):
                        os.mkdir('asset')
                    if not os.path.exists(os.path.join('asset', 'online_bin_energy')):
                        os.mkdir(os.path.join('asset', 'online_bin_energy'))        
                    if (it+1) % 100 == 0:
                        torch.save(new_memory_energy, os.path.join('asset', 'online_bin_energy', f"Task_{task_num}_Iter_{iter}_class_{cl}_memory_bin_energy.pt"))
    
    if args.learning_mode == 'online':
        memory_in_epoch = candidates
        
    total_class_set = total_class_set.union(task_class_set)
    return total_class_set, train_answers / total_len, train_loss_list, memory_in_epoch


def test_total(args,
               device,
               model, 
               loaders, 
               total_class_set, 
               task_num,
               acc_matrix):
    
    task_infos = IntermediateInfos()
    for i,loader in enumerate(loaders):
        task_acc, pred_classes, ys, reps, pred_reps, lowest_img_infos, answer_energy, confused_energy, confused_class = \
            test_by_task(args, device, model, loader, total_class_set, i)
        task_infos.add_infos(task_acc, 'task_acc')
        task_infos.add_infos(pred_classes, 'pred_classes')
        task_infos.add_infos(ys, 'ys')
        task_infos.add_infos(reps, 'reps')
        task_infos.add_infos(pred_reps, 'pred_reps')
        task_infos.add_infos(lowest_img_infos, 'lowest_img_infos')
        if not os.path.exists(os.path.join('asset')):
            os.mkdir(os.path.join('asset'))
        if not os.path.exists(os.path.join('asset', 'answer_energy')):
            os.mkdir(os.path.join('asset', 'answer_energy'))
        if not os.path.exists(os.path.join('asset', 'confused_energy')):
            os.mkdir(os.path.join('asset', 'confused_energy'))    
        if not os.path.exists(os.path.join('asset', 'ys')):
            os.mkdir(os.path.join('asset', 'ys'))
        if not os.path.exists(os.path.join('asset', 'confused_class')):
            os.mkdir(os.path.join('asset', 'confused_class'))
        torch.save(answer_energy, os.path.join('asset', 'answer_energy', f"Task_{task_num}_{i+1}th_test_answer_energy.pt"))
        torch.save(confused_energy, os.path.join('asset', 'confused_energy', f"Task_{task_num}_{i+1}th_test_confused_energy.pt"))
        torch.save(ys, os.path.join('asset', 'ys', f"Task_{task_num}_{i+1}th_test_ys.pt"))
        torch.save(confused_class, os.path.join('asset', 'confused_class', f"Task_{task_num}_{i+1}th_test_confused_class.pt")) 
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
    answer_energy, confused_energy = torch.empty(0), torch.empty(0)

    ids = []
    for i, sample in enumerate(pbar):
        if args.dataset == "splitted_mnist" or args.dataset == "cifar10" or args.dataset == "cifar100" or args.dataset == "tiny_imagenet":
            x, y = sample[0].to(device), sample[1].to(device)
            
        elif args.dataset == "oxford_pet":
            id, x, y = sample[0], sample[1].to(device), sample[2].to(device)
            ids.extend(id)

        elif args.dataset == "permuted_mnist":
            x = sample[0].to(device)
            y = sample[1]+(10*(task_num))
            y = y.to(device)
        
        joint_targets = get_target(total_class_set, y).to(device)
        # energy = torch.empty(0).to(device)
        # rep = torch.empty(0)
        # for cl in range(len(class_set)):
        #     en, r = model(x, joint_targets[:,cl])
        #     energy = torch.cat((energy, en.detach()), dim=-1) # |bxc|
        #     rep = torch.cat((rep, torch.unsqueeze(r.detach().cpu(), 0)), dim=0) # |bx1024xc|
        energy = model(x, joint_targets)
        y = y.detach().cpu()
        energy = energy.detach().cpu() 
        pred_energy, pred_index = torch.min(energy, dim=1) # lowest energy among classes, prediction
        pred_class = class_set[pred_index] 
        
        answer_energy = torch.cat((answer_energy, torch.gather(energy.clone(), 1, y.reshape(-1,1))), dim=0)
        temp_energy = energy.clone()
        temp_energy[:, y] = float('inf')
        confused_pred, confused_class = torch.topk(temp_energy, 1, dim=1, largest=False)
        confused_energy = torch.cat((confused_energy, confused_pred), dim=0)
        
        ###### get prediction & intermediate representation
        # gt_rep = rep[y, np.arange(len(y)), :]
        # pred_rep = rep[pred_class,np.arange(len(pred_class)),:]
        # reps = torch.cat((reps, gt_rep), dim=0)
        # pred_reps = torch.cat((pred_reps, pred_rep))
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
    
    return accumulated_acc, pred_classes, ys, reps.numpy(), pred_reps.numpy(), lowest_img_infos, answer_energy, confused_energy,\
        confused_class
        

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
    optimizer = get_optimizer(args.optimizer, lr=args.lr, parameters=model.parameters(), weight_decay=args.weight_decay)
    criterion = get_criterion(args.criterion, args.use_memory)
    total_class_set = set()
    available_memory_options = ['bin_based', 'random_sample', 'low_energy', 'high_energy', 'min_score', 'max_score', 'confused_pred']
    if args.use_memory and args.learning_mode == 'online':
        coreset = Coreset_Dataset(torch.empty(0), torch.empty(0)) # To Do
    elif args.use_memory and args.learning_mode == 'offline':
        coreset_manager = Offline_Coreset_Manager(args=args,
                                              num_classes_per_tasks=args.num_classes // args.num_tasks,
                                              num_memory_per_class =args.memory_size // args.num_classes,
                                              available_memory_options=available_memory_options)
    if args.wandb:
            wandb.init(project='EBM-Continual',
                       group=args.dataset,
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
                                                                         coresets=coreset_manager.get_coreset() \
                                                                             if args.learning_mode == 'offline' else coreset)
        if args.use_memory and args.learning_mode == 'offline':
            print("Starting memory generation")
            coreset_manager.add_coreset(model =model,
                                        device=device,
                                        memory=memory_in_epoch)
            wasserstein_dist_df = coreset_manager.get_wasserstein_dist_df()
            energy_dist_df = coreset_manager.get_energy_dist_df()
        elif args.use_memory and args.learning_mode == 'online':
            memory_x, memory_y = memory_in_epoch
            new_coreset = Coreset_Dataset(torch.cat(memory_x), torch.cat(memory_y))
            coreset = ConcatDataset([coreset, new_coreset])
            
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
    parser.add_argument('--model', type=str, default='resnet_18', choices=('beginning', 'middle', 'end', 'ebm_mnist', 'end_oxford', \
        'resnet_18', 'resnet_34', 'resnet_50', 'resnet_101', 'resnet_152'))
    parser.add_argument('--norm', type=str, default='continualnorm', choices=('batchnorm', 'continualnorm', 'none'))
    parser.add_argument('--optimizer', type=str, default='adam', choices=('adam', 'adamw', 'sgd'))
    parser.add_argument('--lr', type=float, default=1e-06)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--criterion', type=str, default='nll_energy', choices=('nll_energy', 'contrastive_divergence'))
    parser.add_argument('--learning_mode', type=str, default='offline', choices=('offline', 'online'))
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--use_memory', action='store_true', default=True, help='use memory buffer for CIL if True')
    parser.add_argument('--memory_option', type=str, default='bin_based', choices=('random_sample', 'low_energy', \
                                                                                       'high_energy', 'min_score', 'max_score',\
                                                                                       'confused_pred', 'representation', 'bin_based'))
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--lam', type=float, default=1.0, help='term for balancing current loss and memory loss')
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--memory_size', type=int, default=20)
    parser.add_argument('--save_confusion_fig', default=False, action='store_true')
    parser.add_argument('--save_matrices', default=False, action='store_true')
    parser.add_argument('--save_gt_fig', default=False, action='store_true')
    parser.add_argument('--save_pred_tsne_fig', default=False, action='store_true')
    parser.add_argument('--save_pred_tsne_with_gt_label_fig', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--wandb', default=True, action='store_true')
    parser.add_argument('--run_name', type=str, default='')
    args = parser.parse_args()
    main(args)