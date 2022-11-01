import argparse
import os
import torch
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
from tqdm import tqdm
import wandb
from modules.models import get_model
from modules.dataset import prepare_data
from modules.loss import get_criterion
from modules.coreset import Memory
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
            memory    =None):
    model.train()
    for p in model.parameters():
        p.require_grad = True
    if task_num != 1:
        temp_task_class_set = task_class_set[0]
        for i in range(1, len(task_class_set)):
            temp_task_class_set = temp_task_class_set.union(task_class_set[i])
        task_class_set = [temp_task_class_set]
        # ex) task 1 -> task_class_set : [{0, 1}]
        #     task 2 -> task_class_set : [{0, 1, 2, 3}]
        #     task 3 -> task_class set : [{0, 1, 2, 3, 4, 5}] ...
        
    for e in range(args.epoch):
        total_class_set, train_accs, _, memory = train_one_epoch(args           =args,
                                                                 epoch          =e+1,
                                                                 device         =device,
                                                                 model          =model,
                                                                 loader         =train_loader,
                                                                 optimizer      =optimizer,
                                                                 criterion      =criterion,
                                                                 task_class_set =task_class_set[-1], 
                                                                 total_class_set=total_class_set,
                                                                 task_num       =task_num,
                                                                 memory         =memory)
        task_infos = test_total(args           =args,
                                device         =device,
                                model          =model, 
                                loaders        =test_loaders, 
                                total_class_set=total_class_set,
                                task_num       =task_num,
                                acc_matrix     =acc_matrix)
        
        if args.use_memory and args.learning_mode == 'offline' and e+1 == args.epoch:
            # 마지막 epoch 이후에
            # offline bin sampling
            memory.add_sample_offline(loader        =train_loader, 
                                      task_class_set=task_class_set[-1], 
                                      model         =model, 
                                      device        =device)
        if args.use_memory and e+1 == args.epoch and task_num > 1:
            # task_num이 1 이상인 경우 -> training 후 기존에 저장되어 있던 메모리를 조정해야 하는 경우
            memory.update_memory(task_num+1)
        elif args.use_memory and e+1 == args.epoch and task_num == 1:
            # task_num이 1인 경우     -> training 후 task_id와 저장할 memory 크기만 바꿔주면 되는 경우
            memory.set_task_id(task_num+1)
            memory.merge_samples()
            memory.set_cur_memory_size()
        
        if task_num == args.num_tasks:
            final_memory_x      = memory.x
            final_memory_y      = memory.y
            final_memory_energy = memory.energy
            final_memory_rep    = memory.rep
            torch.save(final_memory_x, 'asset/final_memory/final_memory_x.pt')
            torch.save(final_memory_y, 'asset/final_memory/final_memory_y.pt')
            torch.save(final_memory_energy, 'asset/final_memory/final_memory_energy.pt')
            torch.save(final_memory_energy, 'asset/final_memory/final_memory_rep.pt')
    # final_energies = calculate_final_energy(model=model, 
    #                                         device=device, 
    #                                         loader=train_loader, 
    #                                         task_class_set=task_class_set[-1])
  
    return train_accs, task_infos, total_class_set, model, memory
        
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
                    memory=None):
    pbar = tqdm(loader, total=loader.__len__(), position=0, leave=True, colour='cyan')
    train_loss_list = []
    train_answers, cur_answers, mem_answers, total_len, cur_len, mem_len = 0, 0, 0, 0, 0, 0
    if args.use_memory and task_num != 1:
        memory_data = memory.sample()
        memory_sampler = RandomSampler(data_source=memory_data, 
                                       replacement=False)
        memory_loader = DataLoader(dataset=memory_data,
                                   sampler=memory_sampler,
                                   batch_size=args.batch_size,
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
        cur_data_size = x.size(0)
        joint_targets = get_target(task_class_set, y).to(device).long()
        y_ans_idx     = get_ans_idx(task_class_set, y).to(device)
        total_len    += x.size(0)
        if args.use_memory and task_num > 1:
            mem_sample        = next(iter(memory_loader))
            mem_x             = mem_sample[0].to(device)
            mem_y             = mem_sample[1].to(device)
            mem_joint_targets = get_target(task_class_set, mem_y).to(device) 
            mem_y_ans_idx     = get_ans_idx(task_class_set, mem_y).to(device)
            
            x = torch.cat((x, mem_x), dim=0)
            y = torch.cat((y, mem_y), dim=0)
            
            joint_targets = torch.cat((joint_targets, mem_joint_targets), dim=0)
            y_ans_idx     = torch.cat((y_ans_idx, mem_y_ans_idx), dim=0)
            total_len    += mem_x.size(0)
            
        energy, rep = model(x, joint_targets)
        if args.criterion == 'nll_energy':
            if args.use_memory and task_num > 1:
                cur_energies, mem_energies = energy[:cur_data_size, :], energy[cur_data_size:, :]
                
                cur_loss = criterion(energy=cur_energies,
                                     y_ans_idx=y_ans_idx[:cur_data_size, :],
                                     classes_per_task=args.num_classes//args.num_tasks,
                                     task_class_set=task_class_set,
                                     coreset_mode=False)
                mem_loss = criterion(energy=mem_energies,
                                     y_ans_idx=y_ans_idx[cur_data_size:, :],
                                     classes_per_task=args.num_classes//args.num_tasks,
                                     task_class_set=task_class_set,
                                     coreset_mode=True)
                loss = cur_loss + args.lam*mem_loss
            else:
                loss = criterion(energy=energy,
                                 y_ans_idx=y_ans_idx,
                                 classes_per_task=args.num_classes//args.num_tasks,
                                 task_class_set=task_class_set,
                                 coreset_mode=False)
        elif args.criterion == 'contrastive_divergence':
            if args.use_memory and task_num > 1:
                cur_energies, mem_energies = energy[:cur_data_size, :], energy[cur_data_size:, :]
                
                cur_loss = criterion(energy=cur_energies,
                                     y_ans_idx=y_ans_idx[:cur_data_size, :],
                                     device=device,
                                     class_per_task=args.num_classes//args.num_tasks,
                                     coreset_mode=False)
                mem_loss = criterion(energy=mem_energies,
                                     y_ans_idx=y_ans_idx[cur_data_size:, :],
                                     device=device,
                                     class_per_task=args.num_classes//args.num_tasks,
                                     coreset_mode=True)
                loss = cur_loss + args.lam*mem_loss
            else:
                loss = criterion(energy=energy,
                                 y_ans_idx=y_ans_idx,
                                 device=device,
                                 class_per_task=args.num_classes//args.num_tasks,
                                 coreset_mode=False)
        cur_len       += cur_data_size
        mem_len       += x.size(0) - cur_data_size
        train_answers += calculate_answer(energy, y_ans_idx)
        
        if args.use_memory and task_num > 1:
            cur_answers += calculate_answer(cur_energies, y_ans_idx[:cur_data_size])
            mem_answers += calculate_answer(mem_energies, y_ans_idx[cur_data_size:])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        train_loss_list.append(loss.item())
        if args.use_memory and task_num > 1:
            desc = f"Train Epoch : {epoch}, Loss : {np.mean(train_loss_list):.3f}, Total Accuracy : {train_answers / total_len:.3f}, Cur Accuracy : {cur_answers / cur_len:.3f}, Mem Accuracy : {mem_answers / mem_len:.3f}"
        else:
            desc = f"Train Epoch : {epoch}, Loss : {np.mean(train_loss_list):.3f}, Total Accuracy : {train_answers / total_len:.3f}"
        pbar.set_description(desc)
        
        # To Do
        if args.use_memory and args.learning_mode == 'online':
            for i in range(args.num_classes // args.num_tasks): 
                cl         = torch.tensor(list(task_class_set)[-1*(args.num_classes // args.num_tasks):])[i] 
                idx        = (y == cl).nonzero(as_tuple=True)[0] 
                x_cur      = x[idx].detach().cpu()
                y_cur      = y[idx].detach().cpu()
                energy_cur = energy.gather(dim=1, index=y_ans_idx)
                energy_cur = energy_cur[idx].detach().cpu()
                memory.add_sample_online(x_cur, y_cur, energy_cur, i)

    total_class_set = total_class_set.union(task_class_set)
    return total_class_set, train_answers / total_len, train_loss_list, memory


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
    pbar = tqdm(loader, total=loader.__len__(), position=0, leave=True, colour='magenta')
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
        
        joint_targets                = get_target(total_class_set, y).to(device)
        energy, rep                  = model(x, joint_targets)
        y                            = y.detach().cpu()
        energy                       = energy.detach().cpu() 
        pred_energy, pred_index      = torch.min(energy, dim=1) # lowest energy among classes, prediction
        pred_class                   = class_set[pred_index] 
        
        answer_energy                 = torch.cat((answer_energy, torch.gather(energy.clone(), 1, y.reshape(-1,1))), dim=0)
        temp_energy                   = energy.clone()
        temp_energy[:, y]             = float('inf')
        confused_pred, confused_class = torch.topk(temp_energy, 1, dim=1, largest=False)
        confused_energy               = torch.cat((confused_energy, confused_pred), dim=0)
        
        if i == 0:
            pred_energies = pred_energy
            pred_indices  = pred_index.detach().cpu()
            pred_classes  = pred_class
            ys            = y
        elif i != 0:
            pred_energies = torch.cat((pred_energies, pred_energy), dim=0)
            pred_indices  = torch.cat((pred_indices, pred_index), dim=0)
            pred_classes  = torch.cat((pred_classes, pred_class), dim=0)
            ys            = torch.cat((ys, y), dim=0)
        
        answer_sheet    = pred_classes == ys
        accumulated_acc = np.sum(answer_sheet.numpy()) / len(ys.numpy())
        
        desc = f"Task {task_num+1} Test Acc : {accumulated_acc:.3f}"
        pbar.set_description(desc)
        
    pred_min_energies, pred_min_indices = torch.topk(pred_energies, 5, dim=-1, largest=False, sorted=False)
    pred_min_classes = pred_classes[pred_min_indices]
    y_min_classes    = ys[pred_min_indices]
    pred_ids         = [ids[i] for j in range(len(pred_min_indices)) for i in range(len(ids)) \
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
    
    acc_matrix      = np.zeros((args.num_tasks, args.num_tasks))
    optimizer       = get_optimizer(args.optimizer, lr=args.lr, parameters=model.parameters(), weight_decay=args.weight_decay)
    criterion       = get_criterion(args.criterion, args.use_memory)
    total_class_set = set()
    
    if args.use_memory:
        memory = Memory(args, args.fixed_memory_slot)
        
    if args.wandb:
            wandb.init(project='EBM-Continual',
                       group=args.dataset,
                       name=args.run_name,
                       config=args)
            wandb.watch(model)
    after_train_energies = []
    for task_num in range(len(train_loaders)):
        train_loader = train_loaders[task_num]
        print(f"=================Start Training Task{task_num+1}=================")
        _, task_infos, total_class_set, model, memory = trainer(args           =args,
                                                                device         =device,
                                                                train_loader   =train_loader, 
                                                                test_loaders   =test_loaders[:task_num+1],
                                                                task_class_set =task_class_sets[:task_num+1], 
                                                                total_class_set=total_class_set, 
                                                                model          =model, 
                                                                optimizer      =optimizer,
                                                                criterion      =criterion, 
                                                                task_num       =task_num+1,
                                                                acc_matrix     =acc_matrix,
                                                                memory         =memory)
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
    ##### 임시 구현 #####
    # os.mkdir('asset/after_train_energies')
    # for i in range(len(after_train_energies)):
    #     torch.save(after_train_energies[i], f"asset/Class_{i}_after_train_energies.pt")    
    ####################            
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
    parser.add_argument('--fixed_memory_slot', action='store_true', default=True)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--lam', type=float, default=1.0, help='term for balancing current loss and memory loss')
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--memory_size', type=int, default=20)
    parser.add_argument('--save_confusion_fig', default=True, action='store_true')
    parser.add_argument('--save_matrices', default=False, action='store_true')
    parser.add_argument('--save_gt_fig', default=False, action='store_true')
    parser.add_argument('--save_pred_tsne_fig', default=False, action='store_true')
    parser.add_argument('--save_pred_tsne_with_gt_label_fig', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--run_name', type=str, default='')
    args = parser.parse_args()
    main(args)