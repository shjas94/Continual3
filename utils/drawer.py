import os
import yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

def draw_confusion(root, pred, y, task_num, dataset="oxford_pet"):
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.exists(os.path.join(root, 'confusion_matrix')):
        os.mkdir(os.path.join(root, 'confusion_matrix'))
    title = f'{dataset}_Task_{task_num}_Confusion_Matrix'
    pred_gt = np.unique(np.concatenate([pred, y], axis=0))
    
    with open(os.path.join('utils', 'labels.yml')) as outfile:
        label_map = yaml.safe_load(outfile)
    if dataset == "oxford_pet":
        dataset_label = label_map['OXFORD_LABELS']
    elif dataset == "cifar10":
        dataset_label = label_map['CIFAR10_LABELS']
    elif dataset == "tiny_imagenet":
        dataset_label = label_map['TINYIMAGENET_LABELS']
        
    labels = [list(dataset_label.keys())[list(dataset_label.values()).index(l)]\
            for l in pred_gt]
    cm = confusion_matrix(y, pred)
    f, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, cmap='RdBu_r', ax=ax)
    ax.set_xlabel('Predicted Labels'); ax.set_ylabel('True Labels')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(labels, rotation=40)
    ax.yaxis.set_ticklabels(labels, rotation=40)
    ax.figure.savefig(os.path.join(root, 'confusion_matrix', title + '.png'), bbox_inches='tight')

def draw_tsne_proj(root, embedding, ys, class_per_task, task_num, seed, img_infos, colors=mcolors.CSS4_COLORS, mode='gt', standardization=True, save_lowest=True, dataset='oxford_pet'):
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.exists(os.path.join(root, 'tsne_projection_' + mode)):
        os.mkdir(os.path.join(root, 'tsne_projection_' + mode))
    color_names = ('red', 'blue', 'green', 'yellow', 'black', 'cyan', 'purple', 'lightgreen', 'lavender', 'darkviolet',\
    'royalblue', 'tomato', 'sandybrown', 'palegreen', 'skyblue', 'deepskyblue', 'lime', 'plum', 'orange', 'darkkhaki', \
        'firebrick', 'rosybrown', 'mediumseagreen', 'limegreen', 'dimgrey', 'gainsboro', 'moccasin', 'springgreen', 'palevioletred', 'magenta'\
            'aquamarine', 'aqua', 'forestgreen', 'sienna', 'indianred', 'burlywood', 'darkcyan')
    full_ids = []
    pred_indices = []
    full_low_img_ids = []
    for i in range(len(img_infos)):
        if i == 0:
            full_ids = img_infos[i][4]
        else:
            full_ids.extend(img_infos[i][4])    
    for img_info in img_infos:
        for i in range(len(img_info[2])):
            pred_indices.append(full_ids.index(img_info[2][i]))
    with open(os.path.join('utils', 'labels.yml')) as outfile:
        label_map = yaml.safe_load(outfile)
    if dataset == 'oxford_pet':
        labels = label_map['OXFORD_LABELS']
    elif dataset == 'cifar10':
        labels = label_map['CIFAR10_LABELS']
    title = f'{dataset} {mode} Test {task_num} T-SNE Projection Result'
    t_sne = TSNE(n_components=2, learning_rate='auto', init='random', random_state=seed).fit_transform(embedding)
    if standardization:
        t_sne = MinMaxScaler().fit_transform(t_sne)
    tsnes = [t_sne[ys==i, :] for i in range(task_num*class_per_task)]
    plt.figure(figsize=(20, 20))
    for i, std_emb in enumerate(tsnes):
        label = list(labels.keys())[list(labels.values()).index(i)]
        plt.scatter(std_emb[:,0], std_emb[:,1], color=colors[color_names[i]], label=label, s=100)

    for i in range(len(t_sne)):
        for j in pred_indices:
            if i == j:
                label = list(labels.keys())[list(labels.values()).index(ys[i])]
                plt.scatter(t_sne[i,0], t_sne[i, 1], color=colors[color_names[ys[i]]], marker='*', s=200)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(fontsize=20)
    plt.savefig(os.path.join(root, 'tsne_projection_' + mode, title + '.png'), bbox_inches='tight')
    if mode == 'pred' and save_lowest:
        for id in full_low_img_ids:
            img = Image.open(os.path.join('data', 'images', id))
            if not os.path.exists('lowest_energy_img'):
                os.mkdir('lowest_energy_img')
            img.save(os.path.join(root, 'lowest_energy_img', f'Task_{task_num}_'+id+'.jpg'))
    


def draw_tsne_proj_SBC(root, embedding, ys, class_per_task, task_num, seed, colors=mcolors.CSS4_COLORS, mode='gt', standardization=True, dataset='oxford_pet'):
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.exists(os.path.join(root, 'SBC_tsne_projection_' + mode)):
        os.mkdir(os.path.join(root, 'SBC_tsne_projection_' + mode))
    color_names = ('red', 'blue', 'green', 'yellow', 'black', 'cyan', 'purple', 'lightgreen', 'lavender', 'darkviolet',\
    'royalblue', 'tomato', 'sandybrown', 'palegreen', 'skyblue', 'deepskyblue', 'lime', 'plum', 'orange', 'darkkhaki', \
        'firebrick', 'rosybrown', 'mediumseagreen', 'limegreen', 'dimgrey', 'gainsboro', 'moccasin', 'springgreen', 'palevioletred', 'magenta'\
            'aquamarine', 'aqua', 'forestgreen', 'sienna', 'indianred', 'burlywood', 'darkcyan')
    
    with open(os.path.join('utils', 'labels.yml')) as outfile:
        label_map = yaml.safe_load(outfile)
    if dataset == 'oxford_pet':
        labels = label_map['OXFORD_LABELS']
    elif dataset == 'cifar10':
        labels = label_map['CIFAR10_LABELS']
    title = f'SBC {dataset} {mode} Test {task_num} T-SNE Projection Result'
    t_sne = TSNE(n_components=2, learning_rate='auto', init='random', random_state=seed).fit_transform(embedding)
    if standardization:
        t_sne = MinMaxScaler().fit_transform(t_sne)
    tsnes = [t_sne[ys==i, :] for i in range(task_num*class_per_task)]
    plt.figure(figsize=(20, 20))
    for i, std_emb in enumerate(tsnes):
        label = list(labels.keys())[list(labels.values()).index(i)]
        plt.scatter(std_emb[:,0], std_emb[:,1], color=colors[color_names[i]], label=label, s=100)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(fontsize=20)
    plt.savefig(os.path.join(root, 'SBC_tsne_projection_' + mode, title + '.png'), bbox_inches='tight')


def save_config():
    pass