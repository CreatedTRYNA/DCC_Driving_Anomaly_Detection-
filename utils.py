import csv
from contextlib import contextmanager
from tqdm import tqdm
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc
from typing import Optional
import inspect
from models.resnet import ProjectionHead
from pathlib import Path


class MFGNContrastiveLoss(nn.Module):
    def __init__(self, margin=200.0):
        super(MFGNContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


def contrastive_loss(out_1, out_2):
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    bs = out_1.size(0)
    temp = 0.25
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temp)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)).bool()
    # [2B, 2B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temp)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


def my_contrastiveLoss(output1, output2, label, margin=200.0):
    '''
    :param output1: 非正则化的特征
    :param output2: 非正则化的特征
    :param label: label = 1 means separating the output1 and output2
    :return:
    '''
    # 将向量 x 和 y 的形状分别扩展为 [1, 10, 128] 和 [50, 1, 128]
    output1 = output1.unsqueeze(0)
    output2 = output2.unsqueeze(1)

    # 正则化
    # output1 = torch.norm(output1, p=1, dim=2)
    # output2 = torch.norm(output2, p=1, dim=2)

    # 计算 x 和 y 中所有向量之间的欧几里得距离
    euclidean_distance = torch.cdist(output1, output2, p=2).squeeze(-1)

    # euclidean_distance 的形状为 [50, 10]
    # 取出第一个元素，表示向量 x 与向量 y 之间的距离
    loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                  label * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss_contrastive


def l2_normalize(x, dim=1):
    return x / torch.sqrt(torch.sum(x ** 2, dim=dim).unsqueeze(dim))


def adjust_learning_rate(optimizer, lr_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_rate


class Logger(object):
    """Logger object for training process, supporting resume training"""

    def __init__(self, path, header, resume=False):
        """
        :param path: logging file path
        :param header: a list of tags for values to track
        :param resume: a flag controling whether to create a new
        file or continue recording after the latest step
        """
        self.log_file = None
        self.resume = resume
        self.header = header
        if not self.resume:
            self.log_file = open(path, 'w')
            self.logger = csv.writer(self.log_file, delimiter='\t')
            self.logger.writerow(self.header)
        else:
            self.log_file = open(path, 'a+')
            self.log_file.seek(0, os.SEEK_SET)
            reader = csv.reader(self.log_file, delimiter='\t')
            self.header = next(reader)
            # move back to the end of file
            self.log_file.seek(0, os.SEEK_END)
            self.logger = csv.writer(self.log_file, delimiter='\t')

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for tag in self.header:
            assert tag in values, 'Please give the right value as defined'
            write_values.append(values[tag])
        self.logger.writerow(write_values)
        self.log_file.flush()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _construct_depth_model(base_model):
    # modify the first convolution kernels for Depth input
    modules = list(base_model.modules())
    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
                                 list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]
    # modify parameters, assume the first blob contains the convolution kernels
    motion_length = 1
    params = [x.clone() for x in conv_layer.parameters()]
    kernel_size = params[0].size()
    new_kernel_size = kernel_size[:1] + (1 * motion_length,) + kernel_size[2:]
    new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
    new_conv = nn.Conv3d(1, conv_layer.out_channels, conv_layer.kernel_size, conv_layer.stride,
                         conv_layer.padding, bias=True if len(params) == 2 else False)
    new_conv.weight.data = new_kernels
    if len(params) == 2:
        new_conv.bias.data = params[1].data  # add bias if neccessary
    layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name
    # replace the first convlution layer
    setattr(container, layer_name, new_conv)
    return base_model


def get_fusion_label(csv_path):
    """
    Read the csv file and return labels
    :param csv_path: path of csv file
    :return: ground truth labels
    """
    gt = np.zeros(360000)
    base = -10000
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[-1] == '':
                continue
            if row[1] != '':
                base += 10000
            if row[4] == 'N':
                gt[base + int(row[2]):base + int(row[3]) + 1] = 1
            else:
                continue
    return gt


def evaluate(score, label, whether_plot, **kwargs):
    """
    Compute Accuracy as well as AUC by evaluating the scores
    :param score: scores of each frame in videos which are computed as the cosine similarity between encoded test vector and mean vector of normal driving
    :param label: ground truth
    :param whether_plot: whether plot the AUC curve
    :return: best accuracy, corresponding threshold, AUC
    """
    thresholds = np.arange(0., 1., 0.01)
    best_acc = 0.
    best_threshold = 0.
    for threshold in thresholds:
        prediction = score >= threshold
        correct = prediction == label

        acc = (np.sum(correct) / correct.shape[0] * 100)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
    AUC = auc(fpr, tpr)

    if whether_plot:
        plt.plot(fpr, tpr, color='r')
        # plt.fill_between(fpr, tpr, color='r', y2=0, alpha=0.3)
        plt.plot(np.array([0., 1.]), np.array([0., 1.]), color='b', linestyle='dashed')
        plt.tick_params(labelsize=23)
        # plt.text(0.9, 0.1, f'AUC: {round(AUC, 4)}', fontsize=25)
        plt.xlabel('False Positive Rate', fontsize=25)
        plt.ylabel('True Positive Rate', fontsize=25)
        # plt.show()
        plt.savefig(kwargs["save_fig_path"], bbox_inches='tight')
    return best_acc, best_threshold, AUC


def post_process(score, window_size=6):
    """
    post process the score
    :param score: scores of each frame in videos
    :param window_size: window size
    :param momentum: momentum factor
    :return: post processed score
    """
    processed_score = np.zeros(score.shape)
    for i in range(0, len(score)):
        processed_score[i] = np.mean(score[max(0, i - window_size + 1):i + 1])

    return processed_score


def get_center(model, rank, train_n_loader, train_a_loader, view):
    model.eval()
    model_head = ProjectionHead(128, 18)
    model_head.eval()
    train_n_feature_space = []
    train_a_feature_space = []
    with torch.no_grad():
        if os.path.exists(f'normvec/normal_n_vec_{view}.npy'):
            train_n_feature_space = np.load(f'normvec/normal_n_vec_{view}.npy')
        else:
            for (imgs, _) in tqdm(train_n_loader, desc='Norm-Train set feature extracting'):
                imgs = imgs.to(rank)
                features, _ = model(imgs)
                train_n_feature_space.append(features)
            train_n_feature_space = torch.cat(train_n_feature_space, dim=0).contiguous().cpu().numpy()
            np.save(os.path.join('./normvec/', f'normal_n_vec_{view}.npy'), train_n_feature_space)

        if os.path.exists(f'normvec/normal_a_vec_{view}.npy'):
            train_a_feature_space = np.load(f'normvec/normal_a_vec_{view}.npy')
        else:
            for (imgs, _) in tqdm(train_a_loader, desc='AbNorm-Train set feature extracting'):
                imgs = imgs.to(rank)
                features, _ = model(imgs)
                train_a_feature_space.append(features)
            train_a_feature_space = torch.cat(train_a_feature_space, dim=0).contiguous().cpu().numpy()
            np.save(os.path.join('./normvec/', f'normal_a_vec_{view}.npy'), train_a_feature_space)

        train_n_feature_space = model_head(torch.from_numpy(train_n_feature_space))
        train_a_feature_space = model_head(torch.from_numpy(train_a_feature_space))
    normal_center = torch.FloatTensor(train_n_feature_space).mean(dim=0)
    anormal_center = torch.FloatTensor(train_a_feature_space).mean(dim=0)

    return normal_center.to(rank), anormal_center.to(rank)


def get_score(score_folder, mode):
    """
    !!!Be used only when scores exist!!!
    Get the corresponding scores according to requiements
    :param score_folder: the folder where the scores are saved
    :param mode: top_d | top_ir | front_d | front_ir | fusion_top | fusion_front | fusion_d | fusion_ir | fusion_all
    :return: the corresponding scores according to requirements
    """
    if mode not in ['top_d', 'top_ir', 'front_d', 'front_ir', 'fusion_top', 'fusion_front', 'fusion_d', 'fusion_ir',
                    'fusion_all']:
        print(
            'Please enter correct mode: top_d | top_ir | front_d | front_ir | fusion_top | fusion_front | fusion_d | fusion_ir | fusion_all')
        return
    if mode == 'top_d':
        score = np.load(os.path.join(score_folder + '/score_top_d.npy'))
    elif mode == 'top_ir':
        score = np.load(os.path.join(score_folder + '/score_top_IR.npy'))
    elif mode == 'front_d':
        score = np.load(os.path.join(score_folder + '/score_front_d.npy'))
    elif mode == 'front_ir':
        score = np.load(os.path.join(score_folder + '/score_front_IR.npy'))
    elif mode == 'fusion_top':
        score1 = np.load(os.path.join(score_folder + '/score_top_d.npy'))
        score2 = np.load(os.path.join(score_folder + '/score_top_IR.npy'))
        score = np.mean((score1, score2), axis=0)
    elif mode == 'fusion_front':
        score3 = np.load(os.path.join(score_folder + '/score_front_d.npy'))
        score4 = np.load(os.path.join(score_folder + '/score_front_IR.npy'))
        score = np.mean((score3, score4), axis=0)
    elif mode == 'fusion_d':
        score1 = np.load(os.path.join(score_folder + '/score_top_d.npy'))
        score3 = np.load(os.path.join(score_folder + '/score_front_d.npy'))
        score = np.mean((score1, score3), axis=0)
    elif mode == 'fusion_ir':
        score2 = np.load(os.path.join(score_folder + '/score_top_IR.npy'))
        score4 = np.load(os.path.join(score_folder + '/score_front_IR.npy'))
        score = np.mean((score2, score4), axis=0)
    elif mode == 'fusion_all':
        score1 = np.load(os.path.join(score_folder + '/score_top_d.npy'))
        score2 = np.load(os.path.join(score_folder + '/score_top_IR.npy'))
        score3 = np.load(os.path.join(score_folder + '/score_front_d.npy'))
        score4 = np.load(os.path.join(score_folder + '/score_front_IR.npy'))
        score = np.mean((score1, score2, score3, score4), axis=0)

    return score


def select_device(device='', batch_size=0):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'

        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)

        arg = 'cuda:0'
    elif mps and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():  # prefer MPS if available
        arg = 'mps'
    else:  # revert to CPU
        arg = 'cpu'

    print(p)
    return torch.device(arg)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    # Decorator to make all processes in distributed training wait for each local_master to do something
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    # Print function arguments (optional args dict)
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}

    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    print(s + ', '.join(f'{k}={v}' for k, v in args.items()))
    # LOGGER.info(colorstr(s) + ', '.join(f'{k}={v}' for k, v in args.items()))
