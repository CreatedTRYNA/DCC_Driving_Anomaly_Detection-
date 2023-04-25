import argparse

import torch
from sklearn.metrics import roc_curve, auc

from utils import l2_normalize
import numpy as np
import os
from tqdm import tqdm
import argparse


# import main

# parser = argparse.ArgumentParser()
#
# main.setup_arg_parser(parser)
# args = parser.parse_args()
# os.environ["LRU_CACHE_CAPACITY"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# os.environ["CUDA_VISIBLE_DEVICES"] = args.device


def get_normal_vector(model, train_normal_loader_for_test, cal_vec_batch_size, feature_dim, use_cuda):
    total_batch = int(len(train_normal_loader_for_test))
    print("=====================================Calculating Average Normal Vector=====================================")
    if use_cuda:
        normal_vec = torch.zeros((1, 512)).cuda()
    else:
        normal_vec = torch.zeros((1, 512))
    train_normal_loader_for_test_bar = tqdm(train_normal_loader_for_test)
    for batch, (normal_data, idx) in enumerate(train_normal_loader_for_test_bar):
        if use_cuda:
            normal_data = normal_data.cuda()
        _, outputs = model(normal_data)
        outputs = outputs.detach()
        normal_vec = (torch.sum(outputs, dim=0) + normal_vec * batch * cal_vec_batch_size) / (
                (batch + 1) * cal_vec_batch_size)
        # train_normal_loader_for_test_bar.desc = f'Calculating Average Normal Vector: Batch {batch + 1} / {total_batch}'
        # print(f'Calculating Average Normal Vector: Batch {batch + 1} / {total_batch}')
    normal_vec = l2_normalize(normal_vec)
    return normal_vec


def split_acc_diff_threshold(model_mtv, normal_vec, test_loader, use_cuda,
                             rank, args):
    """
    Search the threshold that split the scores the best and calculate the corresponding accuracy
    """
    total_batch = int(len(test_loader))
    print("================================================Evaluating================================================")
    total_n = 0
    total_a = 0
    threshold = np.arange(0., 1., 0.01)
    total_correct_a = np.zeros(threshold.shape[0])
    total_correct_n = np.zeros(threshold.shape[0])
    # if rank == 0:
    test_bar = tqdm(test_loader)
    similarities = []
    gts = []
    for batch, batch_data in enumerate(test_bar):
        if use_cuda:
            batch_data[0] = batch_data[0].cuda()  # context
            batch_data[1] = batch_data[1].cuda()  # gt
        n_num = torch.sum(batch_data[1]).cpu().detach().numpy()  # gt如果是正常驾驶则为1，否则为0
        total_n += n_num
        total_a += (batch_data[0].size(0) - n_num)
        # outputs = model_mtv(batch_data[0])
        outputs, _ = model_mtv(batch_data[0])
        outputs = outputs.detach()
        similarity = torch.mm(outputs, normal_vec.t())
        similarities.extend(similarity.squeeze().tolist())
        for i in range(len(threshold)):
            # If similarity between sample and average normal vector is smaller than threshold,
            # then this sample is predicted as anormal driving which is set to 0
            prediction = similarity >= threshold[i]
            correct = prediction.squeeze() == batch_data[1]
            total_correct_a[i] += torch.sum(correct[~batch_data[1].bool()])
            total_correct_n[i] += torch.sum(correct[batch_data[1].bool()])
        # print(f'Evaluating: Batch {batch + 1} / {total_batch}')
        gt = batch_data[1].cpu().detach().numpy()
        gts.extend(gt)
        test_bar.desc = f'Evaluating: Batch {batch + 1} / {total_batch}'
        # print('\n')

    fpr, tpr, roc_threshold = roc_curve(gts, similarities)
    rec_auc = auc(fpr, tpr)
    acc_n = [(correct_n / total_n) for correct_n in total_correct_n]
    acc_a = [(correct_a / total_a) for correct_a in total_correct_a]
    acc = [((total_correct_n[i] + total_correct_a[i]) / (total_n + total_a)) for i in range(len(threshold))]
    best_acc = np.max(acc)
    idx = np.argmax(acc)

    best_threshold = idx * 0.01
    return best_acc, best_threshold, acc_n[idx], acc_a[idx], acc, acc_n, acc_a,rec_auc


def split_acc_diff_threshold_for_multi_views(model, normal_vec_list, test_loader_list, use_cuda, rank, views,
                                             total_batch):
    """
    Search the threshold that split the scores the best and calculate the corresponding accuracy
    """
    # total_batch = int(len(zip(*test_loader)[0]))
    print("================================================Evaluating================================================")
    total_n = 0
    total_a = 0
    threshold = np.arange(0., 1., 0.01)
    total_correct_a = np.zeros(threshold.shape[0])
    total_correct_n = np.zeros(threshold.shape[0])

    test_loader = zip(test_loader_list[0], test_loader_list[1])
    # test_loader = zip(test_loader_list[0])
    test_bar = tqdm(test_loader)
    for batch, (batch_views_data) in enumerate(test_bar):  # 有多个view
        # val_data_list= []
        # val_gt_list = []
        similarity_list = [0] * len(views)
        if use_cuda:
            for view in range(len(views)):
                batch_views_data[view][0] = batch_views_data[view][0].cuda()
                batch_views_data[view][1] = batch_views_data[view][1].cuda()

        _, outputs = model(
            [batch_views_data[view][0] for view in range(len(views))])  # model:[view1,view2]->outputs:[view1,view2]
        for view in range(len(views)):
            # val_data_list.append(batch_views_data[view][0])  # 构建一个list类型的data给model
            # val_gt_list.append(batch_views_data[view][1])   # 构建一个list类型的gt给之后验证

            val_data_tensor = batch_views_data[view][0]  # [70,16,1,112,112]
            val_gt_tensor = batch_views_data[view][1]  # 每个view的gt是一样的

            n_num = torch.sum(val_gt_tensor).cpu().detach().numpy()  # gt如果是正常驾驶则为1，否则为0
            total_n += n_num
            total_a += (val_data_tensor.size(0) - n_num)
            # _, outputs = model(val_data_tensor)  # model:[view1,view2]->views_feature:[view1,view2]

            outputs[view] = outputs[view].detach()
            similarity_list[view] = torch.mm(outputs[view], normal_vec_list[
                view].t())  # outputs->[val_bs*views,512], normal_vec -> [1,512]

        similarity = torch.mean(torch.cat(similarity_list, dim=1), dim=1)
        # similarity = similarity_list[0]
        #  由于每个view的gt是一样的，所以这里只要使用mean之后的sim score来测一次acc即可
        for i in range(len(threshold)):
            # If similarity between sample and average normal vector is smaller than threshold,
            # then this sample is predicted as anormal driving which is set to 0
            # TODO
            prediction = similarity >= threshold[i]
            correct = prediction.squeeze() == batch_views_data[0][1]
            total_correct_a[i] += torch.sum(correct[~batch_views_data[0][1].bool()])
            total_correct_n[i] += torch.sum(correct[batch_views_data[0][1].bool()])
        # print(f'Evaluating: Batch {batch + 1} / {total_batch}')

        test_bar.desc = f'Evaluating: Batch {batch + 1} / {total_batch}'
        # print('\n')
    acc_n = [(correct_n / (total_n / len(views))) for correct_n in total_correct_n]
    acc_a = [(correct_a / (total_a / len(views))) for correct_a in total_correct_a]
    acc = [((total_correct_n[i] + total_correct_a[i]) / ((total_n + total_a) / len(views))) for i in
           range(len(threshold))]
    best_acc = np.max(acc)
    idx = np.argmax(acc)
    best_threshold = idx * 0.01
    return best_acc, best_threshold, acc_n[idx], acc_a[idx], acc, acc_n, acc_a


def cal_score(model_front_d, model_front_ir, model_top_d, model_top_ir, normal_vec_front_d, normal_vec_front_ir,
              normal_vec_top_d, normal_vec_top_ir, test_loader_front_d, test_loader_front_ir, test_loader_top_d,
              test_loader_top_ir, score_folder, use_cuda):
    """
    Generate and save scores of top_depth/top_ir/front_d/front_ir views
    """
    assert int(len(test_loader_front_d)) == int(len(test_loader_front_ir)) == int(len(test_loader_top_d)) == int(
        len(test_loader_top_ir))
    total_batch = int(len(test_loader_front_d))
    sim_list = torch.zeros(0)
    sim_1_list = torch.zeros(0)
    sim_2_list = torch.zeros(0)
    sim_3_list = torch.zeros(0)
    sim_4_list = torch.zeros(0)
    label_list = torch.zeros(0).type(torch.LongTensor)
    test_ft_ird_bar = tqdm(zip(test_loader_front_d, test_loader_front_ir, test_loader_top_d, test_loader_top_ir))
    for batch, (data1, data2, data3, data4) in enumerate(test_ft_ird_bar):
        if use_cuda:
            data1[0] = data1[0].cuda()
            data1[1] = data1[1].cuda()
            data2[0] = data2[0].cuda()
            data2[1] = data2[1].cuda()
            data3[0] = data3[0].cuda()
            data3[1] = data3[1].cuda()
            data4[0] = data4[0].cuda()
            data4[1] = data4[1].cuda()

        assert torch.sum(data1[1] == data2[1]) == torch.sum(data2[1] == data3[1]) == torch.sum(data3[1] == data4[1]) == \
               data1[1].size(0)

        out_1 = model_front_d(data1[0])[1].detach()
        out_2 = model_front_ir(data2[0])[1].detach()
        out_3 = model_top_d(data3[0])[1].detach()
        out_4 = model_top_ir(data4[0])[1].detach()

        sim_1 = torch.mm(out_1, normal_vec_front_d.t())
        sim_2 = torch.mm(out_2, normal_vec_front_ir.t())
        sim_3 = torch.mm(out_3, normal_vec_top_d.t())
        sim_4 = torch.mm(out_4, normal_vec_top_ir.t())
        sim = (sim_1 + sim_2 + sim_3 + sim_4) / 4

        sim_list = torch.cat((sim_list, sim.squeeze().cpu()))
        label_list = torch.cat((label_list, data1[1].squeeze().cpu()))
        sim_1_list = torch.cat((sim_1_list, sim_1.squeeze().cpu()))
        sim_2_list = torch.cat((sim_2_list, sim_2.squeeze().cpu()))
        sim_3_list = torch.cat((sim_3_list, sim_3.squeeze().cpu()))
        sim_4_list = torch.cat((sim_4_list, sim_4.squeeze().cpu()))
        test_ft_ird_bar.desc = f'Evaluating: Batch {batch + 1} / {total_batch}'
        # print(f'Evaluating: Batch {batch + 1} / {total_batch}')

    np.save(os.path.join(score_folder, 'score_front_d.npy'), sim_1_list.numpy())
    print('score_front_d.npy is saved')
    np.save(os.path.join(score_folder, 'score_front_IR.npy'), sim_2_list.numpy())
    print('score_front_IR.npy is saved')
    np.save(os.path.join(score_folder, 'score_top_d.npy'), sim_3_list.numpy())
    print('score_top_d.npy is saved')
    np.save(os.path.join(score_folder, 'score_top_IR.npy'), sim_4_list.numpy())
    print('score_top_IR.npy is saved')


def cal_score_for_fir(model_front_ir, normal_vec_front_ir,
                      test_loader_front_ir, score_folder, use_cuda):
    """
    Generate and save scores of top_depth/top_ir/front_d/front_ir views
    """

    total_batch = int(len(test_loader_front_ir))
    sim_2_list = torch.zeros(0)
    test_bar = tqdm(test_loader_front_ir)
    for batch, data2 in enumerate(test_bar):
        if use_cuda:
            data2[0] = data2[0].cuda()
            data2[1] = data2[1].cuda()

        out_2 = model_front_ir(data2[0])[1].detach()
        sim_2 = torch.mm(out_2, normal_vec_front_ir.t())

        sim_2_list = torch.cat((sim_2_list, sim_2.squeeze().cpu()))

        test_bar.desc = f'Evaluating: Batch {batch + 1} / {total_batch}'
        # print(f'Evaluating: Batch {batch + 1} / {total_batch}')

    np.save(os.path.join(score_folder, 'score_front_IR.npy'), sim_2_list.numpy())
    print('score_front_IR.npy is saved')
