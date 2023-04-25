import torch
import torch.nn.functional as F
from torch import nn


class MemoryBank(nn.Module):
    def __init__(self,
                 temp=0.07,
                 queue_size=140,
                 embed_dim=128,
                 use_ctrloss=True):
        super().__init__()
        self.temp = nn.Parameter(torch.ones([]) * temp)
        self.queue_size = queue_size
        self.anorm_feats_queue = torch.randn(embed_dim, self.queue_size)
        # self.register_buffer("norm_feats_queue", torch.randn(embed_dim, self.queue_size))
        # self.register_buffer("anorm_feats_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # self.norm_feats_queue = nn.functional.normalize(self.norm_feats_queue, dim=0)
        self.anorm_feats_queue = nn.functional.normalize(self.anorm_feats_queue, dim=0)



    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
                          for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output

    @torch.no_grad()
    def dequeue_and_enqueue(self, anorm_feat):
        # gather keys before updating queue
        # norm_feats = self.concat_all_gather(norm_feat)
        anorm_feats = self.concat_all_gather(anorm_feat)

        batch_size = anorm_feat.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        # self.norm_feats_queue[:, ptr:ptr + batch_size] = norm_feats.t()
        self.anorm_feats_queue[:, ptr:ptr + batch_size] = anorm_feats.t()
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    # def forward(self, feats_3D, feats_2D):
    #     with torch.no_grad():
    #         self.temp.clamp_(0.001, 0.5)
    #     clip_feat_all = torch.cat([feats_3D.t(), self.clip_feats_queue.clone().detach()], dim=1)
    #     frame_feat_all = torch.cat([feats_2D.t(), self.frame_feats_queue.clone().detach()], dim=1)
    #
    #     sim_i2t = feats_3D @ frame_feat_all / self.temp
    #     sim_t2i = feats_2D @ clip_feat_all / self.temp
    #
    #     sim_targets = torch.zeros(sim_t2i.size()).to(feats_3D.device)
    #     sim_targets.fill_diagonal_(1)
    #
    #     loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
    #     loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()
    #
    #     loss_ita = (loss_i2t + loss_t2i) / 2
    #
    #     self._dequeue_and_enqueue(feats_3D, feats_2D)
    #
    #     return loss_ita

    def contrastiveLoss(self, output1, output2, label, margin=200.0):
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
        output1 = torch.norm(output1, p=1, dim=2)
        output2 = torch.norm(output2, p=1, dim=2)

        # 计算 x 和 y 中所有向量之间的欧几里得距离
        euclidean_distance = torch.cdist(output1, output2, p=2).squeeze(-1)

        # euclidean_distance 的形状为 [50, 10]
        # 取出第一个元素，表示向量 x 与向量 y 之间的距离
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0),2))
        return loss_contrastive

    def forward(self, norm_vec, anorm_vec):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        self.dequeue_and_enqueue(norm_vec, anorm_vec)
        loss = 0
        return loss
