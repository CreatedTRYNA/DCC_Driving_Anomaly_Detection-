import torch
import torch.nn as nn

import model
from model import gen_transformer_layer
from models import resnet, resnet2D,resnet_cat
from utils import _construct_depth_model, contrastive_loss


class MTV(nn.Module):

    def __init__(self, args):
        super(MTV, self).__init__()
        # self.pre_model_path = './premodels/resnet-18-kinetics.pth'
        self.resnet3D = model.generate_model(args)
        self.resnet2D = model.gen_model()
        self.transformerDecoder = gen_transformer_layer(args.num_tr_layer)
        self.MLP = resnet_cat.ProjectionHead(output_dim=128, model_depth=18)

    def forward(self, x):
        # x -> [B,T,C,H,W]
        unnormed_feat_3D, normed_feat_3D = self.resnet3D(x)  # [B,512]
        unnormed_feat_2D, normed_feat_2D = self.resnet2D(x[:, :, 8, :, :])  # [B,512]
        itc_loss = contrastive_loss(normed_feat_3D, unnormed_feat_2D)
        itc_loss += contrastive_loss(unnormed_feat_3D, normed_feat_2D)
        fused_vec = self.transformerDecoder(unnormed_feat_3D.unsqueeze(1), unnormed_feat_2D.unsqueeze(1)).squeeze(1)
        fused_vec = torch.cat((fused_vec,unnormed_feat_3D),dim=1)
        out = self.MLP(fused_vec)  # [B,128]
        return out, itc_loss


class MoCo_MTV(nn.Module):

    def __init__(self, base_encoder, args, dim=128, q_K=200, k_K=140, m=0.001, T=0.07, ):
        super(MoCo_MTV, self).__init__()
        self.k_K = k_K
        self.q_K = q_K
        self.m = m
        self.T = T
        self.encoder_q = base_encoder(args=args)
        # self.encoder_k = base_encoder(args=args)

        # for param_q, param_k in zip(
        #         self.encoder_q.parameters(), self.encoder_k.parameters()
        # ):
        #     param_k.data.copy_(param_q.data)  # initialize
        #     param_k.requires_grad = False  # not update by gradient

        # create the queue

        # self.register_buffer("normal_queue", torch.zeros(dim, q_K))
        # self.normal_queue = nn.functional.normalize(self.normal_queue, dim=0)
        # self.register_buffer("normal_queue_ptr", torch.zeros(1, dtype=torch.long))
        # self.normal_queue_is_full = False
        #
        # self.register_buffer("anormal_queue", torch.zeros(dim, k_K))
        # self.anormal_queue = nn.functional.normalize(self.anormal_queue, dim=0)
        # self.register_buffer("anormal_queue_ptr", torch.zeros(1, dtype=torch.long))
        # self.anormal_queue_is_full = False
    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, querys, keys):
        # gather querys before updating queue
        querys = concat_all_gather(querys)

        query_batch_size = querys.shape[0]

        query_ptr = int(self.normal_queue_ptr)
        assert self.q_K % query_batch_size == 0  # for simplicity
        if query_ptr == self.q_K:
            self.normal_queue_is_full = True
        # replace the querys at ptr (dequeue and enqueue)
        self.normal_queue[:, query_ptr: query_ptr + query_batch_size] = querys.T
        query_ptr = (query_ptr + query_batch_size) % self.q_K  # move pointer

        self.normal_queue_ptr[0] = query_ptr

        # gather keys before updating queue
        keys = concat_all_gather(keys)

        keys_batch_size = keys.shape[0]

        key_ptr = int(self.anormal_queue_ptr)
        assert self.k_K % keys_batch_size == 0  # for simplicity
        if key_ptr == self.k_K:
            self.anormal_queue_is_full = True
        # replace the keys at ptr (dequeue and enqueue)
        self.anormal_queue[:, key_ptr: key_ptr + keys_batch_size] = keys.T
        key_ptr = (key_ptr + keys_batch_size) % self.k_K  # move pointer

        self.anormal_queue_ptr[0] = key_ptr

    def get_normal_queue(self):
        return self.normal_queue

    def forward(self, vd_q, vd_k):
        """
        Input:
            vd_q: a batch of query videos
            vd_k: a batch of key videos
        Output:
            logits, targets
        """
        vd = torch.cat((vd_q, vd_k), dim=0)
        q_k, itc_loss = self.encoder_q(vd)  # queries: NxC
        q = q_k[:10]
        k = q_k[10:]
        # compute query features
        # q, itc_loss_q = self.encoder_q(vd_q)  # queries: NxC
        # q = nn.functional.normalize(q, dim=1)

        # compute key features

        # self._momentum_update_key_encoder()  # update the key encoder

        # # shuffle for making use of BN
        # vd_k, idx_unshuffle = self._batch_shuffle_ddp(vd_k)

        # k, itc_loss_k = self.encoder_q(vd_k)  # queries: NxC
        # k = nn.functional.normalize(k, dim=1)

        # undo shuffle
        # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # dequeue and enqueue
        # self._dequeue_and_enqueue(q, k)
        # itc_loss = (itc_loss_q + itc_loss_k) / 2
        # return q, self.normal_queue, k, itc_loss
        return q, k, itc_loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
