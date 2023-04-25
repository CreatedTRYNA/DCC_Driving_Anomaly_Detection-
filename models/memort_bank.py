import torch
import torch.nn.functional as F
from torch import nn


class MemoryBank(nn.Module):
    def __init__(self,
                 temp=0.07,
                 queue_size=32768,
                 embed_dim=512):
        super().__init__()
        self.temp = temp
        self.queue_size = queue_size

        self.register_buffer("clip_feats_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("frame_feats_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.clip_feats_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.frame_feats_queue = nn.functional.normalize(self.text_queue, dim=0)

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
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = self.concat_all_gather(image_feat)
        text_feats = self.concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.clip_feats_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.frame_feats_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, feats_3D, feats_2D):
        clip_feat_all = torch.cat([feats_3D.t(), self.clip_feats_queue.clone().detach()], dim=1)
        frame_feat_all = torch.cat([feats_2D.t(), self.frame_feats_queue.clone().detach()], dim=1)

        sim_i2t = feats_3D @ frame_feat_all / self.temp
        sim_t2i = feats_2D @ clip_feat_all / self.temp

        sim_targets = torch.zeros(sim_t2i.size()).to(feats_3D.device)
        sim_targets.fill_diagonal_(1)

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        self._dequeue_and_enqueue(feats_3D, feats_2D)

        return loss_ita