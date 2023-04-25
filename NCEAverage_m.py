import torch
import torch.nn as nn

class NCEAverage(nn.Module):
    def __init__(self, feature_dim, len_neg, len_pos, tau, Z_momentum=0.9, Z=-1):
        super(NCEAverage, self).__init__()
        self.len_neg = len_neg
        self.len_pos = len_pos
        self.embed_dim = feature_dim
        self.Z = Z
        self.tau = tau
        self.Z_momentum = Z_momentum
        # self.register_buffer('params', torch.tensor([Z, tau, Z_momentum, ]))
        print(f'[NCE]: params Z {Z}, Z_momentum {Z_momentum}, tau {tau}')

    # @profile()
    def nce_core(self, pos_logits, neg_logits):
        """
        calculate P(vi|normal_v) = exp(torch.mm(vi, normal_v.t()))/tau / Zi
        :param pos_logits: inner product between normal vectors
        :param neg_logits: inner product between normal and anormal vectors
        :return: matrix with shape ((num_normal_v-1)*num_normal_v, num_negative + 1 )
        """
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        outs = torch.exp(logits / self.tau)
        Z = self.Z
        if Z < 0:
            # initialize Z as mean of first batch
            self.Z = outs.mean() * self.len_neg
            Z = self.Z.clone().detach().item()
            print('normalization constant Z is set to {:.1f}'.format(Z))
        else:
            self.Z = self.Z.to(outs.device)
            Z_new = outs.mean() * self.len_neg
            self.Z = (1 - self.Z_momentum) * Z_new + self.Z_momentum * self.Z  ## 这里涉及原地(in-place)操作
            Z = self.Z.clone().detach().item()
        outs = torch.div(outs, Z).contiguous()
        probs = self.extract_probs(outs)
        return outs, probs

    def extract_probs(self, out):
        probs = out / torch.sum(out, dim=1, keepdim=True)
        return probs[:, 0].mean()

    # def forward(self, n_vec, a_vec, indices_n, indices_a, normed_vec):
    def forward(self, n_vec, a_vec):
        n_scores = torch.mm(n_vec, n_vec.t())
        pos_logits = n_scores[~torch.eye(n_scores.shape[0], dtype=bool)].reshape(n_vec.size(0), -1).view(-1, 1)
        n_a_scores = torch.mm(n_vec, a_vec.t())
        neg_logits = n_a_scores.repeat(1, (n_vec.size(0) - 1)).view(pos_logits.size(0), -1)
        outs, probs = self.nce_core(pos_logits, neg_logits)
        return outs, probs
