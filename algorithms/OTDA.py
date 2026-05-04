

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss import ConditionalEntropyLoss
from algorithms.algorithms_base import Algorithm
from utils.module import *
import ot
import numpy as np

class OTDA(Algorithm):
    """
    Optimal Transport Domain Adaptation (OTDA) for Human Activity Recognition
    with Multi-segment & Multi-scale Feature Fusion
    """

    def __init__(self, configs, device, args):
        super(OTDA, self).__init__(configs)

        self.args = args
        self.device = device
        self.avg_mode = configs.avg_mode
        self.period = configs.period
        self.num_classes = configs.num_classes
        
        self.scales = [1, 2, 4]  


        base_fft_mode = self.period // 2 + 1
        self.base_f_out_dim = configs.input_channels * base_fft_mode

        if configs.sequence_len == 5120 and self.avg_mode > 0:
            self.base_f_out_dim = self.avg_mode
            self.freq_pool = nn.AdaptiveAvgPool1d(self.avg_mode)
        
        if configs.sequence_len == 750 and self.avg_mode > 0:
            self.base_f_out_dim = self.avg_mode
            self.freq_pool = nn.AdaptiveAvgPool1d(self.avg_mode)

        if configs.sequence_len == 200 and self.avg_mode > 0:
            self.base_f_out_dim = self.avg_mode
            self.freq_pool = nn.AdaptiveAvgPool1d(self.avg_mode)

        self.t_feature_extractors = nn.ModuleList([
            CNN(configs) for _ in self.scales
        ])
        
        self.f_feature_extractors = nn.ModuleList([
            FrequencyEncoder(configs, downsample_rate=scale, target_out_dim=self.base_f_out_dim) 
            for scale in self.scales
        ])

        self.topk = self.args.topk
        self.t_fusion = TopKAttentionFusion(feat_dim=self.t_feature_extractors[0].out_dim, topk=self.topk)
        self.f_fusion = TopKAttentionFusion(feat_dim=self.base_f_out_dim, topk=self.topk)

        self.concat_dim = self.t_feature_extractors[0].out_dim + self.base_f_out_dim
        self.classifier = ClassifierHead(
            in_dim=self.concat_dim, 
            num_classes=configs.num_classes
        )

        self.cross_entropy = nn.CrossEntropyLoss()
        self.criterion_cond = ConditionalEntropyLoss().to(device)

        self.t_out_dim = self.t_feature_extractors[0].out_dim
        self.f_out_dim = self.base_f_out_dim

        self.cross_modal_enhance = CrossModalMutualEnhancement(
            t_dim=self.t_out_dim,
            f_dim=self.f_out_dim,
            hidden_dim=256
        )

        self.freq_aux_classifier = FrequencyClassifierHead(
            in_dim=self.base_f_out_dim,
            num_classes=self.num_classes
        )

        
        self.optimizer = torch.optim.Adam([
            {'params': self.t_feature_extractors.parameters()},
            {'params': self.f_feature_extractors.parameters()},
            {'params': self.t_fusion.parameters()},
            {'params': self.f_fusion.parameters()},
            {'params': self.cross_modal_enhance.parameters()},
            {'params': self.classifier.parameters()},
            {'params': self.freq_aux_classifier.parameters()},
            ],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    def extract_features(self, x):
        B, C, L = x.size()
        
        segments = torch.split(x, self.period, dim=-1)
        
        t_feats_list = []
        f_feats_list = []

        for seg in segments:
            
            if seg.size(-1) != self.period:
                continue 

            for i, scale in enumerate(self.scales):
                
                if scale > 1:
                    seg_scaled = F.avg_pool1d(seg, kernel_size=scale, stride=scale)
                else:
                    seg_scaled = seg
                
                t_feat = self.t_feature_extractors[i](seg_scaled)
                t_feats_list.append(t_feat)
             
                f_feat = self.f_feature_extractors[i](seg_scaled)
                f_feats_list.append(f_feat)

 
        t_feats_tensor = torch.stack(t_feats_list, dim=1)
        f_feats_tensor = torch.stack(f_feats_list, dim=1)


        t_fused = self.t_fusion(t_feats_tensor)
        f_fused = self.f_fusion(f_feats_tensor)

        t_enhanced, f_enhanced = self.cross_modal_enhance(t_fused, f_fused)
        return t_enhanced, f_enhanced

    
    def build_cost_matrix(self, feat_src, feat_trg, ys_oh, pred_xt_softmax):
        """
        M = eta1 * feature_distance + eta2 * semantic_cost
        """
        M_embed = torch.cdist(feat_src, feat_trg) ** 2
        M_sce = -torch.mm(ys_oh, torch.transpose(torch.log(pred_xt_softmax + 1e-8), 0, 1))
        M = self.args.eta1 * M_embed + self.args.eta2 * M_sce
        return M

    def solve_transport_plan(self, cost_matrix, a, b):

        if torch.is_tensor(a):
            a = a.detach().cpu().numpy().astype(np.float64)
        else:
            a = np.asarray(a, dtype=np.float64)

        if torch.is_tensor(b):
            b = b.detach().cpu().numpy().astype(np.float64)
        else:
            b = np.asarray(b, dtype=np.float64)

        M_cpu = cost_matrix.detach().cpu().numpy().astype(np.float64)

        if self.args.ot_type == "balanced":
            if self.args.epsilon == 0:
                pi = ot.emd(a, b, M_cpu)
            else:
                pi = ot.sinkhorn(a, b, M_cpu, self.args.epsilon)
        elif self.args.ot_type == "unbalanced":
            pi = ot.unbalanced.sinkhorn_knopp_unbalanced(
                a, b, M_cpu, self.args.epsilon, self.args.tau
            )
        elif self.args.ot_type == "partial":
            if self.args.epsilon == 0:
                pi = ot.partial.partial_wasserstein(a, b, M_cpu, self.args.mass)
            else:
                pi = ot.partial.entropic_partial_wasserstein(
                    a, b, M_cpu, m=self.args.mass, reg=self.args.epsilon
                )
        else:
            raise NotImplementedError

        pi = torch.from_numpy(pi).float().to(self.device)
        return pi

    
    def get_uniform_weights(self, batch_size):
        """
        u_i = 1 / B
        """
        return torch.full(
            (batch_size,),
            1.0 / batch_size,
            dtype=torch.float64,
            device=self.device
        )


    def convex_combine_weights(self, adaptive_weights, mix_lambda):
        """
        a = (1-lambda) * u + lambda * a_tilde
        """
        adaptive_weights = adaptive_weights.float()
        adaptive_weights = adaptive_weights / adaptive_weights.sum().clamp_min(1e-12)
        uniform_weights = self.get_uniform_weights(adaptive_weights.size(0))

        mixed = (1.0 - mix_lambda) * uniform_weights + mix_lambda * adaptive_weights
        mixed = mixed / mixed.sum().clamp_min(1e-12)


        # print("---------------")
        # print("---------------")
        # print("mixed weights (convex combination):")
        # print(mixed)
        # print("---------------")
        # print("---------------")

        return mixed

    def get_source_discriminability_weights(self, src_f_logits):

        probs = F.softmax(src_f_logits.detach(), dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        max_entropy = float(np.log(self.num_classes))

        discr = 1.0 - entropy / max_entropy
        discr = discr.clamp(min=0.0, max=1.0)

        discr_norm = discr + self.args.marginal_smooth
        discr_norm = discr_norm / discr_norm.sum().clamp_min(1e-12)

        # a_src_mixed = self.convex_combine_weights(discr_norm, self.args.freq_weight_lambda)
        a_src_mixed = self.convex_combine_weights(discr_norm, self.args.mix_lambda)
        return a_src_mixed, discr, discr_norm


    def get_source_transferability_weights(self, pi_probe):

        row_mass = pi_probe.detach().sum(dim=1)
        transfer_norm = row_mass + self.args.marginal_smooth
        transfer_norm = transfer_norm / transfer_norm.sum().clamp_min(1e-12)

        a_src_mixed = self.convex_combine_weights(transfer_norm, self.args.mix_lambda)
        return a_src_mixed, row_mass, transfer_norm

    def compute_ot_loss_with_marginals(self, feat_src, feat_trg, ys_oh, pred_xt_softmax, a, b):

        cost = self.build_cost_matrix(feat_src, feat_trg, ys_oh, pred_xt_softmax)
        pi = self.solve_transport_plan(cost, a, b)
        ot_loss = torch.sum(pi * cost)
        return ot_loss, pi, cost

        

    def compute_ot_loss(self, feat_src, feat_trg, ys_oh, pred_xt_softmax):

        M_embed = torch.cdist(feat_src, feat_trg) ** 2
        M_sce = -torch.mm(ys_oh, torch.transpose(torch.log(pred_xt_softmax + 1e-8), 0, 1))

        M = self.args.eta1 * M_embed + self.args.eta2 * M_sce

        a = ot.unif(feat_src.size(0)).astype(np.float64) 
        b = ot.unif(feat_trg.size(0)).astype(np.float64)
        M_cpu = M.detach().cpu().numpy().astype(np.float64)
        
        if self.args.ot_type == "balanced":
            if self.args.epsilon == 0:
                pi = ot.emd(a, b, M_cpu)
            else:
                pi = ot.sinkhorn(a, b, M_cpu, self.args.epsilon)
        elif self.args.ot_type == "unbalanced":
            pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M_cpu, self.args.epsilon, self.args.tau)
        elif self.args.ot_type == "partial":
            if self.args.epsilon == 0:
                pi = ot.partial.partial_wasserstein(a, b, M_cpu, self.args.mass)
            else:
                pi = ot.partial.entropic_partial_wasserstein(a, b, M_cpu, m=self.args.mass, reg=self.args.epsilon)
        else:
            raise NotImplementedError
        
        pi = torch.from_numpy(pi).float().to(self.device)
        ot_loss = torch.sum(pi * M)
        return ot_loss

    def update(self, src_x, src_y, trg_x, apply_step=False):

        src_t_fused, src_f_fused = self.extract_features(src_x)
        trg_t_fused, trg_f_fused = self.extract_features(trg_x)

        src_feat_concat = torch.cat([src_t_fused, src_f_fused], dim=1)
        trg_feat_concat = torch.cat([trg_t_fused, trg_f_fused], dim=1)

        src_pred = self.classifier(src_feat_concat)
        trg_pred = self.classifier(trg_feat_concat)

        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)
        pred_xt_softmax = F.softmax(trg_pred, dim=1)
        entropy_trg = self.criterion_cond(trg_pred)

        src_f_logits = self.freq_aux_classifier(src_f_fused)
        freq_aux_cls_loss = self.cross_entropy(src_f_logits, src_y)

        num_classes = self.classifier.fc.out_features
        ys_oh = F.one_hot(src_y, num_classes=num_classes).float()

        # a_src_freq = self.get_uniform_weights(trg_f_fused.size(0))
        a_src_freq, src_discr_scores, src_discr_dist = self.get_source_discriminability_weights(src_f_logits)
        


        b_trg_freq = self.get_uniform_weights(trg_f_fused.size(0))

        ot_loss_f, pi_f, cost_f = self.compute_ot_loss_with_marginals(
            feat_src=src_f_fused,
            feat_trg=trg_f_fused,
            ys_oh=ys_oh,
            pred_xt_softmax=pred_xt_softmax,
            a=a_src_freq,
            b=b_trg_freq
        )

        a_src_time_uniform = self.get_uniform_weights(src_t_fused.size(0))
        b_trg_time_uniform = self.get_uniform_weights(trg_t_fused.size(0))

        ot_loss_probe, pi_probe, cost_t = self.compute_ot_loss_with_marginals(
            feat_src=src_t_fused,
            feat_trg=trg_t_fused,
            ys_oh=ys_oh,
            pred_xt_softmax=pred_xt_softmax,
            a=a_src_time_uniform,
            b=b_trg_time_uniform
        )

        a_src_time_final, src_transfer_scores, src_transfer_dist = self.get_source_transferability_weights(pi_probe)

        pi_t_final = self.solve_transport_plan(cost_t, a_src_time_final, b_trg_time_uniform)
        ot_loss_t = torch.sum(pi_t_final * cost_t)


        # b_trg_time_final, trg_transfer_scores, trg_transfer_dist = self.get_target_transferability_weights(pi_probe)

        # pi_t_final = self.solve_transport_plan(cost_t, a_src_time_uniform, b_trg_time_final)
        # ot_loss_t = torch.sum(pi_t_final * cost_t)

        
        
        loss = self.args.cls_trade_off * src_cls_loss \
               + self.args.ot_t_trade_off * ot_loss_t \
               + self.args.ot_f_trade_off * ot_loss_f \
               + self.args.entropy_trade_off * entropy_trg \
               + self.args.freq_aux_trade_off * freq_aux_cls_loss                
        


        loss = loss / self.args.k
        loss.backward()

        if apply_step:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return {
            'Src_cls_loss': src_cls_loss.item(),
            'OT_loss_t': ot_loss_t.item(),
            'OT_loss_f': ot_loss_f.item(),
            'Cond_entropy_trg': entropy_trg.item(),
            'Freq_aux_cls_loss': freq_aux_cls_loss.item(),
            'Total_loss': loss.item() * self.args.k
        }

    def predict(self, data):
        
        # self.t_feature_extractor.eval()

        for t_enc in self.t_feature_extractors:
            t_enc.eval()
        for f_enc in self.f_feature_extractors:
            f_enc.eval()
        self.t_fusion.eval()
        self.f_fusion.eval()
        self.cross_modal_enhance.eval()
        self.classifier.eval()
        # self.freq_aux_classifier.eval()
        
        with torch.no_grad():
            feat_t, feat_f = self.extract_features(data)
            feat_concat = torch.cat([feat_t, feat_f], dim=1)
            pred = self.classifier(feat_concat)
            
        return pred

    def save_model(self, path):
        torch.save({
            't_encoders': self.t_feature_extractors.state_dict(),
            'f_encoders': self.f_feature_extractors.state_dict(),
            't_fusion': self.t_fusion.state_dict(),
            'f_fusion': self.f_fusion.state_dict(),
            'cross_modal_enhance': self.cross_modal_enhance.state_dict(),
            'classifier': self.classifier.state_dict(),
            'freq_aux_classifier': self.freq_aux_classifier.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.t_feature_extractors.load_state_dict(checkpoint['t_encoders'])
        self.f_feature_extractors.load_state_dict(checkpoint['f_encoders'])
        self.t_fusion.load_state_dict(checkpoint['t_fusion'])
        self.f_fusion.load_state_dict(checkpoint['f_fusion'])
        self.cross_modal_enhance.load_state_dict(checkpoint['cross_modal_enhance'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.freq_aux_classifier.load_state_dict(checkpoint['freq_aux_classifier'])

    def get_domain_acc(self, pred, label):
        pred = torch.argmax(pred, dim=1)
        res = torch.sum(torch.eq(pred, label)) / label.size(0)
        return res

    def extract_for_tsne(self, data):
        """
        Return temporal and frequency features for t-SNE visualization.
        """
        for t_enc in self.t_feature_extractors:
            t_enc.eval()
        for f_enc in self.f_feature_extractors:
            f_enc.eval()
        self.t_fusion.eval()
        self.f_fusion.eval()
        self.cross_modal_enhance.eval()
        self.classifier.eval()
        self.freq_aux_classifier.eval()

        with torch.no_grad():
            feat_t, feat_f = self.extract_features(data)

        return feat_t, feat_f