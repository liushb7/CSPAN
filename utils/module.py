

import torch
from torch import nn
import torch.nn.functional as F

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1 or classname.find("ConvTranspose2d") != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]


class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.t_feat_dim, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.t_feat_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        # (32, 128, 16) → (32, 128, 1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)
        self.out_dim = configs.t_feat_dim

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)

        x_flat = x.reshape(x.shape[0], -1)
        return x_flat

class FrequencyEncoder(nn.Module):

    def __init__(self, configs, downsample_rate=1, target_out_dim=None):
        super(FrequencyEncoder, self).__init__()
        self.input_channels = configs.input_channels
        

        self.period = configs.period // downsample_rate
        self.fft_mode = self.period // 2 + 1
        self.normalize = configs.fft_normalize
        self.out_channels = configs.input_channels
        
        self.scale = (1 / (self.input_channels * self.out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.input_channels, self.out_channels, self.fft_mode, dtype=torch.cfloat))


        # self.flat_dim = self.out_channels * self.fft_mode
        self.flat_dim = self.out_channels * self.fft_mode
        self.use_pool = False

        if configs.sequence_len == 5120 and configs.avg_mode > 0:
            self.use_pool = True
            self.flat_dim = configs.avg_mode // downsample_rate
            self.freq_pool = nn.AdaptiveAvgPool1d(self.flat_dim)
        
        
        if configs.sequence_len == 200 and configs.avg_mode > 0:
            self.use_pool = True
            self.flat_dim = configs.avg_mode // downsample_rate
            self.freq_pool = nn.AdaptiveAvgPool1d(self.flat_dim)
        

        self.target_out_dim = target_out_dim if target_out_dim is not None else self.flat_dim

        self.bottleneck = nn.Sequential(
            nn.Linear(self.flat_dim, self.target_out_dim),
            nn.BatchNorm1d(self.target_out_dim),
            # nn.ReLU(), 
            # nn.Dropout(0.5)
        )
        self.out_dim = self.target_out_dim

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, mode), (in_channel, out_channel, mode) -> (batch, out_channel, mode)
        dim_num = input.dim()
        if dim_num == 3:
            return torch.einsum("bix,iox->box", input, weights)
        elif dim_num == 4:
            # (batch, in_channel, period_num, mode), (in_channel, out_channel, mode) -> (batch, out_channel, period_num, mode)
            return torch.einsum("bixy,ioy->boxy", input, weights)
    def period_data(self, x):

        B = x.size(0)
        N = x.size(1) # Channels
        if x.size(2) % self.period != 0:
            length = ((x.size(-1) // self.period) + 1) * self.period
            padding = torch.zeros([x.shape[0], x.shape[1], (length - (x.size(2)))]).to(x.device)
            out = torch.cat([x, padding], dim=2)
        else:
            length = x.size(2)
            out = x
        # reshape: (B, C, N_periods, Period)
        out = out.reshape(B, N, length // self.period, self.period).contiguous()
        return out

    def get_amplitude(self, x_fft):

        # x_fft: (B, C, N_periods, Mode)
        a = x_fft.abs()
        
        #  (B, C, N_periods, Mode) -> (B, C, Mode)
        if a.dim() == 4:
            a = a.mean(dim=2)
        

        a = a[:, :, :self.fft_mode]
        
        # (B, C, Mode) -> (B, C * Mode)
        a_flat = a.reshape(a.size(0), -1)
        return a_flat
    
    def forward(self, x):
        batchsize = x.size(0)
        x_period = self.period_data(x)
        x_ft = torch.fft.rfft(x_period,norm='ortho', dim=-1)
        
        if self.normalize:
            x_ft = F.normalize(x_ft, dim=-1)
    
        dim_num = x_ft.dim()
        if dim_num == 3:
            out_ft = torch.zeros(batchsize, self.out_channels, self.fft_mode,  device=x.device, dtype=torch.cfloat)
            out_ft[:, :, :] = self.compl_mul1d(x_ft[:, :, :self.fft_mode], self.weights1)
        elif dim_num == 4:
            out_ft = torch.zeros(batchsize, self.out_channels, x_ft.size(2), self.fft_mode,  device=x.device, dtype=torch.cfloat)
            out_ft[:, :, :, :] = self.compl_mul1d(x_ft[:, :, :, :self.fft_mode], self.weights1)
        # for dim_num == 4, out_ft: (batch, out_channel, period_num, mode)
        
        flat_feat = self.get_amplitude(out_ft)

        if self.use_pool:
            flat_feat = self.freq_pool(flat_feat.unsqueeze(1)).squeeze(1)

        feat = self.bottleneck(flat_feat)
        return feat
    

class CrossModalMutualEnhancement(nn.Module):
    """
    1) residual enhancement
    2) constrained alpha / beta
    3) LayerNorm instead of BatchNorm
    4) zero-init last layer for identity start
    5) learnable/small gate gamma
    """

    
    def __init__(self, t_dim, f_dim, hidden_dim=128, init_gamma=1e-4):
        super().__init__()

        self.t_dim = t_dim
        self.f_dim = f_dim

        # h_t -> (alpha_f, beta_f)
        self.t_to_f_fc1 = nn.Linear(t_dim, hidden_dim)
        self.t_to_f_fc2 = nn.Linear(hidden_dim, 2 * f_dim)

        # h_f -> (alpha_t, beta_t)
        self.f_to_t_fc1 = nn.Linear(f_dim, hidden_dim)
        self.f_to_t_fc2 = nn.Linear(hidden_dim, 2 * t_dim)

        # self.norm_t = nn.LayerNorm(t_dim)
        # self.norm_f = nn.LayerNorm(f_dim)
        
        # self.norm_t = nn.BatchNorm1d(t_dim)
        # self.norm_f = nn.BatchNorm1d(f_dim)

        self.gamma_t = nn.Parameter(torch.tensor(init_gamma))
        self.gamma_f = nn.Parameter(torch.tensor(init_gamma))
        
        # self.gamma_t = init_gamma
        # self.gamma_f = init_gamma

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.t_to_f_fc1.weight)
        nn.init.zeros_(self.t_to_f_fc1.bias)

        nn.init.xavier_uniform_(self.f_to_t_fc1.weight)
        nn.init.zeros_(self.f_to_t_fc1.bias)

        nn.init.zeros_(self.t_to_f_fc2.weight)
        nn.init.zeros_(self.t_to_f_fc2.bias)

        nn.init.zeros_(self.f_to_t_fc2.weight)
        nn.init.zeros_(self.f_to_t_fc2.bias)

    def forward(self, h_t, h_f):
        

        # return h_t, h_f

        # -------- h_t -> enhance h_f --------
        z_f = F.relu(self.t_to_f_fc1(h_t))
        params_f = self.t_to_f_fc2(z_f)
        alpha_f_raw, beta_f_raw = torch.chunk(params_f, 2, dim=1)

        alpha_f = torch.tanh(alpha_f_raw)           # in (-1, 1)
        beta_f  = 1e-4 * torch.tanh(beta_f_raw)      # stronger constraint

        h_f_delta = (1.0 + alpha_f) * h_f + beta_f - h_f
        h_f_enh = h_f + self.gamma_f * h_f_delta

        # -------- h_f -> enhance h_t --------
        z_t = F.relu(self.f_to_t_fc1(h_f))
        params_t = self.f_to_t_fc2(z_t)
        alpha_t_raw, beta_t_raw = torch.chunk(params_t, 2, dim=1)

        alpha_t = torch.tanh(alpha_t_raw)
        beta_t  = 1e-4 * torch.tanh(beta_t_raw)

        h_t_delta = (1.0 + alpha_t) * h_t + beta_t - h_t
        h_t_enh = h_t + self.gamma_t * h_t_delta


        return h_t_enh, h_f_enh


    
class TopKAttentionFusion(nn.Module):
    def __init__(self, feat_dim, topk):
        super(TopKAttentionFusion, self).__init__()
        self.topk = topk
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.Tanh(),
            nn.Linear(feat_dim // 2, 1)
        )

    def forward(self, features):
        """
        features: (B, num_features, D)
        """
        B, K, D = features.size()
        actual_topk = min(self.topk, K)
        
        attn_logits = self.attention(features).squeeze(-1) # (B, K)
        attn_weights = F.softmax(attn_logits, dim=-1)      # (B, K)

        if actual_topk < K:
            
            topk_weights, topk_indices = torch.topk(attn_weights, actual_topk, dim=-1)
            
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
            
           
            gathered_features = torch.gather(features, 1, topk_indices.unsqueeze(-1).expand(-1, -1, D))
            
          
            fused_feat = (gathered_features * topk_weights.unsqueeze(-1)).sum(dim=1)
        else:
           
            fused_feat = (features * attn_weights.unsqueeze(-1)).sum(dim=1)

        return fused_feat
        

class ClassifierHead(nn.Module):

    def __init__(self, in_dim, num_classes):
        super(ClassifierHead, self).__init__()
        
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x, get_feat=False):
        predictions = self.fc(x)
        return predictions
        




class TemporalClassifierHead(nn.Module):

    def __init__(self, in_dim, num_classes, bias=True):
        super(TemporalClassifierHead, self).__init__()
        self.head = nn.Linear(in_dim, num_classes, bias=bias)

    def forward(self, x):
        predictions = self.head(x)
        return predictions
    

class FrequencyClassifierHead(nn.Module):

    def __init__(self, in_dim, num_classes, bias=True):
        super(FrequencyClassifierHead, self).__init__()
        self.fc = nn.Linear(in_dim, num_classes, bias=bias)

    def forward(self, x, get_feat=False):
        predictions = self.fc(x)
        if get_feat:
            return predictions, x
        else:
            return predictions

    

class Discriminator(nn.Module):

    def __init__(self, in_dim, disc_hid_dim, layer_num=3):
        """Init discriminator."""
        super(Discriminator, self).__init__()
        if layer_num == 3:
            self.layer = nn.Sequential(
                nn.Linear(in_dim, disc_hid_dim),
                nn.ReLU(),
                nn.Linear(disc_hid_dim, disc_hid_dim),
                nn.ReLU(),
                nn.Linear(disc_hid_dim, 2)
            )
        elif layer_num == 2:
            self.layer = nn.Sequential(
                nn.Linear(in_dim, disc_hid_dim),
                nn.ReLU(),
                nn.Linear(disc_hid_dim, 2)
            )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out




