import torch
import torch.nn as nn
from gtconv import GTConv
from egnn import EGNN
from layer import *

from torch_geometric.nn.aggr import MultiAggregation
device = torch.device('cuda:0')
import torch.nn.functional as F
from mamba_ssm import Mamba
def eva_imp(y_true, y_pred):
    res = (y_pred - y_true) ** 2
    return res

def uni_distill(logits1, logits2):
    prob1 = torch.softmax(logits1, dim=-1)
    prob2 = torch.softmax(logits2, dim=-1)
    mse = torch.mean((prob1 - prob2) ** 2, dim=-1)
    return torch.mean(mse)


def uni_distill_kl(teacher_logits,student_logits, T=1.0):
    """
    KL-based knowledge distillation loss
    """
    # Teacher: target distribution
    # p_teacher = p_teacher.detach()
    p_teacher = F.softmax(teacher_logits / T, dim=-1)
    # Student: predicted distribution (log form)
    log_p_student = F.log_softmax(student_logits / T, dim=-1)

    # KL divergence: KL(teacher || student)
    loss = F.kl_div(log_p_student, p_teacher, reduction='batchmean') * (T * T)
    return loss

def entropy_balance(probs):
    probs = torch.clamp(probs, min=1e-9)
    N = probs.size(1)
    entropy = N * torch.sum(probs * torch.log(probs), dim=1)
    return torch.mean(entropy)

class DMFF(nn.Module):
    def __init__(self, dim_1: int, dim_2: int, dim_3: int, dim_4: int, lstm_dim: int, 
                hidden_dim: int, dropout_rate: float=0.2,n_heads: int=8, bilstm_layers: int = 2, 
                protein_vocab: int = 26, smile_vocab: int = 45,proj_dim: int=256,
                router_hidden: int=256,router_temperature: float=1.0,dropout: float=0.1):
        """
        初始化 DMFF 模型。

        :param dim_1: 第一个模态的维度
        :param dim_2: 第二个模态的维度
        :param dim_3: 第三个模态的维度
        :param dim_4: 第四个模态的维度
        :param lstm_dim: LSTM 维度
        :param hidden_dim: 隐藏层维度
        :param dropout_rate: dropout 比率
        :param n_heads: 注意力头的数量
        :param bilstm_layers: 双向 LSTM 层数
        :param protein_vocab: 蛋白质词汇表大小
        :param smile_vocab: SMILES 词汇表大小
        :param proj_dim: 投影维度
        :param router_hidden: 路由隐藏维度
        :param router_temperature: 路由温度
        """
        super(DMFF, self).__init__()
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.relu = nn.ReLU()
      
        self.bilstm_layers = bilstm_layers
        self.n_heads = n_heads

        # SMILES 相关
        self.smiles_vocab = smile_vocab
        self.smiles_embed = nn.Embedding(smile_vocab + 1, dim_1, padding_idx=0)
        self.sm_init = nn.Linear(dim_1, lstm_dim)
        self.smiles_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                   bidirectional=True, dropout=dropout_rate)
        self.sm_fc = nn.Linear(lstm_dim, dim_1)
        self.enhance1 = SpatialGroupEnhance_for_1D(groups=20)

        # 蛋白质相关
        self.protein_vocab = protein_vocab
        self.protein_embed = nn.Embedding(protein_vocab + 1, dim_1, padding_idx=0)
        self.pr_init = nn.Linear(dim_1, lstm_dim)
        self.protein_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                    bidirectional=True, dropout=dropout_rate)
        self.pr_fc = nn.Linear(lstm_dim, dim_1)
        self.enhance2 = SpatialGroupEnhance_for_1D(groups=200)
        # self.out_attentions1 = LinkAttention(lstm_dim*2, n_heads)
        # self.out_attentions2 = LinkAttention(lstm_dim*2, n_heads)
        # self.out_attentions3 = LinkAttention(lstm_dim*2, n_heads)
        # 输出层
        # Point-wise Feed Forward Network
        self.pwff_1 = nn.Linear(lstm_dim * 6, lstm_dim * 8)
        self.pwff_2 = nn.Linear(lstm_dim * 8, lstm_dim * 6)
        self.out_fc1 = nn.Linear(lstm_dim * 6, dim_1*8)
        self.out_fc2 = nn.Linear(dim_1*8, dim_1)
        self.layer_norm1 = nn.LayerNorm(lstm_dim*2)
        self.layer_norm2 = nn.LayerNorm(dim_2)
        self.layer_norm3 = nn.LayerNorm(dim_3)

        # 其他网络组件
        self.norm = RMSNorm(proj_dim)
        self.sgemb = nn.Linear(31, dim_2)
        self.tgemb = nn.Linear(21, dim_2)
        self.sgnn = GIN(dim_2,dim_2*2,dim_2)
        self.tgnn = GIN(dim_2,dim_2*2,dim_2)
        # self.tgnn = GINE(dim_2,4, dim_2*2,dim_2)
        # self.MGNN = MultiLayerGCN(num_features=dim_2, hidden_dim=dim_2//2, num_classes=dim_2, num_layers=3)


        self.out_2g_fc1 = nn.Linear(dim_2*2, dim_2*8)
        self.out_2g_fc2 = nn.Linear(dim_2*8, dim_2)


        self.tEGNN = EGNN(21,128,dim_3//2)
        self.sEGNN = EGNN(31,128,dim_3//2)
        self.out_3g_fc1 = nn.Linear(dim_3, dim_3*8)
        self.out_3g_fc2 = nn.Linear(dim_3*8, dim_3)

        self.method = 'sum'
        # 每个模态各自一个投影到共享专家空间（论文把模态当作“专家”）
        self.proj_1 = MLP1(dim_1, proj_dim, proj_dim, dropout)
        self.proj_2 = MLP1(dim_2, proj_dim, proj_dim, dropout)
        self.proj_3 = MLP1(dim_3, proj_dim, proj_dim, dropout)
        # 路由：输入为三模态拼接后的 pooled 向量
        self.router = Router(in_dim=proj_dim*3,temperature=router_temperature)

        self.mlp1 = MLP1(proj_dim, proj_dim *4, proj_dim)
        self.mlp2 = MLP1(proj_dim, proj_dim *4, proj_dim)
        self.mlp3 = MLP1(proj_dim, proj_dim *4, proj_dim)
        self.mlp4 = MLP1(proj_dim, proj_dim *4, proj_dim)
        # 分类头：单模态 + 融合（论文中的“保持单模态可判别能力”）
        self.head_1 = nn.Linear(proj_dim, 1)
        self.head_2 = nn.Linear(proj_dim, 1)
        self.head_3 = nn.Linear(proj_dim, 1)
        self.head_fused = nn.Linear(proj_dim, 1)
        # self.crossattn = CrossAttentionMultiHead(hidden_dim, 2)
        self.w = nn.Parameter(torch.tensor(0.5))
        # self.global_pool = MultiAggregation(['sum'], mode="cat")
        self.smile_mamba = Mamba(d_model=lstm_dim*2, d_state=16, d_conv =4, expand =2)
        self.protein_mamba = Mamba(d_model=lstm_dim*2, d_state=16, d_conv =4, expand =2)
    def forward(self, data, reset=False):
        """
        前向传播方法。

        :param data: 输入数据
        :param reset: 是否重置状态
        :return: 模型输出和标签
        """
        
        s_data, t_data = data
        batch_size = len(t_data)
        smiles = t_data.smiles.to(device)
        protein = t_data.protein.to(device)
        smiles = smiles.view(batch_size,540)
        protein = protein.view(batch_size,1000)
        smiles_lengths = t_data.smiles_lengths
        protein_lengths = t_data.protein_lengths

        # SMILES 处理
        smiles = self.smiles_embed(smiles)
        smiles = self.sm_init(smiles)
        smiles = self.enhance1(smiles)
        smiles,_ = self.smiles_lstm(smiles)
        smiles_mamba_out = self.smile_mamba(smiles)
        smiles = smiles + smiles_mamba_out
        smiles = self.layer_norm1(smiles)

        # 蛋白质处理
        protein = self.protein_embed(protein)
        protein = self.pr_init(protein)
        protein = self.enhance2(protein)
        protein,_ = self.protein_lstm(protein)
        protein_mamba_out = self.protein_mamba(protein)
        protein = protein + protein_mamba_out
        protein = self.layer_norm1(protein)

        out = torch.cat([smiles,protein],dim=1)

        # 生成掩码
        smiles_mask = self.generate_masks(smiles, smiles_lengths, self.n_heads)
        protein_mask = self.generate_masks(protein, protein_lengths, self.n_heads)
        out_mask = torch.cat([smiles_mask,protein_mask],dim=1)
        # 池化
        smiles_out = self.masked_mean_pooling(smiles, smiles_mask)
        protein_out = self.masked_mean_pooling(protein, protein_mask)
        out = self.masked_mean_pooling(out,out_mask)
        out = torch.cat([smiles_out, protein_out,out], dim=-1)
        # Point-wise Feed Forward Network
        pwff = self.dropout(self.relu(self.pwff_1(out)))
        pwff = self.dropout(self.relu(self.pwff_2(pwff)))
        out = pwff + out
        out1g = self.dropout(self.relu(self.out_fc1(out)))
        out1g = self.dropout(self.relu(self.out_fc2(out1g)))
        
        s3g = self.sEGNN(s_data.x,s_data.pos,s_data.edge_index,s_data.batch)
        t3g = self.tEGNN(t_data.x,t_data.pos,t_data.edge_index,t_data.batch)
                         # edge_attr=t_data.edge_attr)
        # t3g = self.tEGNN(t_data.x,t_data.pos,t_data.edge_index,t_data.batch)
        out3g = torch.cat([s3g,t3g],dim=-1)
        out3g = self.layer_norm3(out3g)
        pwff = self.dropout(self.relu(self.out_3g_fc1(out3g)))
        pwff = self.dropout(self.relu(self.out_3g_fc2(pwff)))
        out3g = out3g + pwff


        s_data.x = self.sgemb(s_data.x)
        t_data.x = self.tgemb(t_data.x)

        # s2g = self.MGNN(s_data)
        # t2g = self.MGNN(t_data)
        s2g = self.sgnn(s_data)
        t2g = self.tgnn(t_data)
        out2g = torch.cat([s2g,t2g],dim=-1)
        pwff = self.dropout(self.relu(self.out_2g_fc1(out2g)))
        out2g = self.dropout(self.relu(self.out_2g_fc2(pwff)))
        out2g = self.layer_norm2(out2g)

        out1g = self.proj_1(out1g,training=self.training)
        out2g = self.proj_2(out2g,training=self.training)
        out3g = self.proj_3(out3g,training=self.training)
        out1g, out2g, out3g = self.perturb([out1g, out2g, out3g])
        # out1g = self.crossattn(out1g,out1g,out1g)
        # out2g = self.crossattn(out2g,out2g,out2g)
        # out3g = self.crossattn(out3g,out3g,out3g)
        # out  = out1g+out2g+out3g
        out = torch.cat([out1g,out2g,out3g],dim=-1)
        w = self.router(out)
        # outg = self.w * out2g + (1 - self.w) * out3g
        # z_fused = self.crossattn(outg,out1g,out1g)
        w_1, w_2, w_3 = w[:, 0:1], w[:, 1:2], w[:, 2:3]               # (B,1) each
        z_all = torch.stack([out1g, out2g, out3g], dim=1)                    # (B,3,P)
        w_all = w.unsqueeze(-1)                                     # (B,3,1)
        z_fused = (z_all * w_all)                                  # (B,3,P)
        if self.method == 'sum':    
            z_fused = z_fused.sum(dim=1)                           # (B,P)
            z_fused = self.norm(z_fused)
        elif self.method == 'concat':
            z_fused = torch.cat(z_fused[:,0,:],z_fused[:,1,:],z_fused[:,2,:])
            z_fused = self.norm(z_fused)
        
        # z_fused = (out1g+out2g+out3g)/3
        out1g = out1g + self.mlp1(out1g, training = self.training)
        out2g = out2g + self.mlp2(out2g, training = self.training)
        out3g = out3g + self.mlp3(out3g, training = self.training)
        z_fused = z_fused + self.mlp4(z_fused, training = self.training)

        logit_l = self.head_1(out1g).squeeze()
        logit_v = self.head_2(out2g).squeeze()
        logit_a = self.head_3(out3g).squeeze()
        logit_f = self.head_fused(z_fused).squeeze()
      
        out = {
            "logits_f": logit_f,
            "logits_l": logit_l,
            "logits_v": logit_v,
            "logits_a": logit_a,
            "router_w": w,                  # 方便可视化/统计
            "1g_embedding": out1g,
            "2g_embedding": out2g,
            "3g_embedding": out3g,
            "fuse_embedding": z_fused,
        }
        # print(w)
        return out, t_data.y
    
    def perturb(self, outs):
        """
        outs: [out1g, out2g, out3g]  # list of modality feature tensors
        每个 out 的 shape: [batch, dim]
        """


        # ======== 参数控制 ======== #
        sample_ratio = 0.1 # 20% 样本加噪
        noise_ratio = 0.1   # 噪声强度比例


        # ======== 1. 随机选择一个模态 ======== #
        idx = random.randint(0, len(outs) - 1)
        x = outs[idx]  # [batch, dim]

        B = x.size(0)

        # ======== 2. 挑选 20% 的样本行 ======== #
        num_samples = max(1, int(B * sample_ratio))
        ids = torch.randperm(B)[:num_samples]

        # ======== 3. 基于特征尺度构造噪声 ======== #
        std = torch.std(x, dim=0, keepdim=True) + 1e-6
     
        noise = torch.randn((num_samples, x.size(1)), device=x.device)* noise_ratio

        # ======== 4. 应用扰动（只影响被挑选的样本） ======== #
        x[ids] = x[ids] + noise
        outs[idx] = x

        return outs
    def generate_masks(self, adj, adj_sizes, n_heads):
        """
        生成掩码。

        :param adj: 输入张量
        :param adj_sizes: 大小
        :param n_heads: 头数
        :return: 掩码张量
        """
        out = torch.ones(adj.shape[0], adj.shape[1])
        max_size = adj.shape[1]
        if isinstance(adj_sizes, int):
            out[0, adj_sizes:max_size] = 0
        else:
            for e_id, drug_len in enumerate(adj_sizes):
                out[e_id, drug_len:max_size] = 0
        # out = out.unsqueeze(1).expand(-1, n_heads, -1)
        return out.cuda(device=adj.device)

    def masked_mean_pooling(self, x, mask):
        """
        掩码平均池化。

        :param x: 输入张量
        :param mask: 掩码
        :return: 池化后的张量
        """
        mask = mask.unsqueeze(-1)  # [B, L, 1]
        x = x * mask  # zero out padded positions
        sum_x = x.sum(dim=1)  # [B, D]
        lengths = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
        return sum_x / lengths  # [B, D]

#############################################################################
def compute_loss(output, labels, args):
    """
    Compute total loss for DMFF model.

    Args:
        output (dict): 模型输出字典，包含 logits 和 proj 等
        labels (torch.Tensor): (B,1) 或 (B,) 真实标签
        args: 全局参数，包含 fusion_method, device 等
        criterion: 主任务损失函数 (默认 L1Loss，可换成 CrossEntropyLoss)

    Returns:
        total_loss: tensor
        loss_dict: 各部分 loss 组成，方便日志记录
    """
    criterion = nn.MSELoss()

    # --- 单模态 & 多模态任务损失 ---
    loss_task_l = criterion(output['logits_l'], labels)
    loss_task_v = criterion(output['logits_v'], labels)
    loss_task_a = criterion(output['logits_a'], labels)
    loss_task_m = criterion(output['logits_f'], labels)

#     # --- 相似度约束 (eva_imp + dist) ---
    l_dist = eva_imp(output['logits_l'], labels)
    a_dist = eva_imp(output['logits_a'], labels)
    v_dist = eva_imp(output['logits_v'], labels)

    dist = torch.zeros(l_dist.shape[0], 3).to(args.device)

    for i, _ in enumerate(l_dist):
        s = 1/(l_dist[i]+0.1) + 1/(v_dist[i]+0.1) + 1/(a_dist[i]+0.1)
        dist[i][0] = (1/(l_dist[i]+0.1)) / s
        dist[i][1] = (1/(v_dist[i]+0.1)) / s
        dist[i][2] = (1/(a_dist[i]+0.1)) / s

    w = output['router_w']
    # log(f'w:{w} '
    loss_sim = torch.mean(torch.mean((dist.detach() - w) ** 2, dim=-1))
    # log(f"w:{w} dist:{dist}")
    loss_ety = entropy_balance(w)

    # --- 单模态蒸馏损失 ---
    if args.fusion_method == "sum":
        target_proj = (output['1g_embedding'] * w[:,0].view(-1, 1) +
                       output['2g_embedding'] * w[:,1].view(-1, 1) +
                       output['3g_embedding'] * w[:,2].view(-1, 1)).detach()
        loss_ud_1 = uni_distill(output['fuse_embedding'].detach(), output['1g_embedding'])
        loss_ud_2 = uni_distill(output['fuse_embedding'].detach(), output['2g_embedding'])
        loss_ud_3 = uni_distill(output['fuse_embedding'].detach(), output['3g_embedding'])
        loss_ud = loss_ud_1 +loss_ud_2 + loss_ud_3
        # loss_ud = uni_distill(output['fuse_embedding'], target_proj)
    elif args.fusion_method == "concat":
        target_proj = torch.cat([
            output['1g_embedding'] * w[:,0].view(-1, 1),
            output['2g_embedding'] * w[:,1].view(-1, 1),
            output['3g_embedding'] * w[:,2].view(-1, 1)
        ], dim=1).detach()
        loss_ud = uni_distill(output['fuse_embedding'], target_proj)
    else:
        loss_ud = torch.tensor(0.0).to(args.device)

    # --- 总损失 ---
    total_loss = (loss_task_m 
                  + (loss_task_l + loss_task_v + loss_task_a)/3
                  + 0.1*loss_sim 
                  + 0.1*loss_ud)
    # total_loss = loss_task_m

    loss_dict = {
        "loss_task_m": loss_task_m.item(),
        "loss_task_l": loss_task_l.item(),
        "loss_task_v": loss_task_v.item(),
        "loss_task_a": loss_task_a.item(),
        "loss_sim": loss_sim.item(),
        "loss_ety": loss_ety.item(),
        "loss_ud": loss_ud.item(),
        "total": total_loss.item(),
        # "w":w.detach().cpu().numpy(),
        # "dist": dist.detach().cpu().numpy()
    }
    
    return loss_dict,total_loss
#############################################################################
