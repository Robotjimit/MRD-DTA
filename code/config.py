"""
配置文件：包含所有模型参数和训练配置
"""

# =============================
# 模型架构参数
# =============================
class ModelConfig:
    # 维度配置
    dim_1 = 128          # 第一个模态的维度 (SMILES/蛋白质序列)
    dim_2 = 128          # 第二个模态的维度 (2D图结构)
    dim_3 = 128          # 第三个模态的维度 (3D几何)
    dim_4 = 64          # 第四个模态的维度 (预留)
    
    # LSTM配置
    lstm_dim = 64       # LSTM隐藏层维度
    bilstm_layers = 2   # 双向LSTM层数
    
    # 其他网络配置
    hidden_dim = 256    # 隐藏层维度 (dim * 2)
    dropout_rate = 0.2  # dropout比率
    n_heads = 4         # 注意力头数量
    
    # 词汇表大小
    protein_vocab = 26  # 蛋白质词汇表大小
    smile_vocab = 45    # SMILES词汇表大小
    
    # 投影和路由配置
    proj_dim =256      # 投影维度
    router_hidden = 256 # 路由隐藏维度
    router_temperature = 0.2  # 路由温度参数

# =============================
# 训练参数
# =============================
class TrainingConfig:
    # 基础训练参数
    learning_rate = 1e-3
    num_epochs = 100
    batch_size = 128
    seed = 0
    
    # 数据集配置
    dataset_name = 'davis'  # 可选: 'davis', 'kiba', 'pdbbind'
    
    # 交叉验证
    num_folds = 5
    fold_seeds = [18, 283, 839, 12, 74]
    
    # 学习率调度
    scheduler_factor = 0.5
    scheduler_patience = 4
    scheduler_min_lr = 1e-6
    
    # 早停和保存
    save_every_n_epochs = 3
    model_save_dir = './Model/'

# =============================
# 数据分割模式
# =============================
class DataSplitConfig:
    modes = ['default', 'drug_cold', 'target_cold', 'all_cold']
    default_mode = modes[0]

# =============================
# 设备配置
# =============================
class DeviceConfig:
    cuda_device = '0'
    device = f'cuda:{cuda_device}' if cuda_device else 'cpu'

# =============================
# 损失函数权重
# =============================
class LossWeights:
    task_loss_weight = 1.0
    similarity_loss_weight = 0.001
    distillation_loss_weight = 0.2
    entropy_loss_weight = 0.1

# =============================
# 文件路径配置
# =============================
class PathConfig:
    vocab_dir = './Vocab/'
    data_dir = './'  # 数据集根目录
    model_dir = './Model/'
    log_file = 'log.txt'

# =============================
# 便捷函数：获取完整配置
# =============================
def get_model_config():
    """获取模型配置字典"""
    return {
        'dim_1': ModelConfig.dim_1,
        'dim_2': ModelConfig.dim_2,
        'dim_3': ModelConfig.dim_3,
        'dim_4': ModelConfig.dim_4,
        'lstm_dim': ModelConfig.lstm_dim,
        'hidden_dim': ModelConfig.hidden_dim,
        'dropout_rate': ModelConfig.dropout_rate,
        'n_heads': ModelConfig.n_heads,
        'bilstm_layers': ModelConfig.bilstm_layers,
        'protein_vocab': ModelConfig.protein_vocab,
        'smile_vocab': ModelConfig.smile_vocab,
        'proj_dim': ModelConfig.proj_dim,
        'router_hidden': ModelConfig.router_hidden,
        'router_temperature': ModelConfig.router_temperature,
    }

def get_training_config():
    """获取训练配置字典"""
    return {
        'learning_rate': TrainingConfig.learning_rate,
        'num_epochs': TrainingConfig.num_epochs,
        'batch_size': TrainingConfig.batch_size,
        'seed': TrainingConfig.batch_size,
        'dataset_name': TrainingConfig.dataset_name,
        'num_folds': TrainingConfig.num_folds,
        'fold_seeds': TrainingConfig.fold_seeds,
        'scheduler_factor': TrainingConfig.scheduler_factor,
        'scheduler_patience': TrainingConfig.scheduler_patience,
        'scheduler_min_lr': TrainingConfig.scheduler_min_lr,
        'save_every_n_epochs': TrainingConfig.save_every_n_epochs,
        'model_save_dir': TrainingConfig.model_save_dir,
    }

# =============================
# 环境检查
# =============================
def check_config():
    """检查配置的合理性"""
    import torch
    
    # 检查CUDA可用性
    if ModelConfig.dim_1 <= 0 or ModelConfig.dim_2 <= 0 or ModelConfig.dim_3 <= 0:
        raise ValueError("所有维度参数必须大于0")
    
    if TrainingConfig.batch_size <= 0:
        raise ValueError("批次大小必须大于0")
    
    if TrainingConfig.learning_rate <= 0:
        raise ValueError("学习率必须大于0")
    
    print("配置检查通过！")
    print(f"模型维度: {ModelConfig.dim_1}, {ModelConfig.dim_2}, {ModelConfig.dim_3}")
    print(f"训练参数: LR={TrainingConfig.learning_rate}, BS={TrainingConfig.batch_size}")
    print(f"设备: {DeviceConfig.device}")

if __name__ == "__main__":
    check_config()
