def compute_entropy_advantage(c_s_batch, advantages, tau, lambda_scale, power=1.0):
    """
    c_s_batch: [batch_size] 序列置信度（负值）
    advantages: [batch_size, seq_len] Token级优势
    tau: 温度
    lambda_scale: 全局缩放系数
    power: 阶数因子
    返回: [batch_size] 熵优势估计量
    """
    # 1. 计算序列级优势（用于判断符号）
    A_seq = advantages.mean(dim=1)  # [batch_size]

    # 2. 计算基础softmax权重（基于c_s）
    softmax_weights = torch.softmax(c_s_batch / tau, dim=0)  # [batch_size]

    # 3. 根据符号得到原始权重
    pos_mask = A_seq >= 0
    raw_weights = torch.where(pos_mask, 1 - softmax_weights, softmax_weights)  # [batch_size]

    # 应用阶数因子
    if power != 1.0:
        raw_weights = torch.pow(raw_weights, power)

    # 4. 类内归一化：使正负样本各自均值为1
    # 处理空集合情况
    if pos_mask.any():
        pos_mean = raw_weights[pos_mask].mean()
    else:
        pos_mean = 1.0
    if (~pos_mask).any():
        neg_mean = raw_weights[~pos_mask].mean()
    else:
        neg_mean = 1.0

    # 构建缩放因子
    scale = torch.where(pos_mask, 1.0 / pos_mean, 1.0 / neg_mean)
    norm_weights = raw_weights * scale  # 此时正负样本平均均为1

    # 5. 应用全局缩放
    e_s = lambda_scale * norm_weights
    return e_s
