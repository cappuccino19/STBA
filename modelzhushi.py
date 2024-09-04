import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Biaffine(nn.Module):
    def __init__(self, args, in1_features, in2_features, out_features, bias=(True, True)):
        super(Biaffine, self).__init__()
        self.args = args
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = torch.nn.Linear(in_features=self.linear_input_size,
                                      out_features=self.linear_output_size,
                                      bias=False)

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = torch.ones(batch_size, len1, 1).to(self.args.device)
            # ones = torch.ones(batch_size, len1, 1)
            input1 = torch.cat((input1, ones), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = torch.ones(batch_size, len2, 1).to(self.args.device)
            # ones = torch.ones(batch_size, len2, 1)
            input2 = torch.cat((input2, ones), dim=2)
            dim2 += 1
        affine = self.linear(input1)
        affine = affine.view(batch_size, len1 * self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)
        biaffine = torch.bmm(affine, input2)
        biaffine = torch.transpose(biaffine, 1, 2)
        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)
        return biaffine


class base_model(nn.Module):
    def __init__(self, args, pretrained_model_path, hidden_dim: 200, dropout, class_n=16,
                 span_average=False):  # span_average控制是否对跨度进行平均
        super().__init__()
        self.args = args
        # Encoder
        self.bert = BertModel.from_pretrained(pretrained_model_path, force_download=True)
        self.dense = nn.Linear(self.bert.pooler.dense.out_features,
                               hidden_dim)  # 这是一个全连接层，用于将BERT模型的输出特征维度调整为hidden_dim:200
        # self.layer_norm = LayerNorm(hidden_dim*3)
        self.span_average = span_average  # 一个bool值，用于控制是否对跨度进行平均

        # Classifier
        # 这是一个全连接层，用于执行分类任务。它的输入维度是hidden_dim * 3，这可能涉及到之前对跨度表示的处理。输出维度是class_n，对应于分类任务的类别数量
        # self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.classifier = nn.Linear(hidden_dim * 3, class_n)

        self.biaffine = Biaffine(args, in1_features=hidden_dim, in2_features=hidden_dim, out_features=class_n)

        # dropout
        self.layer_drop = nn.Dropout(dropout)

    def forward(self, inputs, weight=None):
        #############################################################################################
        # word representation bert_token: torch.Size([16,42])
        bert_token = inputs['bert_token']  # bert模型的输入标记
        # attention_mask: torch.Size([16, 42])
        attention_mask = (bert_token > 0).int()  # 一个注意力掩码，用于指示哪些标记是有效的,.int()将布尔张量转为整数张量，TRUE被映射为1，False映射为0
        # bert_word_mapback: torch.Size([16, 40])
        bert_word_mapback = inputs['bert_word_mapback']  # 一个与标记相关联的映射，可能用于后续的操作
        # token_length:16,即为batch_size大小
        token_length = inputs['token_length']  # 标记序列的长度
        # bert_length：16，batch_size的大小
        bert_length = inputs['bert_length']  # bert输出的长度

        # 通过调用self.bert，将bert_token和attention_mask传递给BERT模型，以获取BERT模型的输出。
        # 输入给BERT模型的标记bert_token序列，通常是一个整数张量。BERT模型将对这些标记进行编码以生成特征表示
        # .last_hidden_state:表示BERT模型的最后一个隐藏层的输出,形状变为[batch_size, sequence_length, hidden_size]，hidden_size为隐藏状态的维度
        # 行代码将输入的标记序列 bert_token 传递给BERT模型，根据输入和注意力掩码计算出每个标记的特征表示，并将这些表示存储在 bert_out 中
        # bert_out:torch.Size([16, 42, 768])
        bert_out = self.bert(bert_token, attention_mask=attention_mask).last_hidden_state  # \hat{h}

        """
        bert_length：这是BERT模型的输出长度，通常与输入序列的长度不同，因为输入序列可能包含填充标记，而BERT模型的输出需要去除这些填充部分
        sequence_mask(bert_length)：调用 sequence_mask 的函数，并传递了 bert_length 作为参数。
        这个函数的目的是生成一个掩码，其中包含了与 bert_length 相同的长度，表示哪些位置是有效的（True）哪些是无效的（False）
        unsqueeze(dim=-1)：这是一个张量操作，它在张量的最后一个维度上增加了一个新的维度。在这里，dim=-1 表示在最后一个维度上添加新维度。
        这样做是为了使 bert_seq_indi 成为一个三维张量，以便后续的计算
        bert_seq_indi 将是一个形状为 [batch_size, max_bert_length, 1] 的张量
        通过调用sequence_mask,最终，这个函数返回一个形状为 [batch_size, max_len] 的布尔掩码张量，其中 True 表示有效的位置，False 表示无效的位置
        """
        # sequence_mask(bert_length):torch.Size([16,40]),bert_seq_indi:torch.Size([16,40,1]
        bert_seq_indi = sequence_mask(bert_length).unsqueeze(dim=-1)
        """
        [:, 1:max(bert_length) + 1, :]：将 bert_out 在第二个维度（序列长度维度）上进行切片。
        切片的范围是从第一个位置（索引1）到有效的最大长度位置（max(bert_length) + 1）。这样做是为了去除填充部分，仅保留有效序列
        * bert_seq_indi.float()：这是对切片后的 bert_out 张量执行逐元素的乘法操作。bert_seq_indi 张量是之前生成的序列掩码，
        其中 True 表示有效位置，False 表示无效位置。通过将无效位置的元素乘以0，有效位置的元素保持不变，实现了将填充部分设置为零的效果。
        """
        # bert_out[:, 1:max(bert_length) + 1, :]:torch.Size([16, 39, 768])
        # bert_seq_indi.float():torch.Size([16, 40, 1])
        bert_out = bert_out[:, 1:max(bert_length) + 1, :] * bert_seq_indi.float()
        # 这一行代码首先使用 F.one_hot 函数将 bert_word_mapback 转换为独热编码，然后与 bert_seq_indi 相乘，
        # 以根据掩码将填充部分设置为零。最后，通过 .transpose(1, 2) 操作，调整了独热编码的维度，以匹配后续的计算
        # bert_word_mapback:torch.Size([16, 59])
        # F.one_hot(bert_word_mapback):torch.Size([16, 59, 54]),因为F.one_hot 会将编码后的值的最大值作为 one-hot 编码中的类数（类别数量）
        # word_mapback_one_hot:torch.Size([16, 54, 59])
        word_mapback_one_hot = (F.one_hot(bert_word_mapback).float() * bert_seq_indi.float()).transpose(1, 2)
        # 这一行代码执行了矩阵相乘操作，将单词映射的独热编码 word_mapback_one_hot 与 bert_out 相乘，
        # 得到了一个加权的表示。这个操作的目的是根据单词映射为每个位置赋予不同的权重
        # word_mapback_one_hot.float():torch.Size([16, 34, 40])
        # bert_out:torch.Size([16, 54, 200])
        bert_out = torch.bmm(word_mapback_one_hot.float(), self.dense(bert_out))
        # attention_scores = self.attention(bert_out, bert_out)
        # 这一行代码计算了每个位置的权重总和，得到一个大小为 [batch_size, max_bert_length] 的张量 wnt
        wnt = word_mapback_one_hot.sum(dim=-1)

        # 这一行代码对 wnt 张量执行了一个操作，将其中为零的元素替换为1。这是为了避免除以零的情况，确保在计算下一步时不会出现除以零的错误
        wnt.masked_fill_(wnt == 0, 1)
        """
        wnt.unsqueeze(dim=-1)：这是一个操作，将 wnt 张量的最后一个维度（最内层的维度）扩展，以变成一个形状为
        [batch_size, sequence_length, 1] 的张量。这是为了与 bert_out 张量的维度匹配，以便进行逐元素的除法操作
        执行了逐元素的除法操作，将 bert_out 张量中的每个位置的值除以相应位置的 wnt 值。
        这个操作的结果是，bert_out 张量中的每个位置的值都被归一化
        """

        #############################################################################################
        # span representation

        # 计算bert_out 张量中序列长度的最大值，通常对应于批次中的最长序列
        max_seq = bert_out.shape[1]

        # 这是一个函数调用，生成一个掩码张量 token_length_mask，该掩码用于指示哪些位置是有效的标记位置。
        # 具体来说，sequence_mask 函数用于创建一个掩码，它的形状与输入的 token_length 张量相同，
        # 并且掩码的每一行都表示对应位置是否有效（1 表示有效，0 表示无效）
        token_length_mask = sequence_mask(token_length)

        """
        创建一个 candidate_tag_mask 张量，用于表示标签（tag）之间的关系,candidate表示文本
        1.torch.triu(torch.ones(max_seq, max_seq, dtype=torch.int64, device=bert_out.device), diagonal=0)：
        这一部分代码首先生成一个大小为 [max_seq, max_seq] 的单位矩阵，使用 torch.triu 函数将单位矩阵的下三角部分（对角线以下的元素）
        置为0。diagonal=0 参数表示保留主对角线及以上的元素，而将主对角线以下的元素设为0
        2..unsqueeze(dim=0)：通过在第0维上添加一个额外的维度，将单位矩阵扩展为 [1, max_seq, max_seq] 的张量
        3.token_length_mask.unsqueeze(dim=1) 和 token_length_mask.unsqueeze(dim=-1)：这两部分代码分别通过在 
        token_length_mask 张量的第1维和最后一维上添加额外的维度，将其扩展为一个三维张量。 
        4.*：最后，执行了逐元素的乘法操作，将上三角矩阵与 token_length_mask 张量的两个扩展版本相乘。
        这个操作的结果是一个与 bert_out 张量具有相同形状的张量 
        5.这样做的目的是根据标记的有效性生成一个文本标签掩码 candidate_tag_mask，其中 1 表示有效的标签位置，0 表示无效的标签位置
        """
        candidate_tag_mask = torch.triu(torch.ones(max_seq, max_seq, dtype=torch.int64, device=bert_out.device),
                                        diagonal=0).unsqueeze(dim=0) * (
                                         token_length_mask.unsqueeze(dim=1) * token_length_mask.unsqueeze(dim=-1))

        """
        用于生成 boundary_table_features，这是一个包含了标签边界特征的张量
        1.bert_out.unsqueeze(dim=2) 和 bert_out.unsqueeze(dim=1)：这两部分代码分别对 bert_out 张量进行扩展，以增加维度。
        bert_out.unsqueeze(dim=2) 在第2维上添加了一个额外的维度，而 bert_out.unsqueeze(dim=1) 在第1维上添加了一个额外的维度。这两个操
        作分别创建了形状为 [batch_size, sequence_length, 1, hidden_dim] 和 [batch_size, 1, sequence_length, hidden_dim] 的张量。
        2..repeat(1, 1, max_seq, 1)：这两部分代码分别对上面创建的扩展张量进行复制操作，以使它们的形状变为 
        [batch_size, sequence_length, max_seq, hidden_dim]。repeat 函数的参数表示在各个维度上的复制次数，
        其中 1 表示不复制，max_seq 表示在第3维上复制 max_seq 次。
        3.torch.cat([...], dim=-1)：这一部分代码使用 torch.cat 函数将两个扩展和复制后的张量拼接在一起，按照最后一个维度（dim=-1）
        进行拼接。由于两个张量在最后一个维度上拼接，所以结果是一个形状为 [batch_size, sequence_length, max_seq, 2 * hidden_dim] 的张量
        这个张量包含了标签边界特征的信息，其中第一个 hidden_dim 对应于一个位置的 bert_out，而第二个 hidden_dim 对应于另一个位置的 bert_out
        4.* candidate_tag_mask.unsqueeze(dim=-1)：最后，对上述张量执行逐元素的乘法操作，将标签边界特征张量中的无效位置清零。
        这是通过与之前生成的 candidate_tag_mask 张量相乘来实现的，其中 1 表示有效的标签对，0 表示无效的标签对。
        """
        boundary_table_features = torch.cat(
            [bert_out.unsqueeze(dim=2).repeat(1, 1, max_seq, 1), bert_out.unsqueeze(dim=1).repeat(1, max_seq, 1, 1)],
            dim=-1) * candidate_tag_mask.unsqueeze(dim=-1)  # h_i ; h_j

        # 调用form_raw_span_features 函数，该函数用于生成跨度（span）的特征表示。具体来说，它将 bert_out 张量、
        # 候选标签掩码 candidate_tag_mask 和 self.span_average 参数传递给该函数。根据参数，函数可能会计算平均特征或其他特征，
        # 然后将结果存储在 span_table_features 中
        span_table_features = form_raw_span_features(bert_out, candidate_tag_mask,
                                                     is_average=self.span_average)  # sum(h_i,h_{i+1},...,h_{j})

        # 将两个张量 boundary_table_features 和 span_table_features在最后一个维度上进行连接，形成一个新的张量 table_features
        # h_i ; h_j ; sum(h_i,h_{i+1},...,h_{j})
        table_features = torch.cat([boundary_table_features, span_table_features], dim=-1)

        #############################################################################################

        # 首先，从输入的 table_features 中获取特征表示，然后将其传递给分类器模型 self.classifier。在传递之前，还应用了一个
        # self.layer_drop 操作，它是一个 Dropout 层，用于随机丢弃一些特征以减少过拟合。最后，将结果与 candidate_tag_mask 相乘，
        # 以在无效的标签对位置（0值）处清零
        # classifier
        # table_features = self.layer_norm(table_features)
        biaffine_output = self.biaffine(bert_out, bert_out)  # 双向注意力输出
        biaffine_output = biaffine_output * candidate_tag_mask.unsqueeze(dim=-1)

        logits = self.classifier(self.layer_drop(table_features)) * candidate_tag_mask.unsqueeze(dim=-1)
        # attention_scores = self.biaffine_attention(bert_out, bert_out)
        outputs = {
            'logits': logits,  # 将计算得到的 logits 存储在一个字典 outputs 中，并使用 'logits' 作为键
            # 'attention_scores': attention_scores
            'biaffine_output': biaffine_output

        }
        # 接下来，检查输入中是否包含 'golden_label'（用于监督训练），以及 'golden_label' 是否不为 None。如果条件满足，继续执行下面的操作：
        if 'golden_label' in inputs and inputs['golden_label'] is not None:
            # 计算损失值，这里调用了 calcualte_loss 函数，用于计算模型的损失。传入的参数包括预测的 logits、
            # 标签信息 inputs['golden_label'] 以及标签对的掩码 candidate_tag_mask。还可以根据需要使用权重 weight
            loss = calcualte_loss(logits, inputs['golden_label'], candidate_tag_mask, weight=weight)
            outputs['loss'] = loss  # 将计算得到的损失值存储在字典 outputs 中，以 'loss' 作为键

        # 函数返回一个字典 outputs，其中包含了模型的输出信息，包括 logits 和损失（如果进行了监督训练）。
        # 在训练过程中，可以使用这些信息来计算梯度并更新模型的参数，以进行优化
        return outputs


# sequence_mask 函数的作用是生成一个掩码（mask），用于标记哪些位置是有效的，哪些是无效的，通常用于处理可变长度的序列数据
# lengths:这是一个包含序列长度的张量，表示每个序列的实际长度。它的形状通常是 [batch_size]，其中 batch_size 是批量大小
# max_len:这是一个可选的参数，表示生成的掩码的最大长度。如果未提供 max_len，则将使用 lengths 中的最大值作为最大长度
def sequence_mask(lengths, max_len=None):
    # batch_size:16
    batch_size = lengths.numel()  # 通过 lengths.numel() 获取批量大小，即 lengths 张量中元素的数量
    # max_len:59
    max_len = max_len or lengths.max()  # 如果未提供 max_len，则通过 lengths.max() 获取 lengths 张量中的最大值，作为掩码的最大长度
    """
    torch.arange(0, max_len, device=lengths.device)：这是创建一个从0到 max_len-1 的整数序列的张量，其设备与 lengths 张量相同。
    .type_as(lengths)：这是将上述整数序列张量的数据类型转换为与 lengths 张量相同的数据类型，以确保数据类型一致性。
    .unsqueeze(0)：这是在张量的第0维（批量大小维）上添加一个新的维度，以使其形状变为 [1, max_len]。
    .expand(batch_size, max_len)：这是通过复制张量来扩展它，使其形状变为 [batch_size, max_len]，以适应整个批次的序列。
    < (lengths.unsqueeze(1))：最后，将生成的 [batch_size, max_len] 的整数序列张量与 lengths 张量中的每个序列长度进行比较。
    如果整数序列中的值小于相应序列的长度，那么对应位置为 True，否则为 False
    最终，这个函数返回一个形状为 [batch_size, max_len] 的布尔掩码张量，其中 True 表示有效的位置，False 表示无效的位置。
    这个掩码通常用于在处理序列数据时，将填充部分排除在计算之外，以确保模型不会受到填充标记的影响
    """
    #
    return torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) < (lengths.unsqueeze(1))


# 该函数用于生成跨度（span）的原始特征表示，具体操作包括对特征值进行清零、矩阵相乘和平均特征的计算
def form_raw_span_features(v, candidate_tag_mask, is_average=True):
    # 首先，将输入的特征表示 v 扩展为一个四维张量，其中第1维（维度索引为1）上添加了一个额外的维度。
    # 然后，将该扩展的 v 与 candidate_tag_mask 张量相乘，以将无效的标签对的特征值清零。结果存储在 new_v 中
    # 这是因为 candidate_tag_mask 张量中的0表示标签对之间没有关系，因此在相应位置的特征值应清零
    new_v = v.unsqueeze(dim=1) * candidate_tag_mask.unsqueeze(dim=-1)

    """
    1.new_v.transpose(1, -1)：对 new_v 张量执行两次转置操作。
    第一次转置将张量的第1维和最后一维交换位置，第二次转置将张量的第2维和最后一维交换位置
    2.torch.matmul(...)：这一部分代码执行矩阵相乘操作。具体来说，它执行以下操作：
    new_v.transpose(1,-1) 是一个形状为 [batch_size, sequence_length, hidden_dim, max_seq] 的张量。
    candidate_tag_mask.unsqueeze(dim=1).float() 将 candidate_tag_mask 张量的形状变为 
    [batch_size, 1, sequence_length, sequence_length] 并将其转换为浮点型张量。torch.matmul(...) 对这两个张量进行矩阵相乘操作。
    在矩阵相乘过程中，它会对第一个张量的最后两个维度（hidden_dim 和 max_seq）
    和第二个张量的最后两个维度（sequence_length）进行相乘操作。这将生成一个形状为
    [batch_size, sequence_length, hidden_dim, sequence_length] 的张量
    3..transpose(2, 1)：接下来，对结果张量执行两次转置操作。第一次转置将张量的第1维和第2维交换位置。
    第二次转置将张量的第3维和第4维交换位置。这两次转置操作的结果是重新排列了张量维度的版本
    4.span_features 张量包含了跨度的原始特征表示，其中包含了输入特征 v 中相关位置的信息
    """
    span_features = torch.matmul(new_v.transpose(1, -1).transpose(2, -1),
                                 candidate_tag_mask.unsqueeze(dim=1).float()).transpose(2, 1).transpose(2, -1)

    if is_average:  # 平均特征计算：如果 is_average 为 True，则执行以下操作
        _, max_seq, _ = v.shape  # 获取输入特征 v 的序列长度 max_seq
        # 生成一个张量 sub_v，用于表示每个跨度的长度。这个张量的形状通常为 [max_seq, max_seq]，其中每个元素表示对应跨度的长度
        sub_v = torch.tensor(range(1, max_seq + 1), device=v.device).unsqueeze(dim=-1) - torch.tensor(range(max_seq),
                                                                                                      device=v.device)
        # 将小于等于0的值替换为1，然后转置 sub_v 张量，以获得形状为 [max_seq, max_seq] 的新张量，其中每个元素表示对应跨度的长度
        sub_v = torch.where(sub_v > 0, sub_v, 1).T
        # 如果 is_average 为 True，则将 span_features 张量除以 sub_v 张量，以进行跨度特征的平均操作。
        # 这确保了对于每个跨度，特征值都被相应的长度除以，以考虑到跨度的大小
        span_features = span_features / sub_v.unsqueeze(dim=0).unsqueeze(dim=-1)

    return span_features


# 用于计算损失值，主要针对的是多分类任务的损失计算
def calcualte_loss(logits, golden_label, candidate_tag_mask, weight=None):
    # 首先，创建一个交叉熵损失函数（nn.CrossEntropyLoss），可以选择设置权重 weight 用于类别不平衡问题，
    # 并设置 reduction='none'，以便返回每个样本的损失值，而不是对它们进行平均或总和
    loss_func = nn.CrossEntropyLoss(weight=weight, reduction='none')
    """
    1.(loss_func(logits.view(-1, logits.shape[-1]), golden_label.view(-1)).view(golden_label.size())：首先将 logits张量和 
    golden_label 张量展平为二维张量，以便计算交叉熵损失。然后，将损失值的形状重新调整为与 golden_label 张量相同的形状,以便对应到每个样本的损失。
    2.* candidate_tag_mask：将上述损失值与 candidate_tag_mask 相乘，这个掩码通常用于控制哪些标签位置参与损失的计算。
    相乘操作将在无效的标签对位置（0值）处将损失清零。
    3..sum()：最后，对所有样本的损失进行求和操作，以获得总的损失值。这是因为 candidate_tag_mask 控制了哪些位置的损失被考虑进来，
     而其他位置的损失被清零了
    """
    return (loss_func(logits.view(-1, logits.shape[-1]),
                      golden_label.view(-1)
                      ).view(golden_label.size()) * candidate_tag_mask).sum()


