import torch
import numpy as np
from collections import Counter
from torch.utils.data import Dataset

from vocab import *
from scheme.span_tagging import form_raw_table,map_raw_table_to_id

"""
file_name：数据文件的路径，可以是包含文本数据的文件名或包含数据的列表。
vocab：词汇表，用于将文本数据转换为词汇表中的索引。
version：数据版本，有不同的数据处理版本。
tokenizer：用于将文本分词为子词（subwords）的分词器。
max_len：最大序列长度，用于截断或填充序列。
lower：是否将文本转换为小写。
is_clean：是否对数据进行清洗
"""
class ASTE_End2End_Dataset(Dataset):
    def __init__(self, file_name, vocab = None, version = '3D', tokenizer = None, max_len = 128, lower=True, is_clean = True):
        super().__init__()

        self.max_len = max_len
        self.lower = lower
        self.version = version
        # 首先，它检查传递给构造函数的 file_name 参数的类型，如果是字符串类型（即数据集文件的路径），则执行以下步骤：
        if type(file_name) is str:
            with open(file_name,'r',encoding='utf-8') as f:
                lines = f.readlines()
                # 使用列表推导式将每一行的文本数据转换为字典形式，将处理后的数据存储在 self.raw_data 属性中
                self.raw_data = [line2dict(l,is_clean = is_clean) for l in lines]
        else:# 如果 file_name 参数的类型不是字符串，而是其他可迭代类型（如列表），则将传入的数据赋值给 self.raw_data，无需进一步处理
            self.raw_data = file_name

        self.tokenizer = tokenizer
        # 它调用类的 preprocess 方法，将原始数据 self.raw_data 进行进一步处理，将文本数据转换为词汇表索引，
        # 并根据传入的 vocab 和 version 参数进行相应的预处理。处理后的数据存储在 self.data 属性中，供数据集类的其他方法使用
        self.data = self.preprocess(self.raw_data, vocab=vocab, version=version)
    
    def __len__(self):# 返回数据集的样本数量
        return len(self.data)

    def __getitem__(self, idx):# 根据给定的索引 idx 返回数据集中指定索引位置的样本数据
        return self.data[idx]
    
    def text2bert_id(self, token):# 将输入的文本标记（token）序列转换为BERT模型输入所需的词汇表索引
        re_token = []# 定义一个空列表，用于存储经过BERT分词后的标记
        # 也是一个空列表，用于存储每个BERT标记对应的原始标记在原文本中的索引。这个列表的长度会与 re_token 相同，因为它记录了每个BERT标记的来源
        word_mapback = []
        word_split_len = []# 也是一个空列表，用于记录原始标记被BERT分词后的长度。这个列表的长度也与 re_token 相同
        for idx, word in enumerate(token):# 遍历token
            temp = self.tokenizer.tokenize(word)# 使用 self.tokenizer.tokenize(word) 将其分成一个或多个BERT标记
            re_token.extend(temp)# 并将这些标记添加到 re_token 中
            # 将该原始标记在原文本中的索引（idx）添加到 word_mapback 中，次数等于分词后的BERT标记数量
            word_mapback.extend([idx] * len(temp))# len(temp) 表示 temp 中的BERT标记数量，也就是将原始标记分词后的长度
            word_split_len.append(len(temp))# 将每个原始标记分词后的长度添加到 word_split_len 中
        re_id = self.tokenizer.convert_tokens_to_ids(re_token)# 将经过BERT分词后的标记序列 re_token 转换为对应的词汇表索引
        return re_id ,word_mapback ,word_split_len# 返回这些索引
    # preprocess 方法用于对原始数据进行预处理，包括文本分词、词汇表索引转换、标签处理等
    def preprocess(self, data, vocab,version):
        
        token_vocab = vocab['token_vocab']# 从vocab字典中获取了名为'token_vocab'的词汇表对象
        label2id = vocab['label_vocab']['label2id']# 从vocab字典中获取了标签到标签ID的映射,这个映射将用于将原始标签数据转换为模型可以处理的标签索引
        processed = []# 初始化一个空列表 processed，用于存储处理后的数据
        max_len = self.max_len
        # [CLS] 标记的词汇表索引
        CLS_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token])
        SEP_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.sep_token])# [SEP] 标记的词汇表索引
        
        for d in data:# 遍历数据集中的每个样本 d
            # 如果样本中存在三元组标签（'triplets' in d），则调用 form_raw_table 函数生成原始表格数据，并将其映射为标签的整数形式，
            # 存储在 golden_label 变量中，否则 golden_label 为 None
            golden_label = map_raw_table_to_id(form_raw_table(d,version=version),label2id) if 'triplets' in d else None
            # tok
            tok = d['token']# 从数据字典 d 中获取原始文本标记列表，存储在变量 tok 中
            if self.lower:
                tok = [t.lower() for t in tok]# 将原始文本数据转换为小写
            # 调用 self.text2bert_id 方法，将原始文本标记列表 tok 转换为BERT标记的索引 text_raw_bert_indices，以及与每个BERT标记对应的 word_mapback，并忽略第三个返回值
            text_raw_bert_indices, word_mapback, _ = self.text2bert_id(tok)
            # 确保 text_raw_bert_indices 和 word_mapback 的长度不超过 max_len。如果它们的长度超过了 max_len，则会截断多余的部分，保留前 max_len 个元素
            text_raw_bert_indices = text_raw_bert_indices[:max_len]
            word_mapback = word_mapback[:max_len]
            # 计算变量 length，它代表了 word_mapback 中的最后一个元素（即最后一个BERT标记）的索引加上1。这是因为索引是从0开始的，所以需要加1来得到实际的长度
            length = word_mapback[-1 ] +1
            # 使用 assert 语句来确保计算出的 length 与原始文本标记 tok 的长度相匹配。如果它们的长度不匹配，会触发断言错误，表示出现了问题
            assert(length == len(tok))
            # 计算变量 bert_length，它代表了 word_mapback 列表的长度，即BERT标记的数量
            bert_length = len(word_mapback)
            # 通过切片操作，将原始文本标记 tok 截取为与 length 相同长度的子列表。这确保了 tok 的长度与计算得到的 length 一致
            tok = tok[:length]
            # 将文本数据中的每个词汇（t代表一个词汇）映射到词汇表索引，如果词汇不在词汇表中，则使用词汇表的未知词汇索引（token_vocab.unk_index）进行替代
            tok = [token_vocab.stoi.get(t, token_vocab.unk_index) for t in tok]
            
            # 创建一个包含多个字段的字典 temp，然后将这个字典添加到名为 processed 的列表中
            temp = {
                'token':tok,
                'token_length': length,
                'bert_token': CLS_id + text_raw_bert_indices + SEP_id,
                'bert_length': bert_length,
                'bert_word_mapback': word_mapback,
                'golden_label': golden_label

            }
            processed.append(temp)
        return processed
# 用于处理一个小批次（batch）数据的自定义数据处理函数，主要功能是将小批次中的数据整理成模型可以接受的格式
def ASTE_collate_fn(batch):# batch:一个小批次数据，通常包含多个样本，每个样本是一个字典，包含了训练数据的不同部分
    batch_size = len(batch)# 获取小批次的大小，即其中包含的样本数量
    
    re_batch = {}# 创建一个空字典 re_batch，用于存储整理后的数据

    """
    1.for i in range(batch_size)：这个循环迭代处理批次中的每个样本。batch_size 表示批次中的样本数量。
    2.batch[i]['token']：这部分代码提取批次中第 i 个样本的 'token' 数据。每个样本中都包含一个 token 序列，这是一个列表。 
    3.[batch[i]['token'] for i in range(batch_size)]：这是一个列表推导式，用于从整个批次中提取所有样本的 token 序列。
    它创建了一个列表，其中每个元素是一个 token 序列。
    4.get_long_tensor([...])：这部分代码调用了 get_long_tensor 函数，将前面创建的 token 序列列表作为参数传递给它。
    get_long_tensor 函数接受一个列表（tokens_list）作为输入，该列表包含多个 token 序列。
    它将这些 token 序列转换为 PyTorch 的 LongTensor。
    如果这些 token 序列的长度不一致，它会自动进行填充，确保所有 LongTensor 具有相同的长度，这有助于批次化处理。
    5.token = get_long_tensor([...])：最后，将得到的 LongTensor 分配给名为 token 的变量，以便稍后在深度学习模型中使用
    这行代码的作用是将批次中的多个句子（token序列）转换为一个LongTensor张量，以便进行模型的批次化处理
    """
    token = get_long_tensor([ batch[i]['token'] for i in range(batch_size)])# 文本数据的词汇表索引
    token_length = torch.tensor([batch[i]['token_length'] for i in range(batch_size)])# 文本数据的长度
    bert_token = get_long_tensor([batch[i]['bert_token'] for i in range(batch_size)])# BERT 格式的文本数据
    bert_length = torch.tensor([batch[i]['bert_length'] for i in range(batch_size)])# BERT 格式文本的长度
    bert_word_mapback = get_long_tensor([batch[i]['bert_word_mapback'] for i in range(batch_size)])# BERT分词后的词汇表索引
    # 创建全零的包含三个维度的数组，用来存储标签信息
    golden_label = np.zeros((batch_size, token_length.max(), token_length.max()),dtype=np.int64)# 标签数据的词汇表索引
    # 将每个样本的标签数据（如果存在的话）复制到名为 golden_label 的 NumPy 数组中
    if batch[0]['golden_label'] is not None:
        for i in range(batch_size):
            # 从当前样本的 batch[i]['golden_label'] 中获取标签数据。
            # 将标签数据复制到 golden_label 数组的相应位置。具体来说，
            # 它将标签数据复制到 golden_label[i, :token_length[i], :token_length[i]]，其中 i 表示当前样本的索引，
            # token_length[i] 表示当前样本的标记序列长度
            golden_label[i, :token_length[i], :token_length[i]] = batch[i]['golden_label']

    golden_label = torch.from_numpy(golden_label)
    
    re_batch = {
        'token' : token,
        'token_length' : token_length,
        'bert_token' : bert_token,
        'bert_length' : bert_length,
        'bert_word_mapback' : bert_word_mapback,
        'golden_label' : golden_label
    }
    
    return re_batch

# 用于将一个列表中的多个列表的 token 序列转换为一个填充后的 LongTensor
# tokens_list: 一个包含多个列表的列表，每个子列表表示一个 token 序列。
# max_len（可选参数）：指定最终 LongTensor 的最大长度。如果不指定，将使用列表中最长的序列长度
def get_long_tensor(tokens_list, max_len=None):
    """ Convert list of list of tokens to a padded LongTensor. """
    batch_size = len(tokens_list)# 获取 tokens_list 中的子列表数量，即批次大小（batch_size）
    # 计算 tokens_list 中子列表的最大长度，如果指定了 max_len，则使用 max_len 作为最大长度
    token_len = max(len(x) for x in tokens_list) if max_len is None else max_len
    # 创建一个大小为 (batch_size, token_len) 的全零 LongTensor（tokens），其中 batch_size 是批次大小，token_len 是最大长度
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):# 遍历 tokens_list 中的每个子列表 s 和其索引 i
        # rch.LongTensor(s)[:token_len]将子列表 s 转换为一个 PyTorch 的 LongTensor,并进行切片操作，以确保 LongTensor 的长度不超过 token_len
        # tokens[i, : min(token_len,len(s))] = ...将 LongTensor 复制到 tokens 的第 i 行，但只复制 token_len 和 s 长度中较小的那个元素数量
        # 所以这行代码的作用是将子列表 s 中的元素复制到 PyTorch 的 LongTensor tokens 的第 i 行，并确保所复制的元素数量不超过 token_len，以进行填充或截断
        tokens[i, : min(token_len,len(s))] = torch.LongTensor(s)[:token_len]
    # 返回填充后的 LongTensor ‘tokens’，该张量的维度为 (batch_size, token_len)
    # 其中包含了多个 token 序列，并且已经按照批次中最长序列的长度进行了填充。
    return tokens


############################################################################
# data preprocess
def clean_data(l):# 函数用于清洗原始数据，将文本数据和标签数据进行分离
    #使用 strip() 方法去除字符串 l 的首尾空白字符。
    # 使用 split('####') 方法将字符串 l 按 '####' 分隔成两部分，即文本数据和标签数据，返回一个包含两个元素的列表
    token, triplets = l.strip().split('####')
    # 使用 eval() 函数将标签数据部分解析为 Python 对象，通常是一个列表，因为标签数据看起来像 Python 列表。
    # 对解析后的标签数据列表进行去重，使用 set() 将列表转换为集合，然后再将集合转换回列表，并将列表中的元素转换为字符串
    temp_t  = list(set([str(t) for t in eval(triplets) ]))
    # 将清洗后的文本数据和去重后的标签数据重新组合成一个字符串，中间用 '####' 分隔，并在末尾添加换行符 '\n'
    return token + '####' + str([eval(t) for t in temp_t]) + '\n'

def line2dict(l, is_clean=False):# 将输入的文本行l转换为字典格式，其中包括文本数据和标签数据
    if is_clean:# 根据参数 is_clean 的值，可以选择是否调用 clean_data 函数来清洗输入的文本行 l
        l = clean_data(l)
    sentence, triplets = l.strip().split('####')# 使用 '####' 字符串分割 l，将其分成句子和三元组两个部分
    start_end_triplets = []# 创建一个空列表 start_end_triplets，用于存储处理后的三元组信息
    for t in eval(triplets):# 遍历 triplets，它首先使用 eval 函数将 triplets 字符串转换为 Python 对象
        """
        t[0][0] 表示三元组的起始位置的第一个元素，t[0][-1] 表示三元组的起始位置的最后一个元素，
        t[1][0] 表示三元组的结束位置的第一个元素，t[1][-1] 表示三元组的结束位置的最后一个元素
        然后使用 tuple 将其转换为元组。将每个处理后的三元组元组添加到 start_end_triplets 列表中
        """
        start_end_triplets.append(tuple([[t[0][0],t[0][-1]],[t[1][0],t[1][-1]],t[2]]))
    """
     对 start_end_triplets 列表进行排序，排序的依据是每个三元组元组的起始位置的第一个元素 x[0][0] 和结束位置的最后一个元素 x[1][-1]。
     这样，三元组会按照起始位置的第一个元素升序排序，如果起始位置的第一个元素相同，则按照结束位置的最后一个元素升序排序
    """
    start_end_triplets.sort(key=lambda x: (x[0][0],x[1][-1])) # sort ?
    # 最后，函数返回一个字典对象，其中包含两个键值对：
    # 'token': 包含文本行的句子部分，通过 sentence.split(' ') 按空格分割成一个单词列表。
    # 'triplets': 包含经过处理和排序的三元组信息的列表，存储在 start_end_triplets 中
    return dict(token=sentence.split(' '), triplets=start_end_triplets)


#############################################################################
# vocab,构建词汇表
def build_vocab(dataset):# 这是一个函数定义，它接受一个参数 dataset，该参数用于指定数据集的路径或目录
    tokens = []#  创建一个空列表 tokens，用于存储数据集中的所有单词。
    files = ['train_triplets.txt','dev_triplets.txt','test_triplets.txt']# 创建一个包含三个文件名的列表，这些文件包含训练、验证和测试数据的三元组信息
    for file_name in files:# 遍历 files 列表中的每个文件名
        file_path = dataset + '/' + file_name#  构建文件的完整路径，将 dataset 和 file_name 连接在一起
        with open(file_path,'r',encoding='utf-8') as f:
            lines = f.readlines()
        
        for l in lines:#  遍历 lines 列表中的每一行
            # 对于每一行，首先使用 strip() 方法去除首尾空白字符，然后使用 split('####') 方法根据字符串 '####' 进行分割，
            # 获取行中的句子部分。接着，使用 split() 方法根据空格分割句子，将结果存储在 cur_token 中
            cur_token = l.strip().split('####')[0].split()#
            tokens.extend(cur_token)# 将 cur_token 中的单词扩展（添加）到 tokens 列表中。这样，tokens 列表将包含数据集中所有句子中的单词
    return tokens# 返回包含数据集中所有单词的 tokens 列表作为词汇表

# 这是一个函数定义，它接受两个参数：dataset_dir 表示数据集的路径或目录，lower 是一个布尔值参数，用于指定是否将单词转换为小写
def load_vocab(dataset_dir,lower=True):# 函数用于加载词汇表，根据数据构建词汇表并返回
    tokens = build_vocab(dataset_dir)# 调用 build_vocab 函数，传递 dataset_dir 参数，从数据集中提取所有单词，并将它们存储在 tokens 列表中
    if lower:
        tokens = [w.lower() for w in tokens]
    token_counter = Counter(tokens)# 使用 Counter 对象统计 tokens 列表中每个单词的出现次数，并将结果存储在 token_counter 中
    # 创建一个名为 token_vocab 的词汇表对象，该词汇表使用 token_counter 中的词频信息，并指定了两个特殊标记 "<pad>" 和 "<unk>"
    token_vocab = Vocab(token_counter, specials=["<pad>", "<unk>"])
    # 创建一个字典 vocab，其中包含一个键值对，键为 'token_vocab'，值为 token_vocab，将词汇表存储在字典中
    vocab = {'token_vocab':token_vocab}
    return vocab# 返回包含词汇表的字典 vocab