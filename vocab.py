import pickle

class Vocab(object):# 定义一个名为 Vocab 的类，它用于构建词汇表对象
     # counter：一个包含单词及其词频的计数器对象，用于构建词汇表。
     # specials（默认值为 ["<pad>", "<unk>"]）：一个包含特殊标记的列表，用于在词汇表中添加特殊标记，例如 <pad>（用于填充）和 <unk>（用于表示未知单词）
    def __init__(self, counter, specials=["<pad>", "<unk>"]):
        self.pad_index = 0# 将词汇表中的 <pad> 标记的索引设置为 0
        self.unk_index = 1# 将词汇表中的 <unk> 标记的索引设置为 1
        counter = counter.copy()# 创建 counter 的副本，以确保不修改原始计数器对象
        self.itos = list(specials)# 将特殊标记列表 specials 复制到词汇表的 self.itos 属性中，这是一个包含所有词汇表单词的列表
        for tok in specials:# 遍历特殊标记列表中的每个标记
            del counter[tok]# 删除计数器中的特殊标记，以确保它们不会被包含在词汇表中


        # 1.counter.items()：counter 对象的 items() 方法返回一个包含键值对的列表，其中每个键值对是单词和其对应的词频（出现次数）
        # 2.这是排序的关键函数，它指定了排序的依据。在这里，使用了一个匿名函数 lambda，该函数接受一个参数 tup，表示键值对（元组）
        #  tup[0] 表示元组中的第一个元素，即单词（token）。因此，这个匿名函数指定了按照单词字母顺序（从小到大）进行排序
        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        # 在前一步排序的基础上，按照词频从高到低进行排序，以确保词汇表中词频高的单词排在前面
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, _ in words_and_frequencies:# 遍历排序后的单词和词频对
            self.itos.append(word)# 将每个单词添加到词汇表的 self.itos 列表中，构建完整的词汇表

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}# 创建一个从单词到索引的映射字典 self.stoi，其中键是单词，值是该单词在词汇表中的索引

    def __eq__(self, other):# __eq__ 方法：用于比较两个词汇表对象是否相等
        # 首先比较了两个词汇表对象的 stoi 属性（单词到索引的映射）是否相等，然后再比较了 itos 属性（索引到单词的映射，即单词列表）是否相等。
        # 如果两个属性都相等，那么这两个词汇表对象被认为是相等的，此时 __eq__ 方法返回 True，否则返回 False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):# __len__ 方法：返回词汇表的大小，即词汇表中词汇的数量。
        return len(self.itos)

    # # extend 方法：将另一个词汇表的内容扩展到当前词汇表中
    def extend(self, v):#  输入另一个词汇表v
        words = v.itos# 获取词汇表 v 的词汇列表 words
        for w in words:# 遍历 words 中的每个词汇 w
            if w not in self.stoi:
                # 如果词汇 w 不在当前词汇表中，就将它添加到当前词汇表的词汇列表 self.itos 的末尾，并为它分配一个新的索引，该索引是当前词汇表中词汇数量减 1
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
        return self
    # @staticmethod 装饰器用于定义 load_vocab 方法，使其成为一个静态方法。这意味着你可以通过类名来调用这个方法，而不需要创建类的实例
    @staticmethod
    def load_vocab(vocab_path: str):# load_vocab 静态方法：从文件中加载词汇表对象
        with open(vocab_path, "rb") as f:# 打开文件 vocab_path 以二进制只读模式 ("rb")
            print('Loading vocab from:', vocab_path)
            return pickle.load(f)# 从打开的文件中加载词汇表对象并返回

    def save_vocab(self, vocab_path):# save_vocab 方法：将词汇表对象保存到文件中
        with open(vocab_path, "wb") as f:
            print('Saving vocab to:', vocab_path)
            pickle.dump(self, f)# 保存文件


