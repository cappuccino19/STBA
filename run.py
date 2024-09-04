import os
import time
import torch
import random
import argparse
import numpy as np

from transformers import BertTokenizer
from torch.utils.data import DataLoader

from ASTE_dataloader import ASTE_End2End_Dataset,ASTE_collate_fn,load_vocab
from scheme.span_tagging import form_label_id_map, form_sentiment_id_map
from evaluate import evaluate_model,print_evaluate_dict


def totally_parameters(model):# 计算一个 PyTorch 模型中的总参数数量
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params

def ensure_dir(d, verbose=True):# 这个函数检查目录d是否存在，如果不存在则创建它
    if not os.path.exists(d):
        if verbose:
            print("Directory {} do not exist; creating...".format(d))
        os.makedirs(d)

def form_weight_n(n):# 这个函数生成一个长度为n的权重张量
    if n  > 6:
        weight = torch.ones(n)# 如果n大于6，则创建一个全为1的张量
        index_range = torch.tensor(range(n))# 创建一个与 index_range 张量相同大小的张量，index_range 包含了从 0 到 n-1 的整数序列
        # 最后，它将这个权重张量的值进行修改，如果对应的 index_range 的值与 3 进行按位与操作的结果大于 0（即非零），则在原始值上加 1
        weight = weight + ((index_range & 3) > 0)
    else:# 如果输入的 n 不大于 6，那么它会返回一个预定义的张量 [1.0, 2.0, 2.0, 2.0, 1.0, 1.0]
        weight = torch.tensor([1.0,2.0,2.0,2.0,1.0,1.0])

    return weight

# 训练和评估模型，并可以选择是否保存特定的模型
def train_and_evaluate(model_func, args, save_specific=False):
    print('=========================================================================================================')
    set_random_seed(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    dataset_dir = args.dataset_dir + '/' + args.dataset
    saved_dir = args.saved_dir + '/' + args.dataset
    ensure_dir(saved_dir)

    vocab = load_vocab(dataset_dir = dataset_dir)

    label2id, id2label = form_label_id_map(args.version)
    senti2id, id2senti = form_sentiment_id_map()

    vocab['label_vocab'] = dict(label2id=label2id,id2label=id2label)
    vocab['senti_vocab'] = dict(senti2id=senti2id,id2senti=id2senti)

    class_n = len(label2id)
    args.class_n = class_n
    # 根据是否使用权重 (args.with_weight) 创建权重张量 weight，用于处理类别不平衡问题。如果不使用权重，weight 将为 None
    weight = form_weight_n(class_n).to(args.device) if args.with_weight else None
    # weight = form_weight_n(class_n) if args.with_weight else None
    print('> label2id:', label2id)
    print('> weight:', weight)
    print(args)

    print('> Load model...')
    # 创建基础模型 base_model 并将其放置在指定的计算设备上
    base_model = model_func(pretrained_model_path = args.pretrained_model,
                                hidden_dim = args.hidden_dim,
                                dropout = args.dropout_rate,
                                class_n = class_n,
                                args = args,
                                span_average = args.span_average).to(args.device)
                                # span_average = args.span_average)
    # 计算并打印模型的总参数数量，这个函数用于统计模型中的所有可训练参数的数量
    print('> # parameters', totally_parameters(base_model))

    print('> Load dataset...')
    # 创建一个用于训练的数据集对象，这个数据集将在训练模型时使用
    train_dataset = ASTE_End2End_Dataset(file_name = os.path.join(dataset_dir, 'train_triplets.txt'),
                                         version = args.version,
                                        vocab = vocab,
                                        tokenizer = tokenizer)
    valid_dataset = ASTE_End2End_Dataset(file_name = os.path.join(dataset_dir, 'dev_triplets.txt'),
                                         version = args.version,
                                        vocab = vocab,
                                        tokenizer = tokenizer)
    test_dataset = ASTE_End2End_Dataset(file_name = os.path.join(dataset_dir, 'test_triplets.txt'),
                                        version = args.version,
                                        vocab = vocab,
                                        tokenizer = tokenizer)
    # 创建了一个训练数据的数据加载器 train_dataloader，它用于将训练数据划分成小批次并提供给模型进行训练
    # collate_fn: 一个用于对小批次数据进行处理的函数。在这里，ASTE_collate_fn 函数被用来处理小批次数据，要功能是将小批次中的数据整理成模型可以接受的格式
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, collate_fn = ASTE_collate_fn, shuffle = True)
    valid_dataloader = DataLoader(valid_dataset, batch_size = args.batch_size, collate_fn = ASTE_collate_fn, shuffle = False)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, collate_fn = ASTE_collate_fn, shuffle = False)



    optimizer = get_bert_optimizer(base_model,args)

    triplet_max_f1 = 0.0

    best_model_save_path = saved_dir +  '/' + args.dataset + '_' +  args.version + '_' + str(args.with_weight) +'_best.pkl'

    print('> Training...')
    for epoch in range(1, args.num_epoch+1):# 外部循环，用于迭代训练多个epoch
        train_loss = 0.# 累积每个epoch的训练损失
        total_step = 0# 计算总的训练步数

        epoch_begin = time.time()# 记录当前epoch的开始时间，用于计算每个epoch的训练时间
        for batch in train_dataloader:# 遍历训练数据集中的每个批次
            base_model.train()# 将模型设置为训练模式
            optimizer.zero_grad()# 清零优化器的梯度信息，以准备计算新的梯度
            # 将批次数据转换为适合模型输入的格式。这里的batch包含了训练所需的各种数据
            inputs = {k:v.to(args.device) for k,v in batch.items()}
            # inputs = {k:v for k,v in batch.items()}
            # 将输入数据传递给模型并获取模型的输出。base_model是你的深度学习模型，它接受输入数据并返回预测结果
            outputs = base_model(inputs,weight)

            loss = outputs['loss']# 通过模型的前向传播计算得到损失值
            total_step += 1# 记录已经处理的批次数，用于计算平均损失
            train_loss += loss.item()# 将当前批次的损失值累加到训练损失train_loss中。这样，可以在训练结束后计算平均损失
            loss.backward()# 执行反向传播，计算模型参数相对于损失的梯度。梯度是用于调整模型参数的信息
            optimizer.step()# 根据计算得到的梯度，更新模型的参数。这一步是优化器的核心操作，它使用梯度下降算法来调整参数，以减小损失

        # 执行了模型在验证集上的评估操作，并获取了验证集的损失(valid_loss)和其他评估指标(valid_results)
        # 在评估过程中，模型会使用验证数据集进行前向传播，计算损失以及其他指标
        valid_loss, valid_results = evaluate_model(base_model, valid_dataset, valid_dataloader,
                                                   id2senti = id2senti,
                                                   device = args.device,
                                                   version = args.version,
                                                   weight = weight)
        # train_loss / total_step:平均训练损失。  time.time() - epoch_begin：当前周期的训练耗时，以秒为单位
        print('Epoch:{}/{} \ttrain_loss:{:.4f}\tvalid_loss:{:.4f}\ttriplet_f1:{:.4f}% [{:.4f}s]'.format(epoch, args.num_epoch, train_loss / total_step,
                                                                                                       valid_loss, 100.0 * valid_results[0]['triplet']['f1'],
                                                                                                       time.time()-epoch_begin))
        # save model based on the best f1 scores
        if valid_results[0]['triplet']['f1'] > triplet_max_f1:
            triplet_max_f1 = valid_results[0]['triplet']['f1']

            evaluate_model(base_model, test_dataset, test_dataloader,
                            id2senti = id2senti,
                            device = args.device,
                            version = args.version,
                            weight = weight)
            torch.save(base_model, best_model_save_path)


    saved_best_model = torch.load(best_model_save_path)
    if save_specific:
        torch.save(saved_best_model, best_model_save_path.replace('_best','_' + str(args.seed) +'_best'))

    saved_file = (saved_dir + '/' + args.saved_file) if args.saved_file is not None else None

    print('> Testing...')
    # model performance on the test set
    _, test_results = evaluate_model(saved_best_model, test_dataset, test_dataloader,
                                             id2senti = id2senti,
                                             device = args.device,
                                             version = args.version,
                                             weight = weight,
                                             saved_file= saved_file)


    print('------------------------------')

    print('Dataset:{}, test_f1:{:.2f}% | version:{} lr:{} bert_lr:{} seed:{} dropout:{}'.format(args.dataset,test_results[0]['triplet']['f1'] * 100,
                                                                                                  args.version, args.lr, args.bert_lr,
                                                                                                 args.seed, args.dropout_rate))
    print_evaluate_dict(test_results)
    return test_results



# 创建一个优化器（optimizer）对象，通常用于训练神经网络模型
def get_bert_optimizer(model, args):
    # 这是一个包含字符串的列表，用于指定哪些参数不会被应用权重衰减
    no_decay = ['bias', 'LayerNorm.weight']
    # 这是一个包含字符串的列表，用于指定哪些参数属于不同的部分
    diff_part = ['bert.embeddings', 'bert.encoder']

    """
    for nd in no_decay: 这部分是一个列表迭代，它遍历名为 no_decay 的字符串列表中的每个字符串，将当前迭代的字符串赋给变量 nd。
    nd in n: 这是一个条件表达式，用于检查字符串 nd 是否包含在字符串 n 中。如果包含，它会返回 True，否则返回 False。
    nd in n for nd in no_decay: 这部分是一个生成布尔值的迭代器，它遍历 no_decay 列表中的每个字符串，并检查它们是否包含在字符串 n 中。
    它会生成一系列布尔值，表示每个字符串是否满足条件。
    最后，not any(...) 这一部分将检查生成的布尔值序列是否全为 False。如果所有字符串都不包含在 n 中，not any(...) 将返回 True，
    否则返回 False。这就是为什么它被用于筛选出那些不包含在 no_decay 列表中的参数
    """
    optimizer_grouped_parameters = [
        {
            # 第一个参数组包括模型中不属于 no_decay 和属于 diff_part 的参数，使用 args.bert_lr 作为学习率和 args.l2 作为权重衰减
            "params": [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
            "weight_decay": args.l2,
            "lr": args.bert_lr
        },
        {
            "params": [p for n, p in model.named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
            "weight_decay": 0.0,
            "lr": args.bert_lr
        },
        {
            "params": [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
            "weight_decay": args.l2,
            "lr": args.lr
        },
        {
            "params": [p for n, p in model.named_parameters() if
                    any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
            "weight_decay": 0.0,
            "lr": args.lr
        },
    ]
    # eps=args.adam_epsilon: 这是AdamW优化器的参数之一，表示分母项的平滑值，用于避免除以零的情况。通常，它设置为一个很小的常数，如1e-8，以确保数值稳定性
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)

    return optimizer


def set_random_seed(seed):

    os.environ['PYTHONHASHSEED'] =str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic =True

def get_parameters():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset_dir', type=str,default='./data/ASTE-Data-V2-EMNLP2020')
    parser.add_argument('--saved_dir', type=str, default='saved_models')
    parser.add_argument('--saved_file', type=str, default=None)
    parser.add_argument('--pretrained_model', type=str, default='bert-base-uncased')
    parser.add_argument('--dataset', type=str, default='14lap')

    parser.add_argument('--version', type=str, default='3D', choices=['3D'])

    parser.add_argument('--seed', type=int, default=81)

    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bert_lr', type=float, default=2e-5)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    # loss
    parser.add_argument('--with_weight', default=False, action='store_true')
    parser.add_argument('--span_average', default=False, action='store_true')

    args = parser.parse_args()

    return args

def show_results(saved_results):
    all_str = ''
    # for version in ['1D','2D','3D']:
    for version in ['3D']:
        all_str += 'STAGE'+'-'+version + '\t'
        for dataset in ['14lap','14res','15res','16res']:
            k = '{}-{}-True'.format(dataset, version)
            all_str += '|{:.2f}\t{:.2f}\t{:.2f}|\t'.format(saved_results[k]['precision'],saved_results[k]['recall'], saved_results[k]['f1'])
        all_str += '\n'
    print(all_str)



def run():
    from model import base_model
    args = get_parameters()
    args.with_weight = True # default true here

    train_and_evaluate(base_model, args)


def for_reproduce_best_results():# 段代码的主要目的是为了复现并输出模型在不同数据集和不同超参数配置下的最佳结果
    from model import base_model
    seed_list_dict = {# 创建了一个包含不同数据集和标志（flag）的字典seed_list_dict

        '14lap-3D-True': 64,
        '14res-3D-True': 87,
        '15res-3D-True': 1018,
        '16res-3D-True': 1024,

    }

    saved_results = {}# 创建了一个空字典saved_results，用于保存每个配置下的测试结果
    for k,seed in seed_list_dict.items():# 遍历seed_list_dict中的每个配置，为每个配置设置参数，包括随机种子、数据集、版本和是否使用权重（flag）
        dataset, version, flag = k.split('-')
        flag = eval(flag)
        args = get_parameters()

        args.seed = seed
        args.dataset = dataset
        args.version = version
        args.with_weight = flag

        test_results = train_and_evaluate(base_model, args, save_specific=False)

        saved_results[k] = test_results[0]['triplet']

    print(saved_results)
    print('----------------------------------------------------------------')
    for k, r in saved_results.items():
        dataset, version, flag = k.split('-')
        print('{}\t{}\t{:.2f}%'.format(dataset, version, r['f1'] * 100))


def for_reproduce_average_results():
    from model import base_model
    seed_list_dict = {
        '14lap-3D-True':[64,42,45,92,35],
        '14res-3D-True':[87,174,58,46,95],
        '15res-3D-True':[1018,1125,1172,1122,26],
        '16res-3D-True':[1024,2038,1002,244,155],
    }
    saved_results = {}
    for k,seed_list in seed_list_dict.items():
        dataset, version, flag = k.split('-')
        flag = eval(flag)
        args = get_parameters()


        args.dataset = dataset
        args.version = version
        args.with_weight = flag

        saved_results[k] = []

        for seed in seed_list:
            args.seed = seed
            test_results = train_and_evaluate(base_model, args, save_specific=True)

            saved_results[k].append(test_results[0]['triplet'])

    print(saved_results)
    print('----------------------------------------------------------------')
    for k, r_list in saved_results.items():
        dataset, version, flag = k.split('-')
        for i,r in enumerate(r_list):
            print('{}\t{}\t{}\t{:.2f}%'.format(dataset, version, i, r['f1'] * 100))





if __name__ == '__main__':
    # run()
    for_reproduce_best_results()  # best scores
    # for_reproduce_average_results() # 5 runs average