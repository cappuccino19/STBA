import torch
import json
from collections import Counter
from scheme.greedy_inference import loop_version_from_tag_table_to_triplets

# 用于评估模型性能的，特别是在测试集上评估模型的性能
def evaluate_model(model, test_dataset, test_dataloader, id2senti, device='cuda', version = '3D', weight = None,saved_file=None):
    model.eval()# 将模型设置为评估模式
    total_loss = 0.0
    total_step = 0

    # 遍历测试数据集中的每个样本。len(test_dataset.raw_data) 表示测试数据集中样本的总数
    # test_dataset.raw_data[idx]['token'] 获取了第 idx 个样本的原始文本数据（tokens），这是一个字符串列表，表示文本被分成的单词或子词
    saved_token = [test_dataset.raw_data[idx]['token'] for idx in range(len(test_dataset.raw_data))]
    # 从测试数据集中提取的真实标签（gold labels）或目标值
    saved_golds = [test_dataset.raw_data[idx]['triplets'] for idx in range(len(test_dataset.raw_data))]
    
    saved_preds = []
    saved_aspects = []
    saved_opinions = []

    # 执行了模型在测试数据集上的评估操作
    with torch.no_grad():# 上下文管理器，确保在评估期间不会计算梯度，以减少内存使用和加速计算
        for batch in test_dataloader:# 通过循环迭代 test_dataloader，该数据加载器包含了测试数据集的批次数据
            inputs = {k:v.to(device) for k,v in batch.items()}
        
            outputs = model(inputs, weight)

            loss = outputs['loss']
            total_step += 1
            total_loss += loss.item()

            # 对模型的预测结果进行计算，其中 outputs['logits'] 包含了模型对每个类别的分数，torch.argmax 用于找到分数最高的类别索引，即模型的预测类别
            batch_raw_table_id = torch.argmax(outputs['logits'],dim=-1)
            for idx in range(len(batch_raw_table_id)):#  循环遍历每个样本的索引
                # 从 batch_raw_table_id 中获取当前样本的模型预测结果，将模型预测结果 batch_raw_table_id[idx] 转换为标签三元组形式的预测结果 pred_triplets
                # 这是通过调用名为 loop_version_from_tag_table_to_triplets 的函数完成的。该函数的输入包括模型预测的标签表（tag_table）、
                # 将标签表映射回情感标签的字典（id2senti）以及版本信息（version）。该函数的目的是将模型的输出标签表转换为标签三元组（aspect, opinion, sentiment）的形式
                pred_triplets = loop_version_from_tag_table_to_triplets(tag_table = batch_raw_table_id[idx].tolist(), 
                                                            id2senti = id2senti, 
                                                            version=version)
                
                saved_preds.append(pred_triplets['triplets'])
                saved_aspects.append(pred_triplets['aspects'])
                saved_opinions.append(pred_triplets['opinions'])
        
    
    # 这段代码用于将模型的预测结果、原始文本数据和真实标签数据保存到一个文件中
    if saved_file is not None:
        with open(saved_file,'w',encoding='utf-8') as f:
            # 创建一个名为 combined 的列表，其中每个元素都是一个字典，使用 zip 函数将 saved_token、saved_preds、saved_golds、saved_aspects
            # 和 saved_opinions 合并成一个可迭代的对象，并将其用于填充 combined 列表
            combined = [
                dict(token=token, pred=pred, gold=gold, pred_aspect = pred_aspect, pred_opinion=pred_opinion) for token,pred,gold,pred_aspect,pred_opinion in zip(saved_token,saved_preds, saved_golds, saved_aspects, saved_opinions)
            ]
            json.dump(combined, f)# 使用 json.dump 将整个 combined 列表写入到打开的文件中，以将数据保存为 JSON 格式

    loss = total_loss / total_step# 计算平均损失
    # 调用 evaluate_predictions 函数，该函数用于评估模型的预测结果
    # evaluate_predictions 函数将计算各种评估指标，例如准确率、召回率、F1 分数等，以评估模型性能，并将结果保存在 evaluate_dict 中
    evaluate_dict = evaluate_predictions(preds = saved_preds, goldens = saved_golds, preds_aspect = saved_aspects, preds_opinion = saved_opinions)
    model.train()# 将模型设置为训练模式
    return loss, evaluate_dict# 返回计算的损失 loss 和评估结果字典 evaluate_dict


def evaluate_predictions(preds = None, goldens = None, preds_aspect = None, preds_opinion = None):
    counts = Counter()# 统计总体评估的结果
    
    one_counts = Counter()# 统计单一三元组评估的结果
    multi_counts = Counter()#  统计多个三元组评估的结果
    aspect_counts = Counter()# 统计方面评估的结果
    opinion_counts = Counter()# 统计意见评估的结果

    
    ate_counts = Counter()
    ote_counts = Counter()
    
    for pred, gold, pred_aspect,pred_opinion in zip(preds,goldens,preds_aspect,preds_opinion):# 循环遍历每个样本的预测和标签数据
        # 首先调用evaluate_sample函数来计算总体的评估指标，并将结果添加到counts中
        counts = evaluate_sample(pred, gold, counts)
        # 接着，调用get_spereate_triplets函数，将预测结果和标签数据分成不同的三元组类型（one_triplet、new_multi_triplet、a_multi_triplet、o_multi_triplet）
        pred_one,pred_new_multi, pred_a_multi, pred_o_multi = get_spereate_triplets(pred)
        one,new_multi, a_multi, o_multi = get_spereate_triplets(gold)
        # 对每种类型的三元组，分别调用evaluate_sample函数来计算对应的评估指标，并将结果分别添加到one_counts、multi_counts、aspect_counts、opinion_counts中
        one_counts = evaluate_sample(pred_one, one, one_counts)
        multi_counts = evaluate_sample(pred_new_multi, new_multi, multi_counts)
        aspect_counts = evaluate_sample(pred_a_multi, a_multi, aspect_counts)
        opinion_counts = evaluate_sample(pred_o_multi, o_multi, opinion_counts)

        # tuple(x[0]) for x in gold：这一部分首先遍历金标准数据gold中的每个三元组，然后对每个三元组中的方面部分（x[0]表示三元组的第一个元素，即方面部分）创建一个元组。
        # list(set(...))：这一部分将上一步得到的元组列表转换为集合（set），这样可以去除重复的元组，确保每个方面只出现一次
        # [m[0], m[1]] for m in ...：最后，对于每个唯一的方面元组（m表示元组），代码将其转换为包含两个元素的列表，即[方面, 意见]。这些列表被收集到gold_ate列表中
        gold_ate = [[m[0],m[1]] for m in list(set([tuple(x[0]) for x in gold]))]
        gold_ote = [[m[0],m[1]] for m in list(set([tuple(x[1]) for x in gold]))]

        # 这部分首先检查pred_aspect列表的长度是否大于0,接下来，代码检查pred_aspect列表的第一个元素（pred_aspect[0]）的数据类型是否为整数（int）
        if len(pred_aspect) > 0 and type(pred_aspect[0]) is int:
            pred_aspect = [pred_aspect]# 如果pred_aspect的第一个元素不是整数类型，那么它会被转换为一个包含单个元素的列表
            
        if len(pred_opinion) > 0 and  type(pred_opinion[0]) is int:
            pred_opinion = [pred_opinion]
        # 分别调用evaluate_term函数计算了方面（aspect）和意见（opinion）的评估指标，将结果分别存储在ate_counts和ote_counts中
        ate_counts = evaluate_term(pred=pred_aspect, gold=gold_ate, counts = ate_counts)
        ote_counts = evaluate_term(pred=pred_opinion, gold = gold_ote, counts = ote_counts)
    # 代码调用各个评估指标的计算函数，包括output_score_dict和output_score_dict_term，分别计算了总体和子类型的评估指标，并将这些指标以字典的形式返回
    all_scores = output_score_dict(counts)
    one_scores = output_score_dict(one_counts)
    multi_scores = output_score_dict(multi_counts)
    aspect_scores = output_score_dict(aspect_counts)
    opinion_scores = output_score_dict(opinion_counts)
    term_scores = output_score_dict_term(ate_counts, ote_counts)
    
    return all_scores, one_scores, multi_scores, aspect_scores, opinion_scores, term_scores

###############################################################################################
# ASTE (AOPE)
def evaluate_sample(pred, gold, counts = None):
    if counts is None:
        counts = Counter()
    
    correct_aspect = set()
    correct_opinion = set()


    # tuple(x[0]) for x in gold：对于每个样本，我们从标签数据中提取第一个元素 x[0]，这通常包含方面的信息。然后，我们将这个元素转换为元组。
    # list(set(...))：在提取方面信息后，我们使用 set 数据结构来去除重复的方面信息，因为同一个方面可能在多个样本中出现。然后，我们将得到的独特方面信息转换为列表
    # ASPECT.
    aspect_golden = list(set([tuple(x[0]) for x in gold]))
    aspect_predict = list(set([tuple(x[0]) for x in pred]))


    counts['aspect_golden'] += len(aspect_golden)# 将标签中的方面数量添加到计数器 counts 的 'aspect_golden' 键对应的值中
    counts['aspect_predict'] += len(aspect_predict)


    for prediction in aspect_predict:# 遍历模型预测的方面信息列表 aspect_predict 中的每个预测
        # 这一行代码使用列表推导式来检查当前预测 prediction 是否与实际标签中的任何一个方面信息匹配
        # 它遍历了 aspect_golden 列表中的所有实际方面信息，检查是否有任何一个与当前预测匹配
        if any([prediction == actual for actual in aspect_golden]):
            counts['aspect_matched'] += 1# 递增名为 aspect_matched 的计数器，用于记录匹配的方面数量
            correct_aspect.add(prediction)# 将当前预测 prediction 添加到名为 correct_aspect 的集合中，以便后续分析和报告

    # OPINION.
    opinion_golden = list(set([tuple(x[1]) for x in gold]))
    opinion_predict = list(set([tuple(x[1]) for x in pred]))
    
    counts['opinion_golden'] += len(opinion_golden)
    counts['opinion_predict'] += len(opinion_predict)
    
    
    for prediction in opinion_predict:
        if any([prediction == actual for actual in opinion_golden]):
            counts['opinion_matched'] += 1
            correct_opinion.add(prediction)

    triplets_golden = [(tuple(x[0]),tuple(x[1]), x[2]) for x in gold]
    triplets_predict = [(tuple(x[0]),tuple(x[1]), x[2]) for x in pred]
    
    counts['triplet_golden'] += len(triplets_golden)
    counts['triplet_predict'] += len(triplets_predict)
    for prediction in triplets_predict:# 遍历模型预测的三元组列表 triplets_predict 中的每个预测
        # 这一行代码使用列表推导式来检查当前预测 prediction 的前两个元素是否与实际标签中的任何一个三元组的前两个元素匹配。
        # 它遍历了 triplets_golden 列表中的所有实际三元组，检查是否有任何一个与当前预测的前两个元素匹配
        if any([prediction[:2] == actual[:2] for actual in triplets_golden]):
            counts['pair_matched'] += 1# 递增名为 pair_matched 的计数器，用于记录匹配的三元组对数量
        # 检查当前预测 prediction 是否与实际标签中的任何一个完整三元组匹配
        if any([prediction == actual for actual in triplets_golden]):
            counts['triplet_matched'] += 1# 递增名为 triplet_matched 的计数器，用于记录匹配的三元组数量
                

    # Return the updated counts.
    return counts

def output_score_dict(counts):
    # counts['aspect_predict']：表示模型在"aspect"类别上的预测数量，即模型预测的"aspect"标签的数量
    # counts['aspect_golden']：表示"aspect"类别的黄金标签的数量，即实际的"aspect"标签的数量。
    # counts['aspect_matched']：表示在"aspect"类别中正确匹配的数量，即模型的预测与实际的"aspect"标签匹配的数量
    # compute_f1函数根据这些参数计算出"aspect"类别的精确度（precision）、召回率（recall）和F1分数，并将它们存储在一个字典中返回
    # 这些分数可以用来评估模型在"aspect"类别上的性能
    scores_aspect = compute_f1(counts['aspect_predict'], counts['aspect_golden'], counts['aspect_matched'])
    scores_opinion = compute_f1(counts['opinion_predict'], counts['opinion_golden'], counts['opinion_matched'])
    
    scores_pair = compute_f1(counts['triplet_predict'], counts['triplet_golden'], counts['pair_matched'])
    scores_triplet = compute_f1(counts['triplet_predict'], counts['triplet_golden'], counts['triplet_matched'])
    
    return dict(aspect=scores_aspect, opinion=scores_opinion, pair=scores_pair, triplet=scores_triplet)

###############################################################################################

# 评估术语（terms）的匹配情况的。pred（表示模型预测的术语列表）和gold（表示真实的术语列表），以及一个可选的计数器counts
# ATE & OTE
def evaluate_term(pred, gold, counts=None):
    if counts is None:
        counts = Counter()
    # 更新计数器中的统计信息,包括真实标签和预测标签
    counts['golden'] += len(gold)
    counts['predict'] += len(pred)
    
    for prediction in pred:# 遍历模型预测的术语列表
        if any([prediction == actual for actual in gold]):# 检查是否存在与真实术语列表gold中的任何一个匹配
            counts['matched'] += 1# 如果找到匹配项，就将匹配计数器（'matched'）加1
    return counts# 返回更新后的计数器counts


def output_score_dict_term(aspect_counts, opinion_counts):# 输出术语匹配的评分
    # 计算ate和ote的F1分数
    score_ate = compute_f1(aspect_counts['predict'], aspect_counts['golden'], aspect_counts['matched'])
    score_ote = compute_f1(opinion_counts['predict'], opinion_counts['golden'], opinion_counts['matched'])
    return dict(ate=score_ate, ote=score_ote)

###############################################################################################
# for additional experiments
def get_spereate_triplets(triplet):# 将传入的三元组（triplet）根据一些规则分成四类：one_triplet、new_triplet、a_triplet和o_triplet
    one_triplet = []# 包含那些对于两个方面（aspect）和一个观点（opinion）都只有一个词的三元组
    new_triplet = []# 包含那些至少有一个方面或一个观点包含多个词的三元组
    a_triplet = []# 包含那些方面（aspect）中至少有一个包含多个词的三元组
    o_triplet = []# 包含那些观点（opinion）中至少有一个包含多个词的三元组
    for t in triplet:# 循环遍历传入的三元组列表triplet
        # 如果三元组的第一个方面（t[0]）或第一个观点（t[1]）的最后一个词（[-1]索引）与第一个词（[0]索引）不相同，
        # 那么它被分配到new_triplet类别中，表示该三元组包含至少一个方面或观点中有多个词
        if t[0][-1] != t[0][0] or t[1][-1] != t[1][0]:
            new_triplet.append(t)
        else:# 否则，如果第一个方面的最后一个词与第一个词相同，那么它被分配到one_triplet类别中，表示该三元组的方面都只有一个词
            one_triplet.append(t)
        if t[0][-1] != t[0][0]:# 如果第一个方面中的最后一个词与第一个词不同，那么它被分配到a_triplet类别中，表示该三元组的方面至少有一个包含多个词
            a_triplet.append(t)
        if t[1][-1] != t[1][0]:# 如果第一个观点中的最后一个词与第一个词不同，那么它被分配到o_triplet类别中，表示该三元组的观点至少有一个包含多个词
            o_triplet.append(t)
    return one_triplet, new_triplet, a_triplet, o_triplet

# 计算F1分数以及其相关的精确度（precision）和召回率（recall）
# predict：模型的预测数量，表示模型在某个类别上的预测数目。
# golden：实际标签的数量，表示真实情况下该类别的样本数。
# matched：匹配的数量，表示模型的预测中与实际标签相匹配的样本数。
def compute_f1(predict, golden, matched):
    # F1 score.
    precision = matched / predict if predict > 0 else 0# predict大于零，那么精确度等于匹配的数量（matched）除以模型的预测数量（predict）
    recall = matched / golden if golden > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall > 0) else 0
    return dict(precision=precision, recall=recall, f1=f1)


##################################################################################################
# d：包含评分信息的字典，通常是由其他函数生成的，包括精确度（precision）、召回率（recall）和F1分数（F1-score）。
# select_k：一个包含要打印的评分键的列表，如果未提供，则默认打印所有可用的键
# print
def print_dict(d, select_k = None):# 打印一个字典中的评分信息，主要用于打印方面（aspect）、意见（opinion）和三元组（triplet）的评分信息
    if select_k is None:# 首先检查select_k是否为None，如果是，则将其设置为包含所有评分键的列表，这意味着默认情况下将打印所有可用的评分信息
        select_k = list(d.keys())
    
    print_str = '\t  \tP\t\tR\t\tF\n'
    for k in select_k: 
        append_plus = '*' if k in ['aspect','opinion','triplet'] else ''# 创建一个表头
        print_str += '{:^8}\t{:.2f}%\t{:.2f}%\t{:.2f}%\n'.format(append_plus + k.upper(),
                                                                 100.0 * d[k]['precision'], 
                                                                 100.0 * d[k]['recall'], 
                                                                 100.0 *  d[k]['f1'])
    print(print_str)
    
    
def print_evaluate_dict(evaluate_dict):# 打印评估结果字典中的各种评分信息
    type_s = ['all','one','multi','multi_aspect','multi_opinion', 'term']

    for idx,m in enumerate(evaluate_dict):
        print('\n[ ' + type_s[idx], ']')
        if type_s[idx] in ['one','multi','multi_aspect','multi_opinion']:
            select_k = ['triplet']
        elif type_s[idx] in ['all']:
            select_k = ['pair','triplet']
        else:
            select_k = None

        print_dict(m, select_k = select_k)


