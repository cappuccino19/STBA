# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForMaskedLM.from_pretrained("roberta-base")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel,BertTokenizer


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
        affine = affine.view(batch_size, len1*self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)
        biaffine = torch.bmm(affine, input2)
        biaffine = torch.transpose(biaffine, 1, 2)
        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)
        return biaffine

class base_model(nn.Module):
    def __init__(self,args, pretrained_model_path, hidden_dim:200, dropout,class_n =16, span_average = False):
        super().__init__()
        self.args = args
        # Encoder
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        # self.biaffine = Biaffine(args, in1_features=hidden_dim, in2_features=hidden_dim, out_features=class_n)
        self.dense = nn.Linear(self.bert.pooler.dense.out_features, hidden_dim)
        self.layer_norm = LayerNorm(hidden_dim*3)
        self.span_average = span_average
        # Classifier

        self.classifier = nn.Linear(hidden_dim * 3, class_n)
        self.biaffine = Biaffine(args, in1_features=hidden_dim, in2_features=hidden_dim, out_features=class_n)
        # self.attention_layer = SelfAttention(input_dim=hidden_dim, hidden_dim=hidden_dim)
        # self.layer_norm = LayerNorm(hidden_dim * 3)
        # dropout
        self.layer_drop = nn.Dropout(dropout)
        # self.biaffine = Biaffine(args, in1_features=hidden_dim, in2_features=hidden_dim, out_features=class_n)

    def forward(self, inputs, weight=None):
        #############################################################################################
        # word representation bert_token: torch.Size([16,42])
        bert_token = inputs['bert_token']
        attention_mask = (bert_token>0).int()
        bert_word_mapback = inputs['bert_word_mapback']
        token_length = inputs['token_length']
        bert_length = inputs['bert_length']
        bert_out = self.bert(bert_token,attention_mask = attention_mask).last_hidden_state # \hat{h}
        bert_seq_indi = sequence_mask(bert_length).unsqueeze(dim=-1)
        bert_out = bert_out[:, 1:max(bert_length) + 1, :] * bert_seq_indi.float()
        word_mapback_one_hot = (F.one_hot(bert_word_mapback).float() * bert_seq_indi.float()).transpose(1, 2)
        bert_out = torch.bmm(word_mapback_one_hot.float(), self.dense(bert_out))
        wnt = word_mapback_one_hot.sum(dim=-1)
        wnt.masked_fill_(wnt == 0, 1)
        #############################################################################################
        # span representation
        max_seq = bert_out.shape[1]
        token_length_mask = sequence_mask(token_length)
        candidate_tag_mask = torch.triu(torch.ones(max_seq,max_seq,dtype=torch.int64,device=bert_out.device),diagonal=0).unsqueeze(dim=0) * (token_length_mask.unsqueeze(dim=1) * token_length_mask.unsqueeze(dim=-1))
        boundary_table_features = torch.cat([bert_out.unsqueeze(dim=2).repeat(1,1,max_seq,1), bert_out.unsqueeze(dim=1).repeat(1,max_seq,1,1)],dim=-1) * candidate_tag_mask.unsqueeze(dim=-1)  # h_i ; h_j
        span_table_features = form_raw_span_features(bert_out, candidate_tag_mask, is_average = self.span_average) # sum(h_i,h_{i+1},...,h_{j})

        # h_i ; h_j ; sum(h_i,h_{i+1},...,h_{j})
        table_features = torch.cat([boundary_table_features , span_table_features],dim=-1)
        #############################################################################################
        # classifier
        # table_features = self.layer_norm(table_features)
        biaffine_output = self.biaffine(bert_out, bert_out)  # 双向注意力输出
        biaffine_output = biaffine_output * candidate_tag_mask.unsqueeze(dim=-1)
        logits = self.classifier(self.layer_drop(table_features)) *candidate_tag_mask.unsqueeze(dim=-1)
        # attention_scores = self.biaffine_attention(bert_out, bert_out)
        outputs = {
            'logits':logits,
            # 'attention_scores': attention_scores
            'biaffine_output': biaffine_output
        }
        if 'golden_label' in inputs and inputs['golden_label'] is not None:
            loss = calcualte_loss(logits, inputs['golden_label'],candidate_tag_mask, weight = weight)
            outputs['loss'] = loss
        return outputs
def sequence_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) < (lengths.unsqueeze(1))
def form_raw_span_features(v, candidate_tag_mask, is_average = True):
    new_v = v.unsqueeze(dim=1) * candidate_tag_mask.unsqueeze(dim=-1)
    span_features = torch.matmul(new_v.transpose(1,-1).transpose(2,-1), candidate_tag_mask.unsqueeze(dim=1).float()).transpose(2,1).transpose(2,-1)
    if is_average:
        _, max_seq, _ = v.shape
        sub_v = torch.tensor(range(1,max_seq+1), device = v.device).unsqueeze(dim=-1)  - torch.tensor(range(max_seq),device = v.device)
        sub_v  = torch.where(sub_v > 0, sub_v, 1).T
        span_features = span_features / sub_v.unsqueeze(dim=0).unsqueeze(dim=-1)
    return span_features
def calcualte_loss(logits, golden_label,candidate_tag_mask, weight=None):
    loss_func = nn.CrossEntropyLoss(weight = weight, reduction='none')
    return (loss_func(logits.view(-1,logits.shape[-1]),
                      golden_label.view(-1)
                      ).view(golden_label.size()) * candidate_tag_mask).sum()


