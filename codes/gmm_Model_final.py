#!/usr/bin/python3
#完整部署gmm模型的概率，均值，方差，三个维度的特征
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.distributions import Uniform
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from dataloader import *
import random
import pickle
import math
try:
    from apex import amp
except:
    print("apex not installed")
import os
from collections import defaultdict
# import sklearn
# from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
from dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator

query_name_dict = {('e', ('r',)): '1p',
                   ('e', ('r', 'r')): '2p',
                   ('e', ('r', 'r', 'r')): '3p',
                   (('e', ('r',)), ('e', ('r',))): '2i',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                   ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                   (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                   (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                   (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                   ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                   (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                   (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                   (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                   ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                   ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                   }
name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(
    name_query_dict.keys())

class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, weight_embedding):
        return torch.clamp(weight_embedding + self.base_add, self.min_val, self.max_val) #将输入张量的每个元素映射到[min, max]

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing   = smoothing
        self.reduction = reduction
        self.weight    = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
         if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class SelfAttention(nn.Module):

    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.hidden_size = hidden_size

    def forward(self, particles):

        # [batch_size, num_particles, embedding_size]
        K = self.query(particles)
        V = self.query(particles)
        Q = self.query(particles)

        # [batch_size, num_particles, num_particles]
        attention_scores = torch.matmul(Q, K.permute(0,2,1))
        attention_scores = attention_scores / math.sqrt(self.hidden_size)

        # [batch_size, num_particles, num_particles]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # attention_probs = self.dropout(attention_probs)

        # [batch_size, num_particles, embedding_size]
        attention_output = torch.matmul(attention_probs, V)

        return attention_output

class FFN(nn.Module):
    """
    Actually without the FFN layer, there is no non-linearity involved. That is may be why the model cannot fit
    the training queries so well
    """
    def __init__(self, hidden_size, dropout):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)


        self.activation = nn.GELU()
        self.dropout = dropout

    def forward(self, particles):
        return self.linear2(self.dropout(self.activation(self.linear1(self.dropout(particles)))))

class EntityToAnchor(nn.Module):
    def __init__(self, hidden_dim, num_gaussian_component):
        super(EntityToAnchor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_gaussian_component = num_gaussian_component
        self.weight_matrix = nn.Parameter(torch.ones([1, num_gaussian_component, hidden_dim]), requires_grad=True)
        self.mean_offset_matrix = nn.Parameter(torch.zeros([1, num_gaussian_component, hidden_dim]), requires_grad=True)
        self.std_deviation_offset_matrix = nn.Parameter(torch.zeros([1, num_gaussian_component, hidden_dim]), requires_grad=True)

    def forward(self, batch_of_mean_embeddings, batch_of_std_deviation_embeddings):
        batch_size, hidden_dim = batch_of_mean_embeddings.shape
        #[batch_size, num_gaussian_component , hidden_dim]
        batch_of_weight_embedding = self.weight_matrix.repeat(batch_size, 1, 1)
        expanded_batch_of_mean_embeddings = batch_of_mean_embeddings.reshape(batch_size, -1, hidden_dim) + self.mean_offset_matrix
        expanded_batch_of_std_deviation_embeddings = batch_of_std_deviation_embeddings.reshape(batch_size, -1, hidden_dim) + self.std_deviation_offset_matrix
        #[batch_size, num_gaussian_component , hidden_dim*3]
        expanded_batch_of_embeddings = torch.cat([batch_of_weight_embedding, expanded_batch_of_mean_embeddings, expanded_batch_of_std_deviation_embeddings], dim=-1)
        return expanded_batch_of_embeddings

class GMMProjection(nn.Module):
    def __init__(self, hidden_dim, dropout, weight_regularizer):
        super(GMMProjection, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.weight_regularizer = weight_regularizer
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.Wz = nn.Linear(self.hidden_dim * 3, self.hidden_dim * 3)
        self.Uz = nn.Linear(self.hidden_dim * 3, self.hidden_dim * 3)
        self.layer_norm_z = LayerNorm(self.hidden_dim * 3)
        self.Wh = nn.Linear(self.hidden_dim * 3, self.hidden_dim * 3)
        self.Uh = nn.Linear(self.hidden_dim * 3, self.hidden_dim * 3)
        self.layer_norm_h = LayerNorm(self.hidden_dim * 3)

        self.attention_module = SelfAttention(self.hidden_dim * 3)
        self.layer_norm1 = LayerNorm(self.hidden_dim * 3)
        self.layer_norm2 = LayerNorm(self.hidden_dim * 3)

    def forward(self, gmm_embedding, relation_transition_embedding):
        """
            :param gmm_embedding: [batch_size, num_gaussian_component, hidden_dim * 3], weight, mean, variance
            :param relation_transition_embedding: [batch_size, hidden_dim * 2], mean, variance

            :return: [batch_size, num_gaussian_component, entity_dim]
        """
        # [batch_size, 1, hidden_dim * 2]
        relation_transition_embedding = relation_transition_embedding.unsqueeze(1)
        # [batch_size, 1, hidden_dim]
        relation_delta_mean, relation_delta_variance = torch.chunk(relation_transition_embedding, 2, dim=-1)
        # [batch_size, 1, hidden_dim]
        relation_init_weight = torch.ones_like(relation_delta_mean).cuda()
        # [batch_size, 1, hidden_dim * 3]
        weighted_relation_embedding = torch.cat([relation_init_weight, relation_transition_embedding], dim=-1)
        # [batch_size, num_gaussian_component, hidden_dim]
        gmm_weight, gmm_mean, gmm_variance = torch.chunk(gmm_embedding, 3, dim=-1)
        # [batch_size, num_gaussian_component, hidden_dim]
        normalized_gmm_weight = F.softmax(gmm_weight, dim=1)
        # [batch_size, num_gaussian_component, hidden_dim * 3]
        normalized_gmm_embedding = torch.cat([normalized_gmm_weight, gmm_mean, gmm_variance], dim=-1)
        projected_gmm_embedding = normalized_gmm_embedding
        z = self.sigmoid(self.layer_norm_z(self.Wz(self.dropout(weighted_relation_embedding))) + self.Uz(self.dropout(projected_gmm_embedding)))
        h_hat = self.relu(self.layer_norm_h(self.Wh(self.dropout(weighted_relation_embedding))) + self.Uh(self.dropout(projected_gmm_embedding)))
        h = (1 - z) * normalized_gmm_embedding + z * h_hat
        projected_gmm_embedding = h
        projected_gmm_embedding = self.layer_norm1(projected_gmm_embedding)
        projected_gmm_embedding = self.attention_module(self.dropout(projected_gmm_embedding))
        projected_gmm_embedding = self.layer_norm2(projected_gmm_embedding)
        projected_gmm_weight, projected_gmm_mean, projected_gmm_variance = torch.chunk(projected_gmm_embedding, 3, dim=-1)
        # 均一化处理
        projected_gmm_weight = F.softmax(projected_gmm_embedding, dim=1)
        # 标准差同样要求非负
        projected_gmm_variance = torch.abs(projected_gmm_variance)
        projected_gmm_embedding = torch.cat([projected_gmm_weight, projected_gmm_mean, projected_gmm_variance], dim=-1)
        return projected_gmm_embedding

class GMMHigherOrderProjection(nn.Module):
    def __init__(self, hidden_dim, dropout, gmmProjection):
        super(GMMHigherOrderProjection, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.attention_module = SelfAttention(self.hidden_dim * 3)
        self.gmmProjection = gmmProjection
        self.layer_norm1 = LayerNorm(self.hidden_dim * 3)
        self.layer_norm2 = LayerNorm(self.hidden_dim * 3)
        self.outputMLP = FFN(self.hidden_dim * 3, dropout)

    def forward(self, gmm_embedding, relation_transition_embedding):
        normalized_gmm_embedding = gmm_embedding
        projected_gmm_embedding = self.attention_module(normalized_gmm_embedding)
        projected_gmm_embedding = self.layer_norm1(projected_gmm_embedding)
        projected_gmm_embedding = self.outputMLP(projected_gmm_embedding) + projected_gmm_embedding
        projected_gmm_embedding = self.layer_norm2(projected_gmm_embedding)
        projected_gmm_embedding = self.gmmProjection(projected_gmm_embedding, relation_transition_embedding)
        return projected_gmm_embedding

class GMMComplement(nn.Module):
    def __init__(self, hidden_dim, num_gaussian_component, dropout, weight_regularizer):
        super(GMMComplement, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_gaussian_component = num_gaussian_component
        self.dropout = dropout
        self.weight_regularizer = weight_regularizer
        self.attention_module = SelfAttention(self.hidden_dim * 3)
        self.layer_norm1 = LayerNorm(self.hidden_dim * 3)
        self.layer_norm2 = LayerNorm(self.hidden_dim * 3)
        self.layer_norm3 = LayerNorm(self.hidden_dim * 3)
        self.layer_norm4 = LayerNorm(self.hidden_dim * 3)
        self.mlp = FFN(self.hidden_dim * 2, dropout)

    def forward(self, gmm_embedding):
        """
            :param galaxy_embedding: [batch_size, num_planets, hidden_dim * 3]

            :return: [batch_size, num_planets, hidden_dim * 3]
        """
        #negation layer 1
        gmm_weight, gmm_mean, gmm_variance = torch.chunk(gmm_embedding, 3, dim=-1)
        normalized_gmm_weight = F.softmax(gmm_weight, dim=1)
        negationed_gmm_embedding = torch.cat([normalized_gmm_weight, gmm_mean, gmm_variance], dim=-1)
        negationed_gmm_embedding = self.attention_module(self.dropout(negationed_gmm_embedding))
        negationed_gmm_embedding = self.layer_norm1(negationed_gmm_embedding)
        negationed_gmm_embedding = self.mlp(negationed_gmm_embedding) + negationed_gmm_embedding
        negationed_gmm_embedding = self.layer_norm2(negationed_gmm_embedding)
        #negation layer 2
        negationed_gmm_embedding = self.attention_module(self.dropout(negationed_gmm_embedding))
        negationed_gmm_embedding = self.layer_norm3(negationed_gmm_embedding)
        negationed_gmm_embedding = self.mlp(negationed_gmm_embedding) + negationed_gmm_embedding
        negationed_gmm_embedding = self.layer_norm4(negationed_gmm_embedding)
        negationed_gmm_weight, negationed_gmm_mean, negationed_gmm_variance = torch.chunk(negationed_gmm_embedding, 3, dim=-1)
        negationed_gmm_weight = F.softmax(negationed_gmm_weight, dim=1)
        negationed_gmm_variance = torch.abs(negationed_gmm_variance)
        negationed_gmm_embedding = torch.cat([negationed_gmm_weight, negationed_gmm_mean, negationed_gmm_variance], dim=-1)
        return negationed_gmm_embedding

class GMMIntersection(nn.Module):
    def __init__(self, hidden_dim, num_gaussian_component, dropout, weight_regularizer):
        super(GMMIntersection, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_gaussian_component = num_gaussian_component
        self.dropout = dropout
        self.weight_regularizer = weight_regularizer
        self.layer_mlp1 = nn.Linear(self.hidden_dim * 3, self.hidden_dim * 3)
        self.layer_mlp2 = nn.Linear(self.hidden_dim * 3, self.hidden_dim * 3)
        self.attention_module = SelfAttention(self.hidden_dim * 3)
        self.layer_norm1 = LayerNorm(self.hidden_dim * 3)
        self.layer_norm2 = LayerNorm(self.hidden_dim * 3)

    def forward(self, multiple_gmm_embedding, query_multi_hot1, query_multi_hot2, query_multi_hot3 = []):
        """
            :param multiple_gmm_embeddings: [batch_size, num_inputs, num_gaussian_component, hidden_dim * 3]

            :return: [batch_size, num_gaussian_component, hidden_dim * 3]
        """
        if query_multi_hot1 != None:
            if len(query_multi_hot3) > 0:
                intersected_one_hot = query_multi_hot1 + query_multi_hot2 + query_multi_hot3
                intersected_one_hot[intersected_one_hot < 3] = 0
                intersected_one_hot[intersected_one_hot == 3] = 1
                similarWeight1 = query_multi_hot1 - intersected_one_hot
                similarWeight2 = query_multi_hot2 - intersected_one_hot
                similarWeight3 = query_multi_hot3 - intersected_one_hot
                similarWeight1 = torch.sum(similarWeight1, dim=-1, keepdim=True) + 1 # 直接求和和一范数（绝对值求和不知道效果会不会有差别）
                similarWeight2 = torch.sum(similarWeight2, dim=-1, keepdim=True) + 1
                similarWeight3 = torch.sum(similarWeight3, dim=-1, keepdim=True) + 1
                similarWeight1 = 1 / similarWeight1
                similarWeight2 = 1 / similarWeight2
                similarWeight3 = 1 / similarWeight3
            else:
                intersected_one_hot = query_multi_hot1 + query_multi_hot2
                intersected_one_hot[intersected_one_hot < 2] = 0
                intersected_one_hot[intersected_one_hot == 2] = 1
                similarWeight1 = query_multi_hot1 - intersected_one_hot
                similarWeight2 = query_multi_hot2 - intersected_one_hot
                similarWeight1 = torch.sum(similarWeight1, dim=-1, keepdim=True) + 1
                similarWeight2 = torch.sum(similarWeight2, dim=-1, keepdim=True) + 1
                similarWeight1 = 1 / similarWeight1
                similarWeight2 = 1 / similarWeight2
        else:
            similarWeight1 = nn.Parameter(torch.tensor([1]).float(), requires_grad=False).cuda()
            similarWeight2 = nn.Parameter(torch.tensor([1]).float(), requires_grad=False).cuda()
            similarWeight3 = nn.Parameter(torch.tensor([1]).float(), requires_grad=False).cuda()

        batch_size, num_input, num_gaussian_component, entity_dim = multiple_gmm_embedding.shape
        # [batch_size, 1, num_gaussian_component, entity_dim] * num_input
        gmm_embeddings = torch.chunk(multiple_gmm_embedding, num_input, dim=1)

        # cross-attention
        # [batch_size, num_gaussian_component, hidden_dim * 3]
        gmm_embedding_1 = gmm_embeddings[0].squeeze(1)
        # [batch_size, num_gaussian_component, hidden_dim * 3]
        layer_act1 = self.layer_mlp2(F.relu(self.layer_mlp1(gmm_embedding_1)))
        gmm_embedding_2 = gmm_embeddings[1].squeeze(1)
        layer_act2 = self.layer_mlp2(F.relu(self.layer_mlp1(gmm_embedding_2)))
        # [2, batch_size, num_gaussian_component, hidden_dim * 3]
        attention_score = F.softmax(torch.stack([layer_act1 * similarWeight1, layer_act2 * similarWeight2]), dim=0)
        # [batch_size, num_gaussian_component, hidden_dim * 3]
        intersected_gmm_embedding = attention_score[0] * gmm_embedding_1 + attention_score[1] * gmm_embedding_2
        if num_input == 3:
            gmm_embedding_3 = gmm_embeddings[2].squeeze(1)
            layer_act3 = self.layer_mlp2(F.relu(self.layer_mlp1(gmm_embedding_3)))
            # [3, batch_size, num_gaussian_component, hidden_dim * 3]
            attention_score = F.softmax(torch.stack([layer_act1 * similarWeight1, layer_act2 * similarWeight2, layer_act3 * similarWeight3]), dim=0)
            intersected_gmm_embedding = attention_score[0] * gmm_embedding_1 + attention_score[1] * gmm_embedding_2 + attention_score[2] * gmm_embedding_3
        intersected_gmm_embedding = self.layer_norm1(intersected_gmm_embedding)
        # self-attention
        intersected_gmm_embedding = self.attention_module(intersected_gmm_embedding)
        intersected_gmm_embedding = self.layer_norm2(intersected_gmm_embedding)
        intersected_gmm_weight, intersected_gmm_mean, intersected_gmm_variance = torch.chunk(intersected_gmm_embedding, 3, dim=-1)
        intersected_gmm_weight = F.softmax(intersected_gmm_weight, dim=1)
        intersected_gmm_variance = torch.abs(intersected_gmm_variance)
        intersected_gmm_embedding = torch.cat([intersected_gmm_weight, intersected_gmm_mean, intersected_gmm_variance], dim=-1)
        return intersected_gmm_embedding

class Query2GMM(nn.Module):
    def __init__(self, nentity, nrelation, entity_dim,
                 num_gaussian_component = 2,
                 dropout_rate = 0.2, label_smoothing = 0.1,
                 gamma = 24.0, sigma = 0,
                 node_group_one_hot_vector_single=None, group_adj_matrix_single=None,
                 node_group_one_hot_vector_multi=None, group_adj_matrix_multi=None):
        super(Query2GMM, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.label_smoothing = label_smoothing
        self.hidden_dim = entity_dim
        self.entity_dim = self.hidden_dim * 2 # mean value, standard deviation;
        self.relation_dim = self.hidden_dim * 2 # delta mean value, delta standard deviation
        # Number of gaussian components
        self.num_gaussian_component = num_gaussian_component
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )
        self.epsilon = 2.0
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.hidden_dim]),
            requires_grad=False
        )
        # Initialize the entity embeddings and relation transition embeddings with uniform gamma
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim)) # mean value, standard deviation;
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        # 标准差部分要求为正, 但是0.0001貌似因为精度问题被当作非正值。。。。
        nn.init.uniform_(
            tensor=self.entity_embedding[:, self.hidden_dim: 2*self.hidden_dim],
            a=0,
            b=self.embedding_range.item()
        )
        self.weight_regularizer = Regularizer(0.0, 0.01, 5.0)
        self.relation_transition_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim)) # mean value, delta standard deviation;
        nn.init.uniform_(
            tensor=self.relation_transition_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        self.mudulus = nn.Parameter(torch.Tensor([0.5 * self.embedding_range.item()]), requires_grad=True)
        # The model support the operation of projection, intersection, union, and complement
        self.entityToAnchor = EntityToAnchor(self.hidden_dim, self.num_gaussian_component)
        self.gmmProjection = GMMProjection(self.hidden_dim, self.dropout, self.weight_regularizer)
        self.gmmHigherOrderProjection = GMMHigherOrderProjection(self.hidden_dim, self.dropout, self.gmmProjection)
        self.gmmIntersection = GMMIntersection(self.hidden_dim, self.num_gaussian_component, self.dropout, self.weight_regularizer)
        self.gmmComplement = GMMComplement(self.hidden_dim, self.num_gaussian_component, self.dropout, self.weight_regularizer)

        # group infor init
        if group_adj_matrix_single is not None:
            self.group_adj_matrix = torch.tensor(group_adj_matrix_single.tolist(), requires_grad=False).cuda()
        if node_group_one_hot_vector_single is not None:
            self.node_group_one_hot_vector = torch.tensor(node_group_one_hot_vector_single.tolist(), requires_grad=False).cuda()
        # weight of group infor distance in loss func
        self.group_adj_weight = nn.Parameter(torch.tensor([1]).float(), requires_grad=True)

        if group_adj_matrix_multi is not None:
            self.group_adj_matrix_multi = []
            for xxx in group_adj_matrix_multi:
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                self.group_adj_matrix_multi.append(torch.tensor(xxx.tolist(), requires_grad=False).cuda())
        if node_group_one_hot_vector_multi is not None:
            self.node_group_one_hot_vector_multi = []
            for xxx in node_group_one_hot_vector_multi:
                self.node_group_one_hot_vector_multi.append(torch.tensor(xxx.tolist(), requires_grad=False).cuda())
        # self.group_times = len(self.group_adj_matrix_multi)
        # multi相对于可以看成是multi head相对于single head
        # self.group_adj_weight_multi = nn.Parameter(torch.tensor([0.1]).float(), requires_grad=True)

    def forward(self, batch_queries, qtype, inverted=False, mode='single', positive_sample=None, negative_sample=None):
        """
            :param positive_sample: [batch_size]
            :param negative_sample: [batch_size, num_negative_sample]
            :param batch_queries: [batch_size, query_length]
            :param qtype: str

            :return: positive_likelihoods, negative_likelihoods,  particles, weights
            positive_likelihoods: [batch_size, 1]
            negative_likelihoods: [batch_size, num_negative_sample]
        """
        if positive_sample is not None:
            # [batch_size, 1]
            positive_sample = positive_sample.unsqueeze(1)
        if qtype == '1p':
            entity_embedding = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 0])
            relation_embedding = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 1])
            # [batch_size, hidden_dim]
            entity_mean, entity_variance = torch.chunk(entity_embedding, 2, dim=-1)
            # [batch_size, num_gaussian_components, hidden_dim*3]
            anchor_gmm_embedding = self.entityToAnchor(entity_mean, entity_variance)
            gmm_embedding = self.gmmProjection(anchor_gmm_embedding, relation_embedding)

            # group distance calculation
            if mode == 'single':
                # [batch_size, 1, group_dimension]
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=positive_sample[:, 0]).unsqueeze(1)
            elif mode == 'tail-batch':
                batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
                # [batch_size, negative_sample_size, group_dimension)
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=negative_sample.view(-1)).view(batch_size,
                                                                                                                              negative_sample_size,
                                                                                                                              -1)
            # [batch_size, 1, group_dimension]
            one_hot_head = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 0]).unsqueeze(1)
            # [batch_size, group_dimension, group_dimension]
            relation_matrix = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 1])
            # [batch_size, 1, group_dimension]
            inferred_one_hot_tail = torch.matmul(one_hot_head, relation_matrix)
            inferred_one_hot_tail[inferred_one_hot_tail >= 1] = 1  # <1的部分需要调整吗？？？？，以下2u-arg，2p，3p 同样需要考虑这个问题
            # [batch_size, 1/negative_sample_size, group_dimension]
            group_dist = F.relu(one_hot_tail - inferred_one_hot_tail)
            # [batch_size, 1/negative_sample_size]
            group_dist = torch.norm(group_dist, p=1, dim=-1)

        elif qtype == '2u-arg':  # 没看出来造这个结构是为了啥
            entity_embedding = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 0])
            relation_embedding = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 1])
            entity_mean, entity_variance = torch.chunk(entity_embedding, 2, dim=-1)

            batch_size, hidden_dim = relation_embedding.shape
            random_permutation_index = torch.randperm(batch_size)
            # [batch_size, num_gaussian_components, hidden_dim*3]
            anchor_gmm_embedding = self.entityToAnchor(entity_mean, entity_variance)
            gmm_embedding = self.gmmProjection(anchor_gmm_embedding, relation_embedding)

            random_permuted_anchor_gmm_embedding = anchor_gmm_embedding[random_permutation_index]
            random_permuted_relation_embedding = relation_embedding[random_permutation_index]
            random_permuted_gmm_embedding = self.gmmProjection(random_permuted_anchor_gmm_embedding, random_permuted_relation_embedding)

            gmm_embedding = torch.cat([gmm_embedding, random_permuted_gmm_embedding], dim=1)

            # group distance calculation
            if mode == 'single':
                # [batch_size, 1, group_dimension]
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=positive_sample[:, 0]).unsqueeze(1)
            elif mode == 'tail-batch':
                batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
                # [batch_size, negative_sample_size, group_dimension]
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=negative_sample.view(-1)).view(batch_size,
                                                                                                                              negative_sample_size,
                                                                                                                              -1)
            # [batch_size, 1, group_dimension]
            one_hot_head = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 0]).unsqueeze(1)
            # [batch_size, group_dimension, group_dimension]
            relation_matrix = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 1])
            # [batch_size, 1, group_dimension]
            inferred_one_hot_tail = torch.matmul(one_hot_head, relation_matrix)
            inferred_one_hot_tail[inferred_one_hot_tail >= 1] = 1
            # [batch_size, 1/negative_sample_size, group_dimension]
            group_dist = F.relu(one_hot_tail - inferred_one_hot_tail)
            group_dist = torch.norm(group_dist, p=1, dim=-1)

        elif qtype == '2p':
            entity_embedding = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 0])
            relation_embedding1 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 1])
            relation_embedding2 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 2])
            entity_mean, entity_variance = torch.chunk(entity_embedding, 2, dim=-1)

            anchor_gmm_embedding = self.entityToAnchor(entity_mean, entity_variance)
            gmm_embedding = self.gmmProjection(anchor_gmm_embedding, relation_embedding1)
            gmm_embedding = self.gmmHigherOrderProjection(gmm_embedding, relation_embedding2)

            # group distance calculation
            if mode == 'single':
                # [batch_size, 1, group_dimension]
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=positive_sample[:, 0]).unsqueeze(1)
            elif mode == 'tail-batch':
                batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
                # [batch_size, negative_sample_size, group_dimension]
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=negative_sample.view(-1)).view(batch_size,
                                                                                                                              negative_sample_size,
                                                                                                                              -1)
            # [batch_size, 1, group_dimension]
            one_hot_head = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 0]).unsqueeze(1)
            for i in range(1, 1 + 2):
                # [batch_size, group_dimension, group_dimension]
                relation_matrix = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, i])
                # [batch_size, 1, group_dimension]
                one_hot_head = torch.matmul(one_hot_head, relation_matrix)
            # [batch_size, 1, group_dimension]
            inferred_one_hot_tail = one_hot_head
            inferred_one_hot_tail[inferred_one_hot_tail >= 1] = 1
            # [batch_size, 1/negative_sample_size, group_dimension]
            group_dist = F.relu(one_hot_tail - inferred_one_hot_tail)
            group_dist = torch.norm(group_dist, p=1, dim=-1)

        elif qtype == '3p':
            entity_embedding = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 0])
            relation_embedding1 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 1])
            relation_embedding2 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 2])
            relation_embedding3 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 3])
            entity_mean, entity_variance = torch.chunk(entity_embedding, 2, dim=-1)
            # [batch_size, num_gaussian_components, hidden_dim * 3]
            anchor_gmm_embedding = self.entityToAnchor(entity_mean, entity_variance)
            gmm_embedding = self.gmmProjection(anchor_gmm_embedding, relation_embedding1)
            gmm_embedding = self.gmmHigherOrderProjection(gmm_embedding, relation_embedding2)
            gmm_embedding = self.gmmHigherOrderProjection(gmm_embedding, relation_embedding3)

            # group distance calculation
            if mode == 'single':
                # [batch_size, 1, group_dimension]
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=positive_sample[:, 0]).unsqueeze(1)
            elif mode == 'tail-batch':
                batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=negative_sample.view(-1)).view(batch_size,
                                                                                                                              negative_sample_size,
                                                                                                                              -1)
            # [batch_size, 1, group_dimension]
            one_hot_head = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 0]).unsqueeze(1)
            for i in range(1, 1 + 3):
                # [batch_size, group_dimension, group_dimension]
                relation_matrix = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, i])
                # [batch_size, 1, group_dimension]
                one_hot_head = torch.matmul(one_hot_head, relation_matrix)
            # [batch_size, 1, group_dimension]
            inferred_one_hot_tail = one_hot_head
            inferred_one_hot_tail[inferred_one_hot_tail >= 1] = 1
            group_dist = F.relu(one_hot_tail - inferred_one_hot_tail)
            group_dist = torch.norm(group_dist, p=1, dim=-1)

        elif qtype == '2i':
            # group distance calculation
            if mode == 'single':
                # [batch_size, 1, group_dimension]
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=positive_sample[:, 0]).unsqueeze(1)
            elif mode == 'tail-batch':
                batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
                # [batch_size, negative_sample_size, group_dimension]
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=negative_sample.view(-1)).view(batch_size,
                                                                                                                              negative_sample_size,
                                                                                                                              -1)
            # projection 1
            # [batch_size, 1, group_dimension]
            one_hot_head_1 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 0]).unsqueeze(1)
            relation_matrix_1 = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 1])
            # [batch_size, 1, group_dimension]
            inferred_one_hot_tail_1 = torch.matmul(one_hot_head_1, relation_matrix_1)
            inferred_one_hot_tail_1[inferred_one_hot_tail_1 >= 1] = 1
            # projection 2
            # [batch_size, 1, group_dimension]
            one_hot_head_2 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 2]).unsqueeze(1)
            relation_matrix_2 = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 3])
            # [batch_size, 1, group_dimension]
            inferred_one_hot_tail_2 = torch.matmul(one_hot_head_2, relation_matrix_2)
            inferred_one_hot_tail_2[inferred_one_hot_tail_2 >= 1] = 1
            # intersection
            inferred_one_hot_tail = inferred_one_hot_tail_1 + inferred_one_hot_tail_2
            inferred_one_hot_tail[inferred_one_hot_tail < 2] = 0
            inferred_one_hot_tail[inferred_one_hot_tail >= 2] = 1
            # [batch_size, 1/negative_sample_size, group_dimension]
            group_dist = F.relu(one_hot_tail - inferred_one_hot_tail)
            group_dist = torch.norm(group_dist, p=1, dim=-1)

            entity_embedding1 = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 0])
            relation_embedding1 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 1])
            entity_embedding2 = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 2])
            relation_embedding2 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 3])
            entity_mean1, entity_variance1 = torch.chunk(entity_embedding1, 2, dim=-1)
            entity_mean2, entity_variance2 = torch.chunk(entity_embedding2, 2, dim=-1)
            anchor_gmm_embedding1 = self.entityToAnchor(entity_mean1, entity_variance1)
            gmm_embedding1 = self.gmmProjection(anchor_gmm_embedding1, relation_embedding1)
            anchor_gmm_embedding2 = self.entityToAnchor(entity_mean2, entity_variance2)
            gmm_embedding2 = self.gmmProjection(anchor_gmm_embedding2, relation_embedding2)

            multiple_gmm_embeddings = torch.stack([gmm_embedding1, gmm_embedding2], dim=1)
            gmm_embedding = self.gmmIntersection(multiple_gmm_embeddings, inferred_one_hot_tail_1, inferred_one_hot_tail_2)

        elif qtype == '3i':
            # group distance calculation
            if mode == 'single':
                # [batch_size, 1, group_dimension]
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=positive_sample[:, 0]).unsqueeze(1)
            elif mode == 'tail-batch':
                batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=negative_sample.view(-1)).view(batch_size,
                                                                                                                              negative_sample_size,
                                                                                                                              -1)
            # projection 1
            # [batch_size, 1, group_dimension]
            one_hot_head_1 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 0]).unsqueeze(1)
            relation_matrix_1 = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 1])
            # [batch_size, 1, group_dimension]
            inferred_one_hot_tail_1 = torch.matmul(one_hot_head_1, relation_matrix_1)
            inferred_one_hot_tail_1[inferred_one_hot_tail_1 >= 1] = 1
            # projection 2
            # [batch_size, 1, group_dimension]
            one_hot_head_2 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 2]).unsqueeze(1)
            relation_matrix_2 = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 3])
            inferred_one_hot_tail_2 = torch.matmul(one_hot_head_2, relation_matrix_2)
            inferred_one_hot_tail_2[inferred_one_hot_tail_2 >= 1] = 1
            # projection 3
            # [batch_size, 1, group_dimension]
            one_hot_head_3 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 4]).unsqueeze(1)
            relation_matrix_3 = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 5])
            inferred_one_hot_tail_3 = torch.matmul(one_hot_head_3, relation_matrix_3)
            inferred_one_hot_tail_3[inferred_one_hot_tail_3 >= 1] = 1
            # intersection
            inferred_one_hot_tail = inferred_one_hot_tail_1 + inferred_one_hot_tail_2 + inferred_one_hot_tail_3
            inferred_one_hot_tail[inferred_one_hot_tail < 3] = 0
            inferred_one_hot_tail[inferred_one_hot_tail >= 3] = 1
            group_dist = F.relu(one_hot_tail - inferred_one_hot_tail)
            group_dist = torch.norm(group_dist, p=1, dim=-1)

            entity_embedding1 = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 0])
            relation_embedding1 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 1])
            entity_embedding2 = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 2])
            relation_embedding2 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 3])
            entity_embedding3 = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 4])
            relation_embedding3 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 5])
            entity_mean1, entity_variance1 = torch.chunk(entity_embedding1, 2, dim=-1)
            entity_mean2, entity_variance2 = torch.chunk(entity_embedding2, 2, dim=-1)
            entity_mean3, entity_variance3 = torch.chunk(entity_embedding3, 2, dim=-1)

            anchor_gmm_embedding1 = self.entityToAnchor(entity_mean1, entity_variance1)
            gmm_embedding1 = self.gmmProjection(anchor_gmm_embedding1, relation_embedding1)
            anchor_gmm_embedding2 = self.entityToAnchor(entity_mean2, entity_variance2)
            gmm_embedding2 = self.gmmProjection(anchor_gmm_embedding2, relation_embedding2)
            anchor_gmm_embedding3 = self.entityToAnchor(entity_mean3, entity_variance3)
            gmm_embedding3 = self.gmmProjection(anchor_gmm_embedding3, relation_embedding3)

            multiple_gmm_embeddings = torch.stack([gmm_embedding1, gmm_embedding2, gmm_embedding3], dim=1)
            gmm_embedding = self.gmmIntersection(multiple_gmm_embeddings, inferred_one_hot_tail_1, inferred_one_hot_tail_2, inferred_one_hot_tail_3)

        elif qtype == 'ip':
            # group distance calculation
            if mode == 'single':
                # [batch_size, 1, group_dimension]
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=positive_sample[:, 0]).unsqueeze(1)
            elif mode == 'tail-batch':
                batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=negative_sample.view(-1)).view(batch_size,
                                                                                                                              negative_sample_size,
                                                                                                                              -1)
            # projection 1
            # [batch_size, 1, group_dimension]
            one_hot_head_1 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 0]).unsqueeze(1)
            relation_matrix_1 = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 1])
            inferred_one_hot_tail_1 = torch.matmul(one_hot_head_1, relation_matrix_1)
            inferred_one_hot_tail_1[inferred_one_hot_tail_1 >= 1] = 1
            # projection 2
            # [batch_size, 1, group_dimension]
            one_hot_head_2 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 2]).unsqueeze(1)
            relation_matrix_2 = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 3])
            inferred_one_hot_tail_2 = torch.matmul(one_hot_head_2, relation_matrix_2)
            inferred_one_hot_tail_2[inferred_one_hot_tail_2 >= 1] = 1
            # intersection
            # [batch_size, 1, group_dimension]
            inferred_one_hot_tail = inferred_one_hot_tail_1 + inferred_one_hot_tail_2
            inferred_one_hot_tail[inferred_one_hot_tail < 2] = 0
            inferred_one_hot_tail[inferred_one_hot_tail >= 2] = 1
            # projection 3
            relation_matrix_3 = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 4])
            # [batch_size, 1, group_dimension]
            inferred_one_hot_tail = torch.matmul(inferred_one_hot_tail, relation_matrix_3)
            inferred_one_hot_tail[inferred_one_hot_tail >= 0.5] = 1  # 这边阈值可能需要进行调整
            inferred_one_hot_tail[inferred_one_hot_tail < 0.5] = 0
            group_dist = F.relu(one_hot_tail - inferred_one_hot_tail)
            group_dist = torch.norm(group_dist, p=1, dim=-1)

            entity_embedding1 = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 0])
            relation_embedding1 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 1])
            entity_embedding2 = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 2])
            relation_embedding2 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 3])
            relation_embedding3 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 4])
            entity_mean1, entity_variance1 = torch.chunk(entity_embedding1, 2, dim=-1)
            entity_mean2, entity_variance2 = torch.chunk(entity_embedding2, 2, dim=-1)
            anchor_gmm_embedding1 = self.entityToAnchor(entity_mean1, entity_variance1)
            gmm_embedding1 = self.gmmProjection(anchor_gmm_embedding1, relation_embedding1)
            anchor_gmm_embedding2 = self.entityToAnchor(entity_mean2, entity_variance2)
            gmm_embedding2 = self.gmmProjection(anchor_gmm_embedding2, relation_embedding2)

            multiple_gmm_embeddings = torch.stack([gmm_embedding1, gmm_embedding2], dim=1)
            gmm_embedding = self.gmmIntersection(multiple_gmm_embeddings, inferred_one_hot_tail_1, inferred_one_hot_tail_2)

            gmm_embedding = self.gmmHigherOrderProjection(gmm_embedding, relation_embedding3)

        elif qtype == 'pi':
            # group distance calculation
            if mode == 'single':
                # [batch_size, 1, group_dimension]
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=positive_sample[:, 0]).unsqueeze(1)
            elif mode == 'tail-batch':
                batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=negative_sample.view(-1)).view(batch_size,
                                                                                                                              negative_sample_size,
                                                                                                                              -1)
            # projection 1
            # [batch_size, 1, group_dimension]
            one_hot_head_1 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 0]).unsqueeze(1)
            for i in range(1, 1 + 2):
                relation_matrix = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, i])
                one_hot_head_1 = torch.matmul(one_hot_head_1, relation_matrix)
            # [batch_size, 1, group_dimension]
            inferred_one_hot_tail_1 = one_hot_head_1
            inferred_one_hot_tail_1[inferred_one_hot_tail_1 >= 1] = 1
            # projection 2
            # [batch_size, 1, group_dimension]
            one_hot_head_2 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 3]).unsqueeze(1)
            relation_matrix_2 = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 4])
            inferred_one_hot_tail_2 = torch.matmul(one_hot_head_2, relation_matrix_2)
            inferred_one_hot_tail_2[inferred_one_hot_tail_2 >= 1] = 1
            # intersection
            inferred_one_hot_tail = inferred_one_hot_tail_1 + inferred_one_hot_tail_2
            inferred_one_hot_tail[inferred_one_hot_tail < 2] = 0
            inferred_one_hot_tail[inferred_one_hot_tail >= 2] = 1
            group_dist = F.relu(one_hot_tail - inferred_one_hot_tail)
            group_dist = torch.norm(group_dist, p=1, dim=-1)

            entity_embedding1 = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 0])
            relation_embedding1_1 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 1])
            relation_embedding1_2 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 2])
            entity_embedding2 = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 3])
            relation_embedding2 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 4])
            entity_mean1, entity_variance1 = torch.chunk(entity_embedding1, 2, dim=-1)
            entity_mean2, entity_variance2 = torch.chunk(entity_embedding2, 2, dim=-1)
            anchor_gmm_embedding1 = self.entityToAnchor(entity_mean1, entity_variance1)
            gmm_embedding1 = self.gmmProjection(anchor_gmm_embedding1, relation_embedding1_1)
            gmm_embedding1 = self.gmmHigherOrderProjection(gmm_embedding1, relation_embedding1_2)
            anchor_gmm_embedding2 = self.entityToAnchor(entity_mean2, entity_variance2)
            gmm_embedding2 = self.gmmProjection(anchor_gmm_embedding2, relation_embedding2)

            multiple_gmm_embeddings = torch.stack([gmm_embedding1, gmm_embedding2], dim=1)
            gmm_embedding = self.gmmIntersection(multiple_gmm_embeddings, inferred_one_hot_tail_1, inferred_one_hot_tail_2)

        elif qtype == '2u-DNF':
            entity_embedding1 = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 0])
            relation_embedding1 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 1])
            entity_embedding2 = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 2])
            relation_embedding2 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 3])
            entity_mean1, entity_variance1 = torch.chunk(entity_embedding1, 2, dim=-1)
            entity_mean2, entity_variance2 = torch.chunk(entity_embedding2, 2, dim=-1)

            anchor_gmm_embedding1 = self.entityToAnchor(entity_mean1, entity_variance1)
            gmm_embedding1 = self.gmmProjection(anchor_gmm_embedding1, relation_embedding1)
            anchor_gmm_embedding2 = self.entityToAnchor(entity_mean2, entity_variance2)
            gmm_embedding2 = self.gmmProjection(anchor_gmm_embedding2, relation_embedding2)

            gmm_embedding = torch.cat([gmm_embedding1, gmm_embedding2], dim=1)

            # group distance calculation
            if mode == 'single':
                # [batch_size, 1, group_dimension]
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=positive_sample[:, 0]).unsqueeze(1)
            elif mode == 'tail-batch':
                batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=negative_sample.view(-1)).view(batch_size,
                                                                                                                              negative_sample_size,
                                                                                                                              -1)
            # projection 1
            # [batch_size, 1, group_dimension]
            one_hot_head_1 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 0]).unsqueeze(1)
            relation_matrix_1 = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 1])
            inferred_one_hot_tail_1 = torch.matmul(one_hot_head_1, relation_matrix_1)
            inferred_one_hot_tail_1[inferred_one_hot_tail_1 >= 1] = 1
            # projection 2
            # [batch_size, 1, group_dimension]
            one_hot_head_2 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 2]).unsqueeze(1)
            relation_matrix_2 = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 3])
            inferred_one_hot_tail_2 = torch.matmul(one_hot_head_2, relation_matrix_2)
            inferred_one_hot_tail_2[inferred_one_hot_tail_2 >= 1] = 1
            # union
            inferred_one_hot_tail = inferred_one_hot_tail_1 + inferred_one_hot_tail_2
            inferred_one_hot_tail[inferred_one_hot_tail < 1] = 0
            inferred_one_hot_tail[inferred_one_hot_tail >= 1] = 1
            group_dist = F.relu(one_hot_tail - inferred_one_hot_tail)
            group_dist = torch.norm(group_dist, p=1, dim=-1)

        elif qtype == 'up-DNF':
            entity_embedding1 = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 0])
            relation_embedding1 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 1])
            entity_embedding2 = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 2])
            relation_embedding2 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 3])
            relation_embedding3 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 5])
            entity_mean1, entity_variance1 = torch.chunk(entity_embedding1, 2, dim=-1)
            entity_mean2, entity_variance2 = torch.chunk(entity_embedding2, 2, dim=-1)

            anchor_gmm_embedding1 = self.entityToAnchor(entity_mean1, entity_variance1)
            gmm_embedding1 = self.gmmProjection(anchor_gmm_embedding1, relation_embedding1)
            gmm_embedding1 = self.gmmHigherOrderProjection(gmm_embedding1, relation_embedding3)
            anchor_gmm_embedding2 = self.entityToAnchor(entity_mean2, entity_variance2)
            gmm_embedding2 = self.gmmProjection(anchor_gmm_embedding2, relation_embedding2)
            gmm_embedding2 = self.gmmHigherOrderProjection(gmm_embedding2, relation_embedding3)

            gmm_embedding = torch.cat([gmm_embedding1, gmm_embedding2], dim=1)

            # 既然up已经通过DNF换成2个 2p queries，，那group infor的计算是不是也应该遵循DNF之后的形式？
            # group distance calculation
            if mode == 'single':
                # [batch_size, 1, group_dimension]
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=positive_sample[:, 0]).unsqueeze(1)
            elif mode == 'tail-batch':
                batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=negative_sample.view(-1)).view(batch_size,
                                                                                                                              negative_sample_size,
                                                                                                                              -1)
            # projection 1
            # [batch_size, 1, group_dimension]
            one_hot_head_1 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 0]).unsqueeze(1)
            relation_matrix_1 = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 1])
            # [batch_size, 1, group_dimension]
            inferred_one_hot_tail_1 = torch.matmul(one_hot_head_1, relation_matrix_1)
            inferred_one_hot_tail_1[inferred_one_hot_tail_1 >= 1] = 1
            # projection 2
            # [batch_size, 1, group_dimension]
            one_hot_head_2 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 2]).unsqueeze(1)
            relation_matrix_2 = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 3])
            # [batch_size, 1, group_dimension]
            inferred_one_hot_tail_2 = torch.matmul(one_hot_head_2, relation_matrix_2)
            inferred_one_hot_tail_2[inferred_one_hot_tail_2 >= 1] = 1
            # intersection
            # [batch_size, 1, group_dimension]
            inferred_one_hot_tail = inferred_one_hot_tail_1 + inferred_one_hot_tail_2
            inferred_one_hot_tail[inferred_one_hot_tail < 1] = 0  # u操作的阈值与i操作的阈值不同
            inferred_one_hot_tail[inferred_one_hot_tail >= 1] = 1
            # projection 3
            relation_matrix_3 = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 5])
            # [batch_size, 1, group_dimension]
            inferred_one_hot_tail = torch.matmul(inferred_one_hot_tail, relation_matrix_3)
            inferred_one_hot_tail[inferred_one_hot_tail >= 1] = 1  # 与ip的阈值不同
            inferred_one_hot_tail[inferred_one_hot_tail < 1] = 0
            group_dist = F.relu(one_hot_tail - inferred_one_hot_tail)
            group_dist = torch.norm(group_dist, p=1, dim=-1)

        elif qtype == '2in':
            # group distance calculation
            if mode == 'single':
                # [batch_size, 1, group_dimension]
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=positive_sample[:, 0]).unsqueeze(1)
            elif mode == 'tail-batch':
                batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=negative_sample.view(-1)).view(batch_size,
                                                                                                                              negative_sample_size,
                                                                                                                              -1)
            # projection 1
            # [batch_size, 1, group_dimension]
            one_hot_head_1 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 0]).unsqueeze(1)
            relation_matrix_1 = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 1])
            inferred_one_hot_tail_1 = torch.matmul(one_hot_head_1, relation_matrix_1)
            inferred_one_hot_tail_1[inferred_one_hot_tail_1 >= 1] = 1
            # projection 2
            # [batch_size, 1, group_dimension]
            one_hot_head_2 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 2]).unsqueeze(1)
            relation_matrix_2 = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 3])
            inferred_one_hot_tail_2 = torch.matmul(one_hot_head_2, relation_matrix_2)
            inferred_one_hot_tail_2[inferred_one_hot_tail_2 >= 1] = 1
            # negation for 2
            inferred_one_hot_tail_2[inferred_one_hot_tail_2 == 1] = 2
            inferred_one_hot_tail_2[inferred_one_hot_tail_2 < 1] = 1
            inferred_one_hot_tail_2[inferred_one_hot_tail_2 == 2] = 0
            # intersection
            inferred_one_hot_tail = inferred_one_hot_tail_1 + inferred_one_hot_tail_2
            inferred_one_hot_tail[inferred_one_hot_tail < 2] = 0
            inferred_one_hot_tail[inferred_one_hot_tail >= 2] = 1
            group_dist = F.relu(one_hot_tail - inferred_one_hot_tail)
            group_dist = torch.norm(group_dist, p=1, dim=-1)

            entity_embedding1 = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 0])
            relation_embedding1 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 1])
            entity_embedding2 = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 2])
            relation_embedding2 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 3])
            entity_mean1, entity_variance1 = torch.chunk(entity_embedding1, 2, dim=-1)
            entity_mean2, entity_variance2 = torch.chunk(entity_embedding2, 2, dim=-1)
            anchor_gmm_embedding1 = self.entityToAnchor(entity_mean1, entity_variance1)
            gmm_embedding1 = self.gmmProjection(anchor_gmm_embedding1, relation_embedding1)
            anchor_gmm_embedding2 = self.entityToAnchor(entity_mean2, entity_variance2)
            gmm_embedding2 = self.gmmProjection(anchor_gmm_embedding2, relation_embedding2)
            gmm_embedding2 = self.gmmComplement(gmm_embedding2)

            multiple_gmm_embeddings = torch.stack([gmm_embedding1, gmm_embedding2], dim=1)
            gmm_embedding = self.gmmIntersection(multiple_gmm_embeddings, inferred_one_hot_tail_1, inferred_one_hot_tail_2)

        elif qtype == '3in':
            # group distance calculation
            if mode == 'single':
                # [batch_size, 1, group_dimension]
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=positive_sample[:, 0]).unsqueeze(1)
            elif mode == 'tail-batch':
                batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=negative_sample.view(-1)).view(batch_size,
                                                                                                                              negative_sample_size,
                                                                                                                              -1)
            # projection 1
            # [batch_size, 1, group_dimension]
            one_hot_head_1 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 0]).unsqueeze(1)
            relation_matrix_1 = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 1])
            inferred_one_hot_tail_1 = torch.matmul(one_hot_head_1, relation_matrix_1)
            inferred_one_hot_tail_1[inferred_one_hot_tail_1 >= 1] = 1
            # projection 2
            # [batch_size, 1, group_dimension]
            one_hot_head_2 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 2]).unsqueeze(1)
            relation_matrix_2 = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 3])
            inferred_one_hot_tail_2 = torch.matmul(one_hot_head_2, relation_matrix_2)
            inferred_one_hot_tail_2[inferred_one_hot_tail_2 >= 1] = 1
            # projection 3
            # [batch_size, 1, group_dimension]
            one_hot_head_3 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 4]).unsqueeze(1)
            relation_matrix_3 = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 5])
            inferred_one_hot_tail_3 = torch.matmul(one_hot_head_3, relation_matrix_3)
            inferred_one_hot_tail_3[inferred_one_hot_tail_3 >= 1] = 1
            # negation for 3
            inferred_one_hot_tail_3[inferred_one_hot_tail_3 == 1] = 2
            inferred_one_hot_tail_3[inferred_one_hot_tail_3 < 1] = 1
            inferred_one_hot_tail_3[inferred_one_hot_tail_3 == 2] = 0
            # intersection
            inferred_one_hot_tail = inferred_one_hot_tail_1 + inferred_one_hot_tail_2 + inferred_one_hot_tail_3
            inferred_one_hot_tail[inferred_one_hot_tail < 3] = 0
            inferred_one_hot_tail[inferred_one_hot_tail >= 3] = 1
            group_dist = F.relu(one_hot_tail - inferred_one_hot_tail)
            # [batch_size, 1] / [batch_size, negative_sample_size]
            group_dist = torch.norm(group_dist, p=1, dim=-1)

            entity_embedding1 = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 0])
            relation_embedding1 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 1])
            entity_embedding2 = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 2])
            relation_embedding2 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 3])
            entity_embedding3 = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 4])
            relation_embedding3 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 5])
            entity_mean1, entity_variance1 = torch.chunk(entity_embedding1, 2, dim=-1)
            entity_mean2, entity_variance2 = torch.chunk(entity_embedding2, 2, dim=-1)
            entity_mean3, entity_variance3 = torch.chunk(entity_embedding3, 2, dim=-1)
            anchor_gmm_embedding1 = self.entityToAnchor(entity_mean1, entity_variance1)
            gmm_embedding1 = self.gmmProjection(anchor_gmm_embedding1, relation_embedding1)
            anchor_gmm_embedding2 = self.entityToAnchor(entity_mean2, entity_variance2)
            gmm_embedding2 = self.gmmProjection(anchor_gmm_embedding2, relation_embedding2)
            anchor_gmm_embedding3 = self.entityToAnchor(entity_mean3, entity_variance3)
            gmm_embedding3 = self.gmmProjection(anchor_gmm_embedding3, relation_embedding3)
            gmm_embedding3 = self.gmmComplement(gmm_embedding3)

            multiple_gmm_embeddings = torch.stack([gmm_embedding1, gmm_embedding2, gmm_embedding3], dim=1)
            gmm_embedding = self.gmmIntersection(multiple_gmm_embeddings, inferred_one_hot_tail_1, inferred_one_hot_tail_2, inferred_one_hot_tail_3)

        elif qtype == 'inp':
            # group distance calculation
            if mode == 'single':
                # [batch_size, 1, group_dimension]
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=positive_sample[:, 0]).unsqueeze(1)
            elif mode == 'tail-batch':
                batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=negative_sample.view(-1)).view(batch_size,
                                                                                                                              negative_sample_size,
                                                                                                                              -1)
            # projection 1
            # [batch_size, 1, group_dimension]
            one_hot_head_1 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 0]).unsqueeze(1)
            relation_matrix_1 = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 1])
            inferred_one_hot_tail_1 = torch.matmul(one_hot_head_1, relation_matrix_1)
            inferred_one_hot_tail_1[inferred_one_hot_tail_1 >= 1] = 1
            # projection 2
            # [batch_size, 1, group_dimension]
            one_hot_head_2 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 2]).unsqueeze(1)
            relation_matrix_2 = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 3])
            inferred_one_hot_tail_2 = torch.matmul(one_hot_head_2, relation_matrix_2)
            inferred_one_hot_tail_2[inferred_one_hot_tail_2 >= 1] = 1
            # negation for 2
            inferred_one_hot_tail_2[inferred_one_hot_tail_2 == 1] = 2
            inferred_one_hot_tail_2[inferred_one_hot_tail_2 < 1] = 1
            inferred_one_hot_tail_2[inferred_one_hot_tail_2 == 2] = 0
            # intersection
            inferred_one_hot_tail = inferred_one_hot_tail_1 + inferred_one_hot_tail_2
            inferred_one_hot_tail[inferred_one_hot_tail < 2] = 0
            inferred_one_hot_tail[inferred_one_hot_tail >= 2] = 1
            # projection 3
            relation_matrix_3 = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 5])
            inferred_one_hot_tail = torch.matmul(inferred_one_hot_tail, relation_matrix_3)
            inferred_one_hot_tail[inferred_one_hot_tail >= 0.5] = 1  # 这边阈值可能需要进行调整
            inferred_one_hot_tail[inferred_one_hot_tail < 0.5] = 0
            group_dist = F.relu(one_hot_tail - inferred_one_hot_tail)
            # [batch_size, 1] / [batch_size, negative_sample_size]
            group_dist = torch.norm(group_dist, p=1, dim=-1)

            entity_embedding1 = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 0])
            relation_embedding1 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 1])
            entity_embedding2 = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 2])
            relation_embedding2 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 3])
            relation_embedding3 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 5])
            entity_mean1, entity_variance1 = torch.chunk(entity_embedding1, 2, dim=-1)
            entity_mean2, entity_variance2 = torch.chunk(entity_embedding2, 2, dim=-1)
            anchor_gmm_embedding1 = self.entityToAnchor(entity_mean1, entity_variance1)
            gmm_embedding1 = self.gmmProjection(anchor_gmm_embedding1, relation_embedding1)
            anchor_gmm_embedding2 = self.entityToAnchor(entity_mean2, entity_variance2)
            gmm_embedding2 = self.gmmProjection(anchor_gmm_embedding2, relation_embedding2)
            gmm_embedding2 = self.gmmComplement(gmm_embedding2)

            multiple_gmm_embeddings = torch.stack([gmm_embedding1, gmm_embedding2], dim=1)
            gmm_embedding = self.gmmIntersection(multiple_gmm_embeddings)

            gmm_embedding = self.gmmHigherOrderProjection(gmm_embedding, relation_embedding3)

        elif qtype == 'pni':
            # group distance calculation
            if mode == 'single':
                # [batch_size, 1, group_dimension]
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=positive_sample[:, 0]).unsqueeze(1)
            elif mode == 'tail-batch':
                batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=negative_sample.view(-1)).view(batch_size,
                                                                                                                              negative_sample_size,
                                                                                                                              -1)
            # projection 1
            # [batch_size, 1, group_dimension]
            one_hot_head_1 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 0]).unsqueeze(1)
            for i in range(1, 1 + 2):
                relation_matrix = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, i])
                one_hot_head_1 = torch.matmul(one_hot_head_1, relation_matrix)
            # [batch_size, 1, group_dimension]
            inferred_one_hot_tail_1 = one_hot_head_1
            inferred_one_hot_tail_1[inferred_one_hot_tail_1 >= 1] = 1
            # negation for 1
            inferred_one_hot_tail_1[inferred_one_hot_tail_1 == 1] = 2
            inferred_one_hot_tail_1[inferred_one_hot_tail_1 < 1] = 1
            inferred_one_hot_tail_1[inferred_one_hot_tail_1 == 2] = 0
            # projection 2
            # [batch_size, 1, group_dimension]
            one_hot_head_2 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 4]).unsqueeze(1)
            relation_matrix_2 = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 5])
            inferred_one_hot_tail_2 = torch.matmul(one_hot_head_2, relation_matrix_2)
            inferred_one_hot_tail_2[inferred_one_hot_tail_2 >= 1] = 1
            # intersection
            inferred_one_hot_tail = inferred_one_hot_tail_1 + inferred_one_hot_tail_2
            inferred_one_hot_tail[inferred_one_hot_tail < 2] = 0
            inferred_one_hot_tail[inferred_one_hot_tail >= 2] = 1
            group_dist = F.relu(one_hot_tail - inferred_one_hot_tail)
            # [batch_size, 1] / [batch_size, negative_sample_size]
            group_dist = torch.norm(group_dist, p=1, dim=-1)

            entity_embedding1 = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 0])
            relation_embedding1_1 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 1])
            relation_embedding1_2 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 2])
            entity_embedding2 = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 4])
            relation_embedding2 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 5])
            entity_mean1, entity_variance1 = torch.chunk(entity_embedding1, 2, dim=-1)
            entity_mean2, entity_variance2 = torch.chunk(entity_embedding2, 2, dim=-1)
            anchor_gmm_embedding1 = self.entityToAnchor(entity_mean1, entity_variance1)
            gmm_embedding1 = self.gmmProjection(anchor_gmm_embedding1, relation_embedding1_1)
            gmm_embedding1 = self.gmmHigherOrderProjection(gmm_embedding1, relation_embedding1_2)
            gmm_embedding1 = self.gmmComplement(gmm_embedding1)
            anchor_gmm_embedding2 = self.entityToAnchor(entity_mean2, entity_variance2)
            gmm_embedding2 = self.gmmProjection(anchor_gmm_embedding2, relation_embedding2)

            multiple_gmm_embeddings = torch.stack([gmm_embedding1, gmm_embedding2], dim=1)
            gmm_embedding = self.gmmIntersection(multiple_gmm_embeddings, inferred_one_hot_tail_1, inferred_one_hot_tail_2)

        elif qtype == 'pin':
            # group distance calculation
            if mode == 'single':
                # [batch_size, 1, group_dimension]
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=positive_sample[:, 0]).unsqueeze(1)
            elif mode == 'tail-batch':
                batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
                one_hot_tail = torch.index_select(self.node_group_one_hot_vector, dim=0, index=negative_sample.view(-1)).view(batch_size,
                                                                                                                              negative_sample_size,
                                                                                                                              -1)
            # projection 1
            # [batch_size, 1, group_dimension]
            one_hot_head_1 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 0]).unsqueeze(1)
            for i in range(1, 1 + 2):
                relation_matrix = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, i])
                one_hot_head_1 = torch.matmul(one_hot_head_1, relation_matrix)
            # [batch_size, 1, group_dimension]
            inferred_one_hot_tail_1 = one_hot_head_1
            inferred_one_hot_tail_1[inferred_one_hot_tail_1 >= 1] = 1
            # projection 2
            # [batch_size, 1, group_dimension]
            one_hot_head_2 = torch.index_select(self.node_group_one_hot_vector, dim=0, index=batch_queries[:, 3]).unsqueeze(1)
            relation_matrix_2 = torch.index_select(self.group_adj_matrix, dim=0, index=batch_queries[:, 4])
            inferred_one_hot_tail_2 = torch.matmul(one_hot_head_2, relation_matrix_2)
            inferred_one_hot_tail_2[inferred_one_hot_tail_2 >= 1] = 1
            # negation for 2
            inferred_one_hot_tail_2[inferred_one_hot_tail_2 == 1] = 2
            inferred_one_hot_tail_2[inferred_one_hot_tail_2 < 1] = 1
            inferred_one_hot_tail_2[inferred_one_hot_tail_2 == 2] = 0
            # intersection
            inferred_one_hot_tail = inferred_one_hot_tail_1 + inferred_one_hot_tail_2
            inferred_one_hot_tail[inferred_one_hot_tail < 2] = 0
            inferred_one_hot_tail[inferred_one_hot_tail >= 2] = 1
            group_dist = F.relu(one_hot_tail - inferred_one_hot_tail)
            # [batch_size, 1] / [batch_size, negative_sample_size]
            group_dist = torch.norm(group_dist, p=1, dim=-1)

            entity_embedding1 = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 0])
            relation_embedding1_1 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 1])
            relation_embedding1_2 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 2])
            entity_embedding2 = torch.index_select(self.entity_embedding, dim=0, index=batch_queries[:, 3])
            relation_embedding2 = torch.index_select(self.relation_transition_embedding, dim=0, index=batch_queries[:, 4])
            entity_mean1, entity_variance1 = torch.chunk(entity_embedding1, 2, dim=-1)
            entity_mean2, entity_variance2 = torch.chunk(entity_embedding2, 2, dim=-1)
            anchor_gmm_embedding1 = self.entityToAnchor(entity_mean1, entity_variance1)
            gmm_embedding1 = self.gmmProjection(anchor_gmm_embedding1, relation_embedding1_1)
            gmm_embedding1 = self.gmmHigherOrderProjection(gmm_embedding1, relation_embedding1_2)
            anchor_gmm_embedding2 = self.entityToAnchor(entity_mean2, entity_variance2)
            gmm_embedding2 = self.gmmProjection(anchor_gmm_embedding2, relation_embedding2)
            gmm_embedding2 = self.gmmComplement(gmm_embedding2)

            multiple_gmm_embeddings = torch.stack([gmm_embedding1, gmm_embedding2], dim=1)
            gmm_embedding = self.gmmIntersection(multiple_gmm_embeddings, inferred_one_hot_tail_1, inferred_one_hot_tail_2)

        else:
            raise ValueError('query type %s not supported' % qtype)

        if inverted:
            assert (qtype in ['1p', '2p', '3p', '2i', '3i'])
            gmm_embedding = self.gmmComplement(gmm_embedding)

        # [batch_size, num_gaussian_component, hidden_dim]
        gmm_weight, gmm_mean, gmm_variance = torch.chunk(gmm_embedding, 3, dim=-1)
        # [batch_size, num_gaussian_component, hidden_dim]
        normalized_gmm_weight = nn.Softmax(dim=1)(gmm_weight)
        # [nentity, hidden_dim]
        entity_mean, entity_variance = torch.chunk(self.entity_embedding, 2, dim=-1)

        # 用最大内积近似欧氏空间的二范数的
        # [batch_size, num_gaussian_component, nentity]
        mean_similarity = torch.matmul(gmm_mean, entity_mean.T)
        variance_similarity = torch.matmul(gmm_variance, entity_variance.T)
        # [batch_size, nentity]
        total_similarity = torch.sum(mean_similarity + variance_similarity, dim=1)
        weight_score = torch.norm(torch.sum(torch.log(normalized_gmm_weight), dim=1), p=1, dim=-1).unsqueeze(1)
        # [batch_size, nentity]
        prediction_scores = weight_score + total_similarity

        group_score = self.gamma.item() - self.group_adj_weight * group_dist
        return prediction_scores, group_score


    @staticmethod
    def train_step(model, iter, optimizer, use_apex):
        model.train()
        optimizer.zero_grad()
        positive_sample, _, subsampling_weight, batch_queries, query_structures = next(iter)
        positive_sample = torch.tensor(positive_sample).cuda()
        batch_queries = torch.tensor(batch_queries).cuda()
        qtype = query_name_dict[query_structures[0]]

        _p = random.random()
        if qtype == "1p":
            if _p < 0.3333:
                qtype = "2u-arg"

        try:
            label_smoothing = model.label_smoothing
        except:
            label_smoothing = model.module.label_smoothing

        # [batch_size, nentities] [batch_size, 1]
        prediction_scores, group_score = model(batch_queries, qtype, positive_sample=positive_sample)
        loss_fct = LabelSmoothingLoss(smoothing=label_smoothing, reduction='none')
        # [batch_size]
        masked_lm_loss = loss_fct(prediction_scores, positive_sample.view(-1))
        # [batch_size]
        group_score = F.logsigmoid(group_score).squeeze(dim=1)
        # [] 单值
        group_loss = -group_score.mean()
        masked_lm_loss = (masked_lm_loss).mean()
        loss = (masked_lm_loss + group_loss) / 2

        if use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
                optimizer.step()
        else:
            loss.backward()
            optimizer.step()

        log = {
            'qtype': qtype,
            'loss': loss.item(),
        }

        return log

    @staticmethod
    def test_step(model, negative_sample, queries, queries_unflatten, query_structures, easy_answers, hard_answers):
        model.eval()
        queries = torch.tensor(queries).cuda()
        negative_sample = torch.tensor(negative_sample).cuda()
        qtype = query_name_dict[query_structures[0]]
        easy_answer = easy_answers[queries_unflatten[0]]
        hard_answer = hard_answers[queries_unflatten[0]]
        all_answer = easy_answer.union(hard_answer)

        if len(list(hard_answer)) == 0:
            logs = {
                "mr": 30000,
                "mrr": 0,
                "hit_at_1": 0,
                "hit_at_2": 0,
                "hit_at_3": 0,
                "hit_at_5": 0,
                "hit_at_10": 0,
                "num_samples": 0
            }
            return logs

        hard_answer_ids = torch.tensor(list(hard_answer))
        easy_answer_ids = torch.tensor(list(easy_answer))
        all_answer_ids = torch.tensor(list(all_answer))

        # prediction_scores: 未进行归一化的[batch_size, nentity]: 每一个entity与query的最大内积相似度
        # group_scores: 未进行归一化的[batch_size, nentity]: 每一个entity与query的gamma - group distance;因为有负号->每一个entity与query的group相似性
        prediction_scores, group_scores = model(queries, qtype, mode='tail-batch', negative_sample=negative_sample)
        # group_scores = F.logsigmoid(group_scores, dim=-1)
        prediction_scores += group_scores

        # [nentities]
        prediction_scores = prediction_scores.squeeze()
        original_scores = prediction_scores.clone()
        # [nentities]
        not_answer_scores = prediction_scores
        not_answer_scores[all_answer_ids] = - 10000000
        # [1, nentities]
        not_answer_scores = not_answer_scores.unsqueeze(0)
        # [num_hard_answers]
        hard_answer_scores = original_scores[hard_answer_ids]
        # [num_hard_answers, 1]
        hard_answer_scores = hard_answer_scores.unsqueeze(-1)
        answer_is_smaller_matrix = ((hard_answer_scores - not_answer_scores) < 0)

        hard_answer_rankings = answer_is_smaller_matrix.sum(dim=-1) + 1

        rankings = hard_answer_rankings.float()

        mr = torch.mean(rankings).cpu().numpy()
        mrr = torch.mean(torch.reciprocal(rankings)).cpu().numpy()
        hit_at_1 = torch.mean((rankings < 1.5).double()).cpu().numpy()
        hit_at_3 = torch.mean((rankings < 3.5).double()).cpu().numpy()
        hit_at_10 = torch.mean((rankings < 10.5).double()).cpu().numpy()

        # The best scores are implemented and compared with EmQL paper:
        # https://github.com/google-research/language/blob/master/language/emql/eval.py line 132
        # Which is the traditional definition of Hit@N
        mr_best = torch.min(rankings).cpu().numpy()
        mrr_best = torch.max(torch.reciprocal(rankings)).cpu().numpy()
        hit_at_1_best = torch.max((rankings < 1.5).double()).cpu().numpy()
        hit_at_3_best = torch.max((rankings < 3.5).double()).cpu().numpy()
        hit_at_10_best = torch.max((rankings < 10.5).double()).cpu().numpy()

        num_samples = len(all_answer)

        logs_new = {
            "mr": mr,
            "mrr": mrr,
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            "hit_at_10": hit_at_10,
            "num_samples": 1.0,
            "new_num_samples": num_samples,
        }

        logs_tradition = {
            "mr": mr_best,
            "mrr": mrr_best,
            "hit_at_1": hit_at_1_best,
            "hit_at_3": hit_at_3_best,
            "hit_at_10": hit_at_10_best,
            "num_samples": 1.0,
            "new_num_samples": num_samples,
        }

        return logs_new, logs_tradition




