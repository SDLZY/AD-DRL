'''

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from module import Module

class FactorAttributeNceNegBrand(Module):
    def __init__(self,
                 n_users,
                 n_items,
                 num_neg=4,
                 n_factors=4,
                 embed_dim=20,
                 decay_r=1e-4,
                 decay_f=1e-3,
                 decay_a=1e-3,
                 decay_n=1e-3,
                 hidden_layer_dim_a=256,
                 hidden_layer_dim_b=256,
                 dropout_rate_a=0.2,
                 dropout_rate_b=0.2,
                 temp=1,
                 dataset_name='Sports'
                 ):
        super(FactorAttributeNceNegBrand, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_factor = n_factors
        self.embed_dim = embed_dim
        self.num_neg = num_neg
        self.decay_r = decay_r
        self.decay_f = decay_f
        self.decay_a = decay_a
        self.decay_n = decay_n
        self.hidden_layer_dim_a = hidden_layer_dim_a
        self.hidden_layer_dim_b = hidden_layer_dim_b
        self.dropout_rate_a = dropout_rate_a
        self.dropout_rate_b = dropout_rate_b
        self.dataset_name = dataset_name

        self.user_embedding = nn.Embedding(self.n_users, self.embed_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embed_dim)
        if self.dataset_name == 'ToysGames':
            self.textual_mlp_2 = torch.nn.Linear(512, self.embed_dim)
        else:
            self.textual_mlp_2 = torch.nn.Linear(1024, self.embed_dim)
        self.visual_mlp_2 = torch.nn.Linear(1024, self.embed_dim)

        self.modality_attention_net1 = torch.nn.Sequential(
            torch.nn.Linear(int(self.embed_dim / self.n_factor), 3),
            torch.nn.Tanh(),
            torch.nn.Linear(3, 3),
            nn.Softmax(dim=1)
        )

        # Adjustments are made based on the attribute values of different datasets, as detailed in Table 1 of the paper.
        self.price_mlp = torch.nn.Sequential(
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
            torch.nn.Dropout(0.2, inplace=False),
            torch.nn.Linear(int(self.embed_dim / self.n_factor), 5),
        )
        self.salesrank_mlp = torch.nn.Sequential(
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
            torch.nn.Dropout(0.2, inplace=False),
            torch.nn.Linear(int(self.embed_dim / self.n_factor), 5),
        )
        if self.dataset_name == 'ToysGames':
            self.category_mlp = torch.nn.Sequential(
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Dropout(0.2, inplace=False),
                torch.nn.Linear(int(self.embed_dim / self.n_factor), 19),
            )
            self.brand_mlp = torch.nn.Sequential(
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Dropout(0.2, inplace=False),
                torch.nn.Linear(int(self.embed_dim / self.n_factor), 1288),
            )
        elif self.dataset_name == 'Sports':
            self.category_mlp = torch.nn.Sequential(
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Dropout(0.2, inplace=False),
                torch.nn.Linear(int(self.embed_dim / self.n_factor), 18),
            )
            self.brand_mlp = torch.nn.Sequential(
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Dropout(0.2, inplace=False),
                torch.nn.Linear(int(self.embed_dim / self.n_factor), 2081),
            )
        elif self.dataset_name == 'Baby':
            self.brand_mlp = torch.nn.Sequential(
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Dropout(0.2, inplace=False),
                torch.nn.Linear(int(self.embed_dim / self.n_factor), 663),
            )
        else:
            pass

        # Since the Baby dataset in the Amazon is no longer subdivided into categories, the number of factors for Baby is 4, while the number of factors for other datasets is 5.
        if self.dataset_name == 'Baby':
            self.user_mlp = torch.nn.Sequential(
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Dropout(0.2, inplace=False),
                torch.nn.Linear(int(self.embed_dim / self.n_factor), 4),
            )
            self.id_mlp = torch.nn.Sequential(
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Dropout(0.2, inplace=False),
                torch.nn.Linear(int(self.embed_dim / self.n_factor), 4),
            )
            self.text_mlp = torch.nn.Sequential(
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Dropout(0.2, inplace=False),
                torch.nn.Linear(int(self.embed_dim / self.n_factor), 4),
            )
            self.image_mlp = torch.nn.Sequential(
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Dropout(0.2, inplace=False),
                torch.nn.Linear(int(self.embed_dim / self.n_factor), 4),
            )
        else:
            self.user_mlp = torch.nn.Sequential(
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Dropout(0.2, inplace=False),
                torch.nn.Linear(int(self.embed_dim / self.n_factor), 5),
            )
            self.id_mlp = torch.nn.Sequential(
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Dropout(0.2, inplace=False),
                torch.nn.Linear(int(self.embed_dim / self.n_factor), 5),
            )
            self.text_mlp = torch.nn.Sequential(
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Dropout(0.2, inplace=False),
                torch.nn.Linear(int(self.embed_dim / self.n_factor), 5),
            )
            self.image_mlp = torch.nn.Sequential(
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
                torch.nn.Dropout(0.2, inplace=False),
                torch.nn.Linear(int(self.embed_dim / self.n_factor), 5),
            )
        self.T = temp
        self.ce = nn.CrossEntropyLoss()

    def feature_projection_textual(self, textual_feature):
        textual_feature = F.normalize(textual_feature, p=2, dim=1)
        output = self.textual_mlp_2(textual_feature)
        return output

    def feature_projection_visual(self, visual_feature):
        visual_feature = F.normalize(visual_feature, p=2, dim=1)
        output = self.visual_mlp_2(visual_feature)
        return output

    def _create_weight(self, user, item, textual, visual):
        input_ = item + textual + visual
        input = F.normalize(input_, p=2, dim=1)
        output = self.modality_attention_net1(input)
        output = F.softmax(output, dim=1)
        return output

    def high_intra(self, user, item, textual, visual):
        batch_size = item[0].shape[0]
        user_loss, id_loss, textual_loss, visual_loss = 0, 0, 0, 0
        for i in range(len(item)):
            user_logits = self.user_mlp(user[i])
            user_loss += self.ce(user_logits, torch.ones(batch_size, dtype=torch.int64).fill_(i).cuda())

            id_logits = self.id_mlp(item[i])
            id_loss += self.ce(id_logits, torch.ones(batch_size, dtype=torch.int64).fill_(i).cuda())

            textual_logits = self.text_mlp(textual[i])
            textual_loss += self.ce(textual_logits, torch.ones(batch_size, dtype=torch.int64).fill_(i).cuda())

            visual_logits = self.image_mlp(visual[i])
            visual_loss += self.ce(visual_logits, torch.ones(batch_size, dtype=torch.int64).fill_(i).cuda())
        return user_loss + id_loss + textual_loss + visual_loss

    def low_level(self,
                             fusion_factor,
                             price_label,
                             salesrank_label,
                             brand_label,
                             category_label
                             ):
        price_logits = self.price_mlp(fusion_factor[0])
        salesrank_logits = self.salesrank_mlp(fusion_factor[1])
        brand_logits = self.brand_mlp(fusion_factor[2])
        if self.dataset_name != 'Baby':
            category_logits = self.category_mlp(fusion_factor[3])

        price_loss = F.cross_entropy(price_logits, price_label)
        salesrank_loss = F.cross_entropy(salesrank_logits, salesrank_label)
        brand_loss = F.cross_entropy(brand_logits, brand_label)
        if self.dataset_name != 'Baby':
            category_loss = F.cross_entropy(category_logits, category_label)
        if self.dataset_name == 'Baby':
            loss = price_loss + salesrank_loss + brand_loss
        else:
            loss = price_loss + salesrank_loss + brand_loss + category_loss

        return loss

    def attribute_nce(self,
                      feats_t,
                      feats_s
                      ):
        batch_size = feats_t[0].shape[0]
        feats_t_unsqueeze = [feats.unsqueeze(1) for feats in feats_t]
        feats_t_matrix = torch.cat(feats_t_unsqueeze, 1)
        loss_s = 0
        for i in range(0, self.n_factor):
            logits_s = torch.bmm(feats_t_matrix, feats_s[i].unsqueeze(-1)).squeeze(-1)
            loss_s += self.ce(logits_s / self.T, torch.ones(batch_size, dtype=torch.int64).fill_(i).cuda())

        feats_s_unsqueeze = [feats.unsqueeze(1) for feats in feats_s]
        feats_s_matrix = torch.cat(feats_s_unsqueeze, 1)
        loss_t = 0
        for i in range(0, self.n_factor):
            logits_t = torch.bmm(feats_s_matrix, feats_t[i].unsqueeze(-1)).squeeze(-1)
            loss_t += self.ce(logits_t / self.T, torch.ones(batch_size, dtype=torch.int64).fill_(i).cuda())

        return loss_t + loss_s

    def high_inter(self,
                     item_factor_embedding_p,
                     textual_factor_embedding_p,
                     visual_factor_embedding_p
                     ):
        loss_it = self.attribute_nce(item_factor_embedding_p, textual_factor_embedding_p)
        loss_iv = self.attribute_nce(item_factor_embedding_p, visual_factor_embedding_p)
        loss_tv = self.attribute_nce(textual_factor_embedding_p, visual_factor_embedding_p)

        loss_nce = loss_it + loss_iv + loss_tv
        return loss_nce

    def train_forward(self,
                      user_positive_items_pairs,
                      negative_samples,
                      textual_feature_pos,
                      visual_feature_pos,
                      textual_feature_neg,
                      visual_feature_neg,
                      price_label,
                      salesrank_label,
                      brand_label,
                      category_label,
                      price_label_neg,
                      salesrank_label_neg,
                      brand_label_neg,
                      category_label_neg,
                      ):
        batch_size = user_positive_items_pairs.shape[0]
        users = self.user_embedding(user_positive_items_pairs[:, 0])
        pos_items = self.item_embedding(user_positive_items_pairs[:, 1])
        pos_i_t = self.feature_projection_textual(textual_feature_pos)
        pos_i_v = self.feature_projection_visual(visual_feature_pos)

        # negative item embedding (N, K)
        neg_items = self.item_embedding(negative_samples).view(-1, self.embed_dim)
        neg_i_t = self.feature_projection_textual(textual_feature_neg.view(-1, textual_feature_neg.shape[-1]))
        neg_i_v = self.feature_projection_visual(visual_feature_neg.view(-1, visual_feature_neg.shape[-1]))

        textual_f = torch.cat((pos_i_t, neg_i_t), dim=0)
        visual_f = torch.cat((pos_i_v, neg_i_v), dim=0)
        user_n = users.unsqueeze(1).repeat(1, self.num_neg, 1).view(-1, self.embed_dim)
        factor_emb_dim = int(self.embed_dim / self.n_factor)
        textual_factor_embedding = torch.split(textual_f, factor_emb_dim, dim=1)
        textual_factor_embedding_p = torch.split(pos_i_t, factor_emb_dim, dim=1)
        textual_factor_embedding_n = torch.split(neg_i_t, factor_emb_dim, dim=1)
        visual_factor_embedding = torch.split(visual_f, factor_emb_dim, dim=1)
        visual_factor_embedding_p = torch.split(pos_i_v, factor_emb_dim, dim=1)
        visual_factor_embedding_n = torch.split(neg_i_v, factor_emb_dim, dim=1)

        user_factor_embedding_ap = torch.split(users, factor_emb_dim, dim=1)
        user_factor_embedding_an = torch.split(user_n, factor_emb_dim, dim=1)
        p_item_factor_embedding = torch.split(pos_items, factor_emb_dim, dim=1)
        n_item_factor_embedding = torch.split(neg_items, factor_emb_dim, dim=1)

        intra_loss = self.high_intra(user_factor_embedding_ap,
                                             p_item_factor_embedding,
                                             textual_factor_embedding_p,
                                             visual_factor_embedding_p)
        intra_loss_neg = self.high_intra(user_factor_embedding_an,
                                                 n_item_factor_embedding,
                                                 textual_factor_embedding_n,
                                                 visual_factor_embedding_n)

        regularizer = 0

        pos_scores, neg_scores = [], []
        fusion_factor = []
        fusion_factor_neg = []
        for i in range(0, self.n_factor):
            textual_trans = textual_factor_embedding[i]
            p_textual_trans, n_textual_trans = torch.split(textual_trans, [batch_size, self.num_neg * batch_size], dim=0)
            visual_trans = visual_factor_embedding[i]
            p_visual_trans, n_visual_trans = torch.split(visual_trans, [batch_size, self.num_neg * batch_size], dim=0)

            p_weights = self._create_weight(user_factor_embedding_ap[i], p_item_factor_embedding[i], p_textual_trans, p_visual_trans)
            p_fusion = p_weights[:, 0].unsqueeze(1) * F.normalize(p_item_factor_embedding[i], p=2, dim=1) + p_weights[:, 1].unsqueeze(1) * F.normalize(p_textual_trans, p=2, dim=1) + p_weights[:, 2].unsqueeze(1) * F.normalize(p_visual_trans, p=2, dim=1)
            fusion_factor.append(p_fusion)
            p_score = F.softplus(torch.sum(F.normalize(user_factor_embedding_ap[i], p=2, dim=1) * p_fusion, 1))
            pos_scores.append(p_score.unsqueeze(1))

            n_weights = self._create_weight(user_factor_embedding_an[i], n_item_factor_embedding[i], n_textual_trans, n_visual_trans)
            n_fusion = n_weights[:, 0].unsqueeze(1) * F.normalize(n_item_factor_embedding[i], p=2, dim=1) + n_weights[:, 1].unsqueeze(1) * F.normalize(n_textual_trans, p=2, dim=1) + n_weights[:, 2].unsqueeze(1) * F.normalize(n_visual_trans, p=2, dim=1)
            fusion_factor_neg.append(n_fusion)
            n_score = F.softplus(torch.sum(F.normalize(user_factor_embedding_an[i], p=2, dim=1) * n_fusion, 1))
            neg_scores.append(n_score.unsqueeze(1))
        pos_s = torch.cat(pos_scores, dim=1)
        neg_s = torch.cat(neg_scores, dim=1)

        inter_loss = self.high_inter(
            p_item_factor_embedding,
            textual_factor_embedding_p,
            visual_factor_embedding_p
        )
        inter_loss_neg = self.high_inter(
            n_item_factor_embedding,
            textual_factor_embedding_n,
            visual_factor_embedding_n
        )

        low_loss = self.low_level(fusion_factor,
                                                   price_label,
                                                   salesrank_label,
                                                   brand_label,
                                                   category_label)
        low_loss_neg = self.low_level(fusion_factor_neg,
                                                       price_label_neg.view(-1),
                                                       salesrank_label_neg.view(-1),
                                                       brand_label_neg.view(-1),
                                                       category_label_neg.view(-1))



        regularizer += torch.norm(users) ** 2 / 2 + torch.norm(pos_items) ** 2 / 2 + torch.norm(neg_items) ** 2 / 2 \
                      + torch.norm(pos_i_t) ** 2 / 2 + torch.norm(neg_i_t) ** 2 / 2 + torch.norm(pos_i_v) ** 2 / 2 \
                      + torch.norm(neg_i_v) ** 2 / 2
        regularizer = regularizer / batch_size

        pos_score = torch.sum(pos_s, 1)
        negtive_score, _ = torch.max(torch.sum(neg_s, 1).view(batch_size, self.num_neg), 1)

        loss_per_pair = F.softplus(-(pos_score - negtive_score))
        loss = torch.sum(loss_per_pair)

        return loss + self.decay_r * regularizer + self.decay_f * (intra_loss + intra_loss_neg) + self.decay_n * (inter_loss + inter_loss_neg) + self.decay_a * (low_loss + low_loss_neg)


    def inference_forward(self,
                          user_ids,
                          item_ids,
                          textualfeatures,
                          imagefeatures):
        # (N_USER_IDS, 1, K)
        users = self.user_embedding(user_ids).unsqueeze(1)

        # (1, N_ITEM, K)
        items = self.item_embedding(item_ids).unsqueeze(0)
        textual = self.feature_projection_textual(textualfeatures).unsqueeze(0)
        visual = self.feature_projection_visual(imagefeatures).unsqueeze(0)

        item_expand = items.repeat(users.shape[0], 1, 1).view(-1, self.embed_dim)
        textual_expand = textual.repeat(users.shape[0], 1, 1).view(-1, self.embed_dim)
        visual_expand = visual.repeat(users.shape[0], 1, 1).view(-1, self.embed_dim)
        users_expand = users.repeat(1, self.n_items, 1).view(-1, self.embed_dim)

        factor_emb_dim = int(self.embed_dim / self.n_factor)
        user_expad_factor_embedding = torch.split(users_expand, factor_emb_dim, dim=1)
        item_expand_factor_embedding = torch.split(item_expand, factor_emb_dim, dim=1)
        textual_expand_factor_embedding = torch.split(textual_expand, factor_emb_dim, dim=1)
        visual_expand_factor_embedding = torch.split(visual_expand, factor_emb_dim, dim=1)

        factor_scores = []
        factor_sc = []
        factor_ws = []
        for i in range(0, self.n_factor):
            textual_trans = textual_expand_factor_embedding[i]
            visual_trans = visual_expand_factor_embedding[i]

            f_weights = self._create_weight(user_expad_factor_embedding[i], item_expand_factor_embedding[i], textual_trans, visual_trans)
            f_fusion = f_weights[:, 0].unsqueeze(1) * F.normalize(item_expand_factor_embedding[i], p=2, dim=1) + f_weights[:, 1].unsqueeze(1) * F.normalize(textual_trans, p=2, dim=1) + f_weights[:, 2].unsqueeze(1) * F.normalize(visual_trans, p=2, dim=1)
            f_score = F.softplus(torch.sum(F.normalize(user_expad_factor_embedding[i], p=2, dim=1) * f_fusion, 1))
            factor_scores.append(f_score.unsqueeze(1))

        factor_s = torch.cat(factor_scores, dim=1)
        scores = torch.sum(factor_s, 1).view(users.shape[0], -1)
        return scores, factor_sc, factor_ws
