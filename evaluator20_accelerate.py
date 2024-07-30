import math
import heapq
import toolz
import numpy as np
# import tensorflow as tf
from scipy.sparse import lil_matrix
from sklearn.metrics import roc_auc_score
from utils import to_cuda
import torch
from time import time

class RecallEvaluator(object):
    def __init__(self, model, train_dataset, test_dataset):
        """
        Create a evaluator for recall@K evaluation
        :param model: the model we are going to evaluate
        :param train_user_item_matrix: the user-item pairs used in the training set. These pairs will be ignored
               in the recall calculation
        :param test_user_item_matrix: the held-out user-item pairs we make prediction against
        """
        # self.model = model TODO: model还是只在应用eval的时候进行定义吧，这样避免RecallEvaluator中的model和train中的model不一致
        self.textualfeatures = test_dataset.textualfeatures
        self.imagefeatures = test_dataset.imagefeatures
        train_user_item_matrix = train_dataset.dataMatrix
        test_user_item_matrix = test_dataset.dataMatrix
        self.train_user_item_matrix = lil_matrix(train_user_item_matrix)
        self.test_user_item_matrix = lil_matrix(test_user_item_matrix)
        self.test_user_item_matrix_dox = test_user_item_matrix

        self.n_users = max(train_user_item_matrix.shape[0],test_user_item_matrix.shape[0])
        self.n_items = max(train_user_item_matrix.shape[1],test_user_item_matrix.shape[1])
        self.items = [i for i in range(self.n_items)]

        self.user_to_test_set = {u: set(self.test_user_item_matrix.rows[u])
                                 for u in range(test_user_item_matrix.shape[0]) if self.test_user_item_matrix.rows[u]}

        if self.train_user_item_matrix is not None:
            self.user_to_train_set = {u: set(self.train_user_item_matrix.rows[u])
                                      for u in range(train_user_item_matrix.shape[0]) if self.train_user_item_matrix.rows[u]}
            self.max_train_count = max(len(row) for row in self.train_user_item_matrix.rows)
        else:
            self.max_train_count = 0

    def eval(self, model):
        test_recalls = []
        test_ndcg = []
        test_hr = []
        test_pr = []
        test_users = np.asarray(list(set(self.test_user_item_matrix_dox.nonzero()[0])), dtype=np.int64) # TODO: 如果某一个user在测试集中存在不止一次，那可能对其他user不公平
        for user_chunk in toolz.partition_all(20, test_users):
            recalls, ndcgs, hit_ratios, precisions = self.eval_batch(model, user_chunk)
            test_recalls.extend(recalls)
            test_ndcg.extend(ndcgs)
            test_hr.extend(hit_ratios)
            test_pr.extend(precisions)
        recalls = sum(test_recalls) / float(len(test_recalls))
        precisions = sum(test_pr) / float(len(test_pr))
        hit_ratios = sum(test_hr) / float(len(test_hr))
        ndcgs = sum(test_ndcg) / float(len(test_ndcg))
        return recalls, precisions, hit_ratios, ndcgs

    def eval_batch(self, model, users):
        """
        Compute the Top-K recall for a particular user given the predicted scores to items
        :param users: the users to eval the recall
        :param k: compute the recall for the top K items
        :return: hitratio,ndgg@K
        """
        recalls = []
        precisions = []
        hit_ratios = []
        ndcgs = []
        model.eval()

        batch = [torch.tensor(users), torch.tensor(self.items), torch.tensor(self.textualfeatures), torch.tensor(self.imagefeatures)]
        batch = to_cuda(batch)
        user_tops, scores_s, scores_w = model(*batch)

        user_pos_tests, rs = self.batch_rating(users, user_tops.detach().cpu())
        for index in range(len(user_pos_tests)):
            result = self.get_performance(user_pos_tests[index], rs[index])
            recalls.append(result['recall'])
            precisions.append(result['precision'])
            hit_ratios.append(result['hit_ratio'])
            ndcgs.append(result['ndcg'])
        return recalls, ndcgs, hit_ratios, precisions

    def batch_rating(self, users, rating):
        allPos, groundTrue = [], []
        for u in users:
            allPos.append(self.user_to_train_set.get(u, set()))
            groundTrue.append(self.user_to_test_set.get(u, set()))
        exclude_index, exclude_items = [], []
        for range_i, items in enumerate(allPos):
            exclude_index.extend([range_i] * len(items))
            exclude_items.extend(items)
        rating[exclude_index, exclude_items] = -(1 << 10)
        _, rating_K = torch.topk(rating, k=20)

        r = []
        for i in range(len(groundTrue)):
            ground = groundTrue[i]
            predictTopK = rating_K[i]
            pred = list(map(lambda x: x in ground, predictTopK.numpy()))
            pred = np.array(pred).astype("float")
            pred = pred.tolist()
            r.append(pred)
        return groundTrue, r

    def get_performance(self, user_pos_test, r):
        K = 20
        precision=self.precision_at_k(r, K)
        recall=self.recall_at_k(r, K, len(user_pos_test))
        ndcg=self.ndcg_at_k(r, K)
        hit_ratio=self.hit_at_k(r, K)
        return {'recall': np.array(recall), 'precision': np.array(precision),
                'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio)}

    def precision_at_k(self, r, k):
        """Score is precision @ k
        Relevance is binary (nonzero is relevant).
        Returns:
            Precision @ k
        Raises:
            ValueError: len(r) must be >= k
        """
        assert k >= 1
        r = np.asarray(r)[:k]
        return np.mean(r)

    def dcg_at_k(self, r, k, method=1):
        """Score is discounted cumulative gain (dcg)
        Relevance is positive real values.  Can use binary
        as the previous methods.
        Returns:
            Discounted cumulative gain
        """
        r = np.asfarray(r)[:k]
        if r.size:
            if method == 0:
                return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
            elif method == 1:
                return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            else:
                raise ValueError('method must be 0 or 1.')
        return 0.

    def ndcg_at_k(self, r, k, method=1):
        """Score is normalized discounted cumulative gain (ndcg)
        Relevance is positive real values.  Can use binary
        as the previous methods.
        Returns:
            Normalized discounted cumulative gain
        """
        dcg_max = self.dcg_at_k(sorted(r, reverse=True), k, method)
        if not dcg_max:
            return 0.
        return self.dcg_at_k(r, k, method) / dcg_max

    def recall_at_k(self, r, k, all_pos_num):
        r = np.asfarray(r)[:k]
        return np.sum(r) / all_pos_num

    def hit_at_k(self, r, k):
        r = np.array(r)[:k]
        if np.sum(r) > 0:
            return 1.
        else:
            return 0.