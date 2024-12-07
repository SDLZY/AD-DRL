'''
Processing datasets.
'''
import scipy.sparse as sp
from scipy.sparse import lil_matrix
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy


class AmazonDataset(Dataset):
    def __init__(self, path, split, dataset_name, attribute_dataset, n_negative=4, check_negative=True):
        '''
        Amazon Dataset
        :param path: the path of the Dataset
        '''
        super(AmazonDataset, self).__init__()
        self.dataset_name = dataset_name
        self.attribute_dataset = attribute_dataset

        self.dataMatrix, self.data_num = self.load_rating_file_as_matrix(path + f"/{split}.csv")
        self.textualfeatures, self.imagefeatures, = self.load_features(path)

        self.get_pairs(self.dataMatrix)
        self.n_negative = n_negative
        self.check_negative = check_negative
        self.load_attributes_label(path)


    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items, num_total = 0, 0, 0
        df = pd.read_csv(filename, index_col=None, usecols=None)
        for index, row in df.iterrows():
            u, i = int(row['userID']), int(row['itemID'])
            num_users = max(num_users, u)
            num_items = max(num_items, i)
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        for index, row in df.iterrows():
            user, item, rating = int(row['userID']), int(row['itemID']), 1.0
            if (rating > 0):
                mat[user, item] = 1.0
                num_total += 1
        return mat, num_total

    def load_features(self,data_path):
        import os
        # Prepare textual feture data.
        doc2vec_model = np.load(os.path.join(data_path, 'review.npz'), allow_pickle=True)['arr_0'].item()
        vis_vec = np.load(os.path.join(data_path, 'image_feature_ViT.npy'), allow_pickle=True).item()

        filename = data_path + '/train.csv'
        filename_test =  data_path + '/test.csv'
        df = pd.read_csv(filename, index_col=None, usecols=None)
        df_test = pd.read_csv(filename_test, index_col=None, usecols=None)
        num_items = 0
        self.asin_i_dic = {}
        for index, row in df.iterrows():
            asin, i = row['asin'], int(row['itemID'])
            self.asin_i_dic[i] = asin
            num_items = max(num_items, i)
        for index, row in df_test.iterrows():
            asin, i = row['asin'], int(row['itemID'])
            self.asin_i_dic[i] = asin
            num_items = max(num_items, i)

        features = []
        image_features = []
        if self.dataset_name == 'ToysGames':
            for i in range(num_items + 1):
                if self.asin_i_dic[i] not in doc2vec_model:
                    features.append(np.zeros(512))
                else:
                    features.append(doc2vec_model[self.asin_i_dic[i]])
                if self.asin_i_dic[i] not in vis_vec:
                    image_features.append(np.zeros(1024))
                else:
                    image_features.append(np.asarray(vis_vec[self.asin_i_dic[i]]))
        else:
            for i in range(num_items+1):
                if self.asin_i_dic[i] not in doc2vec_model:
                    features.append(np.zeros(1024))
                else:
                    features.append(doc2vec_model[self.asin_i_dic[i]][0])
                if self.asin_i_dic[i] not in vis_vec:
                    image_features.append(np.zeros(1024))
                else:
                    image_features.append(vis_vec[self.asin_i_dic[i]])
        return np.asarray(features,dtype=np.float32),np.asarray(image_features,dtype=np.float32)

    def load_attributes_label(self, data_path):
        self.attribute_label = {}
        self.i_asin_dic = {value: key for key, value in self.asin_i_dic.items()}
        attribute_file = data_path + '/' + self.attribute_dataset + '.csv'

        attribute_df = pd.read_csv(attribute_file)
        for index, row in attribute_df.iterrows():
            if self.dataset_name == 'ToysGames':
                if row['asin'] in self.i_asin_dic:
                    item_id = self.i_asin_dic[row['asin']]
                else:
                    # print(row['asin'] + 'attribute')
                    continue
            else:
                item_id = self.i_asin_dic[row['asin']]
            self.attribute_label[item_id] = {'price_label': row['price_label'], 'salesrank_label': row['salesrank_label'],
                                             'brand_label': row['brand_label'], 'category_label': row['category_label']}

    def get_pairs(self, user_item_matrix):
        self.user_item_matrix = lil_matrix(user_item_matrix)
        self.user_item_pairs = np.asarray(self.user_item_matrix.nonzero()).T
        self.user_item_pairs = self.user_item_pairs.tolist()
        self.user_to_positive_set = {u: set(row) for u, row in enumerate(self.user_item_matrix.rows)}

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, index):
        user_positive_items_pair = deepcopy(self.user_item_pairs[index])
        # sample negative samples
        negative_samples = np.random.randint(
            0,
            self.user_item_matrix.shape[1],
            size=self.n_negative)
        # Check if we sample any positive items as negative samples.
        # Note: this step can be optional as the chance that we sample a positive item is fairly low given a
        # large item set.
        if self.check_negative:
            user = user_positive_items_pair[0]
            for j, neg in enumerate(negative_samples):
                while neg in self.user_to_positive_set[user]:
                    negative_samples[j] = neg = np.random.randint(0, self.user_item_matrix.shape[1])

        # textual and visual features
        textual_feature_pos = self.textualfeatures[user_positive_items_pair[1]]
        visual_feature_pos = self.imagefeatures[user_positive_items_pair[1]]
        textual_feature_neg = self.textualfeatures[negative_samples]
        visual_feature_neg = self.imagefeatures[negative_samples]

        price_label = self.attribute_label[user_positive_items_pair[1]]['price_label']
        salesrank_label = self.attribute_label[user_positive_items_pair[1]]['salesrank_label']
        brand_label = self.attribute_label[user_positive_items_pair[1]]['brand_label']
        category_label = self.attribute_label[user_positive_items_pair[1]]['category_label']

        price_label_neg = [self.attribute_label[i]['price_label'] for i in negative_samples]
        salesrank_label_neg = [self.attribute_label[i]['salesrank_label'] for i in negative_samples]
        brand_label_neg = [self.attribute_label[i]['brand_label'] for i in negative_samples]
        category_label_neg = [self.attribute_label[i]['category_label'] for i in negative_samples]


        return torch.tensor(user_positive_items_pair), torch.tensor(negative_samples), torch.tensor(textual_feature_pos),\
               torch.tensor(visual_feature_pos), torch.tensor(textual_feature_neg), torch.tensor(visual_feature_neg),\
               price_label, salesrank_label, brand_label, category_label,\
               torch.tensor(price_label_neg), torch.tensor(salesrank_label_neg), torch.tensor(brand_label_neg), torch.tensor(category_label_neg)


