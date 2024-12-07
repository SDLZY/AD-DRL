# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
from amazon import AmazonDataset
from model import FactorAttributeNceNegBrand
import os, sys
import argparse
from time import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import create_logger
import shutil
from evaluator20_accelerate import RecallEvaluator

def to_cuda(batch):
    batch = list(batch)

    for i in range(len(batch)):
        if isinstance(batch[i], torch.Tensor):
            batch[i] = batch[i].cuda(non_blocking=True)
        elif isinstance(batch[i], list):
            for j, o in enumerate(batch[i]):
                if isinstance(batch[i], torch.Tensor):
                    batch[i][j] = o.cuda(non_blocking=True)

    return batch

def train(model, optimizer, train_loader, train_dataset, test_dataset, train_num, logger, log_path, args):
    EVALUATION_EVERY_N_BATCHES = train_num // args.batch_size + 1
    cur_best_pre_0 = 0.
    pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], []
    stopping_step = 0
    for epoch_num in range(args.epochs):
        t1 = time()
        # train model
        losses = 0
        model.train()
        # run n mini-batches
        for i in range(5):
            for b, batch in enumerate(train_loader):
                batch = to_cuda(batch)
                optimizer.zero_grad()
                loss = model(*batch)
                loss.backward()

                optimizer.step()
                losses += loss
        t2 = time()
        model.eval()
        testresult = RecallEvaluator(model, train_dataset, test_dataset)
        recalls, precisions, hit_ratios, ndcgs = testresult.eval(model)
        rec_loger.append(recalls)
        pre_loger.append(precisions)
        ndcg_loger.append(ndcgs)
        hit_loger.append(hit_ratios)

        t3 = time()
        print("epochs%d  [%.1fs + %.1fs]: train loss=%.5f, result=recall:%.5f, pres:%.5f, hr:%.5f, ndcg:%.5f" % (
        epoch_num, t2 - t1, t3 - t2, losses / (5 * EVALUATION_EVERY_N_BATCHES), recalls, precisions, hit_ratios, ndcgs))
        logger.info("epochs%d  [%.1fs + %.1fs]: train loss=%.5f, result=recall:%.5f, pres:%.5f, hr:%.5f, ndcg:%.5f" % (
        epoch_num, t2 - t1, t3 - t2, losses / (5 * EVALUATION_EVERY_N_BATCHES), recalls, precisions, hit_ratios, ndcgs))

        cur_best_pre_0, stopping_step, should_stop = early_stopping(recalls, cur_best_pre_0, stopping_step, model, optimizer, log_path,
                                                                    expected_order='acc', flag_step=15)
        if should_stop == True:
            break
        if epoch_num == args.epochs - 1:
            break
    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs)
    idx = list(recs).index(best_rec_0)
    final_perf = "Best Iter = recall:%.5f, pres:%.5f, hr:%.5f, ndcg:%.5f" % (recs[idx], pres[idx], hit[idx], ndcgs[idx])
    print(final_perf)
    logger.info(final_perf)

def test(model, train_dataset, test_dataset, logger, args):
    ckpt_path = f'checkpoints/{args.dataset}/best_param.model'
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    state_dict = checkpoint['state_dict']
    parsed_state_dict = {}
    for k, v in state_dict.items():
        if k in model.state_dict():
            parsed_state_dict[k] = v
        else:
            raise ValueError('failed to match key of state dict smartly!')
    model.load_state_dict(parsed_state_dict)
    model.eval()
    testresult = RecallEvaluator(model, train_dataset, test_dataset)
    recalls, precisions, hit_ratios, ndcgs = testresult.eval(model)
    perf = "result=recall:%.5f, pres:%.5f, hr:%.5f, ndcg:%.5f" % (recalls, precisions, hit_ratios, ndcgs)
    print(perf)
    logger.info(perf)

def early_stopping(log_value, best_value, stopping_step, model, optimizer, log_path, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
        checkpoint_dict = dict()
        checkpoint_dict['state_dict'] = model.state_dict()
        checkpoint_dict['optimizer'] = optimizer.state_dict()
        torch.save(checkpoint_dict, os.path.join(log_path, 'best_param.model'))
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

def parse_args():
    parser = argparse.ArgumentParser(description='Run ADDRL.')
    parser.add_argument('--dataset', nargs='?',default='Sports', help='Choose a dataset.')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    parser.add_argument('--epochs', type=int,default=1000, help = 'total_epochs')
    parser.add_argument('--gpu', nargs='?',default='1', help = 'gpu_id')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning_rate.')
    parser.add_argument('--decay_r', type=float, default=1e-0, help='decay_r.')
    parser.add_argument('--decay_f', type=float, default=1e-0, help='decay_f.')
    parser.add_argument('--decay_a', type=float, default=1e-0, help='decay_a.')
    parser.add_argument('--decay_n', type=float, default=1e-0, help='decay_n.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--n_factors', type=int, default=4,help='Number of factors.')
    parser.add_argument('--num_neg', type=int,default=4, help = 'negative items')
    parser.add_argument('--hidden_layer_dim_a', type=int, default=256, help='Hidden layer dim a.')
    parser.add_argument('--hidden_layer_dim_b', type=int, default=128, help='Hidden layer dim b.')
    parser.add_argument('--dropout_a', type=float, default=0.2, help='dropout_a.')
    parser.add_argument('--dropout_b', type=float, default=0.2, help='dropout_b.')
    parser.add_argument('--emb_dim', type=int, default=128, help='emb_dim.')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay.')
    parser.add_argument('--temp', type=float, default=1, help='temp.')
    parser.add_argument('--attribute_dataset', nargs='?', default='item_attribute_label', help='Choose a attribute dataset.')
    args = parser.parse_args()

    return args


def mainer(args):
    Filename = args.dataset
    Filepath = 'AmazonData/' + Filename
    log_path = f'Result/{args.dataset}/'
    logger = create_logger(log_path)
    logger.info('\nLog path is: ' + log_path)
    logger.info('training args:{}'.format(args))
    shutil.copy('main.py', log_path)
    shutil.copy('model.py', log_path)
    shutil.copy('amazon.py', log_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    train_dataset = AmazonDataset(Filepath, 'train', args.dataset, args.attribute_dataset, n_negative=args.num_neg)
    test_dataset = AmazonDataset(Filepath, 'test', args.dataset, args.attribute_dataset, n_negative=args.num_neg)
    n_users, n_items = max(train_dataset.dataMatrix.shape[0], test_dataset.dataMatrix.shape[0]), max(
        train_dataset.dataMatrix.shape[1], test_dataset.dataMatrix.shape[1])
    print(n_users, n_items)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=6, drop_last=True)

    model = FactorAttributeNceNegBrand(n_users,
                 n_items,
                 num_neg=args.num_neg,
                 n_factors=args.n_factors,
                 embed_dim=args.emb_dim,
                 decay_r=args.decay_r,
                 decay_f=args.decay_f,
                 decay_a=args.decay_a,
                 decay_n=args.decay_n,
                 hidden_layer_dim_a=args.hidden_layer_dim_a,
                 hidden_layer_dim_b=args.hidden_layer_dim_b,
                 dropout_rate_a=args.dropout_a,
                 dropout_rate_b=args.dropout_b,
                 temp=args.temp,
                 dataset_name=args.dataset
                 )
    num_gpus = len(args.gpu.split(','))
    model = torch.nn.DataParallel(model).cuda() if num_gpus > 1 else model.cuda()

    optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate)
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(m.weight)

    if args.mode == 'train':
        train(model, optimizer, train_loader, train_dataset, test_dataset, train_dataset.data_num, logger, log_path, args)
    else:
        test(model, train_dataset, test_dataset, logger, args)

if __name__ == '__main__':
    seed = 12345
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args = parse_args()
    mainer(args)
