import argparse
import gc
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from gmm_Model_final import Query2GMM
from dataloader import *
from tensorboardX import SummaryWriter
import time
import pickle
import collections
import datetime

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
try:
    from apex import amp
except:
    print("apex not installed")

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

def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

def do_train_step(model, train_iterators, optimizer, use_apex, step):

    # training
    logs = []

    for i in range(20):

        # probability_dict = {"1p": 1, "2p": 1, "3p": 1,
        #                     "2i": 1, "3i": 1,
        #                     "ip": 0.2, "pi": 0.2,
        #                     "2u-DNF": 0.2, "up-DNF": 0.2}

        probability_dict = {"1p": 3, "2p": 1.5, "3p": 1.5,
                            "2i": 1, "3i": 1,
                            "2in": 0.1, "3in": 0.1,
                            "inp": 0.1, "pni": 0.1,
                            "pin": 0.1}

        iteration_list = list(train_iterators.items())

        probability_list = [probability_dict[task_name] for task_name, _ in iteration_list]
        task_name_list = [task_name for task_name, _ in iteration_list]

        probability_list = np.array(probability_list)
        probability_list = probability_list / np.sum(probability_list)

        task_name = np.random.choice(task_name_list, p=probability_list)

        iter = train_iterators[task_name]

        if torch.cuda.device_count() > 1:
            log_of_the_step = model.module.train_step(model, iter, optimizer, use_apex)
        else:
            log_of_the_step = model.train_step(model, iter, optimizer, use_apex)

        logs.append(log_of_the_step)

        if task_name in ["1p"]:

            if torch.cuda.device_count() > 1:
                log_of_the_step = model.module.train_inv_step(model, iter, optimizer, use_apex)
            else:
                log_of_the_step = model.train_inv_step(model, iter, optimizer, use_apex)

            logs.append(log_of_the_step)

    return logs

def test_evaluate(model, test_dataset_iter, summary_writer, step, easy_answers, hard_answers):
    total_task_avg_logs = {}
    for task_name, iter in test_dataset_iter.items():
        single_task_logs = {} #记录单个结构的所有query的结果metrics
        for negative_sample, queries, queries_unflatten, query_structures in tqdm(iter):
            if torch.cuda.device_count() > 1:
                test_log_q2b, test_log_emql = model.module.test_step(model, negative_sample, queries, queries_unflatten,
                                                                         query_structures, easy_answers, hard_answers)
            else:
                test_log_q2b, test_log_emql = model.test_step(model, negative_sample, queries, queries_unflatten,
                                                                  query_structures, easy_answers, hard_answers)
            for metric, value in test_log_q2b.items():
                if metric in single_task_logs:
                    single_task_logs[metric].append(value)
                else:
                    single_task_logs[metric] = [value]

        single_task_mean_logs = {}
        for metric, value_list in single_task_logs.items():
            if metric != 'num_samples':
                single_task_average_acc = np.sum(np.array(value_list) *
                                                 np.array(single_task_logs['num_samples'])) / np.sum(single_task_logs['num_samples'])
                single_task_mean_logs[metric] = single_task_average_acc
                summary_writer.add_scalar(task_name + '_' + metric, single_task_average_acc, step)
                logging.info('%s %s at step %d: %f' % (task_name, metric, step, single_task_average_acc))

        if 'num_samples' in total_task_avg_logs:
            total_task_avg_logs['num_samples'].append(np.sum(single_task_logs['num_samples']))
        else:
            total_task_avg_logs['num_samples'] = [np.sum(single_task_logs['num_samples'])]
        for metric, value in single_task_mean_logs.items():
            if metric in total_task_avg_logs:
                total_task_avg_logs[metric].append(value)
            else:
                total_task_avg_logs[metric] = [value]

    total_avg_logs = {}
    for metric, value in total_task_avg_logs.items():
        if metric != 'num_samples':
            average_acc = np.sum(np.array(value) *
                                 np.array(total_task_avg_logs['num_samples'])) / np.sum(total_task_avg_logs['num_samples'])
            total_avg_logs[metric] = average_acc
            summary_writer.add_scalar("_average_test_" + metric, average_acc, step)
            logging.info('test average %s at step %d: %f' % (metric, step, average_acc))

def do_test(model, test_dataset_iter, summary_writer, step, easy_answers, hard_answers):
    # This function will do the evaluation as a whole,
    # not do it in a step-wise way.
    all_average_dev_logs_q2b = {}
    all_average_dev_logs_emql = {}
    all_average_test_logs_q2b = {}
    all_average_test_logs_emql = {}
    for task_name, iter in test_dataset_iter.items(): #每个结构分开测试
        aggregated_logs_q2b = {}
        aggregated_logs_emql = {}
        for negative_sample, queries, queries_unflatten, query_structures in tqdm(iter): # number of negative_sample = 1，一次测一个query
            if torch.cuda.device_count() > 1:
                test_log_q2b, test_log_emql = model.module.test_step(model,
                                                             negative_sample,
                                                             queries,
                                                             queries_unflatten,
                                                             query_structures,
                                                             easy_answers,
                                                             hard_answers)
            else:
                test_log_q2b, test_log_emql = model.test_step(model,
                                                      negative_sample,
                                                      queries,
                                                      queries_unflatten,
                                                      query_structures,
                                                      easy_answers,
                                                      hard_answers)
            for key, value in test_log_q2b.items():
                if key in aggregated_logs_q2b:
                    aggregated_logs_q2b[key].append(value)
                else:
                    aggregated_logs_q2b[key] = [value]
            for key, value in test_log_emql.items():
                if key in aggregated_logs_emql:
                    aggregated_logs_emql[key].append(value)
                else:
                    aggregated_logs_emql[key] = [value]

        aggregated_weighted_mean_logs_q2b = {}
        for key, value in aggregated_logs_q2b.items():
            if key != 'num_samples':
                weighted_average = np.sum(np.array(value) *
                                          np.array(aggregated_logs_q2b['num_samples'])) / np.sum(aggregated_logs_q2b['num_samples'])
                aggregated_weighted_mean_logs_q2b[key] = weighted_average
                summary_writer.add_scalar(task_name + '_' + key, weighted_average, step)
                logging.info('q2b standard: %s %s at step %d: %f' % (task_name, key, step, weighted_average))

        aggregated_weighted_mean_logs_emql = {}
        for key, value in aggregated_logs_emql.items():
            if key != "num_samples":
                weighted_average = np.sum( np.array(value) *
                                           np.array(aggregated_logs_emql["num_samples"]) ) / \
                                   np.sum(aggregated_logs_emql["num_samples"])

                aggregated_weighted_mean_logs_emql[key] = weighted_average
                summary_writer.add_scalar(task_name + "_" + key, weighted_average, step)
                logging.info('emql standard: %s %s at step %d: %f' % (task_name, key, step, weighted_average))
        if "test" in task_name:
            if 'num_samples' in all_average_test_logs_q2b:
                all_average_test_logs_q2b['num_samples'].append(
                    np.sum(aggregated_logs_q2b['num_samples'])
                )
            else:
                all_average_test_logs_q2b['num_samples'] = [np.sum(aggregated_logs_q2b['num_samples'])]
            for key, value in aggregated_weighted_mean_logs_q2b.items():
                if key in all_average_test_logs_q2b:
                    all_average_test_logs_q2b[key].append(value)
                else:
                    all_average_test_logs_q2b[key] = [value]

            if "num_samples" in all_average_test_logs_emql:
                all_average_test_logs_emql["num_samples"].append(
                    np.sum(aggregated_logs_emql["num_samples"])
                )
            else:
                all_average_test_logs_emql["num_samples"] = [np.sum(aggregated_logs_emql["num_samples"])]
            for key, value in aggregated_weighted_mean_logs_emql.items():
                if key in all_average_test_logs_emql:
                    all_average_test_logs_emql[key].append(value)
                else:
                    all_average_test_logs_emql[key] = [value]

        elif "dev" in task_name:
            if "num_samples" in all_average_dev_logs_q2b:
                all_average_dev_logs_q2b["num_samples"].append(
                    np.sum(aggregated_logs_q2b["num_samples"])
                )
            else:
                all_average_dev_logs_q2b["num_samples"] = [np.sum(aggregated_logs_q2b["num_samples"])]
            for key, value in aggregated_weighted_mean_logs_q2b.items():
                if key in all_average_dev_logs_q2b:
                    all_average_dev_logs_q2b[key].append(value)
                else:
                    all_average_dev_logs_q2b[key] = [value]

            if "num_samples" in all_average_dev_logs_emql:
                all_average_dev_logs_emql["num_samples"].append(
                    np.sum(aggregated_logs_emql["num_samples"])
                )
            else:
                all_average_dev_logs_emql["num_samples"] = [np.sum(aggregated_logs_emql["num_samples"])]
            for key, value in aggregated_weighted_mean_logs_emql.items():
                if key in all_average_dev_logs_emql:
                    all_average_dev_logs_emql[key].append(value)
                else:
                    all_average_dev_logs_emql[key] = [value]

    weighted_average_test_logs_q2b = {}
    for key, value in all_average_test_logs_q2b.items():
        if key != 'num_samples':
            weighted_average = np.sum(np.array(value) *
                                      np.array(all_average_test_logs_q2b['num_samples'])) / np.sum(all_average_test_logs_q2b['num_samples'])
            weighted_average_test_logs_q2b[key] = weighted_average
            summary_writer.add_scalar('_average_test_' + key, weighted_average, step)
            logging.info('q2b standard: test average %s at step %d: %f' % (key, step, weighted_average))

    weighted_average_test_logs_emql = {}
    for key, value in all_average_test_logs_emql.items():
        if key != "num_samples":
            weighted_average = np.sum(np.array(value) *
                                      np.array(all_average_test_logs_emql["num_samples"])) / \
                               np.sum(all_average_test_logs_emql["num_samples"])

            weighted_average_test_logs_emql[key] = weighted_average
            summary_writer.add_scalar("_average_test_" + key, weighted_average, step)
            logging.info('emql standard: test average %s at step %d: %f' % (key, step, weighted_average))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-n', '--negative_sample_size', default=1, type=int)

    parser.add_argument('--log_steps', default=6000, type=int, help='train log every xx steps')
    parser.add_argument('--data_name', type=str, required=True)

    parser.add_argument('-d', '--entity_space_dim', default=400, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.002, type=float)
    parser.add_argument('-wc', '--weight_decay', default=0.0000, type=float)

    parser.add_argument('-k', '--num_gaussian_component', default=2, type=int)

    parser.add_argument("--model_name", default="v49.xx", type=str)

    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--dropout_rate", default=0.1, type=float)
    parser.add_argument('-ls', "--label_smoothing", default=0.0, type=float)
    parser.add_argument("--warm_up_steps", default=100, type=int)

    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument('--train_log_steps', default=100, type=int)

    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('--data_path', type=str, default='../data/')

    args = parser.parse_args()
    print(args)

    use_apex = False
    max_train_steps = 300000000 # just for placeholder, 'early-stop' strategy for practice
    previous_steps = 0

    data_name = args.data_name
    data_file_path = args.data_path
    data_path = data_file_path + data_name
    with open('%s/stats.txt' % data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = './q2gmm_logs/gradient_tape/' + current_time + data_name + '/train'
    test_log_dir = './q2gmm_logs/gradient_tape/' + current_time + data_name + '/test'
    train_summary_writer = SummaryWriter(train_log_dir)
    test_summary_writer = SummaryWriter(test_log_dir)

    #构建train.log文件
    save_path = 'logs/' + data_name + '/' + current_time + '/gmm_Model_final_inv'
    log_file = os.path.join(save_path, 'train.log')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.FileHandler(log_file)
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w',
    )

    def load_group_info():
        print("loading group information...")
        with open('%s/node_group_one_hot_vector.pkl' % (data_path + "/"), 'rb') as handle:
            node_group_one_hot_vector_single = pickle.load(handle)
        with open('%s/group_adj_matrix.pkl' % (data_path + "/"), 'rb') as handle:
            group_adj_matrix_single = pickle.load(handle)
        return node_group_one_hot_vector_single, group_adj_matrix_single

    node_group_one_hot_vector_single, group_adj_matrix_single = load_group_info()

    model = Query2GMM(nentity, nrelation, args.entity_space_dim,
                         num_gaussian_component=args.num_gaussian_component,
                         dropout_rate=args.dropout_rate,
                         label_smoothing=args.label_smoothing,
                         gamma=args.gamma,
                      node_group_one_hot_vector_single=node_group_one_hot_vector_single,
                      group_adj_matrix_single=group_adj_matrix_single)

    # Load the model with the given path
    if args.checkpoint_path != "":
        checkpoint_path = args.checkpoint_path
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        previous_steps = checkpoint['steps']

    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.cuda()

    no_decay = ['bias', 'layer_norm', 'embedding', "offset_matrix"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.optimizer == "adam":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

    else:
        print("invalid optimizer name, using adam instead")
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

    # Load the model with the given path
    if args.checkpoint_path != "":
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    use_apex = torch.cuda.device_count() == 1 and use_apex

    if use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    def warmup_lambda(epoch):
        if epoch < args.warm_up_steps:
            return epoch * 1.0 / args.warm_up_steps
        else:
            return 1

    scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)


    batch_size = args.batch_size
    negative_sample_size = args.negative_sample_size
    cpu_num = 4

    test_batch_size = 1

    train_iterators = {}
    test_iterators = {}
    dev_iterators = {}

    def load_data():
        '''
        Load all queries
        '''
        print("loading data...")
        train_queries = pickle.load(open(os.path.join(data_path, "train-queries.pkl"), 'rb'))
        train_answers = pickle.load(open(os.path.join(data_path, "train-answers.pkl"), 'rb'))
        valid_queries = pickle.load(open(os.path.join(data_path, "valid-queries.pkl"), 'rb'))
        valid_hard_answers = pickle.load(open(os.path.join(data_path, "valid-hard-answers.pkl"), 'rb'))
        valid_easy_answers = pickle.load(open(os.path.join(data_path, "valid-easy-answers.pkl"), 'rb'))
        test_queries = pickle.load(open(os.path.join(data_path, "test-queries.pkl"), 'rb'))
        test_hard_answers = pickle.load(open(os.path.join(data_path, "test-hard-answers.pkl"), 'rb'))
        test_easy_answers = pickle.load(open(os.path.join(data_path, "test-easy-answers.pkl"), 'rb'))


        return train_queries, train_answers, valid_queries,\
               valid_hard_answers, valid_easy_answers, test_queries, \
               test_hard_answers, test_easy_answers



    train_queries, train_answers, valid_queries, valid_hard_answers,\
    valid_easy_answers, test_queries, test_hard_answers, test_easy_answers = load_data()

    print("training query structures:")
    logging.info('training query structures:')
    for query_structure in train_queries:
        print(query_name_dict[query_structure] + ": " + str(len(train_queries[query_structure])))
        logging.info('%s: %s' % (query_name_dict[query_structure], str(len(train_queries[query_structure]))))

    print("====== Create Training Iterators  ======")
    for query_structure in train_queries:
        if "DM" in query_name_dict[query_structure]:
            continue
        tmp_train_queries = list(train_queries[query_structure])
        tmp_train_queries = [(query, query_structure) for query in tmp_train_queries]
        new_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDataset(tmp_train_queries, nentity, nrelation, negative_sample_size, train_answers),
            batch_size=batch_size,
            shuffle=True,
            num_workers=cpu_num,
            collate_fn=TrainDataset.collate_fn
        ))
        train_iterators[query_name_dict[query_structure]] = new_iterator

    print("====== Create Testing Dataloader ======")
    logging.info('test query structures:')
    for query_structure in test_queries:
        if "DM" in query_name_dict[query_structure]:
            continue
        logging.info(query_name_dict[query_structure] + ": " + str(len(test_queries[query_structure])))

        tmp_queries = list(test_queries[query_structure])
        tmp_queries = [(query, query_structure) for query in tmp_queries]
        test_dataloader = DataLoader(
            TestDataset(
                tmp_queries,
                nentity,
                nrelation,
            ),
            batch_size=test_batch_size,
            num_workers=cpu_num,
            collate_fn=TestDataset.collate_fn
        )
        test_iterators["test_" + query_name_dict[query_structure]] = test_dataloader


    print("====== Create Validation Dataloader ======")
    logging.info('validation query structures:')
    for query_structure in valid_queries:
        if "DM" in query_name_dict[query_structure]:
            continue
        logging.info(query_name_dict[query_structure] + ": " + str(len(valid_queries[query_structure])))

        tmp_queries = list(valid_queries[query_structure])
        tmp_queries = [(query, query_structure) for query in tmp_queries]
        valid_dataloader = DataLoader(
            TestDataset(
                tmp_queries,
                nentity,
                nrelation,
            ),
            batch_size=test_batch_size,
            num_workers=cpu_num,
            collate_fn=TestDataset.collate_fn
        )
        # print("dev_" + query_name_dict[query_structure])
        dev_iterators["dev_" + query_name_dict[query_structure]] = valid_dataloader

    #training start:
    for step in tqdm(range(max_train_steps)):
        total_step = step + previous_steps
        logs = do_train_step(model, train_iterators, optimizer, use_apex, total_step) #训练20次
        scheduler.step()

        aggregated_logs = {}
        for log in logs:

            for key, value in log.items():
                if key == "qtype":
                    continue
                train_summary_writer.add_scalar(log["qtype"]+ "_" + key, value, total_step)

                if key in aggregated_logs:
                    aggregated_logs[key].append(value)
                else:
                    aggregated_logs[key] = [value]

        aggregated_mean_logs = {key: np.mean(value) for key, value in aggregated_logs.items()}

        for key, value in aggregated_mean_logs.items():

            train_summary_writer.add_scalar( "_average_" + key, value, total_step)
            if total_step % args.train_log_steps == 0:
                logging.info('average %s at %d: %f' % (key, total_step, value))

        save_step = args.log_steps
        model_name = args.model_name
        if step % save_step == 0 and step != 0:
            general_checkpoint_path = "./q2gmm_logs/" + model_name +"_"+ str(total_step) +"_"+ data_name +".bin"

            if torch.cuda.device_count() > 1:
                torch.save({
                    'steps': total_step,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, general_checkpoint_path)
            else:
                torch.save({
                    'steps': total_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, general_checkpoint_path)

        if step % save_step == 0:
            print("Evaluating test dataset:")
            do_test(model, test_iterators, test_summary_writer, total_step, test_easy_answers, test_hard_answers)
            # do_test(model, dev_iterators, test_summary_writer, total_step, valid_easy_answers, valid_hard_answers)

        if step % 50 == 0:
            gc.collect()

if __name__ == "__main__":
    main()
