import os
import sys
import math
import pprint

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data
import time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import tasks, util


separator = ">" * 30
line = "-" * 30


def train_and_validate(cfg, model, train_data, valid_data, test_data,filtered_data=None):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    train_triplets = torch.cat([train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(train_triplets, cfg.train.batch_size, sampler=sampler)

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    #step = math.ceil(cfg.train.num_epoch / 10)
    step = 1
    best_result = float("-inf")
    best_epoch = -1

    batch_id = 0
    for i in range(0, cfg.train.num_epoch, step):
        parallel_model.train()
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            losses = []
            sampler.set_epoch(epoch)
            start = time.time()
            for batch in train_loader:
                batch = tasks.negative_sampling(train_data, batch, cfg.task.num_negative,
                                                strict=cfg.task.strict_negative)
                pred = parallel_model(train_data, batch)

                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                neg_weight = torch.ones_like(pred)
                if cfg.task.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1)
                else:
                    neg_weight[:, 1:] = 1 / cfg.task.num_negative

                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("binary cross entropy: %g" % loss)
                losses.append(loss.item())
                batch_id += 1

            if util.get_rank() == 0:
                avg_loss = sum(losses) / len(losses)
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning("average binary cross entropy: %g" % avg_loss)

            end = time.time()
            # print("train time")
            # print(end - start)
        epoch = min(cfg.train.num_epoch, i + step)
        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()


        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")

        mrr, results = test(cfg, model, valid_data, filtered_data=filtered_data)
        for key in results.keys():
            score = results[key]
            metric = key
            logger.warning("%s: %g" % (metric, score))

        # if epoch >= 1:
        #     path, weight, count = visualize(cfg, model, train_data, filtered_data=filtered_data)

        if mrr > best_result:
            best_result = mrr
            best_epoch = epoch
            logger.warning(separator)
            logger.warning("--------------test----------")
            start_t = time.time()
            mrr_test, results_test = test(cfg, model, test_data, filtered_data=filtered_data)
            end_t = time.time()
            print("test time")
            print(end_t - start_t)
            best_test = results_test.copy()
            for key in results_test.keys():
                score = results_test[key]
                metric = key
                logger.warning("%s: %g" % (metric, score))

    logger.warning(separator)
    logger.warning("final best test")
    for key in best_test.keys():
        score = best_test[key]
        metric = key
        logger.warning("%s: %g" % (metric, score))


    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()

def visualize(cfg, model, train_data, filtered_data=None):
    world_size = util.get_world_size()
    rank = util.get_rank()

    # test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    # sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    # test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

    train_triplets = torch.cat([train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(train_triplets, cfg.train.batch_size, sampler=sampler)

    model.train()
    paths = []
    weights = []
    triples = []
    count = 0
    num_batch = 0
    count_r = 0
    for batch in train_loader:
        num_batch = num_batch + len(batch)
        for i in range(len(batch)):
            r = batch[i][2]
            h = batch[i][0]
            if h == 242:
                print("!!!!!!!!!!!")
                path, weight = model.visualize(train_data, batch[i].unsqueeze(0))
            if r == 106:
                h = batch[i][0]
                if h == 1198:
                    print("!!!!!!!!!!!!!")
                    path, weight = model.visualize(train_data, batch[i].unsqueeze(0))
            # path, weight, paths_wo, weight_wo = model.visualize(train_data, batch[i].unsqueeze(0))
            # # print("-----------triplets-------")
            # # print(batch[i])
            # # batch_i = str(num_batch) + "_" + str(i)
            # # print("---------------------")
            # # for j in range(len(path)):
            # #     last_tail = path[j][-1][1]
            # #     if last_tail != batch[i][1]:
            # #         print(batch_i)
            # r = batch[i][2]
            # h = batch[i][0]
            # if h == 1198:
            #     print("!!!!!!!!!!!")
            # if r == 106:
            #     h = batch[i][0]
            #     if h == 1198:
            #         print("!!!!!!!!!!!!!")
            #     count_r = count_r + 1
            #     if len(path) < 2:
            #         continue
            #     best_path = path[0]
            #     query = batch[i]
            #     # if len(best_path) < 2:
            #     #     break
            #     s_p = 0
            #     for j in range(len(best_path)):
            #         relation = best_path[j][2]
            #         if relation == (180) or relation == (362):
            #             s_p = s_p + 1
            #     if (s_p >= 2) & (len(best_path) < 3):
            #         a = best_path[-1][2] == (r + 180)
            #         b = best_path[-2][2] == (r + 180)
            #         c = best_path[-1][2] == (r + 540)
            #         d = best_path[-2][2] == (r + 540)
            #         if (a & b) or (c & d):
            #             count = count + 1

            # if len(path) < 2:
            #     continue
            # best_path = path[0]
            # # if len(best_path) < 2:
            # #     break
            # s_p = 0
            # for j in range(len(best_path)):
            #     relation = best_path[j][2]
            #     if relation == (r + 180) or relation == (r + 540):
            #         s_p = s_p + 1
            # if (s_p >= 2) & (len(best_path)<3):
            #     a = best_path[-1][2] == (r+180)
            #     b = best_path[-2][2] == (r+180)
            #     c = best_path[-1][2] == (r+540)
            #     d = best_path[-2][2] == (r+540)
            #     if (a & b) or (c & d):
            #         count = count + 1
            # paths.append(path)
            # weights.append(weight)
    #
    #print(triples)
    print(num_batch)
    return paths, weights, count

@torch.no_grad()
def test(cfg, model, test_data, filtered_data=None):
    world_size = util.get_world_size()
    rank = util.get_rank()

    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, cfg.train.batch_size, sampler=sampler)

    model.eval()
    rankings = []
    num_negatives = []
    training = True
    for batch in test_loader:
        t_batch, h_batch = tasks.all_negative(test_data, batch)
        t_pred = model(test_data, t_batch)
        h_pred = model(test_data, h_batch)

        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)

    results = dict()
    if rank == 0:
        for metric in cfg.task.metric:
            if metric == "mr":
                score = all_ranking.float().mean()
                results["mr"] = score
            elif metric == "mrr":
                score = (1 / all_ranking.float()).mean()
                results["mrr"] = score
            elif metric.startswith("hits@"):
                values = metric[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    # unbiased estimation
                    fp_rate = (all_ranking - 1).float() / all_num_negative
                    score = 0
                    for i in range(threshold):
                        # choose i false positive from num_sample - 1 negatives
                        num_comb = math.factorial(num_sample - 1) / \
                                   math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                    results[metric] = score
                else:
                    score = (all_ranking <= threshold).float().mean()
                    results[metric] = score
            #logger.warning("%s: %g" % (metric, score))
    mrr = (1 / all_ranking.float()).mean()

    return mrr, results


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    is_inductive = cfg.dataset["class"].startswith("Ind")
    dataset = util.build_dataset(cfg)
    cfg.model.num_relation = dataset.num_relations
    model = util.build_model(cfg)

    device = util.get_device(cfg)
    model = model.to(device)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)
    if is_inductive:
        # for inductive setting, use only the test fact graph for filtered ranking
        filtered_data = None
    else:
        # for transductive setting, use the whole graph for filtered ranking
        filtered_data = Data(edge_index=dataset.data.target_edge_index, edge_type=dataset.data.target_edge_type)
        filtered_data = filtered_data.to(device)

    train_and_validate(cfg, model, train_data, valid_data, test_data, filtered_data=filtered_data)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")
    test(cfg, model, valid_data, filtered_data=filtered_data)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    test(cfg, model, test_data, filtered_data=filtered_data)
