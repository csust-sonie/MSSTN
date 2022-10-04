import os
from statistics import mode
import torch
import random
import numpy as np
import torch.nn as nn
from options import args
from torch.optim.lr_scheduler import StepLR
from datareader.data_util import tDataUtil

from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from metrices import masked_mae_np, masked_mape_np, masked_rmse_np
from evaluation import evaluation

from models.model_v6 import Archer
from models.STGCN import STGCN


if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.cuda_visiable)
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def set_log(logPattern):
    if args.log:
        logger.add(logPattern)
    options = vars(args)
    if args.log:
        logger.info(options)
    else:
        print(options)


if args.tensorboard:
    writer = SummaryWriter()
else:
    writer = None


def output_metrices(name, metrices, epoch=0, outToStd=True):
    if outToStd:
        print("{} mae: {}, mape: {}, rmse: {}".format(
            name, metrices[0], metrices[1], metrices[2]))
    if args.log:
        logger.info("{} mae: {}, mape: {}, rmse: {}".format(
            name, metrices[0], metrices[1], metrices[2]))
    if args.tensorboard:
        writer.add_scalar(name + 'mae', metrices[0], epoch)
        writer.add_scalar(name + 'mape', metrices[1], epoch)
        writer.add_scalar(name + 'rmse', metrices[2], epoch)


def train(model, data_util: tDataUtil, optimizer, criterion):
    batch_loss = []
    train_mae, train_mape, train_rmse = [], [], []

    model.train()
    for src, targets in data_util.train_loader:
        optimizer.zero_grad()

        src = src.to(device)
        targets = targets.to(device)

        outputs = model(src)

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        batch_loss.append(loss.item())

        target_unnorm = data_util.unnormal(targets.detach().cpu().numpy())
        out_unnorm = data_util.unnormal(outputs.detach().cpu().numpy())

        train_mae.append(masked_mae_np(target_unnorm, out_unnorm, 0))
        train_rmse.append(masked_rmse_np(target_unnorm, out_unnorm, 0))
        train_mape.append(masked_mape_np(target_unnorm, out_unnorm, 0))

    return np.mean(batch_loss), np.mean(train_mae), np.mean(train_mape), np.mean(train_rmse)


def train_main(model, data_util: tDataUtil, optimizer, criterion):
    set_seed(5)

    A = data_util.adj

    best_valid_rmse = 1000
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(1, args.epochs + 1):
        print(f"training epoch: {epoch}")
        loss_metrices = train(model, data_util, optimizer, criterion)
        print(f"Train loss : {loss_metrices[0]}")
        if args.log:
            logger.info(f"train loss: {loss_metrices[0]}")
        if args.tensorboard:
            writer.add_scalar("train loss", loss_metrices[0], epoch)
        output_metrices('train', metrices=loss_metrices[1:], epoch=epoch)

        print("Evaluating...")
        eval_metrices = evaluation(
            model, data_util, is_test=True, device=device)
        output_metrices("Evaluate", eval_metrices, epoch=epoch)

        if eval_metrices[2] < best_valid_rmse:
            best_valid_rmse = eval_metrices[2]
            print('New best rmse results!')
            torch.save(model.state_dict(),
                       f'./best_model_params/net_params_{args.dataset}_best.pkl')

        scheduler.step()

    model.load_state_dict(torch.load(
        f'./best_model_params/net_params_{args.dataset}_best.pkl'))
    test_metrices = evaluation(model, data_util, is_test=False, device=device)
    output_metrices("Test", test_metrices)


if __name__ == '__main__':
    loss = nn.L1Loss()
    data_util = tDataUtil(args)
    model = Archer(data_util.num_node, 9, args.n_history, args.n_predict, 1, 32, 1, data_util.adj).to(device)

    # model = STGCN(data_util.num_node, 1, args.n_history,
    #               args.n_predict, data_util.adj).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_main(model, data_util, optimizer, loss)
