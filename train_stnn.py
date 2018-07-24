import os
import random
import json
from collections import defaultdict, OrderedDict

import configargparse
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn


from datasets import dataset_factory
from utils import DotDict, Logger, rmse
from stnn import SaptioTemporalNN


#######################################################################################################################
# Options - CUDA - Random seed
#######################################################################################################################
p = configargparse.ArgParser()
# -- data
p.add('--datadir', type=str, help='path to dataset', default='data')
p.add('--dataset', type=str, help='dataset name', default='heat')
# -- xp
p.add('--outputdir', type=str, help='path to save xp', default='output')
p.add('--xp', type=str, help='xp name', default='stnn')
# -- model
p.add('--mode', type=str, help='STNN mode (default|refine|discover)', default='default')
p.add('--nz', type=int, help='laten factors size', default=1)
p.add('--activation', type=str, help='dynamic module activation function (identity|tanh)', default='identity')
p.add('--khop', type=int, help='spatial depedencies order', default=1)
p.add('--nhid', type=int, help='dynamic function hidden size', default=0)
p.add('--nlayers', type=int, help='dynamic function num layers', default=1)
p.add('--dropout_f', type=float, help='latent factors dropout', default=.0)
p.add('--dropout_d', type=float, help='dynamic function dropout', default=.0)
p.add('--lambd', type=float, help='lambda between reconstruction and dynamic losses', default=.1)
# -- optim
p.add('--lr', type=float, help='learning rate', default=3e-3)
p.add('--beta1', type=float, default=.0, help='adam beta1')
p.add('--beta2', type=float, default=.999, help='adam beta2')
p.add('--eps', type=float, default=1e-9, help='adam eps')
p.add('--wd', type=float, help='weight decay', default=1e-6)
p.add('--wd_z', type=float, help='weight decay on latent factors', default=1e-7)
p.add('--l2_z', type=float, help='l2 between consecutives latent factors', default=0.)
p.add('--l1_rel', type=float, help='l1 regularization on relation discovery mode', default=0.)
# -- learning
p.add('--batch_size', type=int, default=1000, help='batch size')
p.add('--patience', type=int, default=150, help='number of epoch to wait before trigerring lr decay')
p.add('--nepoch', type=int, default=10000, help='number of epochs to train for')
# -- gpu
p.add('--device', type=int, default=-1, help='-1: cpu; > -1: cuda device id')
# -- seed
p.add('--manualSeed', type=int, help='manual seed')

# parse
opt = DotDict(vars(p.parse_args()))
opt.mode = opt.mode if opt.mode in ('refine', 'discover') else None

# cudnn
if opt.device > -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.device)
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
# seed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.device > -1:
    torch.cuda.manual_seed_all(opt.manualSeed)


#######################################################################################################################
# Data
#######################################################################################################################
# -- load data
setup, (train_data, test_data), relations = dataset_factory(opt.datadir, opt.dataset, opt.khop)
train_data = train_data.to(device)
test_data = test_data.to(device)
relations = relations.to(device)
for k, v in setup.items():
    opt[k] = v

# -- train inputs
t_idx = torch.arange(opt.nt_train, out=torch.LongTensor()).unsqueeze(1).expand(opt.nt_train, opt.nx).contiguous()
x_idx = torch.arange(opt.nx, out=torch.LongTensor()).expand_as(t_idx).contiguous()
# dynamic
idx_dyn = torch.stack((t_idx[1:], x_idx[1:])).view(2, -1).to(device)
nex_dyn = idx_dyn.size(1)
# decoder
idx_dec = torch.stack((t_idx, x_idx)).view(2, -1).to(device)
nex_dec = idx_dec.size(1)

#######################################################################################################################
# Model
#######################################################################################################################
model = SaptioTemporalNN(relations, opt.nx, opt.nt_train, opt.nd, opt.nz, opt.mode, opt.nhid, opt.nlayers,
                         opt.dropout_f, opt.dropout_d, opt.activation, opt.periode).to(device)


#######################################################################################################################
# Optimizer
#######################################################################################################################
params = [{'params': model.factors_parameters(), 'weight_decay': opt.wd_z},
          {'params': model.dynamic.parameters()},
          {'params': model.decoder.parameters()}]
if opt.mode in ('refine', 'discover'):
    params.append({'params': model.rel_parameters(), 'weight_decay': 0.})
optimizer = optim.Adam(params, lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.wd)
if opt.patience > 0:
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt.patience)


#######################################################################################################################
# Logs
#######################################################################################################################
logger = Logger(opt.outputdir, opt.xp, 100)
with open(os.path.join(opt.outputdir, opt.xp, 'config.json'), 'w') as f:
    json.dump(opt, f, sort_keys=True, indent=4)


#######################################################################################################################
# Training
#######################################################################################################################
lr = opt.lr
pb = trange(opt.nepoch)
for e in pb:
    # ------------------------ Train ------------------------
    model.train()
    # --- decoder ---
    idx_perm = torch.randperm(nex_dec).to(device)
    batches = idx_perm.split(opt.batch_size)
    logs_train = defaultdict(float)
    for i, batch in enumerate(batches):
        optimizer.zero_grad()
        # data
        input_t = idx_dec[0][batch]
        input_x = idx_dec[1][batch]
        x_target = train_data[input_t, input_x]
        # closure
        x_rec = model.dec_closure(input_t, input_x)
        mse_dec = F.mse_loss(x_rec, x_target)
        # backward
        mse_dec.backward()
        # step
        optimizer.step()
        # log
        logger.log('train_iter.mse_dec', mse_dec.item())
        logs_train['mse_dec'] += mse_dec.item() * len(batch)
    # --- dynamic ---
    idx_perm = torch.randperm(nex_dyn).to(device)
    batches = idx_perm.split(opt.batch_size)
    for i, batch in enumerate(batches):
        optimizer.zero_grad()
        # data
        input_t = idx_dyn[0][batch]
        input_x = idx_dyn[1][batch]
        # closure
        z_inf = model.factors[input_t, input_x]
        z_pred = model.dyn_closure(input_t - 1, input_x)
        # loss
        mse_dyn = z_pred.sub(z_inf).pow(2).mean()
        loss_dyn = mse_dyn * opt.lambd
        if opt.l2_z > 0:
            loss_dyn += opt.l2_z * model.factors[input_t - 1, input_x].sub(model.factors[input_t, input_x]).pow(2).mean()
        if opt.mode in('refine', 'discover') and opt.l1_rel > 0:
            # rel_weights_tmp = model.rel_weights.data.clone()
            loss_dyn += opt.l1_rel * model.get_relations().abs().mean()
        # backward
        loss_dyn.backward()
        # step
        optimizer.step()
        # clip
        # if opt.mode == 'discover' and opt.l1_rel > 0:  # clip
        #     sign_changed = rel_weights_tmp.sign().ne(model.rel_weights.data.sign())
        #     model.rel_weights.data.masked_fill_(sign_changed, 0)
        # log
        logger.log('train_iter.mse_dyn', mse_dyn.item())
        logs_train['mse_dyn'] += mse_dyn.item() * len(batch)
        logs_train['loss_dyn'] += loss_dyn.item() * len(batch)
    # --- logs ---
    logs_train['mse_dec'] /= nex_dec
    logs_train['mse_dyn'] /= nex_dyn
    logs_train['loss_dyn'] /= nex_dyn
    logs_train['loss'] = logs_train['mse_dec'] + logs_train['loss_dyn']
    logger.log('train_epoch', logs_train)
    # ------------------------ Test ------------------------
    model.eval()
    with torch.no_grad():
        x_pred, _ = model.generate(opt.nt - opt.nt_train)
        score_ts = rmse(x_pred, test_data, reduce=False)
        score = rmse(x_pred, test_data)
    logger.log('test_epoch.rmse', score)
    logger.log('test_epoch.ts', {t: {'rmse': scr.item()} for t, scr in enumerate(score_ts)})
    # checkpoint
    logger.log('train_epoch.lr', lr)
    pb.set_postfix(loss=logs_train['loss'], rmse_test=score)
    logger.checkpoint(model)
    # schedule lr
    if opt.patience > 0 and score < 1:
        lr_scheduler.step(score)
    lr = optimizer.param_groups[0]['lr']
    if lr <= 1e-5:
        break
logger.save(model)
