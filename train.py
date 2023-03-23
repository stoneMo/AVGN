from multiprocessing import reduction
import os
import argparse
import builtins
import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score

import utils
from model import AVGN
from datasets import get_train_dataset, get_test_dataset


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='avgn_vggsound', help='experiment name (used for checkpointing and logging)')

    # Data params
    parser.add_argument('--trainset', default='vggsound', type=str, help='trainset')
    parser.add_argument('--testset', default='vggsound', type=str, help='testset')
    parser.add_argument('--train_data_path', default='', type=str, help='Root directory path of train data')
    parser.add_argument('--test_data_path', default='', type=str, help='Root directory path of test data')
    parser.add_argument('--test_gt_path', default='', type=str)
    parser.add_argument('--num_test_samples', default=-1, type=int)
    parser.add_argument('--num_class', default=221, type=int)

    # mo-vsl hyper-params
    parser.add_argument('--model', default='movsl')
    parser.add_argument('--imgnet_type', default='vitb8')
    parser.add_argument('--audnet_type', default='vitb8')

    parser.add_argument('--out_dim', default=512, type=int)
    parser.add_argument('--num_negs', default=None, type=int)
    parser.add_argument('--tau', default=0.03, type=float, help='tau')

    parser.add_argument('--attn_assign', type=str, default='soft', help="type of audio grouping assignment")
    parser.add_argument('--dim', type=int, default=512, help='dimensionality of features')
    parser.add_argument('--depth_aud', type=int, default=3, help='depth of audio transformers')
    parser.add_argument('--depth_vis', type=int, default=3, help='depth of visual transformers')

    # training/evaluation parameters
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument("--lr_schedule", default='cte', help="learning rate schedule")
    parser.add_argument("--init_lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="warmup epochs")
    parser.add_argument("--seed", type=int, default=12345, help="random seed")
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight Decay')
    parser.add_argument("--clip_norm", type=float, default=0, help="gradient clip norm")
    parser.add_argument("--dropout_img", type=float, default=0, help="dropout for image")
    parser.add_argument("--dropout_aud", type=float, default=0, help="dropout for audio")

    # Distributed params
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--node', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:12345')
    parser.add_argument('--multiprocessing_distributed', action='store_true')

    return parser.parse_args()


def main(args):
    mp.set_start_method('spawn')
    args.dist_url = f'tcp://{args.node}:{args.port}'
    print('Using url {}'.format(args.dist_url))

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node
        mp.spawn(main_worker,
                 nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))

    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # Setup distributed environment
    if args.multiprocessing_distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # Create model dir
    model_dir = os.path.join(args.model_dir, args.experiment_name)
    os.makedirs(model_dir, exist_ok=True)
    utils.save_json(vars(args), os.path.join(model_dir, 'configs.json'), sort_keys=True, save_pretty=True)

    # tb writers
    tb_writer = SummaryWriter(model_dir)

    # logger
    log_fn = f"{model_dir}/train.log"
    def print_and_log(*content, **kwargs):
        # suppress printing if not first GPU on each node
        if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
            return
        msg = ' '.join([str(ct) for ct in content])
        sys.stdout.write(msg+'\n')
        sys.stdout.flush()
        with open(log_fn, 'a') as f:
            f.write(msg+'\n')
    builtins.print = print_and_log

    # Create model
    if args.model.lower() == 'avgn':
        model = AVGN(args.tau, args.out_dim, args.dropout_img, args.dropout_aud, args)
    else:
        raise ValueError

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.multiprocessing_distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
    print(model)

    # Optimizer
    optimizer, scheduler = utils.build_optimizer_and_scheduler_adam(model, args)

    # Resume if possible
    start_epoch, best_precision, best_ap, best_f1 = 0, 0., 0., 0.
    if os.path.exists(os.path.join(model_dir, 'latest.pth')):
        ckp = torch.load(os.path.join(model_dir, 'latest.pth'), map_location='cpu')
        start_epoch, best_precision, best_ap, best_f1 = ckp['epoch'], ckp['best_Precision'], ckp['best_AP'], ckp['best_F1']
        model.load_state_dict(ckp['model'])
        optimizer.load_state_dict(ckp['optimizer'])
        print(f'loaded from {os.path.join(model_dir, "latest.pth")}')

    # Dataloaders
    traindataset = get_train_dataset(args)
    train_sampler = None
    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(traindataset)
    train_loader = torch.utils.data.DataLoader(
        traindataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True,
        persistent_workers=args.workers > 0)

    testdataset = get_test_dataset(args)
    test_loader = torch.utils.data.DataLoader(
        testdataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, drop_last=False,
        persistent_workers=args.workers > 0)
    print("Loaded dataloader.")

    # =============================================================== #
    # Training loop
    if args.testset in {'vgginstruments_multi', 'music_duet', 'vggsound_duet'}:
        precision, ap, f1 = validate_multi(test_loader, model, args)
    else:
        precision, ap, f1 = validate(test_loader, model, args)
    print(f'Precision (epoch {start_epoch}): {precision}')
    print(f'AP (epoch {start_epoch}): {ap}')
    print(f'F1 (epoch {start_epoch}): {f1}')
    print(f'best_Precision: {best_precision}')
    print(f'best_AP: {best_ap}')
    print(f'best_F1: {best_f1}')

    metric_list = [[] for _ in range(3)]

    for epoch in range(start_epoch, args.epochs):
        if args.multiprocessing_distributed:
            train_loader.sampler.set_epoch(epoch)

        # Train
        train(train_loader, model, optimizer, epoch, args, tb_writer)

        # Evaluate
        if args.testset in {'vgginstruments_multi', 'music_duet'}:
            precision, ap, f1 = validate_multi(test_loader, model, args)
        else:
            precision, ap, f1 = validate(test_loader, model, args)
        if precision >= best_precision:
            best_precision, best_ap, best_f1 = precision, ap, f1
        print(f'Precision (epoch {epoch+1}): {precision}')
        print(f'AP (epoch {epoch+1}): {ap}')
        print(f'F1 (epoch {epoch+1}): {f1}')
        print(f'best_Precision: {best_precision}')
        print(f'best_AP: {best_ap}')
        print(f'best_F1: {best_f1}')

        tb_writer.add_scalar('Precision', precision, epoch)
        tb_writer.add_scalar('AP', ap, epoch)
        tb_writer.add_scalar('F1', f1, epoch)

        metric_list[0].append(precision)
        metric_list[1].append(ap)
        metric_list[2].append(f1)

        # Checkpoint
        if args.rank == 0:
            ckp = {'model': model.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   'epoch': epoch+1,
                   'best_Precision': best_precision,
                   'best_AP': best_ap,
                   'best_F1': best_f1}
            torch.save(ckp, os.path.join(model_dir, 'latest.pth'))
            if precision == best_precision:
                torch.save(ckp, os.path.join(model_dir, 'best.pth'))
            print(f"Model saved to {model_dir}")

    np.save(os.path.join(model_dir, 'metrics.npy'), np.array(metric_list))


def train(train_loader, model, optimizer, epoch, args, writer):
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_mtr = AverageMeter('Loss', ':.3f')
    loss_loc_mtr = AverageMeter('Loc Loss', ':.3f')
    loss_token_mtr = AverageMeter('Token Loss', ':.3f')
    loss_pred_mtr = AverageMeter('Pred Loss', ':.3f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, loss_mtr, loss_loc_mtr, loss_token_mtr, loss_pred_mtr],
        prefix="Epoch: [{}]".format(epoch),
    )

    end = time.time()
    for i, (image, spec, anno, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        global_step = i + len(train_loader) * epoch
        utils.adjust_learning_rate(optimizer, epoch + i / len(train_loader), args)

        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)
            label = anno['class'].cuda(args.gpu, non_blocking=True)

        loc_loss, _, cls_token_loss, cls_pred_loss = model(image.float(), spec.float(), cls_target=label, mode='train')
        loss = loc_loss + cls_token_loss + cls_pred_loss

        loss_mtr.update(loss.item(), image.shape[0])
        loss_loc_mtr.update(loc_loss.item(), image.shape[0])
        loss_token_mtr.update(cls_token_loss.item(), image.shape[0])
        loss_pred_mtr.update(cls_pred_loss.item(), image.shape[0])

        optimizer.zero_grad()
        loss.backward()

        # gradient clip
        if args.clip_norm != 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)  # clip gradient

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        writer.add_scalar('loss', loss_mtr.avg, global_step)
        writer.add_scalar('Loc loss', loss_loc_mtr.avg, global_step)
        writer.add_scalar('Token loss', loss_token_mtr.avg, global_step)
        writer.add_scalar('Pred loss', loss_pred_mtr.avg, global_step)
        # writer.add_scalar('batch_time', batch_time.avg, global_step)
        # writer.add_scalar('data_time', data_time.avg, global_step)

        if i % 10 == 0 or i == len(train_loader) - 1:
            progress.display(i)
        del loss


def validate(test_loader, model, args):
    model.train(False)
    evaluator = utils.EvaluatorFull()
    for step, (image, spec, bboxes, name) in enumerate(test_loader):
        if torch.cuda.is_available():
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)
            label = bboxes['class'].cuda(args.gpu, non_blocking=True)

        avl_map = model(image.float(), spec.float(), cls_target=label, mode='test')[1].unsqueeze(1)
        avl_map = F.interpolate(avl_map, size=(224, 224), mode='bicubic', align_corners=False)
        avl_map = avl_map.data.cpu().numpy()

        av_min, av_max = -1. / args.tau, 1. / args.tau
        min_max_norm = lambda x, xmin, xmax: (x - xmin) / (xmax - xmin)

        for i in range(spec.shape[0]):
            gt_map = bboxes['gt_map'][i].data.cpu().numpy()
            bb = bboxes['bboxes'][i]
            bb = bb[bb[:, 0] >= 0].numpy().tolist()

            n = avl_map[i, 0].size
            scores = min_max_norm(avl_map[i, 0], av_min, av_max)
            pred = utils.normalize_img(scores)
            conf = np.sort(scores.flatten())[-n//4:].mean()
            thr = np.sort(pred.flatten())[int(n*0.5)]
            # evaluator.cal_CIOU(bb, conf, pred, gt_map, thr)
            evaluator.update(bb, gt_map, conf, pred, thr, name[i])

    # cIoU = evaluator.finalize_AP50(evaluator.ciou)
    # AUC = evaluator.finalize_AUC(evaluator.ciou)
    precision = evaluator.precision_at_30()
    # ap = evaluator.ap_at_30()
    ap = evaluator.piap_average()
    f1 = evaluator.f1_at_30()
    return precision, ap, f1


def validate_multi(test_loader, model, args):
    model.train(False)
    evaluator_0 = utils.EvaluatorFull()
    evaluator_1 = utils.EvaluatorFull()
    for step, (image, spec, bboxes, name) in enumerate(test_loader):
        if torch.cuda.is_available():
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)
            label = bboxes['class'].cuda(args.gpu, non_blocking=True)

        num_mixtures = image.shape[1]
        avl_map_list = []
        for j in range(num_mixtures):        
            avl_map = model(image[:,j].float(), spec.float(), cls_target=label[:,j], mode='test')[1].unsqueeze(1)
            avl_map = F.interpolate(avl_map, size=(224, 224), mode='bicubic', align_corners=False)
            avl_map_list.append(avl_map)

        avl_map = torch.stack(avl_map_list, dim=1).data.cpu().numpy()
        av_min, av_max = -1. / args.tau, 1. / args.tau
        min_max_norm = lambda x, xmin, xmax: (x - xmin) / (xmax - xmin)

        for i in range(spec.shape[0]):
            gt_map = bboxes['gt_map'][i].data.cpu().numpy()     # (2, 224, 224)
            bb = bboxes['bboxes'][i]
            bb = bb[bb[:, 0] >= 0].numpy().tolist()

            for j in range(num_mixtures):
                n = avl_map[i, j, 0].size
                scores = min_max_norm(avl_map[i, j, 0], av_min, av_max)
                pred = utils.normalize_img(scores)
                conf = np.sort(scores.flatten())[-n//4:].mean()
                thr = np.sort(pred.flatten())[int(n*0.5)]
                # evaluator.cal_CIOU(bb, conf, pred, gt_map, thr)
                if j == 0:
                    evaluator_0.update(bb, gt_map[j], conf, pred, thr, name[i])
                elif j == 1:
                    evaluator_1.update(bb, gt_map[j], conf, pred, thr, name[i])

    # cIoU = evaluator.finalize_AP50(evaluator.ciou)
    # AUC = evaluator.finalize_AUC(evaluator.ciou)
    precision = (evaluator_0.precision_at_10() + evaluator_1.precision_at_10()) / 2
    # ap = evaluator.ap_at_30()
    ap = (evaluator_0.piap_average() + evaluator_1.piap_average()) / 2
    f1 = (evaluator_0.f1_at_30() + evaluator_1.f1_at_30()) / 2
    return precision, ap, f1


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", fp=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.fp = fp

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        msg = '\t'.join(entries)
        print(msg, flush=True)
        if self.fp is not None:
            self.fp.write(msg+'\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == "__main__":
    main(get_arguments())
