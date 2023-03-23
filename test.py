import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils
import numpy as np
import argparse
from model import AVGN
from datasets import get_test_dataset, inverse_normalize
import cv2

import torch.multiprocessing as mp
import torch.distributed as dist


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='avgn_vggss', help='experiment name (experiment folder set to "args.model_dir/args.experiment_name)"')
    parser.add_argument('--save_visualizations', action='store_true', help='Set to store all VSL visualizations (saved in viz directory within experiment folder)')

    # Dataset
    parser.add_argument('--testset', default='flickr', type=str, help='testset (flickr or vggss)')
    parser.add_argument('--test_data_path', default='', type=str, help='Root directory path of data')
    parser.add_argument('--test_gt_path', default='', type=str)
    parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')
    parser.add_argument('--num_class', default=37, type=int)

    # mo-vsl hyper-params
    parser.add_argument('--model', default='avgn')
    parser.add_argument('--out_dim', default=512, type=int)
    parser.add_argument('--num_negs', default=None, type=int)
    parser.add_argument('--tau', default=0.03, type=float, help='tau')

    parser.add_argument('--attn_assign', type=str, default='soft', help="type of audio grouping assignment")
    parser.add_argument('--dim', type=int, default=512, help='dimensionality of features')
    parser.add_argument('--depth_aud', type=int, default=3, help='depth of audio transformers')
    parser.add_argument('--depth_vis', type=int, default=3, help='depth of visual transformers')

    # evaluation parameters
    parser.add_argument('--alpha', default=0.4, type=float, help='alpha')
    parser.add_argument("--dropout_img", type=float, default=0, help="dropout for image")
    parser.add_argument("--dropout_aud", type=float, default=0, help="dropout for audio")

    parser.add_argument('--m_img', default=1.0, type=float, metavar='M', help='momentum for imgnet')
    parser.add_argument('--m_aud', default=1.0, type=float, metavar='M', help='momentum for audnet')

    parser.add_argument('--use_momentum', action='store_true')
    parser.add_argument('--relative_prediction', action='store_true')
    parser.add_argument('--use_mom_eval', action='store_true')
    parser.add_argument('--pred_size', default=0.5, type=float)
    parser.add_argument('--pred_thr', default=0.5, type=float)

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


def main_worker(local_rank, ngpus_per_node, args):
    args.gpu = local_rank
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Model dir
    model_dir = os.path.join(args.model_dir, args.experiment_name)
    viz_dir = os.path.join(model_dir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)

    # Setup distributed environment
    if args.multiprocessing_distributed:
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + local_rank
            print(args.dist_url, args.world_size, args.rank)
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # Create model
    if args.model.lower() == 'avgn':
        audio_visual_model = AVGN(args.tau, args.out_dim, args.dropout_img, args.dropout_aud, args)
    else:
        raise ValueError

    from torchvision.models import resnet18
    object_saliency_model = resnet18(pretrained=True)
    object_saliency_model.avgpool = nn.Identity()
    object_saliency_model.fc = nn.Sequential(
        nn.Unflatten(1, (512, 7, 7)),
        NormReducer(dim=1),
        Unsqueeze(1)
    )
    # object_saliency_model.fc = nn.Unflatten(1, (512, 7, 7))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            audio_visual_model.cuda(args.gpu)
            object_saliency_model.cuda(args.gpu)
            if args.multiprocessing_distributed:
                audio_visual_model = torch.nn.parallel.DistributedDataParallel(audio_visual_model, device_ids=[args.gpu])
                object_saliency_model = torch.nn.parallel.DistributedDataParallel(object_saliency_model, device_ids=[args.gpu])

    # Load weights
    ckp_fn = os.path.join(model_dir, 'best.pth')
    if os.path.exists(ckp_fn):
        ckp = torch.load(ckp_fn, map_location='cpu')
        audio_visual_model.load_state_dict({k.replace('module.', ''): ckp['model'][k] for k in ckp['model']})
        print(f'loaded from {os.path.join(model_dir, "best.pth")}')
    else:
        print(f"Checkpoint not found: {ckp_fn}")

    # Dataloader
    testdataset = get_test_dataset(args)
    if args.multiprocessing_distributed:
        sampler = torch.utils.data.DistributedSampler(testdataset, num_replicas=ngpus_per_node, rank=args.rank, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(testdataset)
    testdataloader = DataLoader(testdataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers)
    print("Loaded dataloader.")

    validate(testdataloader, audio_visual_model, object_saliency_model, model_dir, args)


@torch.no_grad()
def validate(testdataloader, audio_visual_model, object_saliency_model, model_dir, args):
    audio_visual_model.train(False)
    object_saliency_model.train(False)

    evaluator_av = utils.EvaluatorFull(default_conf_thr=0.5, pred_size=args.pred_size, pred_thr=args.pred_thr, results_dir=f"{model_dir}/av")
    evaluator_obj = utils.EvaluatorFull(default_conf_thr=0.5, pred_size=args.pred_size,  pred_thr=args.pred_thr, results_dir=f"{model_dir}/obj")
    evaluator_av_obj = utils.EvaluatorFull(default_conf_thr=0., pred_size=args.pred_size,  pred_thr=args.pred_thr, results_dir=f"{model_dir}/av_obj")
    for step, (image, spec, bboxes, name) in enumerate(testdataloader):
        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)

        # Compute S_AVL
        heatmap_av = audio_visual_model(image.float(), spec.float(), mode='test')[1].unsqueeze(1)
        heatmap_av = F.interpolate(heatmap_av, size=(224, 224), mode='bicubic', align_corners=True)
        heatmap_av = heatmap_av.data.cpu().numpy()

        # Compute S_OBJ
        img_feat = object_saliency_model(image)
        heatmap_obj = F.interpolate(img_feat, size=(224, 224), mode='bicubic', align_corners=True)
        heatmap_obj = heatmap_obj.data.cpu().numpy()

        av_min, av_max = -1. / args.tau, 1. / args.tau
        obj_min, obj_max = 0., 2.5
        min_max_norm = lambda x, xmin, xmax: (x - xmin) / (xmax - xmin)

        # Compute eval metrics and save visualizations
        for i in range(spec.shape[0]):
            gt_map = bboxes['gt_map'][i].data.cpu().numpy()
            bb = bboxes['bboxes'][i]
            bb = bb[bb[:, 0] >= 0].numpy().tolist()

            n = heatmap_av[i, 0].size
            scores_av = min_max_norm(heatmap_av[i, 0], av_min, av_max)
            scores_obj = min_max_norm(heatmap_obj[i, 0], obj_min, obj_max)
            scores_av_obj = scores_av * args.alpha + scores_obj * (1 - args.alpha)

            conf_av = np.sort(scores_av.flatten())[-n//4:].mean()
            conf_obj = np.sort(scores_obj.flatten())[-n//4:].mean()
            conf_av_obj = np.sort(scores_av_obj.flatten())[-n//4:].mean()

            if args.relative_prediction:
                pred_av = utils.normalize_img(scores_av)
                pred_obj = utils.normalize_img(scores_obj)
                pred_av_obj = utils.normalize_img(scores_av_obj)

                thr_av = np.sort(pred_av.flatten())[int(n * args.pred_size)]
                thr_obj = np.sort(pred_obj.flatten())[int(n * args.pred_size)]
                thr_av_obj = np.sort(pred_av_obj.flatten())[int(n * args.pred_size)]
            else:
                pred_av = scores_av
                pred_obj = scores_obj
                pred_av_obj = scores_av_obj

                thr_av = thr_obj = thr_av_obj = args.pred_thr

            evaluator_av.update(bb, gt_map, conf_av, pred_av, thr_av, name[i])
            evaluator_obj.update(bb, gt_map, conf_obj, pred_obj, thr_obj, name[i])
            evaluator_av_obj.update(bb, gt_map, conf_av_obj, pred_av_obj, thr_av_obj, name[i])

            if args.save_visualizations:
                evaluator_av.save_viz(image[i], bb, pred_av, name[i])
                evaluator_obj.save_viz(image[i], bb, pred_obj, name[i])
                evaluator_av_obj.save_viz(image[i], bb, pred_av_obj, name[i])

        print(f'{step+1}/{len(testdataloader)}: AV+OGL-Prec@30={evaluator_av_obj.precision_at_30():.3f} AVL-Prec@30={evaluator_av.precision_at_30():.3f} OGL-Prec@30={evaluator_obj.precision_at_30():.3f}')

    evaluator_av.save_results()
    evaluator_obj.save_results()
    evaluator_av_obj.save_results()

    print('='*20 + ' AVL ' + '='*20)
    stats_av = evaluator_av.finalize_stats()
    print('\n'.join([f' - {k}: {stats_av[k]}' for k in sorted(stats_av.keys()) if stats_av[k] is not np.nan]))

    print('='*20 + ' OGL ' + '='*20)
    stats_obj = evaluator_obj.finalize_stats()
    print('\n'.join([f' - {k}: {stats_obj[k]}' for k in sorted(stats_obj.keys()) if stats_obj[k] is not np.nan]))

    print('='*20 + ' AV+OGL ' + '='*20)
    stats_av_obj = evaluator_av_obj.finalize_stats()
    print('\n'.join([f' - {k}: {stats_av_obj[k]}' for k in sorted(stats_av_obj.keys()) if stats_av_obj[k] is not np.nan]))


class NormReducer(nn.Module):
    def __init__(self, dim):
        super(NormReducer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.abs().mean(self.dim)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


if __name__ == "__main__":
    main(get_arguments())
