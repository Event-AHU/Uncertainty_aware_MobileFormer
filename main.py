import argparse
import os
import random
import shutil
import time
import warnings
import logging

import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

from utils.utils import cutmix, cutmix_criterion
from utils.config import config
from model import MobileFormer
from datasets.video_dataset import video_Dataset as video_dataset
from datasets.spatial_transforms import *
from datasets.temporal_transforms import *
from mobileformer.mobile_former import mobile_former_96m, mobile_former_151m, mobile_former_214m, mobile_former_294m, mobile_former_508m
from mobileformer.mobile_former import ExponentialMovingAverage
import cv2
import pdb
from timm import create_model as create
from pytorch_grad_cam import GradCAM, \
                            ScoreCAM, \
                            GradCAMPlusPlus, \
                            AblationCAM, \
                            XGradCAM, \
                            EigenCAM, \
                            EigenGradCAM, \
                            LayerCAM, \
                            FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
def reshape_transform(tensor, height=14, width=14):
    # 去掉cls token
    # pdb.set_trace()
    result = tensor[:, 1:, :].reshape(tensor.size(0),
    height, width, tensor.size(2))

    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)
    return result


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--name', default='mf508', type=str,
                    help='model name')
parser.add_argument('--root', default='./data/Celex5', type=str,
                    help='root dir')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--num_cls', default=1000, type=int,
                    help='number of classes')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--base_model', default='resnet50', type=str)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr_warmup_epochs', default=5, type=int)
parser.add_argument('--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0.20, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--log_dir', default='', type=str, metavar='PATH',
                    help='path to log')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:55554', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--model-ema', action='store_true',
                    help='use ema.')
parser.add_argument('--model-ema-steps', default=32, type=int)
parser.add_argument('--model-ema-decay', default=0.99998, type=float)
parser.add_argument("--clip_len", default=8, type=int)
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--cutmix', action='store_true',
                    help='Use cutmix data augument')
parser.add_argument('--cutmix-prob', default=0.5, type=float,
                    help='cutmix probility')
parser.add_argument('--beta', default=1.0, type=float)

best_acc1 = 0


def main():
    args = parser.parse_args()
    # logging.basicConfig(filename=args.log_dir, level=logging.DEBUG)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    print('n_per_node:', ngpus_per_node)
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        print('world_size:', args.world_size)
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    print('gpu', gpu)
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
            print('rank:', args.rank)
        print('init ...', args.dist_url)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    model =mobile_former_508m()
    # model = create('vit_base_patch16_224',pretrained=True,num_classes=150)
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        print('ddp mode')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        # if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        #     model.features = torch.nn.DataParallel(model.features)
        #     model.cuda()
        # else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs/10, eta_min=0, last_epoch=- 1, verbose=False)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    model_ema = None
    cudnn.benchmark = True
    print('############################### Dataset loading ###############################')
    input_mean=[.485, .456, .406]
    input_std=[.229, .224, .225]
    normalize = GroupNormalize(input_mean, input_std)
        
    scales = [1, .875, .75, .66]
    trans_train = transforms.Compose([
                GroupScale(256),
                GroupMultiScaleCrop(224, scales),
                Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
                ToTorchFormatTensor(div=(args.base_model not in ['BNInception', 'InceptionV3'])),
                normalize
        ])

    trans_test  = torchvision.transforms.Compose([
                            GroupScale(256),
                            GroupCenterCrop(224),
                            Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
                            ToTorchFormatTensor(div=(args.base_model not in ['BNInception', 'InceptionV3'])),
                            normalize
                            ])
    temporal_transform_train = torchvision.transforms.Compose([
                                        TemporalUniformCrop_train(args.clip_len)
                                        ]) 
    temporal_transform_test = torchvision.transforms.Compose([
                                        TemporalUniformCrop_val(args.clip_len)
                                        ]) 

    train_dataset = video_dataset(args.root, mode='train',split='txt',spatial_transform=trans_train,
         temporal_transform = temporal_transform_train)
    val_dataset = video_dataset(args.root, mode='test',split='txt',spatial_transform=trans_test,
         temporal_transform = temporal_transform_test)
        
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)


    print('###############################  Dataset loaded  ##############################')
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, scheduler, epoch, args)

        # evaluate on validation set
        # if epoch%5==0:
        acc1 = validate(val_loader, model, criterion, args)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                # 'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best)
        # scheduler.step()
    print("best_acc1:",best_acc1)


def train(train_loader, model, criterion, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.16f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avg_loss = AverageMeter('Avg_Loss', ':.16f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    total_loss = 0
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)
        # pdb.set_trace()
        if args.cutmix and np.random.rand(1) < args.cutmix_prob:
            images, target_a, target_b, lam = cutmix(images, target, args.beta)
            output = model(images)
            loss = cutmix_criterion(criterion, output, target_a, target_b, lam)
        else:
            # pdb.set_trace()
            # target_hot = F.one_hot(target, num_classes=150 )
            output = model(images)
            # pdb.set_trace()
            # print(torch.sum(output))
            loss = criterion(output, target)
        #pdb.set_trace() #在这打断点 top可视化得分
        total_loss += float(loss)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
            progress.display(i)
    # pdb.set_trace()
    avg_loss = total_loss/len(train_loader)
    print("epoch: {}, Avg_Loss {}".format(epoch, avg_loss))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    
# 创建 GradCAM 对象
    # pdb.set_trace()
    # cam = GradCAM(model=model,
    #             target_layers=[model.blocks[11].norm1],
    #             # 这里的target_layer要看模型情况，
    #             # 比如还有可能是：target_layers = [model.blocks[-1].ffn.norm]
    #             reshape_transform=reshape_transform,
    #             use_cuda=True
    #             )

    with torch.no_grad():
        end = time.time()

        for i, (images, target) in enumerate(val_loader):
            # pdb.set_trace()
            # images = torch.squeeze(images,1)
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output

            
            # img = images[:,0,:,:,:]

            # img = images.cpu()
            # img = torch.squeeze(img,0)
            # img = img.permute(1,2,0)
            # img = np.array(img, dtype=np.uint8)
            # img = cv2.resize(img, (224, 224))


            # img = preprocess_image(img, mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

            # # # 看情况将图像转换为批量形式
            # # # input_tensor = input_tensor.unsqueeze(0)
            # # img = img.cuda()
            # # 计算 grad-cam
            # target_category = None # 可以指定一个类别，或者使用 None 表示最高概率的类别
            # grayscale_cam = cam(input_tensor=img, targets=target_category)
            # grayscale_cam = grayscale_cam[0, :]

            # # 将 grad-cam 的输出叠加到原始图像上
            # img = (img.squeeze(0)).permute(1,2,0)
            # visualization = show_cam_on_image(np.array(img) / 255, grayscale_cam)

            # # 保存可视化结果
            # cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
            # cv2.imwrite('cam.jpg', visualization)
            # pdb.set_trace()

            # from ptflops import get_model_complexity_info
            # from thop import profile
            # from thop import clever_format
            # pdb.set_trace()
            # # model_1=model.backbone 

            # # device = torch.device("cuda:0")

            # # model_1.to(device)
            # flops, params = get_model_complexity_info(model, (8,3,224,224), as_strings=True, print_per_layer_stat=True)
            # print("Flops: {}".format(flops))
            # print("Params: " + params)

            # input = torch.randn(1,8,3,224,224)
            # input = input.cuda()
            # # input2 = torch.randn(1, 16, 260, 346, 3)
            # # input=[]
            # # input.append(input1)
            # # input.append(input2)


            # flops, params = profile(model, inputs=(input, ))

            # flops1, params1 = clever_format([flops, params], "%.3f")


            # print('FLOPs = ' + str(flops/1000**3) + 'G')
            # print('Params = ' + str(params/1000**2) + 'M')
            output = model(images)

            # loss = criterion(output, target)
            # measure accuracy and record loss
            #pdb.set_trace()   #测试这里打断点
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
            '''x = output
            y = target
            x_new = x.cpu().detach().numpy()
            y_new =  y.unsqueeze(1).cpu().detach().numpy()

            if i == 0:
                x_total = x_new
                y_total = y_new                              
            else:
                 x_total = np.vstack((x_total,x_new))
                 y_total = np.vstack((y_total,y_new))
                
                
            np.savetxt('./Ncaltech_x_file_4.txt',x_total,fmt='%.32f')
            np.savetxt('./Ncaltech_label_file_4.txt',y_total,fmt='%f') ''' #图6特征可视化
        # TODO: this should also be done with the ProgressMeter  
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg


def save_checkpoint(state, is_best, filename='Ncal_.pth_up.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'Ncal_best_BAS.pth.tar')


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
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    print('lr', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
