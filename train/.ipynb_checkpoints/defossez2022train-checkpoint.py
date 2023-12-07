import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from torch.utils.tensorboard  import SummaryWriter
import models, utils
from utils.data import save_pickle,load_pickle
from utils.metric import AverageMeter, evaluate
from params import defossez2022decoding_params
from utils.data.meg.gwilliams2022neural import load_gwilliams2022neural_origin
from models.defossez2022decoding import defossez2022decoding
from models.wav2vec import wav2vec
from train.dataset import Trainset, Testset


def validate(epoch,model_audio,model_meg, val_loader, writer):
    model_audio.eval()
    model_meg.eval()
    top1 = AverageMeter()
    top2 = AverageMeter()

    for data, chan_pos, subject_id, audio in tqdm(val_loader):
        bsz = subject_id.shape[0]
        if torch.cuda.is_available():
            data = data.cuda()
            chan_pos = chan_pos.cuda()
            subject_id = subject_id.cuda()
            audio = audio.cuda()

        input_meg = (data, chan_pos, subject_id)
        output_meg = model_meg(input_meg)
        output_audio = model_audio(audio)
        output_meg = output_meg.flatten(1,2)
        output_audio = output_audio.flatten(1,2)
        temp = nn.Parameter(torch.tensor(1.)).exp()
        sim = torch.einsum('i d, j d -> i j', output_meg, output_audio) * temp
        labels = torch.arange(bsz)
        labels = labels.cuda()
        # update metric
        acc1, acc2 = evaluate(sim, labels, topk=(1, 2))
        top1.update(acc1.item(), bsz)
        top2.update(acc2.item(), bsz)

    writer.add_scalar('Val/Acc@1', top1.avg, epoch)
    writer.add_scalar('Val/Acc@2', top2.avg, epoch)
    print(' Val Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print(' Val Acc@2 {top2.avg:.3f}'.format(top2=top2))
    return 


def train(epoch, model_audio,model_meg,optimizer, train_loader, writer):
    model_audio.train()
    model_meg.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    iteration = len(train_loader) * epoch

    for data, chan_pos, subject_id, audio in tqdm(train_loader):
        bsz = subject_id.shape[0]
        if torch.cuda.is_available():
            data = data.cuda()
            chan_pos = chan_pos.cuda()
            subject_id = subject_id.cuda()
            audio = audio.cuda()

        optimizer.zero_grad()
        input_meg = (data, chan_pos, subject_id)
        output_meg = model_meg(input_meg)
        output_audio = model_audio(audio) 
        output_meg = output_meg.flatten(1,2)
        output_audio = output_audio.flatten(1,2)
        temp = nn.Parameter(torch.tensor(1.)).exp()
        sim = torch.einsum('i d, j d -> i j', output_meg, output_audio) * temp
        labels = torch.arange(bsz)
        labels = labels.cuda()
        loss = F.cross_entropy(sim, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc2 = evaluate(sim, labels, topk=(1, 2))
        top1.update(acc1.item(), bsz)
        top2.update(acc2.item(), bsz)
        
        loss.backward()
        # print(model_meg.mlp_time[0].weight.grad)
        # print(model_audio.mlp_feature[0].weight.grad)
        optimizer.step()

        iteration += 1
        if iteration % 50 == 0:
            writer.add_scalar('Train/Loss', losses.avg, iteration)
            writer.add_scalar('Train/Acc@1', top1.avg, iteration)
            writer.add_scalar('Train/Acc@2', top2.avg, iteration)

    print(' Epoch: %d'%(epoch))
    print(' Train Acc@1 {top1.avg:.3f}'.format(top1=top1))
    print(' Train Acc@2 {top2.avg:.3f}'.format(top2=top2))
    return 


def run(args,d2dparam):
    save_folder = os.path.join('../experiments',  args.exp_name)
    ckpt_folder = os.path.join(save_folder,  'ckpt')
    log_folder = os.path.join(save_folder,  'log')
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=log_folder)

    # define dataset and dataloader
    base = os.path.join(os.getcwd(), os.pardir)
    path_data = os.path.join(base, "data", "meg.gwilliams2022neural")
    train_dataset = Trainset()
    val_dataset = Testset()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(
         val_dataset, batch_size=args.batchsize, shuffle=False, num_workers=1)
    
    # define network 
    model_audio = wav2vec()
    model_meg = defossez2022decoding(d2dparam.model)
    if torch.cuda.is_available():
        model_audio = model_audio.cuda()
        model_meg = model_meg.cuda()

    # define optimizer
    optimizer = optim.Adam([{'params':model_audio.seq_feature.parameters()},
                                     {'params':model_meg.parameters()}],lr=1e-3)

    if args.cont:
        # load latest checkpoint
        ckpt_lst = os.listdir(ckpt_folder)
        ckpt_lst.sort(key=lambda x: int(x.split('_')[-1]))
        read_path = os.path.join(ckpt_folder, ckpt_lst[-1])
        print('load checkpoint from %s'%(read_path))
        checkpoint = torch.load(read_path)
        model_audio.load_state_dict(checkpoint['model_audio'])
        model_meg.load_state_dict(checkpoint['model_meg'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.total_epoch):
        train(epoch, model_audio,model_meg, optimizer, train_loader, writer)
        
        if epoch % args.save_freq == 0:
            state = {
                'model_audio': model_audio.state_dict(),
                'model_meg': model_meg.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(ckpt_folder, 'ckpt_epoch_%s'%(str(epoch)))
            torch.save(state, save_file)

        with torch.no_grad():
            validate(epoch,model_audio, model_meg, val_loader, writer)
    filename = str(args.exp_name)
    torch.save(model_audio.state_dict(), '/root/NLP/models/model_audio'+filename+'.pt')
    torch.save(model_meg.state_dict(), '/root/NLP/models/model_meg'+filename+'.pt')
    return 


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--exp_name', '-e', type=str, default='./experiments', help="The checkpoints and logs will be save in ./checkpoint/$EXP_NAME")
    arg_parser.add_argument('--lr', '-l', type=float, default=1e-3, help="Learning rate")
    arg_parser.add_argument('--save_freq', '-s', type=int, default=1, help="frequency of saving model")
    arg_parser.add_argument('--total_epoch', '-t', type=int, default=20, help="total epoch number for training")
    arg_parser.add_argument('--cont', '-c', action='store_true', help="whether to load saved checkpoints from $EXP_NAME and continue training")
    arg_parser.add_argument('--batchsize', '-b', type=int, default=10, help="batch size")
    args = arg_parser.parse_args()
    
    utils.model.set_seeds(42)
    dataset = "gwilliams2022neural"
    n_channels = 208
    n_subjects = 2
    n_features = 128
    d2d_params_inst = defossez2022decoding_params(n_channels, n_subjects, n_features)

    run(args,d2d_params_inst)