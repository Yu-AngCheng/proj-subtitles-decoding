import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import math
from torch.utils.tensorboard import SummaryWriter
from utils.metric import AverageMeter, evaluate
from utils.model import set_seeds
from models.audioencoder.audioencoder import AudioEncoder
from models.seegencoder.seegencoder import SEEGEncoder
from dataset.dataset import CustomDataset


def validate(epoch, audio_encoder, seeg_encoder, val_loader, writer, device):
    audio_encoder.eval()
    seeg_encoder.eval()
    top1 = AverageMeter()
    top2 = AverageMeter()

    for audio_data, seeg_data, seeg_padding_mask in tqdm(val_loader):
        batch_size = audio_data.shape[0]

        audio_data = audio_data.to(device)
        seeg_data = seeg_data.to(device)
        seeg_padding_mask = seeg_padding_mask.to(device)

        # Forward
        audio_embedding = audio_encoder(audio_data)
        seeg_embedding = seeg_encoder(seeg_data, seeg_padding_mask)

        # Flatten the output for later similarity computation
        audio_embedding = audio_embedding.flatten(1, 2)
        seeg_embedding = seeg_embedding.flatten(1, 2)

        # Compute similarity
        sim = torch.einsum('i d, j d -> i j', audio_embedding, seeg_embedding) * math.e

        # Compute loss
        labels = torch.arange(batch_size).to(device)

        # update metric
        acc1, acc2 = evaluate(sim, labels, topk=(1, 2))
        top1.update(acc1.item(), batch_size)
        top2.update(acc2.item(), batch_size)

    writer.add_scalar('Val/Acc@1', top1.avg, epoch)
    writer.add_scalar('Val/Acc@2', top2.avg, epoch)
    print(f'Val Acc@1 {top1.avg:.3f}')
    print(f'Val Acc@2 {top2.avg:.3f}')


def train(epoch, audio_encoder, seeg_encoder, optimizer, train_loader, writer, device):
    audio_encoder.train()
    seeg_encoder.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    iteration = len(train_loader) * epoch

    for audio_data, seeg_data, seeg_padding_mask in tqdm(train_loader):
        batch_size = audio_data.shape[0]

        audio_data = audio_data.to(device)
        seeg_data = seeg_data.to(device)
        seeg_padding_mask = seeg_padding_mask.to(device)

        optimizer.zero_grad()

        # Forward
        audio_embedding = audio_encoder(audio_data)
        seeg_embedding = seeg_encoder(seeg_data, seeg_padding_mask)

        # Flatten the output for later similarity computation
        audio_embedding = audio_embedding.flatten(1, 2)
        seeg_embedding = seeg_embedding.flatten(1, 2)

        # Compute similarity
        sim = torch.einsum('i d, j d -> i j', audio_embedding, seeg_embedding) * math.e

        # Compute loss
        labels = torch.arange(batch_size).to(device)
        loss = F.cross_entropy(sim, labels)

        # update metric
        losses.update(loss.item(), batch_size)
        acc1, acc2 = evaluate(sim, labels, topk=(1, 2))
        top1.update(acc1.item(), batch_size)
        top2.update(acc2.item(), batch_size)

        loss.backward()
        optimizer.step()

        iteration += 1
        if iteration % 50 == 0:
            writer.add_scalar('Train/Loss', losses.avg, iteration)
            writer.add_scalar('Train/Acc@1', top1.avg, iteration)
            writer.add_scalar('Train/Acc@2', top2.avg, iteration)

    print(f'Epoch: {epoch}')
    print(f'Train Acc@1 {top1.avg:.3f}')
    print(f'Train Acc@2 {top2.avg:.3f}')


def run(args):
    exp_folder = os.path.join('./experiments', args.exp_name)
    ckpt_folder = os.path.join(exp_folder, 'ckpt')
    log_folder = os.path.join(exp_folder, 'log')
    os.makedirs(exp_folder, exist_ok=True)
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)
    writer = SummaryWriter(log_dir=log_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the dataloaders
    audio_dir = args.audio_dir
    seeg_dir = args.seeg_dir
    train_ratio = args.train_ratio
    train_dataset = CustomDataset(audio_dir=audio_dir, seeg_dir=seeg_dir, train_ratio=train_ratio, is_train=True)
    test_dataset = CustomDataset(audio_dir=audio_dir, seeg_dir=seeg_dir, train_ratio=train_ratio, is_train=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=8)

    # define the audio encoder
    audio_encoder = AudioEncoder().to(device)

    # define the seeg encoder
    num_input_channels = 57
    num_output_channels = 128
    input_length = 4096
    output_length = 199  # 199 is the default output length from the audio encoder
    num_heads = 3
    num_encoder_layers = 6
    dim_feedforward = 2048
    seeg_encoder = SEEGEncoder(num_input_channels=num_input_channels, num_output_channels=num_output_channels,
                               input_length=input_length, output_length=output_length, num_heads=num_heads,
                               num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward).to(device)

    # define the optimizer
    optimizer = optim.Adam([{'params': audio_encoder.conv_block_1d.parameters()},
                            {'params': seeg_encoder.parameters()}], lr=args.lr)

    if args.cont:   # load checkpoint to continue training
        ckpt_lst = os.listdir(ckpt_folder)
        ckpt_lst.sort(key=lambda x: int(x.split('_')[-1]))
        read_path = os.path.join(ckpt_folder, ckpt_lst[-1])
        print(f'load checkpoint from {read_path}')
        checkpoint = torch.load(read_path)
        audio_encoder.load_state_dict(checkpoint['audio_encoder'])
        seeg_encoder.load_state_dict(checkpoint['seeg_encoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.total_epoch):
        train(epoch, audio_encoder, seeg_encoder, optimizer, train_loader, writer, device)

        if epoch % args.save_freq == 0:
            state = {
                'audio_encoder': audio_encoder.state_dict(),
                'seeg_encoder': seeg_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            ckpt_file = os.path.join(ckpt_folder, f'ckpt_epoch_{epoch}.pth')
            torch.save(state, ckpt_file)

        with torch.no_grad():
            validate(epoch, audio_encoder, seeg_encoder, test_loader, writer, device)
    torch.save(audio_encoder.state_dict(), os.path.join(ckpt_folder, f'audio_encoder_epoch_{epoch}.pth'))
    torch.save(seeg_encoder.state_dict(), os.path.join(ckpt_folder, f'seeg_encoder_epoch_{epoch}.pth'))
    writer.close()


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--exp_name', '-e', type=str, default='lr_1e-3-batch_10-train_ratio-0.8',
                            help="The checkpoints and logs will be save in /experiments/$EXP_NAME")
    arg_parser.add_argument('--lr', '-l', type=float, default=1e-3, help="Learning rate")
    arg_parser.add_argument('--save_freq', '-s', type=int, default=1, help="frequency of saving model")
    arg_parser.add_argument('--total_epoch', '-t', type=int, default=20, help="total epoch number for training")
    arg_parser.add_argument('--cont', '-c', action='store_true',
                            help="whether to load saved the latest checkpoint from $EXP_NAME and continue training")
    arg_parser.add_argument('--batch_size', '-b', type=int, default=10, help="batch size")
    arg_parser.add_argument('--audio_dir', '-ad', type=str, default='./data/audio_1-4seconds',
                            help="path to the audio data folder")
    arg_parser.add_argument('--seeg_dir', '-sd', type=str, default='./data/seeg_1-4seconds',
                            help="path to the seeg data folder")
    arg_parser.add_argument('--train_ratio', '-r', type=float, default=0.8,
                            help="the ratio of training data to all data")
    args = arg_parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    set_seeds(42)
    run(args)
