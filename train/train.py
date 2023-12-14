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


def eval(audio_encoder, seeg_encoder, eval_loader, device, split):
    audio_encoder.eval()
    seeg_encoder.eval()

    audio_embeddings = None
    seeg_embeddings = None

    with torch.no_grad():
        for audio, seeg, seeg_padding_mask in tqdm(eval_loader):
            audio = audio.to(device)
            seeg = seeg.to(device)
            seeg_padding_mask = seeg_padding_mask.to(device)

            # Forward
            audio_embedding = audio_encoder(audio)
            seeg_embedding = seeg_encoder(seeg, seeg_padding_mask)

            # Flatten the output for later similarity computation
            audio_embedding = audio_embedding.flatten(1, 2)
            seeg_embedding = seeg_embedding.flatten(1, 2)

            if audio_embeddings is None:
                audio_embeddings = audio_embedding
                seeg_embeddings = seeg_embedding
            else:
                audio_embeddings = torch.cat((audio_embeddings, audio_embedding), dim=0)
                seeg_embeddings = torch.cat((seeg_embeddings, seeg_embedding), dim=0)

        # Compute similarity
        sim = torch.einsum('i d, j d -> i j', audio_embeddings, seeg_embeddings) * math.e
        labels = torch.arange(audio_embeddings.shape[0]).to(device)

        # Compute accuracy
        acc1, acc2 = evaluate(sim, labels, topk=(1, 2))
        print(f'{split} Acc@1 {acc1.item():.3f}')
        print(f'{split} Acc@2 {acc2.item():.3f}')
        return acc1.item(), acc2.item()


def train(epoch, audio_encoder, seeg_encoder, optimizer, train_loader, writer, device):
    audio_encoder.train()
    seeg_encoder.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    iteration = len(train_loader) * epoch

    for audio, seeg, seeg_padding_mask in tqdm(train_loader):
        batch_size = audio.shape[0]

        audio = audio.to(device)
        seeg = seeg.to(device)
        seeg_padding_mask = seeg_padding_mask.to(device)

        optimizer.zero_grad()

        # Forward
        audio_embedding = audio_encoder(audio)
        seeg_embedding = seeg_encoder(seeg, seeg_padding_mask)

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
    data_file = args.data_file
    train_ratio = args.train_ratio
    train_dataset = CustomDataset(data_file=data_file, train_ratio=train_ratio, split='train')
    test_dataset = CustomDataset(data_file=data_file, train_ratio=train_ratio, split='test')
    val_dataset = CustomDataset(data_file=data_file, train_ratio=train_ratio, split='val')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # define the audio encoder
    num_output_channels = args.num_output_channels
    audio_encoder = AudioEncoder(output_channels=num_output_channels).to(device)

    # define the seeg encoder
    num_input_channels = 84
    input_length = 6443
    output_length = 314  # 314 is the default output length from the audio encoder
    num_heads = args.num_heads
    num_encoder_layers = args.num_encoder_layers
    dim_feedforward = args.dim_feedforward
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

    best_val_acc1 = 0
    best_epoch = 0

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

        # Validation
        acc1, _ = eval(audio_encoder, seeg_encoder, val_loader, device, split='val')

        # Save the best model
        if acc1 > best_val_acc1:
            best_val_acc1 = acc1
            best_epoch = epoch
            torch.save(audio_encoder.state_dict(), os.path.join(ckpt_folder, f'audio_encoder_epoch_{epoch}.pth'))
            torch.save(seeg_encoder.state_dict(), os.path.join(ckpt_folder, f'seeg_encoder_epoch_{epoch}.pth'))

    writer.close()

    # Load the best model
    audio_encoder.load_state_dict(torch.load(os.path.join(ckpt_folder, f'audio_encoder_epoch_{best_epoch}.pth')))
    seeg_encoder.load_state_dict(torch.load(os.path.join(ckpt_folder, f'seeg_encoder_epoch_{best_epoch}.pth')))

    # Test
    eval(audio_encoder, seeg_encoder, test_loader, device, split='test')


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
    arg_parser.add_argument('--data_file', '-d', type=str, default='./data/data_segmented.npy',
                            help="path to the .npy file containing the data")
    arg_parser.add_argument('--train_ratio', '-r', type=float, default=0.7,
                            help="the ratio of training data to all data. 1/3 of the remaining data will be used for "
                                 "testing and 2/3 for validation")
    arg_parser.add_argument('--num_workers', '-w', type=int, default=4, help="number of workers for dataloader")
    arg_parser.add_argument('--num_output_channels', '-o', type=int, default=64,
                            help="number of output channels for the seeg encoder")
    arg_parser.add_argument('--num_heads', '-hh', type=int, default=3, help="number of heads for the seeg encoder")
    arg_parser.add_argument('--num_encoder_layers', '-n', type=int, default=6, help="number of encoder layers for the "
                                                                                    "seeg encoder")
    arg_parser.add_argument('--dim_feedforward', '-f', type=int, default=2048, help="dim_feedforward for the seeg "
                                                                                    "encoder")
    args = arg_parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    set_seeds(42)
    run(args)
