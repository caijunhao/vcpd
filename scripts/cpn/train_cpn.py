from cpn.datasets import CPNDataset, ToTensor, Permute, Float32
from cpn.trainer import cpn_trainer
from cpn.model import CPN
from cpn.criterions import CPNLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch
import argparse
import json
import os
torch.autograd.set_detect_anomaly(True)


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    with open(args.config, 'r') as config_file:
        cfg = json.load(config_file)
    train_transform = transforms.Compose([Permute(), Float32(), ToTensor()])
    test_transform = transforms.Compose([Permute(), Float32(), ToTensor()])
    train_dataset = CPNDataset(args.train_dir, cfg, padding=True, shuffle=True, transform=train_transform)
    test_dataset = CPNDataset(args.test_dir, cfg, padding=False, shuffle=False, transform=test_transform)
    train_loader = DataLoader(train_dataset, cfg['train']['batch_size'], shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, 1, shuffle=False, num_workers=args.num_workers)
    cpn = CPN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch = 0
    global_step = 0
    if args.model_path:
        model_name, epoch, global_step = args.model_path.split('/')[-1].split('_')  # only keep file name
        epoch = int(epoch) + 1
        global_step = epoch * len(train_dataset) // cfg['train']['batch_size']
        cpn.load_network_state_dict(device=device, pth_file=args.model_path)
    loss = CPNLoss(cfg['train'])
    optimizer = optim.Adam(cpn.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg['train']['gamma'])
    writer = SummaryWriter(log_dir=args.log)
    cpn_trainer(model=cpn,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                criterion=loss,
                optimizer=optimizer,
                scheduler=scheduler,
                writer=writer,
                path=args.log,
                curr_epoch=epoch,
                curr_step=global_step,
                num_epochs=args.num_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='path to the config file.')
    parser.add_argument('--log',
                        type=str,
                        required=True,
                        help='path to the save log files')
    parser.add_argument('--train_dir',
                        required=True,
                        type=str,
                        help='path to the training data set.')
    parser.add_argument('--test_dir',
                        required=True,
                        type=str,
                        help='path to the test data set.')
    parser.add_argument('--model_path',
                        default='',
                        type=str,
                        help='path to the pretrained model.')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='number of workers to load data.')
    parser.add_argument('--num_epochs',
                        default=50,
                        type=int,
                        help='number of training epochs.')
    parser.add_argument('--cuda_device',
                        default='0',
                        type=str,
                        help='id of nvidia device.')
    main(parser.parse_args())
