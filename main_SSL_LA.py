import argparse
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from core_scripts.startup_config import set_random_seed
from data_utils_SSL import genSpoof_list, Dataset_ASVspoof2019_train, Dataset_ASVspoof2021_eval
from eval_metric_LA import compute_eer
from model import Model


__author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"


def evaluate_dev_set(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    bona_scores = []
    spoof_scores = []
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    with torch.no_grad():
        for batch_x, batch_y in dev_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            batch_out = model(batch_x)
            batch_loss = criterion(batch_out, batch_y)
            val_loss += (batch_loss.item() * batch_size)

            batch_score = batch_out[:, 1].detach().cpu().numpy().ravel()
            batch_label = batch_y.detach().cpu().numpy().ravel()
            bona_scores.extend(batch_score[batch_label == 1].tolist())
            spoof_scores.extend(batch_score[batch_label == 0].tolist())

    val_loss /= max(num_total, 1.0)
    bona_scores = np.asarray(bona_scores)
    spoof_scores = np.asarray(spoof_scores)
    eer, threshold = compute_eer(bona_scores, spoof_scores)
    return val_loss, 100.0 * eer, threshold


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)
    model.eval()

    if os.path.exists(save_path):
        os.remove(save_path)

    with torch.no_grad():
        for batch_x, utt_id in data_loader:
            batch_x = batch_x.to(device)
            batch_out = model(batch_x)
            batch_score = batch_out[:, 1].data.cpu().numpy().ravel()

            with open(save_path, 'a+') as fh:
                for f, cm in zip(utt_id, batch_score.tolist()):
                    fh.write(f'{f} {cm}\n')
    print('Scores saved to {}'.format(save_path))


def train_epoch(train_loader, model, optimizer, device):
    running_loss = 0.0
    num_total = 0.0
    model.train()

    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for batch_x, batch_y in train_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)

        running_loss += (batch_loss.item() * batch_size)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    running_loss /= max(num_total, 1.0)
    return running_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system with SSL+logMel fusion')
    parser.add_argument('--database_path', type=str, default='/your/path/to/data/ASVspoof_database/LA/', help='Database root directory.')
    parser.add_argument('--protocols_path', type=str, default='database/', help='Protocol directory path')

    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1234)')
    parser.add_argument('--model_path', type=str, default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None, help='Comment to describe the saved model')
    parser.add_argument('--track', type=str, default='LA', choices=['LA', 'PA', 'DF'], help='LA/PA/DF')
    parser.add_argument('--eval_output', type=str, default=None, help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False, help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False, help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--mel_bins', type=int, default=128)
    parser.add_argument('--mel_n_fft', type=int, default=512)
    parser.add_argument('--mel_win_length', type=int, default=400)
    parser.add_argument('--mel_hop_length', type=int, default=160)
    parser.add_argument('--fusion_heads', type=int, default=4)
    parser.add_argument('--ssl_model_path', type=str, default='/root/autodl-tmp/xlsr_300m_hf',
                        help='Local path or HuggingFace name for XLS-R/Wav2Vec2 front-end.')
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', default=True, help='use cudnn-deterministic? (default true)')
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', default=False, help='use cudnn-benchmark? (default false)')

    parser.add_argument('--algo', type=int, default=5,
                        help='Rawboost algos. 0:none, 1:LnL, 2:ISD, 3:SSI, 4:(1+2+3), 5:(1+2), 6:(1+3), 7:(2+3), 8:(1||2)')
    parser.add_argument('--nBands', type=int, default=5)
    parser.add_argument('--minF', type=int, default=20)
    parser.add_argument('--maxF', type=int, default=8000)
    parser.add_argument('--minBW', type=int, default=100)
    parser.add_argument('--maxBW', type=int, default=1000)
    parser.add_argument('--minCoeff', type=int, default=10)
    parser.add_argument('--maxCoeff', type=int, default=100)
    parser.add_argument('--minG', type=int, default=0)
    parser.add_argument('--maxG', type=int, default=0)
    parser.add_argument('--minBiasLinNonLin', type=int, default=5)
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20)
    parser.add_argument('--N_f', type=int, default=5)
    parser.add_argument('--P', type=int, default=10)
    parser.add_argument('--g_sd', type=int, default=2)
    parser.add_argument('--SNRmin', type=int, default=10)
    parser.add_argument('--SNRmax', type=int, default=40)

    args = parser.parse_args()

    if not os.path.exists('models'):
        os.mkdir('models')

    set_random_seed(args.seed, args)

    track = args.track
    assert track in ['LA', 'PA', 'DF'], 'Invalid track given'

    prefix = 'ASVspoof_{}'.format(track)
    prefix_2019 = 'ASVspoof2019.{}'.format(track)
    prefix_2021 = 'ASVspoof2021.{}'.format(track)

    model_tag = 'model_{}_{}_{}_{}_{}_logmel_fusion'.format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)
    os.makedirs(model_save_path, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    model = Model(args, device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model = model.to(device)
    print('nb_params:', nb_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print('Model loaded : {}'.format(args.model_path))

    if args.eval:
        file_eval = genSpoof_list(dir_meta=os.path.join(args.protocols_path + '{}_cm_protocols/{}.cm.eval.trl.txt'.format(prefix, prefix_2021)), is_train=False, is_eval=True)
        print('no. of eval trials', len(file_eval))
        eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval, base_dir=os.path.join(args.database_path + 'ASVspoof2021_{}_eval/'.format(args.track)))
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        sys.exit(0)

    d_label_trn, file_train = genSpoof_list(dir_meta=os.path.join(args.protocols_path + '{}_cm_protocols/{}.cm.train.trn.txt'.format(prefix, prefix_2019)), is_train=True, is_eval=False)
    print('no. of training trials', len(file_train))
    train_set = Dataset_ASVspoof2019_train(args, list_IDs=file_train, labels=d_label_trn,
                                           base_dir=os.path.join(args.database_path + '{}_{}_train/'.format(prefix_2019.split('.')[0], args.track)),
                                           algo=args.algo)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=True)
    del train_set, d_label_trn

    d_label_dev, file_dev = genSpoof_list(dir_meta=os.path.join(args.protocols_path + '{}_cm_protocols/{}.cm.dev.trl.txt'.format(prefix, prefix_2019)), is_train=False, is_eval=False)
    print('no. of validation trials', len(file_dev))
    dev_set = Dataset_ASVspoof2019_train(args, list_IDs=file_dev, labels=d_label_dev,
                                         base_dir=os.path.join(args.database_path + '{}_{}_dev/'.format(prefix_2019.split('.')[0], args.track)),
                                         algo=0)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, num_workers=8, shuffle=False)
    del dev_set, d_label_dev

    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    best_eer = float('inf')
    best_epoch = -1
    best_ckpt = os.path.join(model_save_path, 'best_model.pth')
    stats_path = os.path.join(model_save_path, 'best_result.txt')

    for epoch in range(num_epochs):
        running_loss = train_epoch(train_loader, model, optimizer, device)
        val_loss, dev_eer, dev_threshold = evaluate_dev_set(dev_loader, model, device)

        writer.add_scalar('train_loss', running_loss, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('dev_eer', dev_eer, epoch)

        improved = dev_eer < best_eer
        if improved:
            best_eer = dev_eer
            best_epoch = epoch
            torch.save(model.state_dict(), best_ckpt)
            with open(stats_path, 'w') as fh:
                fh.write('best_epoch={}\n'.format(best_epoch))
                fh.write('best_dev_eer={:.6f}\n'.format(best_eer))
                fh.write('best_dev_threshold={:.10f}\n'.format(dev_threshold))
                fh.write('train_loss={:.10f}\n'.format(running_loss))
                fh.write('val_loss={:.10f}\n'.format(val_loss))

        print('Epoch {:03d} | train_loss {:.6f} | val_loss {:.6f} | dev_EER {:.4f}% | best_EER {:.4f}% @ epoch {}'.format(
            epoch, running_loss, val_loss, dev_eer, best_eer, best_epoch
        ))

    writer.close()
    print('Training finished. Best model saved to {}'.format(best_ckpt))
