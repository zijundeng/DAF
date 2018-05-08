import datetime
import os

import scipy.io as sio
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from config import training_root
from dataset import ImageFolder
from misc import AvgMeter, check_mkdir
from model import DAF

torch.cuda.set_device(0)

ckpt_path = './ckpt'
exp_name = 'DAF'
args = {
    'iter_num': 1200,
    'train_batch_size': 4,
    'lr': 5e-3,
    'lr_step': 600,
    'lr_decay': 50,
    'weight_decay': 1e-2,
    'momentum': 0.9
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

train_set = ImageFolder(training_root, None, transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=12, shuffle=True)

bce_logit = nn.BCEWithLogitsLoss().cuda()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')


def main():
    net = DAF().cuda().train()

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)


def train(net, optimizer):
    curr_iter = args['last_iter']
    while True:
        train_loss_record, loss0_record, loss1_record, loss2_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        loss3_record, loss0_2_record, loss1_2_record = AvgMeter(), AvgMeter(), AvgMeter()
        loss2_2_record, loss3_2_record = AvgMeter(), AvgMeter()

        for i, data in enumerate(train_loader):
            if curr_iter == args['lr_step']:
                optimizer.param_groups[0]['lr'] = 2 * args['lr'] / args['lr_decay']
                optimizer.param_groups[1]['lr'] = args['lr'] / args['lr_decay']

            inputs, labels = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()
            outputs0, outputs1, outputs2, outputs3, outputs0_2, outputs1_2, outputs2_2, outputs3_2 = net(inputs)
            loss0 = bce_logit(outputs0, labels)
            loss1 = bce_logit(outputs1, labels)
            loss2 = bce_logit(outputs2, labels)
            loss3 = bce_logit(outputs3, labels)
            loss0_2 = bce_logit(outputs0_2, labels)
            loss1_2 = bce_logit(outputs1_2, labels)
            loss2_2 = bce_logit(outputs2_2, labels)
            loss3_2 = bce_logit(outputs3_2, labels)

            loss = loss0 + loss1 + loss2 + loss3 + loss0_2 + loss1_2 + loss2_2 + loss3_2
            loss.backward()
            optimizer.step()

            train_loss_record.update(loss.data[0], batch_size)
            loss0_record.update(loss0.data[0], batch_size)
            loss1_record.update(loss1.data[0], batch_size)
            loss2_record.update(loss2.data[0], batch_size)
            loss3_record.update(loss3.data[0], batch_size)
            loss0_2_record.update(loss0_2.data[0], batch_size)
            loss1_2_record.update(loss1_2.data[0], batch_size)
            loss2_2_record.update(loss2_2.data[0], batch_size)
            loss3_2_record.update(loss3_2.data[0], batch_size)

            log = '[iter %d], [train loss %.5f], [loss0 %.5f], [loss1 %.5f], [loss2 %.5f], [loss3 %.5f], [loss0_2 %.5f], [loss1_2 %.5f], [loss2_2 %.5f], [loss3_2 %.5f], [lr %.13f]' % \
                  (curr_iter, train_loss_record.avg, loss0_record.avg, loss1_record.avg, loss2_record.avg,
                   loss3_record.avg, loss0_2_record.avg, loss1_2_record.avg, loss2_2_record.avg, loss3_2_record.avg,
                   optimizer.param_groups[1]['lr'])
            print log
            open(log_path, 'a').write(log + '\n')

            if curr_iter > args['iter_num']:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                return


if __name__ == '__main__':
    main()
