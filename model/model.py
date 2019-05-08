import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm

from net import Net
from tools import accuracy

class Model(object):
    def __init__(self, opt):
        # hyperparameters you may set in command line
        self.xx = opt.xx
        self.xxx = opt.xxx
        self.iter_size = opt.iter_size
        self.grad_clip = opt.grad_clip
        
        self.net = Net()
        if torch.cuda.is_available():
            self.net.cuda()
            cudnn.benchmark = True

        # set your loss function
        self.criterion = nn.CrossEntropyLoss()

        params = list(self.net.parameters())
        self.params = params
        # choose your optimizer
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate, weight_decay=opt.weight_decay)
        self.Eiters = 0
    
    def state_dict(self):
        # get state_dict for save model
        state_dict = self.net.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        # load state dict from checkpoint or pretrain
        model_dict = self.net.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(state_dict)
        self.net.load_state_dict(model_dict)
    
    def train_start(self):
        self.net.train()

    def val_start(self):
        self.net.eval()

    def forward(self, feature):
        out = self.net(feature)
        return out
    
    def forward_loss(self, out, label, **kwargs):
        loss = self.criterion(out, label)
        acc = accuracy(out, label) # notice: out should be reshaped to (n*c) and label to (n)
        
        # log your train status on your screen
        self.logger.update('Loss', loss.data[0], out.size(0)) # out.size(0) is the number of this batch data for calculate average loss
        self.logger.update('acc', acc[0].data[0], out.size(0))
        return loss
    
    def train_forwad(self, feature, label):
        self.Eiters += 1
        if torch.cuda.is_available():
            feature = feature.cuda()
            label = label.cuda()
        
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        out = self.forward(feature)
        loss = self.forward_loss(out, label)
        loss.backward()
        if self.Eiters % self.iter_size == 0:
            if self.iter_size != 1:
                for g in self.optimizer.param_groups:
                    for p in g['params']:
                        p.grad /= self.iter_size
            if self.grad_clip > 0:
                total_norm = clip_grad_norm(self.params, self.grad_clip)
                if total_norm > self.grad_clip:
                    print('clipping gradient: {} with coef {}'.format(total_norm, self.grad_clip / total_norm))
            self.optimizer.step()
            self.optimizer.zero_grad()
    
    def val_forward(self, feature, label):
        if torch.cuda.is_available():
            feature = feature.cuda()
            label = label.cuda()
        
        out = self.forward(feature)
        loss = self.forward_loss(out, label)
        return loss

