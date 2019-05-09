import argparse
import torch

from model.model import Model
from data.dataloader import PreprocessDataset

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default='runs/run_conv/38_checkpoint.pth.tar', type=str, help='')
parser.add_argument('--iter_size', default=1, type=int, help='Size of a training mini-batch.')
parser.add_argument('--grad_clip', default=1., type=float, help='Gradient clipping threshold.')
opt = parser.parse_args()

checkpoint_path = opt.checkpoint_path
model = Model(opt)
pretrain_model = torch.load(checkpoint_path)
model.load_state_dict(pretrain_model['model'])
model.val_start()

test_dataset = PreprocessDataset(opt, 'test')

def my_job(cnt):
    name = test_dataset.id_list[cnt]
    feature, _,  = test_dataset.__getitem__(cnt)
    with torch.no_grad():
        out = model.forward(feature.view(1, 1024, -1).cuda())
    
    ''' save the results or evaluate directly '''

for cnt in range(test_dataset):
    my_job(cnt)
        

