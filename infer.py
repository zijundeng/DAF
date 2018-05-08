import numpy as np
import os

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import testing_root
from misc import check_mkdir
from misc import crf_refine
from model import DAF

torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = './ckpt'
exp_name = 'DAF'
args = {
    'snapshot': ''
}

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_pil = transforms.ToPILImage()


def main():
    net = DAF().cuda()

    if len(args['snapshot']) > 0:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))

    net.eval()

    for idx, img_name in enumerate(os.listdir(os.path.join(testing_root, 'us'))):
        print 'predicting %d' % (idx + 1)
        check_mkdir(os.path.join(ckpt_path, exp_name, 'prediction_' + args['snapshot']))
        img = Image.open(os.path.join(testing_root, 'us', img_name)).convert('RGB')
        img_var = Variable(img_transform(img).unsqueeze(0)).cuda()
        prediction = np.array(to_pil(net(img_var).data.squeeze(0).cpu()))
        prediction = crf_refine(np.array(img), prediction)

        Image.fromarray(prediction).save(os.path.join(ckpt_path, exp_name, 'prediction_' + args['snapshot'], img_name))


if __name__ == '__main__':
    main()
