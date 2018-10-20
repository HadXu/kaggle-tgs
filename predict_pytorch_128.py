import pandas as pd
from sklearn.model_selection import StratifiedKFold

import torch

from transform import *
from data_pytorch import TsgDataset
from models.model_pytorch import UNetResNet34_128
from metrics import intersection_over_union_thresholds
from utils import RLenc

name = 'unet_resnet34_torch_128'

BATCH_SIZE = 32


def valid_augment(index, image, mask):
    image = do_center_pad_to_factor(image, factor=32)
    mask = (mask >= 0.5).astype(np.float32)
    return index, image, mask


def do_eval(net, dataset):
    net.set_mode('eval')

    probs = np.zeros((len(dataset), 101, 101))
    truths = np.zeros((len(dataset), 101, 101))

    for i in range(len(dataset)):
        with torch.no_grad():
            index, image, y_mask, _ = dataset[i]

            hflip_image = np.array(image)[:, ::-1]
            images = np.array([image, hflip_image])
            images = torch.Tensor(images).cuda()

            logit = net(images)
            prob = logit.sigmoid()

            prob = prob.cpu().data.numpy().squeeze()
            mask = prob[0]
            hflip_mask = prob[1][:, ::-1]
            prob = (mask + hflip_mask) / 2
            prob = prob[13:128 - 14, 13:128 - 14]

            probs[i, :, :] = prob
            truths[i, :, :] = y_mask

    iou = intersection_over_union_thresholds(
        np.int32(truths > 0.5), np.int32(probs > 0.5))

    return iou


def do_test(net, dataset):
    net.set_mode('eval')

    probs = np.zeros((len(dataset), 101, 101))

    for i in range(len(dataset)):
        with torch.no_grad():
            index, image, y_mask, _ = dataset[i]

            hflip_image = np.array(image)[:, ::-1]
            images = np.array([image, hflip_image])
            images = torch.Tensor(images).cuda()

            logit = net(images)
            prob = logit.sigmoid()

            prob = prob.cpu().data.numpy().squeeze()
            mask = prob[0]
            hflip_mask = prob[1][:, ::-1]
            prob = (mask + hflip_mask) / 2
            prob = prob[13:128 - 14, 13:128 - 14]

            probs[i, :, :] = prob

    return probs


data = pd.read_csv('../input/data_ids_with_class.csv')
data_ids = data['id'].values
data_class = data['class'].values

test_ids = pd.read_csv('../input/test_ids.csv')['id'].values
test_dataset = TsgDataset(root='../input/test', image_ids=test_ids, augment=valid_augment, mode='test')

test_probs = np.zeros((len(test_dataset), 101, 101))

kfold = 5
cv = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=1926)

for cv_num, (train_idx, val_idx) in enumerate(cv.split(data_ids, data_class)):
    print('cv:', cv_num)

    train_ids, val_ids = data_ids[train_idx], data_ids[val_idx]

    valid_dataset = TsgDataset(root='../input/train', image_ids=val_ids, augment=valid_augment, mode='valid')

    net = UNetResNet34_128().cuda()

    net.load_state_dict(torch.load('../weights/{}_{}_lovasz_loss_clr_{}.th'.format(name, cv_num, 5)))
    iou = do_eval(net, valid_dataset)
    print('valid iou:', iou)
    test_probs += do_test(net, test_dataset)

test_probs /= kfold

pred_dict = {idx: RLenc(np.where(test_probs[i] >= 0.5, 1, 0)) for i, idx in enumerate(test_ids)}

sub = pd.DataFrame.from_dict(pred_dict, orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('results/cv_submission_128.csv')

np.save("results/test_probs_128.npy", test_probs)
