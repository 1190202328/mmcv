import socket

import wandb

wandb.init(
    project="mmcv_test",
    entity="jingjiang",
    notes=socket.gethostname(),
    name="mmcv_test",
    job_type="training",
    reinit=True
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import EpochBasedRunner
from mmcv.utils import get_logger
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.transformer import BaseTransformerLayer


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        # conv + bn + relu
        self.conv1 = ConvModule(3, 128, 3, norm_cfg=dict(type='BN'))
        # conv + bn + relu
        self.conv2 = ConvModule(128, 256, 3, norm_cfg=dict(type='BN'))
        self.transformer = BaseTransformerLayer()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        print(x.shape)
        x = self.transformer(x)
        print(x.shape)

        raise Exception
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_step(self, data, optimizer):
        images, labels = data
        predicts = self(images)  # -> self.__call__() -> self.forward()
        loss = self.loss_fn(predicts, labels)
        wandb.log({"loss": loss})
        return {'loss': loss}

    def val_step(self, data, opimizer):
        images, labels = data
        with torch.no_grad():
            predicts = self(images)
            predicts = torch.argmax(predicts, dim=1)
        correct = labels.shape[0]
        acc = (labels == predicts).sum()
        wandb.log({'acc': acc.cpu().data / correct})
        return {'gt': labels, 'pred': predicts}


if __name__ == '__main__':
    wandb.config = {
        "lr": 0.001,
        "momentum": 0.9,
        "epochs": 4,
        "batch_size": 128
    }
    model = Model()
    if torch.cuda.is_available():
        # only use gpu:0 to train
        # Solved issue https://github.com/open-mmlab/mmcv/issues/1470

        model = MMDataParallel(model.cuda(), device_ids=[0])

    # dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = CIFAR10(
        root='data', train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_set, batch_size=wandb.config['batch_size'], shuffle=True, num_workers=2)
    test_set = CIFAR10(
        root='data', train=False, download=True, transform=transform)
    test_loader = DataLoader(
        test_set, batch_size=wandb.config['batch_size'], shuffle=True, num_workers=2)

    optimizer = optim.SGD(model.parameters(), lr=wandb.config['lr'], momentum=wandb.config['momentum'])
    logger = get_logger('mmcv')
    # runner is a scheduler to manage the training
    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir='./work_dir',
        logger=logger,
        max_epochs=wandb.config['epochs'])

    # learning rate scheduler config
    lr_config = dict(policy='step', step=[2, 3])
    # configuration of optimizer
    optimizer_config = dict(grad_clip=None)
    # configuration of saving checkpoints periodically
    checkpoint_config = dict(interval=1)
    # save log periodically and multiple hooks can be used simultaneously
    log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

    # register hooks to runner and those hooks will be invoked automatically
    runner.register_training_hooks(
        lr_config=lr_config,
        optimizer_config=optimizer_config,
        checkpoint_config=checkpoint_config,
        log_config=log_config)

    wandb.watch(model)
    print('***')
    print(type(runner))
    runner.run([train_loader, test_loader], [('train', 1), ('val', 1)])

    # runner.run([test_loader], [('test', 1)])
    # 运行命令 单卡 python train.py
    # 运行命令[x]，目前没有实现 多卡 python -m torch.distributed.launch --nproc_per_node=4 train.py
