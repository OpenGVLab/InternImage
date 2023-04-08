import warnings
import argparse
import logging

import pytorch_lightning as pl
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
import torch
from torch import optim as optim
from accelerate.logging import get_logger
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from dataset import build_loader2
from models import build_model
from config import get_config
from utils import load_pretrained

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


def parse_option():
    parser = argparse.ArgumentParser(
        'InternImage training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--dataset', type=str, help='dataset name', default=None)
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                        'full: cache all data, '
                        'part: sharding the dataset into nonoverlapping pieces and only cache one piece'
                        )
    parser.add_argument('--pretrained', help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)'
                        )
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--save_dir', default='saved', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--save-ckpt-num', default=1, type=int)

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config


def build_criterion(config):
    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(
            smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    return criterion


class InternImageClassification(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = build_model(config)
        self.config = config
        self.criterion = build_criterion(config)
        load_pretrained(config, self.model, logging)

    def training_step(self, batch, batch_idx):
        samples, targets = batch
        outputs = self.model(samples)
        loss = self.criterion(outputs, targets)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=2e-5)
        return optimizer


def main():
    args, config = parse_option()
    _, _, _, train_loader, val_loader, _, _ = build_loader2(config)
    model = InternImageClassification(config)

    trainer = pl.Trainer(
        max_epochs=1,
        gpus=4,
        accelerator="gpu",
        strategy=DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            cpu_checkpointing=True,
        ),
        precision=16,
    )
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == '__main__':
    main()
