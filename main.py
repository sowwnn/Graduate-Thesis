import os
import time
from monai.data import decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import wandb
import json


from libs.data.MSD_data import get_loader
from train import run

with open("/kaggle/working/datalist.json") as f:
    datalist = json.load(f)

config = {
    "max_epochs": 50,
    "name":"test",
    "lr":1e-4,
    "results_dir":"/kaggle/results",
}
# wandb.init(project="test", config['name']='test', config=config)
        
loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), config['lr'], weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['max_epochs'])

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

logger = None
run(model, train_loader, val_loader, optimizer, loss_function, lr_scheduler, dice_metric, dice_metric_batch, logger, config)