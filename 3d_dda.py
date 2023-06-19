import torch
from torch import nn
import json
import os
import time
import numpy as np
import argparse
# import SimpleITK as sitk
import matplotlib.pyplot as plt
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, Activations, AsDiscrete, Transpose, Orientation, Flip

from model.st.get_baseline import get_model
from libs.data.dataset import dataloader
from libs.data.st_dataset import dataloader as st_dataloader

def merge(pred):
    
    pred = torch.squeeze(pred,0)
    c,w,h,d = pred.shape
    out = torch.zeros((5,w,h,d))
    out[1] = pred[0]*2
    out[2] = pred[1]*1
    out[4] = pred[2]*3
    out = torch.argmax(out,0)
    out =  Transpose((2,1,0))(out)
    return out

def predict(model, val_loader, datalist):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    post_trans = Compose([Activations(sigmoid="True"), AsDiscrete(threshold=0.5)])
    with torch.no_grad():
        for idx,inputs in enumerate(val_loader):
            name =  datalist['test'][idx]['image'][0].split('/')[-1].split('_t1ce.nii')[0]
            inputs = inputs["image"].to(device)
            
            print(f"Process: {name} ", end=" ")
            
            val_outputs = sliding_window_inference(inputs=inputs,  roi_size=(128, 128, 128),  sw_batch_size=1,  predictor=model,  overlap=0.5,)
            val_outputs = post_trans(val_outputs)
            val_outputs = merge(val_outputs)
            # out = sitk.GetImageFromArray(val_outputs.cpu().numpy().astype(np.uint16))
            # sitk.WriteImage(out, f"/kaggle/working/predict/{name}.nii.gz")
            
            print("Done!")
            plt.imshow(val_outputs[87,:,:].cpu().detach())
            plt.show()
    print("All DONE!!!")

def main(data):
    ## Init model and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(data['model_name'], data['att'])
    model = model.to(device)

    ##Init dataloader
    datalist = data['datalist']
    with open(datalist) as f:
        datalist = json.load(f)

    loader = dataloader(datalist['test'], 1, 'test', True) 

    ##Init method and params
    config = data['config']

    checkpoint = torch.load(data['model_trained'],map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])

    logger = None
    if config['log'] == True:
        logger = wandb.init(project=data['project'], name = config['name'], config=config, dir="/kaggle/input/pretrain/BrainTumour_Seg/")

    ## Run
    predict(model, loader, datalist)
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--input",
        type=str,
        default="/content/exp.json",
        help="expriment configuration",
    )
    args = parser.parse_args()
    f = open(args.input)
    data = json.load(f)

    main(data)
    