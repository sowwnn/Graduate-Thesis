import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import time
import wandb

from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, Activations, AsDiscrete, Transpose, Orientation, Flip

from monai.networks.nets import DynUNet, SegResNet
from model.v2.DynUnet_DDA import DynUNet_DDA
from libs.data.dataset import dataloader
from model.fusion.baseline import get_trained
from model.fusion.fusion_model import Fusion



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


def val_step(model, dyn, seg, val_loader):
    model.eval()
    post_trans = Compose([Activations(sigmoid="True"), AsDiscrete(threshold=0.5)])
    with torch.no_grad():
        for idx,inputs in enumerate(val_loader):
            name =  datalist['test'][idx]['image'][0].split('/')[-1].split('_t1ce.nii')[0]
            inputs = inputs["image"].to(device)
            
            print(f"Process: {name} ", end=" ")

            f_dyn =  sliding_window_inference(inputs=inputs,  roi_size=(128, 128, 128),  sw_batch_size=1,  predictor=dyn,  overlap=0.5,)
            f_seg =  sliding_window_inference(inputs=inputs,  roi_size=(128, 128, 128),  sw_batch_size=1,  predictor=seg,  overlap=0.5,)

            
            inputs = torch.cat([f_dyn, f_seg],1)
            val_outputs = sliding_window_inference(inputs=inputs,  roi_size=(128, 128, 128),  sw_batch_size=1,  predictor=model,  overlap=0.5,)
            val_outputs = post_trans(val_outputs)
            val_outputs = merge(val_outputs)
            out = sitk.GetImageFromArray(val_outputs.cpu().numpy().astype(np.uint16))
            sitk.WriteImage(out, f"/kaggle/working/predict/{name}.nii.gz")
            
            print("Done!")
            plt.imshow(val_outputs[87,:,:].cpu().detach())
            plt.show()
    print("All DONE!!!")


def main(data):
    ## Init model and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dyn, seg = get_trained(data['dynunet_trained'], data['segresnet_trained'])
    loader = dataloader(datalist, 1, stage="test")


    model = Fusion([16,32], 48, 48).to('cuda').to(device)
    checkpoint = torch.load(data['fusion_trained'] ,map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    del checkpoint 

    val_step(model, dyn, seg, loader)


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
    