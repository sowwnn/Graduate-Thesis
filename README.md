# Graduate Thesis: Brain Tumor Segmentaion using Deep Learning.
In my thesis, I proposed two method to segmentation 3D Brain MRI.
First method using Attention mechanism to forcus on nessesary position.
Second one using Fusion method to combine multiple trained model.

Detail: [here](https://www.sowwn.dev/bachelorthesis)

____
## How to use?
### You need to clone my repo and setup it.

```bash
git clone https://github.com/RC-Sho0/Graduate-Thesis.git
```
Move to source code folder.

```bash
cd Graduate-Thesis
```
Set it up
```bash
python utils/setup.py <!your wandb key or empty>
```

Prepair you datalist
```bash
!python libs/data/prepare_datalist.py --path "<Your folder contain dataset>" --output "/{path of file}/datalist.json" --stage "train" --split 'true'
```

### For training
#### With my first method names 3D Dual-Domain Attention, you need to configure information like **exemple/exp.json**
```json
{
    "model_name": "dynunet", //[segresnet, dynunet, vnet, swinunetr, dynunet_dda]
    "att": [], //Only use if model_name is dynunet_dda else [] 
    "project": "st_baseline",
    "model_trained": null, //null for training stage, trained path for testing stage 
    "datalist": "temp/datalist.json", //your datalist
    "config":{
        "loss": "mse",
        "max_epochs": 120,
        "name":"dda_+",
        "lr":3e-4,
        "tmax": 30,
        "results_dir":"temp/results", //results dir
        "log": false //true if you want show on your wandb
    }
}   
```
**Training:**
```bash
python seg_train.py --input <your exp.json file>
```


#### For 3D Dual-Fusion Attention method use just need to upload fusion_train.ipynb in kaggle and training 🤣
----
## Predict
#### 3D Dual-Domain Attention
Fill model_trained in exp.json then run
```bash
python libs/data/prepare_datalist.py --path "<Your folder contain dataset>" --output "/{path of file}/datalist.json" --stage "test" 

python 3d_dda.py --input <your exp.json file>
```
#### 3D Dual-Fusion Attention
**You need to add 2 more variable in exp.json is:**
```json
...
    "model_name": "fusion", 
    ...
    "model_trained": null, //null for
    "dynunet_trained": <path of dynunet trained>,
    "segresnet_trained": <path of segresnet trained>,
...
```
**than run**
```zsh
python 3d_dda.py --input <your exp.json file>
```

***That all :3***

------
If you like that, please Star my repo 🌟 And if you want to support let follows my github 🎆

Authorized by [Sho0](https://www.sowwn.dev/about)


