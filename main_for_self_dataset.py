# pytorch libs

from plotting import plot_curves
import torch
from torch import nn
import torchvision
import os

# numpy
import numpy as np

# torch metrics

from torchmetrics import Accuracy

from torch.utils.data import DataLoader

from torchvision import transforms
import wandb
import config

wandb.login(key=config.API_KEY)
print("[LOG]: Login Succesfull.")


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] current used device: {device}")


# Getting DATASET

# defining transform
transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.TrivialAugmentWide(num_magnitude_bins=3),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    # transforms.Resize(224),
    transforms.ToTensor()
])


dataset = torchvision.datasets.ImageFolder("./data/custom/self_dataset_sketch",transform=transform_train)

# print(f"[INFO] dataset size: {len(dataset)}")
# print(f"[INFO] dataset classes length: {len(dataset.classes)}")
# print(f"[INFO] dataset class to idx mapping: {dataset.class_to_idx}")
# print(f"[INFO] datset[0] shape: {dataset[0][0].shape}")

torch.manual_seed(42)
torch.cuda.manual_seed(42)
random_seed = 42

split_size = int(0.8 * len(dataset))

train_sample_idx = torch.randperm(len(dataset)).tolist()[:split_size]
test_sample_idx = torch.randperm(len(dataset)).tolist()[split_size:]

from torch.utils.data import Subset

train_dataset = Subset(dataset=dataset,indices=train_sample_idx)
test_dataset = Subset(dataset=dataset,indices=test_sample_idx)

# converting data into torch dataloader
import os
BATCH_SIZE = 64
NUM_WORKERS = 4

train_dataloader = DataLoader(
    train_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers=NUM_WORKERS
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size = BATCH_SIZE,
    shuffle = False,
    num_workers=NUM_WORKERS
)

# Testing one batch from train dataloader
# print("\n\n1st batch form train dataloader")
# print(next(iter(train_dataloader))[1])

# Importing model
from models import EarlyStopping, Resnet18, Resnet50, VGG16, MobileNetV2

def get_resnet_18_model():
    model = Resnet18()
    return model

def get_resnet_50_model():
    model = Resnet50()
    return model


def get_vgg16_model():
    model = VGG16()
    return model


def get_mobilenet_v2_model():
    model = MobileNetV2()
    return model



# load custom weights function #
def load_custom_weights(model:nn.Module, path:str, num_classes:int=10):
    global mode
    try:
        model.load_state_dict(torch.load(path))
        
        # freezing all layers
        if mode == "freeze":
            print(f"[INFO] Freezing base layers")
            for param in model.parameters():
                param.requires_grad = False
            
        # changing last layer
        print(f"[INFO] model class name: {model.__class__.__name__}")
        if model.__class__.__name__ == "Resnet18":
            model.resnet.fc = nn.Linear(512, num_classes, bias=True)
        if model.__class__.__name__ == "Resnet50":
            model.resnet.fc = nn.Linear(2048, num_classes, bias=True)
        if model.__class__.__name__ == "VGG16":
            model.vgg.classifier = nn.Sequential(
                nn.Linear(in_features=25088, out_features=4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=4096, out_features=4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=4096, out_features=num_classes, bias=True),
            )
        if model.__class__.__name__ == "MobileNetV2":
            model.mobilenet.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=1280, out_features=num_classes, bias=True),
            )
        model.eval()
        # print(f"[INFO] model.fc: {model.fc}")
        print(f"[INFO] model loaded with custom weights.")
    except:
        raise Exception("Error while loading model weights. Please check model architecture.")

# Train Info
# Early stopping
early_stopping = EarlyStopping(tolerance=3, min_delta=0.001)

# Training model on train data
from engine import train
from timeit import default_timer as timer 

# Hyperparms
lr = [1e-3,1e-4] # learning rate
betas=[(0.8, 0.888)] # coefficients used for computing running averages of gradient and its square
eps = [1e-8] # term added to the denominator to improve numerical stability
weight_decay = [1e-3] # weight decay (L2 penalty)

# init. epochs
NUM_EPOCHS = [10,15]

parms_combs = [(l,b,e,w_d,epochs) for l in lr for b in betas for e in eps for w_d in weight_decay for epochs in NUM_EPOCHS]

# init. loss function, accuracy function and optimizer
loss_fn = nn.CrossEntropyLoss()
acc_fn = Accuracy(task="multiclass", num_classes=7).to(device=device)

cur,total = 1, len(lr)*len(betas)*len(eps)*len(weight_decay)*len(NUM_EPOCHS)
for h_parms in parms_combs:
    wandb.init(
        # set the wandb project where this run will be logged
        project="cv-project-2",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": h_parms[0],
        "weight_decay": h_parms[3],
        "architecture": "MobilenetV2",
        "comment": "MobilenetV2 with custom weights from Imagenet Sketch without freezing base layers",
        "dataset": "DATASET-SELF-SPLIT-80-20",
        "epochs": h_parms[4],
        }
    )

    ### INIT MODEL STARTS ###
    # traning same model for each parms
    mode = "unfreeze" # freeze / unfreeze
    model = get_mobilenet_v2_model()
    load_custom_weights(model=model, path="./models/MobileNetV2_epoch_50_optim_adam_lr_0.0001_betas_(0.8, 0.888)_eps_1e-08_weight_decay_0.001.pth", num_classes=7)
    model = model.to(device=device)
    ### INIT MODEL END ###

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=h_parms[0], betas=h_parms[1], eps=h_parms[2],weight_decay=h_parms[3]
    )

    # importing and init. the timer for checking model training time
    from timeit import default_timer as timer

    start_time = timer()
    print(f"current exp / total: {cur} / {total}")
    print(f"Training with: lr: {h_parms[0]}, betas: {h_parms[1]}, eps: {h_parms[2]}, weight_decay: {h_parms[3]}")

    model_results = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        acc_fn=acc_fn,
        epochs=h_parms[4],
        save_info=f"lr_{h_parms[0]}_betas_{h_parms[1]}_eps_{h_parms[2]}_weight_decay_{h_parms[3]}_dataset_DATASET-SELF-SPLIT-80-20_freeze_base_layers",
        device=device
    )

    # end timer
    end_time = timer()
    # printing time taken
    print(f"total training time: {end_time-start_time:.3f} sec.")
    # print("model stats:")
    # print(model_0_results)
    print(f"LOSS & Accuracy Curves\n"
        f"lr: {h_parms[0]}, betas: {h_parms[1]}, eps: {h_parms[2]}, weight_decay: {h_parms[3]}")
    plot_curves(model_results,f"{model.__class__.__name__}_epoch_{h_parms[4]}_optim_adam_"
                +
                f"lr_{h_parms[0]}_betas_{h_parms[1]}_eps_{h_parms[2]}_weight_decay_{h_parms[3]}_dataset_DATASET-SELF-SPLIT-80-20")
    cur+=1
    print()