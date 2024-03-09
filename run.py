#!/usr/bin/env python3

import argparse
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import tqdm 

def main(args):

    # Define transformation for the validation data
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the ImageNet validation dataset
    train_dataset = datasets.ImageFolder(os.path.join(args["data"], "train"), transform=transform)

    # Use a pre-trained Wide ResNet model
    if args["big"] == "wide_resnet50_2":
        model = models.wide_resnet50_2(pretrained=True)
    else:
        raise ValueError(f"Wrong model name given. I don't know the model {args['big']}.")

    # Set the model to evaluation mode
    model.eval()

    # Define the device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the appropriate device
    model.to(device)

    # Create a DataLoader for batching
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args["b"], shuffle=False, pin_memory=True, num_workers = 6)
    
    # Predict the class of each image in the validation dataset in batches
    n_correct = 0
    for i, (images, labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc = "Getting predictions of the big model"):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            preds = model(images)
            n_correct += (preds.argmax(1) == labels).sum()
    
    print(f"Accuracy is {n_correct/len(train_loader) * 100.0}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a multi-label classification problem on a series of patients. Training and evaluation are performed on a per-patient basis, i.e. we train on patients {1,2,3} and test on patient 4.')
    parser.add_argument("--data", help='Path to ImageNet data.', required=False, type=str, default="/mnt/ssd/data/ImageNet")
    parser.add_argument("--small", help='Path to ImageNet data.', required=False, type=str, default="")
    parser.add_argument("--big", help='Path to ImageNet data.', required=False, type=str, default="wide_resnet50_2")
    parser.add_argument("-b", help='Batch size.', required=False, type=int, default=64)
    parser.add_argument("--rejector", help='Rejector.', required=False, type=str, default="DecisionTreeClassifier")
    parser.add_argument("-p", help='Budget to try.', required=False, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--out", help='Name / Path of output csv.', required=False, type=str, default="imagenet.csv")
    args = vars(parser.parse_args())
    
    main(args)