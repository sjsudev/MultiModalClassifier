from __future__ import print_function, division
from flask import Flask, jsonify
from TorchClassifier.myTorchEvaluator import get_predictions
from TorchClassifier.Datasetutil.Torchdatasetutil import loadTorchdataset
from TorchClassifier.myTorchModels.TorchCNNmodels import createTorchCNNmodel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os

# Initialize Flask App
app = Flask(__name__)
model = None 
device = None

dataloaders, dataset_sizes, class_names, img_shape = loadTorchdataset('MNIST', 'torchvisiondataset', './../ImageClassificationData', 28, 28, 32)

numclasses = len(class_names)
model_ft = createTorchCNNmodel('mlpmodel1', numclasses, img_shape)

modelpath=os.path.join('./outputs/', 'model_best.pt')
model_ft.load_state_dict(torch.load(modelpath))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

@app.route('/predict', methods=['POST'])
def predict():
    images, labels, probs = get_predictions(model_ft, dataloaders['test'], device)

    return jsonify({'labels': labels})

if __name__ == '__main__':
    app.run()