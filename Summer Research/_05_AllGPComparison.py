import torch
import math
import gpytorch
import time
from numpy.random import randn, seed
import random
from statistics import mean

from GPModels.GP import ExactGPModel
from GPModels.RFF import RFFGPModel
from GPModels.ORF import ORFGPModel
from GPModels.StructuredORF import StructuredORFGPModel
    

def oneDimensional(grid_size,noise_std):
    x = torch.linspace(0, 1, grid_size)
    
    true = torch.cos(1 * math.pi * x)
    noisy = true + noise_std * torch.randn(x.size(0))
    return(x,true,noisy)

def twoDimensional(grid_size,noise_std):
    x1 = torch.linspace(0, 1, grid_size)
    x2 = torch.linspace(0, 1, grid_size)
    x1, x2 = torch.meshgrid(x1, x2, indexing='ij')
    x = torch.stack([x1.reshape(-1),x2.reshape(-1)],dim=-1)
    
    true = torch.cos(1 * math.pi * x[:, 0]) + torch.cos(2 * math.pi * x[:, 1])
    noisy = true + noise_std * torch.randn(x.size(0))
    return(x,true,noisy)

def threeDimensional(grid_size,noise_std):
    x1 = torch.linspace(0, 1, grid_size)
    x2 = torch.linspace(0, 1, grid_size)
    x3 = torch.linspace(0, 1, grid_size)
    x1, x2, x3 = torch.meshgrid(x1, x2, x3, indexing='ij')
    x = torch.stack([x1.reshape(-1),x2.reshape(-1),x3.reshape(-1)],dim=-1)
    
    true = torch.cos(1 * math.pi * x[:, 0]) + torch.cos(2 * math.pi * x[:, 1]) + torch.cos(3 * math.pi * x[:,2])
    noisy = true + noise_std * torch.randn(x.size(0))
    return(x,true,noisy)

def trainGP(model,likelihood,train_x,train_y,name,training_iter,early_stop_loss,showTraining):
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    start = time.time()
    for iterations in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        if showTraining: print(f"{name} Iter {iterations+1}/{training_iter} - Loss: {loss.item():.3f}   noise: {likelihood.noise.item():.3f}")
        optimizer.step()
        if loss.item() < early_stop_loss:
            if showTraining: print(f"{name} stopping early: loss < {early_stop_loss}")
            break
    return time.time()-start, iterations+1

def evaluateGP(model, likelihood, test_x, true_y):
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        start = time.time()
        preds = model(test_x)
        mean = preds.mean
        totalTime = time.time() - start

        metrics = {}
        mse = torch.mean((preds.mean - true_y) ** 2).item()
        metrics = {
                "MSE": mse,
                "Time": totalTime
            }
    return mean, metrics

def main(dimensions,grid_size,noise_std,training_iter,early_stop_loss,features,KTrials,showTraining=False):
    grid_size = grid_size
    test_grid_size=grid_size
    noise_std = noise_std
    training_iter = training_iter
    early_stop_loss=early_stop_loss
    features=features
    if dimensions==1:
        train_x, true_y, train_y = oneDimensional(grid_size,noise_std)
        test_x = torch.linspace(0, 1, test_grid_size)
        test_y = torch.cos(1 * math.pi * test_x)
    if dimensions==2:
        train_x, true_y, train_y = twoDimensional(grid_size,noise_std)
        
        test_x1 = torch.linspace(0, 1, test_grid_size)
        test_x2 = torch.linspace(0, 1, test_grid_size)
        test_x1, test_x2 = torch.meshgrid(test_x1, test_x2, indexing='ij')
        test_x = torch.stack([test_x1.reshape(-1), test_x2.reshape(-1)], dim=-1)
        test_y = torch.cos(1 * math.pi * test_x[:, 0]) + torch.cos(2 * math.pi * test_x[:, 1])
    if dimensions==3:
        train_x, true_y, train_y = threeDimensional(grid_size,noise_std)
        
        test_x1 = torch.linspace(0, 1, test_grid_size)
        test_x2 = torch.linspace(0, 1, test_grid_size)
        test_x3 = torch.linspace(0, 1, test_grid_size)
        test_x1, test_x2, test_x3 = torch.meshgrid(test_x1, test_x2, test_x3, indexing='ij')
        test_x = torch.stack([test_x1.reshape(-1), test_x2.reshape(-1), test_x3.reshape(-1)], dim=-1)
        test_y = (torch.cos(1 * math.pi * test_x[:, 0]) + torch.cos(2 * math.pi * test_x[:, 1]) + torch.cos(3 * math.pi * test_x[:, 2]))
        
        
    GPTrainingTimes=[]
    GPIterationCounts=[]
    GPMSEs=[]
    GPTestingTimes=[]
    
    RFFTrainingTimes=[]
    RFFIterationCounts=[]
    RFFMSEs=[]
    RFFTestingTimes=[]
    
    ORFTrainingTimes=[]
    ORFIterationCounts=[]
    ORFMSEs=[]
    ORFTestingTimes=[]
    
    StructuredORFTrainingTimes=[]
    StructuredORFIterationCounts=[]
    StructuredORFMSEs=[]
    StructuredORFTestingTimes=[]
    
    torch.manual_seed(5)
    random.seed(5)
    seed(5)
    
    for trial in range(KTrials):
        print(f"Trial: {trial+1}/{KTrials}")
        #GP Model
        GPLikelihood = gpytorch.likelihoods.GaussianLikelihood()
        GPModel = ExactGPModel(train_x,train_y,GPLikelihood)
        
        GPTrainingTime, GPIterations = trainGP(GPModel,GPLikelihood,train_x,train_y,"GP",training_iter,0.05,showTraining)
        GPTrainingTimes.append(GPTrainingTime)
        GPIterationCounts.append(GPIterations)
        GPMean,GPMetrics = evaluateGP(GPModel,GPLikelihood,test_x,test_y)
        GPMSEs.append(GPMetrics["MSE"])
        GPTestingTimes.append(GPMetrics["Time"])
        
        #RFF
        RFFLikelihood = gpytorch.likelihoods.GaussianLikelihood()
        RFFModel = RFFGPModel(train_x,train_y,RFFLikelihood,num_rff_features=features,input_dim=dimensions)
        
        RFFTrainingTime, RFFIterations = trainGP(RFFModel,RFFLikelihood,train_x,train_y,"RFF",training_iter,early_stop_loss,showTraining)
        RFFTrainingTimes.append(RFFTrainingTime)
        RFFIterationCounts.append(RFFIterations)
        RFFMean,RFFMetrics = evaluateGP(RFFModel,RFFLikelihood,test_x,test_y)
        RFFMSEs.append(RFFMetrics["MSE"])
        RFFTestingTimes.append(RFFMetrics["Time"])
        
        #ORF
        ORFLikelihood = gpytorch.likelihoods.GaussianLikelihood()
        ORFModel = ORFGPModel(train_x,train_y,ORFLikelihood,num_orf_features=features,input_dim=dimensions)
        
        ORFTrainingTime, ORFIterations = trainGP(ORFModel,ORFLikelihood,train_x,train_y,"ORF",training_iter,early_stop_loss,showTraining)
        ORFTrainingTimes.append(ORFTrainingTime)
        ORFIterationCounts.append(ORFIterations)
        ORFMean,ORFMetrics = evaluateGP(ORFModel,ORFLikelihood,test_x,test_y)
        ORFMSEs.append(ORFMetrics["MSE"])
        ORFTestingTimes.append(ORFMetrics["Time"])
        
        #Structured ORF
        StructuredORFLikelihood = gpytorch.likelihoods.GaussianLikelihood()
        StructuredORFModel = StructuredORFGPModel(train_x,train_y,StructuredORFLikelihood,num_orf_features=features,input_dim=dimensions)
        
        StructuredORFTrainingTime, StructuredORFIterations = trainGP(StructuredORFModel,StructuredORFLikelihood,train_x,train_y,"StructuredORF",training_iter,early_stop_loss,showTraining)
        StructuredORFTrainingTimes.append(StructuredORFTrainingTime)
        StructuredORFIterationCounts.append(StructuredORFIterations)
        StructuredORFMean,StructuredORFMetrics = evaluateGP(StructuredORFModel,StructuredORFLikelihood,test_x,test_y)
        StructuredORFMSEs.append(StructuredORFMetrics["MSE"])
        StructuredORFTestingTimes.append(StructuredORFMetrics["Time"])
        
    print(f"\nGP: \n Training | Average Time : {mean(GPTrainingTimes)}, Average Iterations : {mean(GPIterationCounts)} \n Testing | Average Time: {mean(GPTestingTimes)}, Average MSE: {mean(GPMSEs)}")
    print(f"\nRFF: \n Training | Average Time : {mean(RFFTrainingTimes)}, Average Iterations : {mean(RFFIterationCounts)} \n Testing | Average Time: {mean(RFFTestingTimes)}, Average MSE: {mean(RFFMSEs)}")
    print(f"\nORF: \n Training | Average Time : {mean(ORFTrainingTimes)}, Average Iterations : {mean(ORFIterationCounts)} \n Testing | Average Time: {mean(ORFTestingTimes)}, Average MSE: {mean(ORFMSEs)}")
    print(f"\nStructuredORF: \n Training | Time : {mean(StructuredORFTrainingTimes)}, Iterations : {mean(StructuredORFIterationCounts)} \n Testing | Average Time: {mean(StructuredORFTestingTimes)}, Average MSE: {mean(StructuredORFMSEs)}")
    

dimensions = 2
grid_size = 10
noise_std = 0.1
training_iter = 100
early_stop_loss = 0.05
features = 100
KTrials=5
main(dimensions,grid_size,noise_std,training_iter,early_stop_loss,features,KTrials=5)