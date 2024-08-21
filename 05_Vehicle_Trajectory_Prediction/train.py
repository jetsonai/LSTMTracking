from __future__ import print_function

from os.path import join
import math

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from model import HighwayNet
from utils import NGSimDataset, fixSeed, AverageMeter, maskedMSE


######################################################################################################################################################################################


def main() :
    ## Network Arguments
    args = {}
    args["path"] = "data/trajectory"
    args["useCUDA"] = torch.cuda.is_available()
    args["encoderSize"] = 64
    args["decoderSize"] = 128
    args["inputLength"] = 16
    args["outputLength"] = 20
    args["gridSize"] = (265,3)
    args["inputEmbeddingSize"] = 32

    # Fix Seed
    fixSeed()

    # Initialize network
    net = HighwayNet(args)
    if args["useCUDA"] :
        net = net.cuda()
        
    # Compute Number of Model Parameters
    numParameter = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Number of Trainable Parameters : {numParameter:,}")

    # Set Hyperparameter
    lr = 1e-3
    trainEpochs = 2
    batchSize = 256
    cavRatio = 0.4
    trackHist = 30
    
    # Fix Seed
    fixSeed()
    
    # Create DataLoader Instance
    trainSet = NGSimDataset(join(args["path"], "TrainSet_us101.mat"), trackHist=trackHist, cavRatio=cavRatio)
    trainDataLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True, drop_last=True, collate_fn=trainSet.collateFunction)
    
    # Create Optimizer Instance
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Create Instances for Saving Best Model
    trainLoss = AverageMeter()
    bestLoss = math.inf

    for epoch in range(trainEpochs) :
        # Reset AverageMeter Instance
        trainLoss.reset()
        
        # Create Train DataLoader TQDM Instance
        trainBar = tqdm(trainDataLoader)
        
        # Set to Train Mode
        net.train()
        
        for data in trainBar :
            # Unpack Variables
            flag, hist, nbrs, mask, _, _, fut, outputMask, _, _, _, _, _ = data
            
            # Happends when No Target HDV in Front 
            if flag == 0 : 
                continue
            
            # Assign Device
            if args["useCUDA"] :
                hist, nbrs, mask, fut, outputMask = hist.cuda(), nbrs.cuda(), mask.cuda(), fut.cuda(), outputMask.cuda()

            # Get Prediction
            futurePred, _ = net(hist, nbrs, mask)
            
            # Compute Loss
            loss = maskedMSE(futurePred, fut, outputMask)

            # Update Model Weights
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
            optimizer.step()

            # Average Computed Loss
            trainLoss.update(loss.detach().cpu().item(), batchSize)
            trainBar.set_description(desc=f"[{epoch}/{trainEpochs}] [Train] < Loss:{trainLoss.avg:.4f} >")

        # Save Best Model
        if trainLoss.avg < bestLoss :
            bestLoss = trainLoss.avg
            torch.save(net.state_dict(), "best-model.pth")

        # Save Latest Model
        torch.save(net.state_dict(), "latest-model.pth")


if __name__ == "__main__" :
    main()