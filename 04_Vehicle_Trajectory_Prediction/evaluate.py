from __future__ import print_function

from os.path import join

import pandas as pd
import numpy as np

import torch
from model import HighwayNet
from torch.utils.data import DataLoader

from tqdm import tqdm

from utils import NGSimDataset, AverageMeter, maskedMSETest, maskedRMSETest


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

    # Set Hyperparameter
    batchSize = 256
    cav = 0.4 # Set as -1, when No Cavs at All
    trackHist = 30 # 30 or 60

    # Initialize network
    net = HighwayNet(args)
    net.load_state_dict(torch.load("best-model.pth"))
    if args["useCUDA"] :
        net = net.cuda()

    # Create DataLoader Instance
    testSet = NGSimDataset(join(args["path"], "TestSet_us101.mat"), trackHist=trackHist, encSize =64, cavRatio=cav)
    testDataLoader = DataLoader(testSet, batch_size=batchSize, shuffle=False, drop_last=False, collate_fn=testSet.collateFunction)

    # Create List Instance for Adding Results
    vehID, targetID, targetLoc, predX, predY, T, dsID, weightAttn = [], [], [], [], [], [], [], []
    
    # Create AverageMeter Instance
    testMSE, testRMSE = AverageMeter(), AverageMeter()

    # Reset AverageMeter Instance
    testMSE.reset(), testRMSE.reset()
    
    # Create Test DataLoader TQDM Instance
    testBar = tqdm(testDataLoader)
    
    for data in testBar :
        # Unpack Variables
        flag, hist, nbrs, mask, _, _, fut, outputMask, vehID, t, ds, targetID, targetLoc = data
        
        # Happends when No Target HDV in Front 
        if flag == 0 : 
            continue
        
        vehID.append(vehID) # CAV ID
        targetID.append(targetID) # target HDV ID
        targetLoc.append(targetLoc) # target HDV location
        T.append(t) # current time
        dsID.append(ds)
        
        if args["useCUDA"] :
            hist, nbrs, mask, fut, outputMask = hist.cuda(), nbrs.cuda(), mask.cuda(), fut.cuda(), outputMask.cuda()

        futurePred, wtA = net(hist, nbrs, mask)

        mseLoss, _ = maskedMSETest(futurePred, fut, outputMask)
        rmseLoss, _ = maskedRMSETest(futurePred, fut, outputMask)

        predX.append(futurePred[:,:,0].detach().cpu().numpy())
        predY.append(futurePred[:,:,1].detach().cpu().numpy())
        weightAttn.append(wtA[:, :, 0].detach().cpu().numpy())
        
        # Average Computed Loss
        testMSE.update(mseLoss.detach().cpu().item(), batchSize), testRMSE.update(rmseLoss.detach().cpu().item(), batchSize)
        testBar.set_description(desc=f"[Test] < MSE Loss:{testMSE.avg:.4f} | RMSE Loss:{testRMSE.avg:.4f} >")
    
    vehID = pd.DataFrame(sum(vehID, []))
    targetID = pd.DataFrame(sum(targetID, []))
    targetLoc = pd.DataFrame(sum(targetLoc, []))
    T = pd.DataFrame(sum(T, []))
    dsID = pd.DataFrame(sum(dsID, []))
    predX = pd.DataFrame(np.concatenate(predX, axis=1))
    predY = pd.DataFrame(np.concatenate(predY, axis=1))
    weightAttn = pd.DataFrame(np.concatenate(weightAttn, axis=0))