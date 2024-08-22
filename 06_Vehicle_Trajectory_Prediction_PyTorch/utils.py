from __future__ import print_function, division

import os
import random

import hdf5storage

import numpy as np

import torch
from torch.utils.data import Dataset


######################################################################################################################################################################################


class NGSimDataset(Dataset) :
    def __init__(self, mat, trackHist=30, trajLen=40, downSample=2, encSize=64, gridSize=(265,3), cavRatio=0.7) :
        # Inheritance
        super(NGSimDataset, self).__init__()
        
        # Load Dataset
        self.D = hdf5storage.loadmat(mat)["res_traj"]
        print(f"[Data Loaded!] < Col -> ['res_traj'] | Size -> {self.D.shape} >")
        self.T = hdf5storage.loadmat(mat)["res_t"]
        print(f"[Data Loaded!] < Col -> ['res_t'] | Size -> {self.T.shape} >")

        # Initialize Variable
        self.trackHist = trackHist  # Length of Track History
        self.trajLen = trajLen  # Length of Predicted Trajectory
        self.downSample = downSample  # Downsampling Rate of All Sequences
        self.encSize = encSize # Size of Encoder LSTM
        self.gridSize = gridSize # Size of Social Context Grid
        self.cavRatio = cavRatio # Ratio of CAVs


    def __len__(self) :
        return len(self.D)


    def __getitem__(self, idx) :
        # Get Individual Data
        dsId = self.D[idx,0].astype(int)
        vehId = self.D[idx,1].astype(int)
        t = self.D[idx, 2]
        grid = self.D[idx,10:]
        neighbors = []

        hist = self.getHistory(vehId, t, vehId, dsId)
        fut = self.getFuture(vehId, t, dsId)
        indx = 0

        for i in grid : 
            if random.uniform(0,1) <= self.cavRatio : 
                neighbors.append(self.getHistory(i.astype(int), t,vehId,dsId)) 
            else : 
                neighbors.append(np.empty([0,2]))  
            indx = indx + 1 
        
        indx = 0 
        flag = 0 
        finalTargetID = 0
        locationTargetID = 0
        
        for i in grid :
            if indx == 397 : 
                neighbors[indx] = self.getHistory(vehId, t, vehId, dsId)

            if indx > 397 and indx <= 529 :
                tem = self.getHistory(i.astype(int), t, vehId, dsId)
                if len(tem) != 0 : 
                    targetHDVId = i.astype(int)
                    egoCAVId = vehId
                    hist = self.getHistory(targetHDVId, t, vehId, dsId) 
                    fut = self.getFutureTargetHDV(targetHDVId, egoCAVId, t, dsId)
                    neighbors[indx] = tem 
                    flag = 1 

            if flag == 1 : 
                finalTargetID = targetHDVId
                locationTargetID = indx
                break
            indx = indx + 1 

        lonEnc = np.zeros([2])
        latEnc = np.zeros([3])

        return flag, hist, fut, neighbors, latEnc, lonEnc, vehId, t, dsId, finalTargetID, locationTargetID


    def getHistory(self, vehId, t, refVehId, dsId) :
        if vehId == 0 :
            return np.empty([0,2])
        else:
            if self.T.shape[1] <= vehId-1 : 
                return np.empty([0,2])

            refTrack = self.T[dsId-1][refVehId-1].transpose()
            vehTrack = self.T[dsId-1][vehId-1].transpose()
            refPos = refTrack[np.where(refTrack[:,0]==t)][0,1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:,0]==t).size == 0 :
                 return np.empty([0,2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:,0]==t).item()-self.trackHist)
                enpt = np.argwhere(vehTrack[:,0]==t).item() + 1
                hist = vehTrack[stpt:enpt:self.downSample,1:3] - refPos

            if len(hist) < self.trackHist//self.downSample + 1 :
                return np.empty([0,2])
            
            return hist


    def getFuture(self, vehId, t, dsId) :
        vehTrack = self.T[dsId-1][vehId-1].transpose()
        refPos = vehTrack[np.where(vehTrack[:,0]==t)][0,1:3]
        stpt = np.argwhere(vehTrack[:,0]==t).item() + self.downSample
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:,0]==t).item() + self.trajLen + 1)
        fut = vehTrack[stpt:enpt:self.downSample,1:3] - refPos
        
        return fut


    def getFutureTargetHDV(self, targetHDVId, egoCAVId, t,dsId) :
        HDVTrack = self.T[dsId-1][targetHDVId-1].transpose()
        CAVTrack = self.T[dsId-1][egoCAVId-1].transpose()
        refPos = CAVTrack[np.where(CAVTrack[:,0]==t)][0,1:3]

        stpt = np.argwhere(HDVTrack[:,0]==t).item() + self.downSample
        enpt = np.minimum(len(HDVTrack), np.argwhere(HDVTrack[:,0]==t).item() + self.trajLen + 1)
        fut = HDVTrack[stpt:enpt:self.downSample,1:3] - refPos

        return fut


    def collateFunction(self, samples) :
        nbrBatchSize = 0
        usefulSamples = 0
        
        for flag, _, _, nbrs, _, _, _, _, _, _, _ in samples :
            if flag == 1 :
                nbrBatchSize += sum([len(nbrs[i])!=0 for i in range(len(nbrs))])
                usefulSamples += 1

        maxlen = self.trackHist//self.downSample + 1

        if nbrBatchSize == 0 : 
            nbrBatchSize = 20
        nbrsBatch = torch.zeros(maxlen, nbrBatchSize, 2)

        pos = [0,0]
        
        if usefulSamples == 0 :
            usefulSamples = 20

        maskBatch = torch.zeros(usefulSamples, self.gridSize[1], self.gridSize[0], self.encSize)
        maskBatch = maskBatch.byte()

        histBatch = torch.zeros(maxlen, usefulSamples, 2)
        futBatch = torch.zeros(self.trajLen//self.downSample, usefulSamples, 2)
        outputMaskBatch = torch.zeros(self.trajLen//self.downSample, usefulSamples, 2)
        latEncBatch = torch.zeros(usefulSamples, 3)
        lonEncBatch = torch.zeros(usefulSamples, 2)

        count, vehID, time, dsID, targetID, targetLoc, i = 0, [], [], [], [], [], 0
        
        for _, (flag, hist, fut, nbrs, latEnc, lonEnc, vehId, t, ds,finalTargetID, locationTargetID) in enumerate(samples) :
            if flag == 0 :
                continue

            histBatch[0:len(hist),i,0] = torch.from_numpy(hist[:,0]) 
            histBatch[0:len(hist),i,1] = torch.from_numpy(hist[:,1])
            
            futBatch[0:len(fut),i,0] = torch.from_numpy(fut[:,0])
            futBatch[0:len(fut),i,1] = torch.from_numpy(fut[:,1])
            
            outputMaskBatch[0:len(fut), i,:] = 1 
            
            latEncBatch[i,:] = torch.from_numpy(latEnc)
            lonEncBatch[i,:] = torch.from_numpy(lonEnc)
            
            vehID.append(vehId), time.append(t), dsID.append(ds)
            targetID.append(finalTargetID), targetLoc.append(locationTargetID)

            for id, nbr in enumerate(nbrs) :
                if len(nbr) !=0 :
                    nbrsBatch[0:len(nbr),count,0] = torch.from_numpy(nbr[:,0])
                    nbrsBatch[0:len(nbr),count,1] = torch.from_numpy(nbr[:,1])
		
                    pos[0] = id%self.gridSize[0] 
                    pos[1] = id//self.gridSize[0]
                    
                    maskBatch[i,pos[1],pos[0],:] = torch.ones(self.encSize).byte()
                    count += 1 
            i += 1

        return i, histBatch, nbrsBatch, maskBatch, latEncBatch, lonEncBatch, futBatch, outputMaskBatch, vehID, time, dsID, targetID, targetLoc


def outputActivation(x) :
    muX, muY = x[:,:,0:1], x[:,:,1:2]
    sigX, sigY = torch.exp(x[:,:,2:3]), torch.exp(x[:,:,3:4])
    rho = torch.tanh(x[:,:,4:5])
    
    out = torch.cat([muX, muY, sigX, sigY, rho], dim=2)
    
    return out


def maskedMSE(yPred, yTarget, mask) :
    acc = torch.zeros_like(mask)
    
    muX, muY = yPred[:,:,0], yPred[:,:,1]
    x, y = yTarget[:,:,0], yTarget[:,:,1]
    
    out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)
    
    acc[:,:,0] = out 
    acc[:,:,1] = out
    
    acc = acc*mask
    
    loss = torch.sum(acc)/torch.sum(mask) 
    
    return loss


def maskedMSETest(yPred, yTarget, mask) :
    acc = torch.zeros_like(mask)
    
    muX, muY = yPred[:,:,0], yPred[:,:,1]
    x, y = yTarget[:,:,0], yTarget[:,:,1]
    
    out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)
    
    acc[:,:,0] = out
    acc[:,:,1] = out
    
    acc = acc*mask
    
    loss, counts = torch.sum(acc[:,:,0], dim=1), torch.sum(mask[:,:,0], dim=1)
    
    return loss.mean(), counts


def maskedRMSETest(yPred, yTarget, mask) :
    acc = torch.zeros_like(mask)
    
    muX, muY = yPred[:,:,0], yPred[:,:,1]
    x, y = yTarget[:,:,0], yTarget[:,:,1]
    
    out = torch.pow((torch.pow(x-muX, 2) + torch.pow(y-muY, 2)), 0.5)
    
    acc[:,:,0] = out
    acc[:,:,1] = out
    
    acc = acc*mask
    
    loss, counts = torch.sum(acc[:,:,0], dim=1), torch.sum(mask[:,:,0], dim=1)
    
    return loss.mean(), counts


class AverageMeter(object) :
    def __init__(self) :
        self.reset()
        
    def reset(self) :
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1) :
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count
        

def fixSeed(seed=42) :
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True