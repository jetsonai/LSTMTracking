from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import outputActivation


######################################################################################################################################################################################


class HighwayNet(nn.Module) :
    def __init__(self, args) :
        # Inheritance
        super(HighwayNet, self).__init__()

        # Unpack Arguments
        self.args = args
        self.useCUDA = args["useCUDA"]

        # Sizes of Network Layers
        self.encoderSize = args["encoderSize"]
        self.decoderSize = args["decoderSize"]
        self.inputLength = args["inputLength"]
        self.outputLength = args["outputLength"]
        self.gridSize = args["gridSize"]
        self.inputEmbeddingSize = args["inputEmbeddingSize"]

        # Input Embedding Layer
        self.inputEmb = torch.nn.Linear(2, self.inputEmbeddingSize)

        # Encoder & Decoder LSTM
        self.encLSTM1 = torch.nn.LSTM(self.inputEmbeddingSize, self.encoderSize, 1)
        self.encLSTM2 = torch.nn.LSTM(self.inputEmbeddingSize, self.encoderSize, 1)
        self.spatialEmb = nn.Linear(5, self.encoderSize)
        self.pre4att = nn.Linear(self.encoderSize, 1)
        self.decLSTM = torch.nn.LSTM(self.encoderSize, self.decoderSize) # 64, 128

        # Output Layers
        self.output = torch.nn.Linear(self.decoderSize, 5)

        # Activations:
        self.tanh = nn.Tanh()
        self.leakyReLU = torch.nn.LeakyReLU(0.1)


    def attention(self, lstmOutWeight, lstmOut) :
        alpha = F.softmax(lstmOutWeight, 1)
        newHiddenState = torch.bmm(lstmOut.permute(0, 2, 1), alpha).squeeze(2)
        newHiddenState = F.relu(newHiddenState)

        return newHiddenState, alpha


    def decode(self, enc) :
        enc = enc.repeat(self.outputLength, 1, 1)
        
        hDec, _ = self.decLSTM(enc) 
        hDec = hDec.permute(1, 0, 2)

        futurePred = self.output(hDec) 
        futurePred = futurePred.permute(1, 0, 2) 
        futurePred = outputActivation(futurePred)

        return futurePred


    def decodeByStep(self, enc) :
        predTrajList = []

        decInput = enc

        for _ in range(self.outputLength) :
            decInput = decInput.unsqueeze(0)
            hDec, _ = self.decLSTM(decInput)
            
            hPred = hDec.squeeze()
            futurePred = self.output(hPred)
            
            predTrajList.append(futurePred.view(futurePred.size()[0], -1))
            
            embInput = futurePred
            decInput = self.spatialEmb(embInput)

        predTrajList = torch.stack(predTrajList, dim=0)
        predTrajList = outputActivation(predTrajList)

        return predTrajList


    def forward(self, hist, nbrs, masks) :
        _, (histEnc, _) = self.encLSTM1(F.leaky_relu(self.inputEmb(hist), 0.2))
        histEnc = histEnc.squeeze().unsqueeze(2)

        _, (nbrsEnc, _) = self.encLSTM2(F.leaky_relu(self.inputEmb(nbrs), 0.2))
        nbrsEnc = nbrsEnc.view(nbrsEnc.shape[1], nbrsEnc.shape[2])

        socEnc = torch.zeros_like(masks).float()
        socEnc = socEnc.masked_scatter_(masks, nbrsEnc)
        socEnc = socEnc.permute(0,3,2,1) 
        socEnc = socEnc.contiguous().view(socEnc.shape[0], socEnc.shape[1], -1) 

        newHS = torch.cat((socEnc, histEnc), 2)
        newHS = newHS.permute(0, 2, 1) 
        
        weight = self.pre4att(self.tanh(newHS)) 
        newHidden, attnWeights = self.attention(weight, newHS) 
    
        enc = newHidden 
        futurePred = self.decode(enc) 
        
        return futurePred, attnWeights