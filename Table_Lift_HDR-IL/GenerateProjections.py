#from tqdm import tqdm_notebook as tqdm

#import seaborn as sns

#import A1PrimitiveData
#import A3TrainSoftmax
import TrainDynamicModels as B3TrainODE
import utils

#sns.color_palette("bright")

import csv
import torch
from matplotlib import pyplot as plt
import numpy as np
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('-trainiters', default=1, help='Number of training iterations to output')
parser.add_argument('-startindex', default=0, help='Row to start generation')
parser.add_argument('-datasize', default=55, help='Number of rows in each demonstration')
parser.add_argument('-features', default=21, help='Number of features')
args = parser.parse_args()



device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

"""Load all models"""

datasize = args.datasize
trainsize = args.datasize*args.trainiters
start = args.startindex #61*0

features = args.features

models = {

0: B3TrainODE.graspmodel,
1: B3TrainODE.liftmodel,
2: B3TrainODE.extendmodel,
3: B3TrainODE.placemodel,
4: B3TrainODE.retractmodel,
5: B3TrainODE.sidemodel,

}

#seqmodel = A3TrainSoftmax.seqmodel


path1 = B3TrainODE.path1
path2 = B3TrainODE.path2
path3 = B3TrainODE.path3
path4 = B3TrainODE.path4
path5 = B3TrainODE.path5
path6 = B3TrainODE.path6



models[0].load_state_dict(torch.load(path1))
models[1].load_state_dict(torch.load(path2))
models[2].load_state_dict(torch.load(path3))
models[3].load_state_dict(torch.load(path4))
models[4].load_state_dict(torch.load(path5))
models[5].load_state_dict(torch.load(path6))



train_set = utils.BaxterDataset()


with torch.no_grad():


    outputsize = torch.zeros([args.datasize, 1, features])
    #outputsize2 = torch.zeros([12, 1, features])


    visited = torch.zeros([1, 1, features]).cuda()
    truth = torch.zeros([1, 1, features]).cuda()
    varianceslist = torch.zeros([1, 1, features]).cuda()

    #enter as a range


    while start < trainsize:

        print(datasize, "datasize", start, trainsize)

        a, _ = train_set[start:start + 1]
        a = a.unsqueeze(1).cuda()


        print(a)
        print(trainsize)

        a1, _ = train_set[start:start + datasize]
        a1 = a1.unsqueeze(1).cuda()

        start = start + datasize
        print(start, "s")

        variances = torch.zeros([1, 1, features]).cuda()


        #while a.size()[0] < 71:
            #c = torch.zeros((a.size()[0], 1, 6))
            #primitive = seqmodel.getCurrentPrimitive(a.float(), c)


        #while a.size()[0] < 59:

        test = [0,1,2,3,4,5]


        for primitive in test:


            print("size", a.size())
            mean, var = models[primitive].generate_mean_variance(a[-1].unsqueeze(1).float(), outputsize, outputsize)
            print(var)
            print(a.size(), mean.size())
            a = torch.cat((a.float(), mean.float()), 0)
            variances = torch.cat((variances.float(), var.float()), 0)
            print("grasp")



        visited = torch.cat((visited.float(), a[0:datasize, :, :].float()), 0)
        truth = torch.cat((truth.float(), a1[0:datasize, :, :].float()), 0)
        varianceslist = torch.cat((varianceslist.float(), variances[0:datasize, :, :].float()), 0)

        headers = utils.outputHeaders()


a = a[0:trainsize - start, :, :]
a1 = a1[0:trainsize - start, :, :]
variances = variances[0:trainsize - start, :, :]



"""Write results to csv"""
with open('projection_data/out.csv', 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(headers)

    for i in range(1, visited.size()[0]):

        thewriter.writerow(visited[i][0].tolist() + truth[i][0].tolist())






