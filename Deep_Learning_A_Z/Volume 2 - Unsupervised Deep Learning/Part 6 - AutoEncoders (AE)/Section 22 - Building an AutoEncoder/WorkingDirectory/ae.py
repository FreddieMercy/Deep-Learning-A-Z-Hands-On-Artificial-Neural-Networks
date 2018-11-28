# AutoEncoders

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(in_features=nb_movies, out_features = 20)#20 comes from instrucotr's experiment(s)
        self.fc2 = nn.Linear(20,10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
        
    def forward(self, input_x):
        input_x = self.activation(self.fc1(input_x))
        input_x = self.activation(self.fc2(input_x))
        input_x = self.activation(self.fc3(input_x))
        input_x = self.fc4(input_x)
        return input_x
    
    
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) # "lr" and "weight_decay" come from instrucotr's experiment(s)

nb_epoch = 200
for epoch in range(nb_epoch):
    train_loss = 0
    users_rated_at_least_1 = 0.
    
    for id_user in range(nb_users):
        input_set = Variable(training_set[id_user]).unsqueeze(0) #make "single value" to ["single value"] (vector size == 1)
        target = input_set.clone()
        
        if torch.sum(target.data > 0) > 0:
            output = sae.forward(input_set)
            target.require_grad = False #wtf???
            output[target == 0] = 0
            loss = criterion(output, input_set)
            mean_corrector = nb_movies/float(torch.sum(target.data>0)+1e-10) #"1e-10" is system required, to avoid the denuminator is not 0
            loss.backward() #decide direction (increase/decrease)
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            users_rated_at_least_1+=1.
            optimizer.step() #decide intensity?
            

    print('epoch: '+str(epoch+1)+' loss: '+str(train_loss/users_rated_at_least_1))    
    
test_loss = 0
users_rated_at_least_1 = 0.
    
for id_user in range(nb_users):
    input_set = Variable(training_set[id_user]).unsqueeze(0) #make "single value" to ["single value"] (vector size == 1)
    #target = Variable(test_set[id_user])
    target = Variable(test_set[id_user]).unsqueeze(0) #make "single value" to ["single value"] (vector size == 1)
                                                      #not accurate!!!
        
    if torch.sum(target.data > 0) > 0:
        output = sae(input_set)
        target.require_grad = False #wtf???
        output[target == 0] = 0
        loss = criterion(output, input_set)
        mean_corrector = nb_movies/float(torch.sum(target.data>0)+1e-10) #"1e-10" is system required, to avoid the denuminator is not 0
            
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        users_rated_at_least_1+=1.
            

print('test loss: '+str(test_loss/users_rated_at_least_1))    
    

            
            
            
        