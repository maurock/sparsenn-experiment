import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import math
from utils import *
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
# Set parameters
training_size = 5000
test_size = 1000
hidden_size = 25
input_size = 2
lr = 0.005
epochs = 100
batch_size = 50
training = False
criteria = 2

class CorrelationMatrix():
    def __init__(self, matrix_activation_1, matrix_activation_2):  
        self.matrix_activation_1 = matrix_activation_1
        self.matrix_activation_2 = matrix_activation_2
        self.matrix = np.zeros((self.matrix_activation_1.matrix.shape[0], self.matrix_activation_2.matrix.shape[0]))
        self.mask_weights = torch.ones((self.matrix.shape[1], self.matrix.shape[0]))

    def generate_correlation_matrix(self):
        for p in range(self.matrix_activation_1.matrix.shape[0]):
            for q in range(self.matrix_activation_2.matrix.shape[0]):
                self.matrix[p][q] = (np.dot(self.matrix_activation_1.matrix[p], self.matrix_activation_2.matrix[q])/(np.linalg.norm(self.matrix_activation_1.matrix[p])*np.linalg.norm(self.matrix_activation_2.matrix[q])))
        
class ActivationMatrix():
    '''
    Activation matrix. It stores the activations of a specific layer.
    The size is (number neurons, number samples)
    '''
    def __init__(self):
        self.matrix = np.array([])
    
    def add_sample(self,activations):
        if self.matrix.shape[0]==0:
            self.matrix = np.array(activations)
        else:
            self.matrix = np.hstack((self.matrix,np.array(activations)))  
        

    
# returns torch tensors for training and test set
def create_training_test_set():
    # Function:    z = 2x^3 + 3x^2 - 5x  -2y^2 - 7y -2 
    x_total = np.array([-5+random.random()*8 for _ in range(training_size+test_size)])
    y_total = np.array([-5+random.random()*6 for _ in range(training_size+test_size)])
    z_total = (2*np.power(x_total,3) + 3*np.power(x_total,2) - 5*x_total  -2*np.power(y_total,2) - 7*y_total -2)
    
    # Add Gaussian error
    z_error_total = []
    for _ in range(training_size+test_size):
        z_error_total.append(z_total[_]+np.random.normal(loc=0.0, scale=1.0, size=None)*15)
    
    # Split training and test set
    np.random.seed(42)
    total_data_numpy = np.column_stack((x_total,y_total,z_error_total)) 
    np.random.shuffle(total_data_numpy)   
    training_data = torch.FloatTensor(total_data_numpy[:training_size])
    test_data = torch.FloatTensor(total_data_numpy[training_size:])
    total_data = torch.FloatTensor(total_data_numpy)
    
    return total_data, training_data, test_data
    
    
if __name__ == "__main__": 
    if training:
        # Get training and test data
        _, training_data, test_data  = create_training_test_set()
        training_data = training_data.to(device)
        test_data = test_data.to(device)
        
        # Set parameters for training
        MLP = Dense_MLP(input_size, hidden_size).to(device)
        mse_loss = torch.nn.MSELoss()
        optim = torch.optim.Adam(MLP.parameters(), lr=lr)
        
        # Train network
        total_loss_train_list=[]
        total_loss_test_list=[]
        train_x = training_data[:, :2].view(batch_size, -1, 2)
        train_y = training_data[:, [2]].view(batch_size, -1, 1)
        test_x = test_data[:, :2].view(1, -1, 2)
        test_y = test_data[:, [2]].view(1, -1, 1)
        for epoch in range(epochs):
            total_train_loss = 0
            total_test_loss = 0
            for idx, input_data in enumerate(train_x):
                optim.zero_grad()        
                pred = MLP(input_data)
                train_loss = mse_loss(train_y[idx],pred)                  
                train_loss.backward()
                optim.step()
                total_train_loss += train_loss.item()            
            total_loss_train_list.append(total_train_loss/train_x.shape[0])    
            with torch.no_grad():
                for idx, input_data in enumerate(test_x):
                    optim.zero_grad()        
                    pred = MLP(input_data)
                    test_loss = mse_loss(test_y[idx],pred)                  
                    total_test_loss += test_loss.item()     
            total_loss_test_list.append(total_test_loss/test_x.shape[0]) 
            print(f"Loss (train): {total_train_loss/train_x.shape[0]}     Loss (test): {total_test_loss/test_x.shape[0]}      Epoch: {epoch}")
        torch.save(MLP.state_dict(), "weights.pt")            
        plot_2D(np.arange(len(total_loss_train_list)), total_loss_train_list,c='b',title="Training loss")
        plot_2D(np.arange(len(total_loss_test_list)), total_loss_test_list,c='r', title= "Test loss")
        plot_2D(np.arange(len(total_loss_train_list)), total_loss_train_list,c='b',  show=False)
        plot_2D(np.arange(len(total_loss_test_list)), total_loss_test_list,c='r', title= "Train + test loss", legend = ['Training', 'Test'])
        
        # Test: plot results
        with torch.no_grad():
            for idx, input_data in enumerate(test_x):
                pred = MLP(input_data)
                
                # Plot x, prediction, ground truch
                data_x_plot = input_data[:,0].view(-1).cpu().numpy()
                pred_plot = pred.view(-1).cpu().numpy()
                ground_truth_plot = test_y.view(-1).cpu().numpy()
                scatter_2D(x=data_x_plot, z=pred_plot, show=False)
                scatter_2D(x=data_x_plot, z=ground_truth_plot, show=True)
                     
                # Plot y, prediction, ground truch
                data_y_plot = input_data[:,1].view(-1).cpu().numpy()
                scatter_2D(x=data_y_plot, z=pred_plot, show=False)
                scatter_2D(x=data_y_plot, z=ground_truth_plot, show=True)

    #################### 
    ##   CRITERIA 1   ##
    ####################
    
    # =============================================================================
    # Correlation matrix built using absolute values.
    # 1.1: Remove correlation from 1 to 0.
    # Removal strategy: 1% of weights are removed per iteration, up to 100 iterations
    # =============================================================================
    
    if criteria==1:
        class Mask():
            def __init__(self, correlation_matrix):
                self.mask_weights = torch.ones((correlation_matrix.shape[1], correlation_matrix.shape[0]))
                self.correlation_matrix = copy.copy(correlation_matrix)
                self.correlation_matrix_temp = copy.copy(correlation_matrix)
        
            def get_mask_weights(self):
                iterations = int(mask_weights.shape[0] * mask_weights.shape[1] * 0.01)
                for _ in range(iterations):
                    max_value = np.argmax(self.correlation_matrix_temp)
                    row = math.floor(max_value/self.correlation_matrix.shape[1])
                    col = max_value % self.correlation_matrix.shape[1]      
                    self.mask_weights[col,row] = 0            
                    self.correlation_matrix_temp[row,col] = 0
            
            
        class Dense_MLP(torch.nn.Module):
            def __init__(self, input_size, hidden_size):
                super(Dense_MLP, self).__init__()
                self.input_size = input_size
                self.hidden_size  = hidden_size
                self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
                self.relu = torch.nn.ReLU()
                self.sigmoid = torch.nn.Sigmoid()
                self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
                self.fc3 = torch.nn.Linear(self.hidden_size, 1)
            
            def forward(self, x, mask_weights=None,activation_1=None, activation_2=None):
                if mask_weights != None:
                    with torch.no_grad():
                        self.fc2.weight = torch.nn.Parameter(self.fc2.weight*mask_weights)
                hidden1 = self.fc1(x)
                sigmoid = self.sigmoid(hidden1)
                if activation_1 != None:
                    activation_1.add_sample(sigmoid.permute(1,0).detach().numpy())
                hidden2 = self.fc2(sigmoid)
                sigmoid = self.sigmoid(hidden2)
                if activation_2 != None:
                    activation_2.add_sample(sigmoid.permute(1,0).detach().numpy())
                output = self.fc3(sigmoid)
                return output
        
        # Build correlation matrix    
        model = Dense_MLP(input_size, hidden_size)
        model.load_state_dict(torch.load('weights.pt'))
        mse_loss = torch.nn.MSELoss()
        data, _, _ = create_training_test_set()
        data_x = data[:, :2].view(-1, 1, 2)
        data_y = data[:, 2].view(-1,1, 1)
        iterations = 30
        mask_weights = torch.ones((25,25))
        for iteration in range(iterations):
            activation_1 = ActivationMatrix()
            activation_2 = ActivationMatrix()
            score = 0
            with torch.no_grad():
                for idx, input_data in enumerate(data_x):
                    pred = model(input_data, mask_weights = mask_weights, activation_1=activation_1, activation_2=activation_2)
                    loss = mse_loss(data_y[idx] ,pred)
                    score += loss.item()
            print(f'Score: {score/data_x.shape[0]}')
            correlation_matrix = CorrelationMatrix(activation_1,activation_2)
            correlation_matrix.generate_correlation_matrix()
            correlation_mean = np.mean(correlation_matrix.matrix)  
            if iteration==0:
                weights_corr_matrix = correlation_matrix.matrix
            else:
                weights_corr_matrix = mask.correlation_matrix_temp
            mask = Mask(weights_corr_matrix)
            mask.get_mask_weights()
            mask_weights = mask.mask_weights
            plt.hist(x=correlation_matrix.matrix.reshape(correlation_matrix.matrix.size,-1), bins=30, density=True, histtype='stepfilled')         
            plt.title(f"Iteration {iteration}")
            plt.show()
            # Set highest weights to 0
            mask.get_mask_weights()
        

    #################### 
    ##   CRITERIA 2   ##
    ####################
    
    # =============================================================================
    # Correlation matrix built using absolute values.
    # 2.1: Remove correlation from 1 to 0.
    # Removal strategy: 1% of weights are removed per iteration, up to 100 iterations
    # Removal applies to the weights of the previous layer, since this has a bigger impact 
    # on the overall results
    # =============================================================================
    
    if criteria == 2:
        
        class Mask():
            def __init__(self, correlation_matrix):
                self.mask_weights = torch.ones((25, 2))
                self.correlation_matrix = copy.copy(correlation_matrix)
                self.correlation_matrix_temp = copy.copy(correlation_matrix)
        
            def get_mask_weights(self, model):
                #iterations = int(mask_weights.shape[0] * mask_weights.shape[1] * 0.01)
                iterations = 5
                for _ in range(iterations):
                    max_value = np.argmin(self.correlation_matrix_temp)
                    row = math.floor(max_value/self.correlation_matrix.shape[1])  # neuron in first layer
                    col = max_value % self.correlation_matrix.shape[1]
                    with torch.no_grad():
                        weights = model.fc1.weight
                        values, indices = torch.min(torch.abs(weights[row]), 0)
                        self.mask_weights[row,indices] = 0
                    self.correlation_matrix_temp[row,col] = 2
            
            
        class Dense_MLP(torch.nn.Module):
            def __init__(self, input_size, hidden_size):
                super(Dense_MLP, self).__init__()
                self.input_size = input_size
                self.hidden_size  = hidden_size
                self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
                self.relu = torch.nn.ReLU()
                self.sigmoid = torch.nn.Sigmoid()
                self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
                self.fc3 = torch.nn.Linear(self.hidden_size, 1)
            
            def forward(self, x, mask_weights=None,activation_1=None, activation_2=None):
                if mask_weights != None:
                    with torch.no_grad():
                        self.fc1.weight = torch.nn.Parameter(self.fc1.weight*mask_weights)
                hidden1 = self.fc1(x)
                sigmoid = self.sigmoid(hidden1)
                if activation_1 != None:
                    activation_1.add_sample(sigmoid.permute(1,0).detach().numpy())
                hidden2 = self.fc2(sigmoid)
                sigmoid = self.sigmoid(hidden2)
                if activation_2 != None:
                    activation_2.add_sample(sigmoid.permute(1,0).detach().numpy())
                output = self.fc3(sigmoid)
                return output        
        
        # Build correlation matrix    
        model = Dense_MLP(input_size, hidden_size)
        model.load_state_dict(torch.load('weights.pt'))
        mse_loss = torch.nn.MSELoss()
        data, _, _ = create_training_test_set()
        data_x = data[:, :2].view(-1, 1, 2)
        data_y = data[:, 2].view(-1,1, 1)
        iterations = 30
        mask_weights = torch.ones((25,2))
        for iteration in range(iterations):
            activation_1 = ActivationMatrix()
            activation_2 = ActivationMatrix()
            score = 0
            with torch.no_grad():
                for idx, input_data in enumerate(data_x):
                    pred = model(input_data, mask_weights = mask_weights, activation_1=activation_1, activation_2=activation_2)
                    loss = mse_loss(data_y[idx] ,pred)
                    score += loss.item()
            print(f'Score: {score/data_x.shape[0]}')
            correlation_matrix = CorrelationMatrix(activation_1,activation_2)
            correlation_matrix.generate_correlation_matrix()
            correlation_mean = np.mean(correlation_matrix.matrix)  
            if iteration==0:
                weights_corr_matrix = correlation_matrix.matrix
            else:
                weights_corr_matrix = mask.correlation_matrix_temp
            mask = Mask(weights_corr_matrix)
            mask.get_mask_weights(model)
            mask_weights = mask.mask_weights
            plt.hist(x=correlation_matrix.matrix.reshape(correlation_matrix.matrix.size,-1), bins=30, density=True, histtype='stepfilled')         
            plt.title(f"Iteration {iteration}")
            plt.show()