import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from helpers import *
import pandas as pd

class EuropeDataset(Dataset):
    def __init__(self, csv_file: str):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
        """
        #### YOUR CODE HERE ####
        try: 
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {csv_file} not found.")
        self.features = torch.from_numpy (df[['long', 'lat']].to_numpy()).float()
        self.labels = torch.from_numpy(df['country'].to_numpy()).long()
        # Load the data into a tensors
        # The features shape is (n,d)
        # The labels shape is (n)
        # The feature dtype is float
        # THe labels dtype is long
        #### END OF YOUR CODE ####

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        #### YOUR CODE HERE ####
        return self.features.shape[0]

    def __getitem__(self, idx) :
        """
        Args:
            idx (int): Index of the data row
        
        Returns:
            dictionary or list corresponding to a feature tensor and it's corresponding label tensor
        """
        #### YOUR CODE HERE ####
        return self.features[idx], self.labels[idx]
    

class MLP(nn.Module):
    def __init__(self, num_hidden_layers, hidden_dim, output_dim, with_batchnorm=False):
        super(MLP, self).__init__()
        """
        Args:
            num_hidden_layers (int): The number of hidden layers, in total you'll have an extra layer at the end, from hidden_dim to output_dim
            hidden_dim (int): The hidden layer dimension
            output_dim (int): The output dimension, should match the number of classes in the dataset
        """
        #### YOUR CODE HERE ####
        self.hiddem_layer = []
        input_dim = 2  # since we have longitude and latitude as input features
        for _ in range(num_hidden_layers):
            self.hiddem_layer.append(nn.Linear(input_dim, hidden_dim))
            if with_batchnorm:
                self.hiddem_layer.append(nn.BatchNorm1d(hidden_dim))
            self.hiddem_layer.append(nn.ReLU())
            input_dim = hidden_dim
        self.hiddem_layer.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*self.hiddem_layer)
    def forward(self, x):
        #### YOUR CODE HERE ####
       return self.model(x)


def train(train_dataset, val_dataset, test_dataset, model, lr=0.001, epochs=50, batch_size=256):    

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)    
    
    #### YOUR CODE HERE ####
    # initialize your criterion and optimizer here
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []
    best_val_acc = 0.0
    best_loss_val = float('inf')
    best_model = None
    ep_where_best = 0
    for ep in range(epochs):
        #
        model.train()
        for batch_X, batch_y in trainloader:
            #### YOUR CODE HERE ####
            # perform training loop here
            optimizer.zero_grad()
            predictions  = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            #### YOUR CODE HERE ####
            # perform validation loop and test loop here
            train_loss, train_acc = measure_current_model_on_set(trainloader, model, criterion)
            val_loss, val_acc = measure_current_model_on_set(valloader, model, criterion)
            test_loss, test_acc = measure_current_model_on_set(testloader, model, criterion)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            test_accs.append(test_acc)
        if val_loss<best_loss_val:
            best_loss_val = val_loss
            best_model = model.state_dict()
            ep_where_best = ep
        print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(ep, train_accs[-1], val_accs[-1], test_accs[-1]))  
    model.load_state_dict(best_model)
    print(f'Best model found at epoch {ep_where_best} with validation loss {best_loss_val:.4f}')
    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, best_model,best_loss_val

def measure_current_model_on_set(dataloader, model, criterion):
    # implement a function that measures accuracy and loss on a given dataloader and model
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0.0
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            predictions = model(batch_X)
            total_loss += criterion(predictions, batch_y).item() * batch_X.size(0)
            total_correct += (predictions.argmax(dim=1) == batch_y).sum().item()
            total_samples += batch_X.size(0)
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy
    #### END OF YOUR CODE ####

if __name__ == '__main__':
    # seed for reproducibility
    torch.manual_seed(0)    

    train_dataset = EuropeDataset('input_data/train.csv')
    val_dataset = EuropeDataset('input_data/validation.csv')
    test_dataset = EuropeDataset('input_data/test.csv')
    output_dim = len(train_dataset.labels.unique())
    #### YOUR CODE HERE #####
    # Find the number of classes, e.g.:
    # output_dim = len(train_dataset.labels.unique()) 
    lrs= [0.01, 0.001, 0.00001]
    epochs = [50,100]
    model = MLP(6, 16, output_dim, with_batchnorm=False)
    models = {key : None for key in lrs}
    for lr in lrs:
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train(train_dataset, val_dataset, test_dataset, model, lr=lr, epochs=50, batch_size=256)
   



    train_data = pd.read_csv('input_data/train.csv')
    val_data = pd.read_csv('input_data/validation.csv')
    test_data = pd.read_csv('input_data/test.csv')
    plot_decision_boundaries(model, test_data[['long', 'lat']].values, test_data['country'].values, 'Decision Boundaries', implicit_repr=False)

def plot_and_save_results(train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train')
    ax1.plot(val_losses, label='Validation')
    ax1.plot(test_losses, label='Test')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss over Epochs')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Train')
    ax2.plot(val_accs, label='Validation')
    ax2.plot(test_accs, label='Test')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy over Epochs')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()