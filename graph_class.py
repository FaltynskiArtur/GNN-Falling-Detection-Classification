import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNWithResiduals(torch.nn.Module):
    def __init__(self, activation_fn=F.relu, input_dim=3, hidden_dims=[32, 64, 128, 256], num_classes=4):
        super(GNNWithResiduals, self).__init__()
        # Graph convolutional layers
        self.conv1 = GCNConv(input_dim, hidden_dims[0])
        self.conv2 = GCNConv(hidden_dims[0], hidden_dims[1])
        self.conv3 = GCNConv(hidden_dims[1], hidden_dims[2])
        self.conv4 = GCNConv(hidden_dims[2], hidden_dims[3])

        # Residual connections
        self.res1 = torch.nn.Linear(input_dim, hidden_dims[0])
        self.res2 = torch.nn.Linear(hidden_dims[0], hidden_dims[1])
        self.res3 = torch.nn.Linear(hidden_dims[1], hidden_dims[2])
        self.res4 = torch.nn.Linear(hidden_dims[2], hidden_dims[3])

        # Fully connected layers for classification
        self.lin1 = torch.nn.Linear(hidden_dims[3], hidden_dims[2])
        self.lin2 = torch.nn.Linear(hidden_dims[2], hidden_dims[1])
        self.lin3 = torch.nn.Linear(hidden_dims[1], num_classes)  # Updated num_classes to 4

        # Additional layers
        self.dropout = torch.nn.Dropout(p=0.5)
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_dims[0])
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_dims[1])
        self.batch_norm3 = torch.nn.BatchNorm1d(hidden_dims[2])
        self.batch_norm4 = torch.nn.BatchNorm1d(hidden_dims[3])
        self.activation_fn = activation_fn

    def forward(self, data):
        # Extract graph data
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First graph convolution with residual connection
        res = self.res1(x)
        x = self.conv1(x, edge_index) + res
        x = self.batch_norm1(x)
        x = self.activation_fn(x)

        # Second graph convolution with residual connection
        res = self.res2(x)
        x = self.conv2(x, edge_index) + res
        x = self.batch_norm2(x)
        x = self.activation_fn(x)

        # Third graph convolution with residual connection
        res = self.res3(x)
        x = self.conv3(x, edge_index) + res
        x = self.batch_norm3(x)
        x = self.activation_fn(x)

        # Fourth graph convolution with residual connection
        res = self.res4(x)
        x = self.conv4(x, edge_index) + res
        x = self.batch_norm4(x)
        x = self.activation_fn(x)

        # Global pooling and fully connected layers
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.lin1(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.activation_fn(x)
        x = self.lin3(x)

        # Apply log-softmax for classification output
        return F.log_softmax(x, dim=1)
