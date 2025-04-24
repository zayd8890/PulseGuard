import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple PyTorch model for medical data classification
class Model(nn.Module):
    def __init__(self, input_dim, activation, num_class):
        super(Model, self).__init__()
        
        # Define layers
        self.layer1 = nn.Linear(input_dim, 1024)
        self.activation = activation
        self.dropout1 = nn.Dropout(0.5)

        self.layer2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)

        self.layer3 = nn.Linear(512, 64)
        self.dropout3 = nn.Dropout(0.25)

        self.layer4 = nn.Linear(64, num_class)

    def forward(self, x):
        # Define forward pass
        x = self.dropout1(self.activation(self.layer1(x)))
        x = self.dropout2(self.activation(self.layer2(x)))
        x = self.dropout3(self.activation(self.layer3(x)))
        x = self.layer4(x)
        return x

# Define activation function
def get_activation_function(activation_name):
    if activation_name == 'relu':
        return nn.ReLU()
    elif activation_name == 'sigmoid':
        return nn.Sigmoid()
    elif activation_name == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f"Activation function '{activation_name}' is not supported.")

def load_bp_model():
    """Load and initialize the blood pressure prediction model"""
    # Initialize the model
    input_size = 5  # Your input features: PPG, ABP, Age, Gender, Height
    activation = get_activation_function('relu')  # Use the activation function used during training
    num_classes = 5  # Your output classes: Normal BP, Elevated BP, etc.

    # Create model instance
    model = Model(input_dim=input_size, activation=activation, num_class=num_classes)

    # Load the state dictionary
    model_path = r'E:\CNR_2025\models\model-4-large.pth'
    model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode
    model.eval()
    
    return model

def predict(model, data):
    """Make a prediction using the blood pressure model"""
    # Convert data to tensor
    x = torch.FloatTensor(data)

    # Forward pass
    with torch.no_grad():
        outputs = model(x)

    # Apply softmax to get probabilities
    probs = F.softmax(outputs, dim=0)

    # Get prediction and confidence
    confidence, predicted = torch.max(probs, 0)

    return predicted.item(), confidence.item()

# Define class names
CLASS_NAMES = ["Elevated BP", "Hypertension Stage 1", "Hypertension Stage 2", "Hypertensive Crisis", "Normal BP"]