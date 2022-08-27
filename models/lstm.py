
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, device, input_size=85, hidden_layer_size=85, output_size=85, num_layer=1,  dropout=0.1):
        super().__init__()

        # Save size of hidden layer and number of layers
        self.num_layers = num_layer
        self.hidden_layer_size = hidden_layer_size

        # Make lstm model
        self.lstm = nn.LSTM( input_size, hidden_layer_size, num_layer, dropout=dropout).to(device)

        # Make one linear layer
        self.linear = nn.Linear(hidden_layer_size, output_size)
        #self.linear2 = nn.Linear(output_size, output_size)
        #self.linear3 = nn.Linear(output_size, output_size)

        # Initialize the hidden cell
        self.hidden_cell = (torch.zeros(num_layer, 1, self.hidden_layer_size).to(device),
                            torch.zeros(num_layer, 1, self.hidden_layer_size).to(device))

    def forward(self, input_seq):

        # Forward the hidden layer, and get the output from the input but add a dimension for the batch input
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        # Return the output to the original dimensions and forward to linear layer
        # predictions3 = self.linear(lstm_out.view(len(input_seq), -1))
        # predictions2 = self.linear(predictions3)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))

        # Return the last prediction
        return predictions[-1]
