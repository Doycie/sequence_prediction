
import torch
import torch.nn as nn


class GRU(nn.Module):

    def __init__(self, device, input_size=85, hidden_layer_size=85, output_size=85, num_layer=1,  dropout=0.1):
        super().__init__()

        self.hidden_layer_size = hidden_layer_size

        self.gru = nn.GRU(input_size, hidden_layer_size, num_layer, dropout=dropout).to(device)

        self.hidden_cell = (torch.zeros(num_layer, 1, self.hidden_layer_size).to(device),
                            torch.zeros(num_layer, 1, self.hidden_layer_size).to(device))

        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
    
        # h0 = torch.zeros(self.layer_dim, self.input_dim, self.hidden_dim).requires_grad_()
        out, self.hidden_cell = self.gru(x.view(len(x), 1, -1))
        predictions = self.fc(out.view(len(x), -1))
        return predictions[-1]

