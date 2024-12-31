from torch import nn
import math
import torch
from models.semantic_fusion_ts.linear_layer import LinearLayer
from models.semantic_fusion_ts.add_and_norm import AddAndNorm
from models.semantic_fusion_ts.gated_linear_unit import GLU
import pdb

class GatedResidualNetwork(nn.Module):
    def __init__(self, 
                input_size,
                hidden_layer_size,
                output_size=None,
                dropout_rate=None,
                use_time_distributed=True,
                return_gate=False,
                batch_first=False,
                tower="add"
                ):

        super(GatedResidualNetwork, self).__init__()
        if output_size is None:
            output = hidden_layer_size
        else:
            output = output_size
        
        self.output = output
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.return_gate = return_gate

        self.linear_layer = LinearLayer(input_size, output, use_time_distributed, batch_first)

        self.hidden_linear_layer1 = LinearLayer(input_size, hidden_layer_size, use_time_distributed, batch_first)
        self.hidden_context_layer = LinearLayer(hidden_layer_size, hidden_layer_size, use_time_distributed, batch_first)
        self.hidden_linear_layer2 = LinearLayer(hidden_layer_size, hidden_layer_size, use_time_distributed, batch_first)
        self.hidden_linear_layer3 = LinearLayer(hidden_layer_size * 2, hidden_layer_size, use_time_distributed, batch_first)
        self.hidden_linear_layer4 = LinearLayer(hidden_layer_size + 768, hidden_layer_size, use_time_distributed, batch_first)

        self.elu1 = nn.ELU()
        self.glu = GLU(hidden_layer_size, output, dropout_rate, use_time_distributed, batch_first)
        self.add_and_norm = AddAndNorm(hidden_layer_size=output)

        self.tower = tower   # cat

    def forward(self, x, context=None):
        # Setup skip connection
        if self.output_size is None:
            skip = x
        else:
            skip = self.linear_layer(x)

        # Apply feedforward network
        hidden = self.hidden_linear_layer1(x)
        if self.tower == "add":
            if context is not None:
                #pdb.set_trace()
                hidden = hidden + self.hidden_context_layer(context)
            hidden = self.elu1(hidden)
            hidden = self.hidden_linear_layer2(hidden)
        else: 
            if context is not None:
                #pdb.set_trace()
                context_layer_output = self.hidden_context_layer(context)
                context_layer_output = context_layer_output.expand(-1, hidden.size(1), -1)
                hidden = torch.cat((hidden, context_layer_output), dim=-1)
                hidden = self.elu1(hidden)
                hidden = self.hidden_linear_layer3(hidden)
                #hidden = self.hidden_linear_layer4(hidden)
            else:
                hidden = self.elu1(hidden)
                hidden = self.hidden_linear_layer2(hidden)

            #input_size = hidden.size(-1)
            #output_size = self.hidden_layer_size
            #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
            #hidden_linear_layer2 = nn.Linear(input_size, output_size).to(device)
            #hidden = hidden_linear_layer2(hidden)

        gating_layer, gate = self.glu(hidden)
        if self.return_gate:
            return self.add_and_norm(skip, gating_layer), gate
        else:
            return self.add_and_norm(skip, gating_layer)
