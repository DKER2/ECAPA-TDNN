import torch
import torch.nn as nn
import torch.nn.functional as F
import s3prl.hub as hub
class SSLUpstream(nn.Module):
    def __init__(self, upstream_model='xlsr_53', hidden_state=16, finetune=False):
        super(SSLUpstream, self).__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model) # loading ssl model from s3prl
        self.n_encoder_layer = len(self.upstream.model.encoder.layers)
        assert hidden_state > 0 and hidden_state <= self.n_encoder_layer 
        self.hidden_state = 'hidden_state_{}'.format(hidden_state)
        if not finetune:
            for param in self.upstream.parameters():
                param.requires_grad = False 

    def forward(self, x, x_len):
        x_input = [torch.narrow(wav, 0, 0, x_len[i]) for (i, wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x_input)
        # s3prl sometimes misses the ouput of some hidden states so need a while loop to assure we can get the output of the desired hidden state 
        while self.hidden_state not in x.keys():
            x = self.upstream(x_input)
        x = x[self.hidden_state]
        return x