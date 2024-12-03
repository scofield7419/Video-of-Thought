
# import paddle
# from paddlemix.models.blip2.Qformer import BertLMHeadModel
# from paddlenlp.transformers.bert.configuration import BertConfig
# from paddle.nn import Transformer

import torch 
import torch.nn as nn
from .qformer import BertLMHeadModel, BertConfig


class MLP(nn.Module):
    def __init__(self, in_features=None, out_features=None, num_layers=1):
        super().__init__()
        modules = [nn.Linear(in_features=in_features, out_features=out_features)]

        for _ in range(1, num_layers):
            modules.append(nn.GELU())
            modules.append(nn.Linear(in_features=out_features, out_features=out_features))

        self.layer =  nn.Sequential(*modules)
    
    def forward(self, x):
        return self.layer(x)
    
    @property
    def config(self):
        return {"mm_projector_type": "mlp"}
    
    @property
    def device(self):
        return self.layer[0].weight.device
    
    @property
    def dtype(self):
        return self.layer[0].weight.dtype


class QFormer(nn.Module):
    def __init__(self, in_features, out_features, num_query_token, cross_attention_freq=1, num_hidden_layers=2):
        super().__init__()

        self.in_fc = nn.Linear(in_features, out_features)

        qformer_config = BertConfig.from_pretrained('bert-base-uncased')
        qformer_config.encoder_width = out_features
        qformer_config.add_cross_attention = True
        qformer_config.num_hidden_layers = num_hidden_layers
        qformer_config.cross_attention_freq = cross_attention_freq
        qformer_config.gradient_checkpointing = False
        qformer_config.query_length = num_query_token
        qformer_config.use_fusedlinear =False

        self.Qformer = BertLMHeadModel.from_pretrained('bert-base-uncased', config=qformer_config)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, qformer_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=qformer_config.initializer_range)

        self.out_fc = nn.Linear(out_features, out_features)

    def forward(self, x, input_embs):
        x = x + input_embs
        x = self.in_fc(x)
        image_atts = torch.ones(x.size()[:-1], dtype=torch.long).to(x.device)
        # print(x.size())
        query_tokens = self.query_tokens.expand(x.shape[0], -1, -1)
        # print(image_atts.size())
        # print(query_tokens.size())
        outputs = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=x,
            encoder_attention_mask=image_atts,
            return_dict=True,
        ).last_hidden_state
        # print(outputs.size())
        outputs = self.out_fc(outputs)
        return outputs
    
    @property
    def config(self):
        return {"mm_projector_type": "qformer"}
    
    @property
    def device(self):
        return self.query_tokens.device
    
    @property
    def dtype(self):
        return self.query_tokens.dtype


class TransformersProjector(nn.Module):
    def __init__(self, in_features, out_features, num_query_token, **kwargs):
        super().__init__()
        hidden_dim = 512
        self.in_fc = nn.Linear(in_features, hidden_dim)
        self.tfm = nn.Transformer(batch_first=True, norm_first=True,
                                      d_model=hidden_dim, num_encoder_layers=4, num_decoder_layers=4,
                                      dim_feedforward=hidden_dim * 4, dropout=0.0, nhead=4)
        self.out_fc = nn.Linear(hidden_dim, out_features)

        self.query_embs = nn.Parameter(torch.randn(1, num_query_token, hidden_dim))
        self.query_embs.data.normal_(mean=0.0, std=0.0)

    def forward(self, x, input_embs):
        x = x + input_embs
        # print('layer x: ', x)
        x = self.in_fc(x)
        # print('layer fc x: ', x.shape)
        # print('layer fc query_embs: ', self.query_embs.shape)
        x = self.tfm(x, self.query_embs.repeat(x.shape[0], 1, 1))
        # print('layer tfm x: ', x)
        outputs = self.out_fc(x)
        return outputs

    @property
    def config(self):
        return {"mm_projector_type": "transformer"}

    @property
    def device(self):
        return self.query_embs.device
    
    @property
    def dtype(self):
        return self.query_embs.dtype