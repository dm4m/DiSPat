import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
import os
from torch_geometric.nn import GCNConv, GATConv


GRAPH = "SPR"
# GRAPH = 'GCN'
# GRAPH = 'GAT'


class SelfAttention(nn.Module):
    def __init__(
            self,
            config,
    ):
        super().__init__()
        self.self = BartAttention(config.hidden_size, config.num_attention_heads, config.attention_probs_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states,
                attention_mask=None, output_attentions=False, extra_attn=None):
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self(
            hidden_states=hidden_states, attention_mask=attention_mask, output_attentions=output_attentions,
            extra_attn=extra_attn,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states=None,
            past_key_value=None,
            attention_mask=None,
            output_attentions: bool = False,
            extra_attn=None,
            only_attn=False,
    ):
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if extra_attn is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, tgt_len)
            extra_attn = extra_attn.unsqueeze(1)
            attn_weights += extra_attn
            attn_weights = attn_weights.view(bsz*self.num_heads, tgt_len, tgt_len)

        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"

        if attention_mask is not None:
            assert attention_mask.size() == (
                bsz,
                1,
                tgt_len,
                src_len,
            ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        if only_attn:
            return attn_weights_reshaped

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
                .transpose(1, 2)
                .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class GraphLayer(nn.Module):
    def __init__(self, config, last=False):
        super(GraphLayer, self).__init__()
        self.config = config

        class _Actfn(nn.Module):
            def __init__(self):
                super(_Actfn, self).__init__()
                if isinstance(config.hidden_act, str):
                    self.intermediate_act_fn = ACT2FN[config.hidden_act]
                else:
                    self.intermediate_act_fn = config.hidden_act

            def forward(self, x):
                return self.intermediate_act_fn(x)

        if GRAPH == 'SPR':
            self.hir_attn = SelfAttention(config)
        elif GRAPH == 'GCN':
            self.hir_attn = GCNConv(config.hidden_size, config.hidden_size)
        elif GRAPH == 'GAT':
            self.hir_attn = GATConv(config.hidden_size, config.hidden_size, 1)

        self.last = last
        if last:
            self.cross_attn = BartAttention(config.hidden_size, 8, 0.1, True)
            self.cross_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.output_layer = nn.Sequential(nn.Linear(config.hidden_size, config.intermediate_size),
                                          _Actfn(),
                                          nn.Linear(config.intermediate_size, config.hidden_size),
                                          )
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, label_emb, extra_attn, self_attn_mask, bs):
        if GRAPH == 'SPR':
            label_emb = self.hir_attn(label_emb,
                                      attention_mask=self_attn_mask, extra_attn=extra_attn)[0]
        elif GRAPH == 'GCN' or GRAPH == 'GAT':
            label_emb = self.hir_attn(label_emb.squeeze(0), edge_index=extra_attn)
        if self.last:
            label_emb = label_emb.expand(bs, -1, -1)
            return label_emb
        label_emb = self.output_layer_norm(self.dropout(self.output_layer(label_emb)) + label_emb)
        if self.last:
            label_emb = self.dropout(self.classifier(label_emb))
        return label_emb


class GraphEncoder(nn.Module):
    def __init__(self, config, tokenizer, graph=False, max_num_class=50, layer=1, data_path=None, threshold=0.01, tau=1):
        super(GraphEncoder, self).__init__()
        self.config = config
        self.tau = tau
        self.tokenizer = tokenizer
        self.max_num_class = max_num_class
        self.hier_id_embedding = nn.Embedding(self.max_num_class+1, self.config.hidden_size, self.max_num_class)
        self.citation_embedding = nn.Embedding(self.max_num_class + 1, 1, 0)
        self.hir_layers = nn.ModuleList([GraphLayer(config, last=i == layer - 1) for i in range(layer)])
        self.graph = graph
        self.threshold = threshold

    def forward(self, 
                label_name, 
                hier_id, 
                cit_list, 
                embeddings, 
                bs, device):
        self.label_name = label_name
        self.label_name = nn.Parameter(self.label_name, requires_grad=False).to(device)
        self.label_num = len(self.label_name[0])

        if self.graph:
            self.hier_id = hier_id
            self.cit_list = cit_list
            if GRAPH == 'SPR':
                self.cit_list = self.cit_list.view(bs, 1, -1)
                self.cit_list = nn.Parameter(self.cit_list, requires_grad=False).to(device)
                self.hier_id = nn.Parameter(self.hier_id, requires_grad=False).to(device)

        label_mask = (self.label_name != self.tokenizer.pad_token_id).float()
        label_mask = torch.where(label_mask == 0, torch.tensor(1e-8).to(device), label_mask)
        label_emb = embeddings(self.label_name).to(device)
        label_emb = (label_emb * label_mask.unsqueeze(-1)).sum(dim=2) / label_mask.sum(dim=2).unsqueeze(-1)
        label_attn_mask = torch.ones(bs, 1, label_emb.size(1), device=label_emb.device)
        extra_attn = None
        self_attn_mask = torch.matmul(torch.transpose((label_attn_mask * 1.), 1, 2) , (label_attn_mask * 1.)).unsqueeze(1)
        expand_size = label_emb.size(-2) // self.label_name.size(1)
        
        if self.graph:
            if GRAPH == 'SPR':
                label_emb += self.hier_id_embedding(self.hier_id[:, :, None]).view(bs, -1, self.config.hidden_size).to(device)
                extra_attn = self.citation_embedding(self.cit_list).to(device)
                extra_attn = extra_attn.view(bs, self.label_num, 1, self.label_num, 1)
                extra_attn = extra_attn.reshape(bs, self.label_num * expand_size, -1)
            elif GRAPH == 'GCN' or GRAPH == 'GAT':
                extra_attn = self.edge_list
        for hir_layer in self.hir_layers:
            label_emb = hir_layer(label_emb, extra_attn, self_attn_mask, bs)

        return label_emb

        
