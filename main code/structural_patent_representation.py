from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEncoder
from transformers.file_utils import ModelOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from graphencoder import GraphEncoder


class BertPoolingLayer(nn.Module):
    def __init__(self, config, avg='cls'):
        super(BertPoolingLayer, self).__init__()
        self.avg = avg

    def forward(self, x):
        if self.avg == 'cls':
            x = x[:, 0, :]
        else:
            x = x.mean(dim=1)
        return x

class BertOutputLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class NTXent(nn.Module):
    def __init__(self, config, tau=1.):
        super(NTXent, self).__init__()
        self.tau = tau
        self.norm = 1.
        self.transform = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, x, labels=None):
        x = self.transform(x)
        n = x.shape[0]
        x = F.normalize(x, p=2, dim=1) / np.sqrt(self.tau)
        # 2B * 2B
        sim = x @ x.t()
        sim[np.arange(n), np.arange(n)] = -1e9
        logprob = F.log_softmax(sim, dim=1)
        m = 2
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)
        loss = -logprob[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m - 1) / self.norm
        return loss

class BaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    last_hidden_state = None
    pooler_output = None
    hidden_states = None
    past_key_values = None
    attentions = None
    cross_attentions = None
    input_embeds = None

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config, max_num_class):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.max_num_class = max_num_class
        self.max_position_embeddings = config.max_position_embeddings
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0,
            embedding_weight=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[2]
        if position_ids is None:
            position_ids = torch.arange(self.max_position_embeddings).unsqueeze(0).unsqueeze(0).expand(input_shape[0], self.max_num_class, seq_length + past_key_values_length).to(input_ids.device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if embedding_weight is not None:
            if len(embedding_weight.size()) == 2:
                embedding_weight = embedding_weight.unsqueeze(-1)
            inputs_embeds = inputs_embeds * embedding_weight
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids).squeeze(1)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, inputs_embeds

class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need`_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.

    .. _`Attention is all you need`: https://arxiv.org/abs/1706.03762

    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, max_num_class):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config, max_num_class)
        self.encoder = BertEncoder(config)
        self.pooler = None
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            embedding_weight=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``: ``1`` for
            tokens that are NOT MASKED, ``0`` for MASKED tokens.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if not self.config.is_decoder:
            use_cache = False
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output, inputs_embeds = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            embedding_weight=embedding_weight,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output, inputs_embeds) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            inputs_embeds=inputs_embeds,
        )

class SPR(BertPreTrainedModel):
    def __init__(self, config, tokenizer, enc_hidden_dim, graph=False, max_num_class=20, layer=1, data_path=None, threshold=0.01, tau=1, topK=5):
        super(SPR, self).__init__(config)
        self.tokenizer = tokenizer
        self.enc_hidden_dim = enc_hidden_dim
        self.max_num_class = max_num_class
        self.topK = topK
        self.bert = BertModel(config, self.max_num_class)
        self.graph_encoder = GraphEncoder(config, self.tokenizer, graph, self.max_num_class, layer=layer, data_path=data_path, threshold=threshold, tau=tau)
        self.topK = topK
        self.init_weights()
    
    def label_name_complement(self, src_label_name):
        claim_num = len(src_label_name)
        sentence_length = len(src_label_name[0]) # 512
        if claim_num < self.max_num_class:
            tmp = np.zeros((self.max_num_class-claim_num, sentence_length), dtype=int)
            src_label_name.extend(tmp)
        else:
            # keys = list(src_label_name.keys())[:self.max_num_class]  # 获取前20个键
            # values = list(src_label_name.values())[:self.max_num_class]  # 获取前20个值
            # src_label_name = dict(zip(keys, values))  # 创建一个包含前20个键值对的新字典
            src_label_name = src_label_name[:self.max_num_class]  # 创建一个包含前20个值
        return torch.tensor(src_label_name).unsqueeze(0)
    def hier_id_complement(self, hier_id):
        claim_num = len(hier_id)
        if claim_num < self.max_num_class:
            tmp = self.max_num_class * np.ones((self.max_num_class-claim_num), dtype=int)
            hier_id.extend(tmp)
        else:
            # keys = list(hier_id.keys())[:self.max_num_class]  # 获取前20个键
            # values = list(hier_id.values())[:self.max_num_class]  # 获取前20个值
            # hier_id = dict(zip(keys, values))  # 创建一个包含前20个键值对的新字典
            hier_id = hier_id[:self.max_num_class]
        return torch.tensor(hier_id).unsqueeze(0)
    def cit_list_complement(self, cit_list):
        #在数组A的边缘填充constant_values指定的数值
        #（3,2）表示在A的第[0]轴填充（二维数组中，0轴表示行），即在0轴前面填充3个宽度的0，比如数组A中的95,96两个元素前面各填充了3个0；在后面填充2个0，比如数组A中的97,98两个元素后面各填充了2个0
        #（2,3）表示在A的第[1]轴填充（二维数组中，1轴表示列），即在1轴前面填充2个宽度的0，后面填充3个宽度的0
        # np.pad(A,((3,2),(2,3)),'constant',constant_values = (0,0))  #constant_values表示填充值，且(before，after)的填充值等于（0,0）
        claim_num = len(cit_list)
        if claim_num < self.max_num_class:
            cit_list = np.pad(cit_list, ((0,self.max_num_class-claim_num),(0,self.max_num_class-claim_num)), 'constant', constant_values = (0,0))
        else:
            cit_list = cit_list[:self.max_num_class, :self.max_num_class]
        return torch.tensor(cit_list).unsqueeze(0)

    def forward(
            self,
            examples, 
            return_dict=None,
            device='cpu',
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if not isinstance(examples, list):
            examples = [examples]
        bs = len(examples)
        sps_hidden = []
        all_label_name = []
        all_hier_id = []
        all_cit_list = []
        for e in examples:
            all_label_name.append(self.label_name_complement(e.src_label_name))
            all_hier_id.append(self.hier_id_complement(e.hier_id))
            all_cit_list.append(self.cit_list_complement(e.cit_list))
            for i in range(self.topK):
                all_label_name.append(self.label_name_complement(e.sim[i]['label_name']))
                all_hier_id.append(self.hier_id_complement(e.sim[i]['hier_id']))
                all_cit_list.append(self.cit_list_complement(e.sim[i]['cit_list']))
        all_label_name = torch.cat(all_label_name, dim=0)
        all_hier_id = torch.cat(all_hier_id, dim=0) 
        all_cit_list = torch.cat(all_cit_list, dim=0) 
        tmp_bs = all_label_name.size()[0]
        all_hidden = self.graph_encoder(all_label_name, 
                                        all_hier_id, 
                                        all_cit_list, 
                                        lambda x: self.bert.embeddings(x)[0], 
                                        tmp_bs, device)
        for i in range(bs):
            if i == 0:
                hidden = all_hidden[i].unsqueeze(0)
                sp_hidden = all_hidden[(i+1):(i+self.topK+1), :, :]
            else:
                hidden = torch.cat([hidden, all_hidden[i*(self.topK+1)].unsqueeze(0)], dim=0)
                sp_hidden = all_hidden[(i*(self.topK+1)+1):(i*(self.topK+1)+self.topK+1), :, :]
            sps_hidden.append(sp_hidden)
        all_hidden = None
        all_label_name = None
        all_hier_id = None
        all_cit_list = None
        return{
            "hidden": hidden,
            "sps_hidden": sps_hidden,
        }
