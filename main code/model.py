from __future__ import division

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer
from structural_patent_representation import SPR
from disentanglement import disentangle_gate
from classifiers import Approval_Classifier




class Model(nn.Module):

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict)

    @classmethod
    def load(cls, load_path):
        params = torch.load(load_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = cls(args)
        model.load_state_dict(params['state_dict'])
        if args.cuda:
            if "gpu" in args:
                model = model.cuda(args.gpu)
            else:
                model = model.cuda()
        return model

    def decode(self, inputs, encoder_outputs, encoder_hidden):
        return self.decoder.forward(
            inputs=inputs,
            encoder_outputs=encoder_outputs,
            encoder_hidden=encoder_hidden
        )

    def base_information(self):
        origin = super().base_information()
        return origin \
               + "_con:{}\n" \
                 "_clf:{}\n" .format(str(self.args._con),
                                      str(self.args._clf)
                                      )

    def __init__(self, args, data_path=None): 
        super(Model, self).__init__(args)
        if args.cuda:
            if "gpu" in args:
                self.device = 'cuda:' + str(args.gpu)
            else:
                self.device = 'cuda'
        else:
            self.device = 'cpu'
        print("self.device:",self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        self.graphormer = SPR.from_pretrained('bert-base-chinese', self.tokenizer, args.enc_hidden_dim, args.graph, args.max_num_class, args.layer, data_path, args.threshold, args.tau, args.topK)
        self.disentangle_module = disentangle_gate(args.max_num_class, args.enc_hidden_dim)
        self.unk_rate = args.unk_rate
        self.step_unk_rate = 0.0
        self.direction_num = 2 if args.bidirectional else 1
        self.enc_hidden_dim = args.enc_hidden_dim
        self.enc_layer_dim = args.enc_hidden_dim * self.direction_num
        self.enc_hidden_factor = self.direction_num * args.enc_num_layers
        self.dec_hidden_factor = args.dec_num_layers
        args.use_attention = False
        self.topK = args.topK
        if args.mapper_type == "link":
            self.dec_layer_dim = self.enc_layer_dim
        elif args.use_attention:
            self.dec_layer_dim = self.enc_layer_dim
        else:
            self.dec_layer_dim = args.dec_hidden_dim
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.clf_input_dim = args.clf_input_dim
        self.clf_hidden_dim = args.clf_hidden_dim
        self.clf_output_dim = args.clf_output_dim
        self.classifier = Approval_Classifier(self.clf_input_dim, self.clf_hidden_dim, self.clf_output_dim, args._clf)        

    def get_gpu(self):
        model_list = [self.word_encoder, self.bridger, self.decoder, self.syn_mean, self.syn_logv, self.syn_to_h,
                      self.sem_mean, self.sem_logv,
                      self.sem_to_h, self.sup_syn, self.sup_sem, self.syn_adv, self.syn_infer, self.sem_adv,
                      self.sem_infer]
        for model in model_list:
            device = torch.device("cuda:1" if torch.cuda.is_available else "cpu")
            model = torch.nn.DataParallel(model)
            model.to(device)

    def forward(self, examples, is_dis=False):
        if not isinstance(examples, list):
            examples = [examples]
        batch_size = len(examples)
        ret = self.graphormer(examples, return_dict=True, device=self.device)
        ret = self.hidden_to_latent(ret=ret, is_sampling=self.training, device=self.device) 
        ret = self.latent_for_init(ret=ret)
        spe_hidden = ret['spe_hidden']
        sim_hidden = ret['sim_hidden']
        sps_hidden = ret['sps_hidden']
        mul_spe_loss, mul_sim_loss = self.get_mul_loss(
            specific_repr=spe_hidden,
            similar_repr=sim_hidden,
            similar_patents_repr=sps_hidden
        )
        classifier_loss, predicted_probs, predicted_labels, label_batch = self.classifier(
            spe_hidden = spe_hidden,
            sim_hidden = sim_hidden,
            label_batch = torch.LongTensor([e.tgt for e in examples]).to(self.device)
        )
        ret['mul_spe_loss'] = mul_spe_loss
        ret['mul_sim_loss'] = mul_sim_loss
        ret['mul_loss'] = mul_spe_loss + mul_sim_loss
        ret['clf_loss'] = classifier_loss
        ret['predicted_probs'] = predicted_probs
        ret['predicted_labels'] = predicted_labels
        ret['label_batch'] = label_batch
        ret['batch_size'] = batch_size
        return ret

    def get_loss(self, examples, train_iter, is_dis=False, **kwargs):
        self.step_unk_rate = wd_anneal_function(unk_max=self.unk_rate, anneal_function=self.args.unk_schedule,
                                                step=train_iter, x0=self.args.x0,
                                                k=self.args.k)
        explore = self.forward(examples, is_dis)
        batch_size = explore['batch_size']
        mul_spe_loss = explore['mul_spe_loss'] / batch_size
        mul_sim_loss = explore['mul_sim_loss'] / batch_size
        mul_loss = explore['mul_loss'] / batch_size
        clf_loss = explore['clf_loss'] / batch_size
        predicted_probs = explore['predicted_probs']
        predicted_labels = explore['predicted_labels']
        label_batch = explore['label_batch']
        return {
            'MUL SPE Loss': mul_spe_loss,
            'MUL SIM Loss': mul_sim_loss,
            'MUL Loss': mul_loss,
            'CLF Loss': clf_loss,
            'Predicted_Probs': predicted_probs,
            'Predicted_Labels': predicted_labels,
            'Label_Batch': label_batch,
            'Loss': mul_loss + clf_loss,
        }

    def get_mul_loss(self, specific_repr, similar_repr, similar_patents_repr):
        batchsize = specific_repr.size()[0]     
        spe_loss = 0
        sim_loss = 0
        target_spe = -torch.ones(30, 1).to(self.device)
        target_spe = -torch.ones(similar_patents_repr[0].size()[0], 1)
        target_spe = -torch.ones(similar_patents_repr[0].size()[0], 1).to(self.device)
        target_sim = torch.ones(similar_patents_repr[0].size()[0], 1).to(self.device)
        for i in range(batchsize):
            score_spe = self.cosine_loss(torch.mean(specific_repr[i], dim=0).unsqueeze(0).repeat(similar_patents_repr[i].size()[0], 1), torch.mean(similar_patents_repr[i], dim=1), target_spe)
            score_sim = self.cosine_loss(torch.mean(similar_repr[i], dim=0).unsqueeze(0).repeat(similar_patents_repr[i].size()[0], 1), torch.mean(similar_patents_repr[i], dim=1), target_sim)
            spe_loss += score_spe
            sim_loss += score_sim
        return self.args._con * spe_loss / self.args.topK, self.args._con * sim_loss / self.args.topK

    def hidden_to_latent(self, ret, is_sampling=True, device='cpu'):
        hidden = ret['hidden']
        specific_latent, similar_latent = self.disentangle_module(hidden)
        ret['spe_z'] = specific_latent
        ret['sim_z'] = similar_latent
        return ret

    def latent_for_init(self, ret):
        specific_latent = ret['spe_z']
        similar_latent = ret['sim_z']
        ret['spe_hidden'] = specific_latent
        ret['sim_hidden'] = similar_latent
        return ret

 
def kl_anneal_function(anneal_function, step, k, x0):
        if anneal_function == "fixed":
            return 1.0
        elif anneal_function == 'logistic':
            return float(1 / (1 + np.exp(-k * (step - x0))))
        elif anneal_function == 'sigmoid':
            return float(1 / (1 + np.exp(0.001 * (x0 - step))))
        elif anneal_function == 'negative-sigmoid':
            return float(1 / (1 + np.exp(-0.001 * (x0 - step))))
        elif anneal_function == 'linear':
            return min(1, step / x0)
        
def wd_anneal_function(unk_max, anneal_function, step, k, x0):
    return unk_max * kl_anneal_function(anneal_function, step, k, x0)