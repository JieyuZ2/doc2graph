import os
import time, random, pickle, math, json
from itertools import chain, combinations
import numpy as np
from typing import *
import torch
from torch.nn import Parameter
from torch import optim, nn
import torch.nn.functional as F
from torch.autograd import Variable


class LossLogger:
    def __init__(self):
        self.loss_list = []

    def add(self, i):
        self.loss_list.append(i)

    def get(self):
        mean = np.mean(self.loss_list)
        self.loss_list = []
        return mean


def kl_weight(it):
    """
    Credit to: https://github.com/kefirski/pytorch_RVAE/
    0 -> 1
    """
    return (math.tanh((it - 10)/100) + 1)/2


def temp(it):
    """
    Softmax temperature annealing
    1 -> 0
    """
    return 1-kl_weight(it) + 1e-5  # To avoid overflow


def l2_matrix_norm(m):
    """
    Frobenius norm calculation

    Args:
       m: {Variable} ||AAT - I||

    Returns:
        regularized value


    """
    return torch.mean(torch.sum(torch.sum(m ** 2, 1), 1) ** 0.5)


def self_attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output, scores


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        # attention
        self.dim = dim
        self.W_h = nn.Linear(dim, dim, bias=False)
        self.W_c = nn.Linear(1, dim, bias=False)
        self.v = nn.Linear(dim, 1, bias=False)
        self.decode_proj = nn.Linear(dim, dim)

        # for m in self.modules():
        #     self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, s_t_hat, encoder_outputs, enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())

        encoder_feature = self.W_h(encoder_outputs)
        dec_fea = self.decode_proj(s_t_hat) # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.expand(b, t_k, n).contiguous() # B x t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded # B * t_k x 2*hidden_dim
        # att_features = att_features.view(-1, n)  # B * t_k x 2*hidden_dim

        # coverage_input = coverage.view(-1, 1)  # B * t_k x 1
        coverage_feature = self.W_c(coverage.unsqueeze(-1))  # B * t_k x 2*hidden_dim
        att_features = att_features + coverage_feature

        e = F.tanh(att_features) # B * t_k x 2*hidden_dim
        scores = self.v(e).squeeze()
        scores = scores.masked_fill(enc_padding_mask == 0, -1e9)# B * t_k x 1
        # scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1) # B x t_k
        attn_dist = attn_dist_

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, self.dim)  # B x 2*hidden_dim

        coverage = coverage + attn_dist.squeeze(1)

        return c_t, attn_dist, coverage


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight.data)
        # if self.bias is not None:
        #     self.bias.data.fill_(0.0)

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = torch.matmul(inputs, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class AbstractNetGen(nn.Module):
    def __init__(self, n_node, n_vocab, n_labels, embed_dim, p_word_dropout=0.5, unk_idx=0, pad_idx=1,
                 start_idx=2, eos_idx=3, pretrained_embeddings=None, freeze_embeddings=False, teacher_forcing=True, device=False):
        super(AbstractNetGen, self).__init__()
        self.UNK_IDX = unk_idx
        self.PAD_IDX = pad_idx
        self.START_IDX = start_idx
        self.EOS_IDX = eos_idx

        self.n_vocab = n_vocab
        self.n_labels = n_labels
        self.p_word_dropout = p_word_dropout
        self.n_node = n_node
        self.teacher_forcing = teacher_forcing

        self.device = device

        """
        Word embeddings layer
        """
        if pretrained_embeddings is None:
            self.emb_dim = embed_dim
            self.word_emb = nn.Embedding(n_vocab, embed_dim, self.PAD_IDX)
        else:
            self.emb_dim = pretrained_embeddings.size(1)
            self.word_emb = nn.Embedding(n_vocab, self.emb_dim, self.PAD_IDX)

            # Set pretrained embeddings
            self.word_emb.weight.data.copy_(pretrained_embeddings)

            if freeze_embeddings:
                self.word_emb.weight.requires_grad = False

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def word_dropout(self, inputs):
        """
        Do word dropout: with prob `p_word_dropout`, set the word to '<unk>'.
        """
        if isinstance(inputs, Variable):
            data = inputs.data.clone()
        else:
            data = inputs.clone()

        # Sample masks: elems with val 1 will be set to <unk>
        mask = torch.from_numpy(
            np.random.binomial(1, p=self.p_word_dropout, size=tuple(data.size()))
                .astype('uint8')
        )

        padding_mask = self.dataset.create_padding_mask(inputs=inputs, device=self.device, padding=self.PAD_IDX)

        mask = mask.to(self.device)
        mask = mask * padding_mask.byte()

        # Set to <unk>
        data[mask] = self.UNK_IDX

        return Variable(data)

    def forward_encoder(self, inputs):
        """
        Inputs is batch of sentences: seq_len x mbsize
        """
        inputs = self.word_emb(inputs)
        return self.forward_encoder_embed(inputs)

    def forward_encoder_embed(self, inputs):
        """
        Inputs is embeddings of: seq_len x mbsize x emb_dim
        """
        outputs, (h, c) = self.encoder(inputs)
        bsize = inputs.size()[0]
        h = h.view(1, bsize, -1)
        c = c.view(1, bsize, -1)

        h = F.relu(self.encoder_fc_h(h))
        c = F.relu(self.encoder_fc_c(c))
        return outputs, (h, c)

    def forward_decoder(self, inputs, z):
        """
        Inputs must be embeddings: seq_len x mbsize
        """
        bsize, length = inputs.size()

        # 1 x mbsize x z_dim
        init_h = z.unsqueeze(0)
        c = self.decoder_c_linear(init_h).relu()
        h = self.decoder_h_linear(init_h).relu()
        d = self.decoder_d_linear(init_h).relu().transpose(1, 0)

        if self.teacher_forcing:
            dec_inputs = self.word_dropout(inputs)
            inputs_emb = self.word_emb(dec_inputs)  # seq_len x mbsize x emb_dim
            inputs_emb = torch.cat([inputs_emb, d.repeat(1, length, 1)], -1)
            # inputs_emb = torch.cat([inputs_emb, init_h.repeat(seq_len, 1, 1)], 2)
            outputs, _ = self.decoder(inputs_emb, (h, c))
            y = self.decoder_fc(outputs)
            # y = y @ self.word_emb.weight.t() #.data.t()
        else:
            start_word = torch.LongTensor([self.START_IDX]).repeat(bsize).view(-1, 1)
            start_word = start_word.to(self.device)
            emb = self.word_emb(start_word)
            emb = torch.cat([emb, d], -1)
            ys = []
            for i in range(length):
                output, (h, c) = self.decoder(emb, (h, c))
                output = self.decoder_fc(output)
                ys.append(output)
                y = F.softmax(output, dim=2)
                emb = (y @ self.word_emb.weight).data
                emb = torch.cat([emb, d], -1)
            y = torch.cat(ys, dim=1)

        return y

    def generate_node(self, z, hidden_state, mask):
        bsize, n, d_k = z.size()
        h, c = hidden_state

        s = torch.zeros(bsize, 1, d_k).to(self.device)
        coverage = torch.zeros(bsize, n).to(self.device)
        outputs, atts, states, g_emb = [], [], [], []
        closs = 0

        for i in range(self.n_node):
            _, (h, c) = self.generate_lstm(s, (h, c))
            x = torch.cat((h, c), dim=-1).view(bsize, 1, -1)
            c_t, attn_dist, coverage_next = self.attention_network(x, z, mask, coverage)
            states.append(x)
            c_t = c_t.unsqueeze(dim=1)
            outputs.append(c_t)
            atts.append(attn_dist)
            closs = closs + torch.sum(torch.min(attn_dist.squeeze(1), coverage), 1).mean()
            s = c_t.data
            coverage = coverage_next

        text = torch.cat(outputs, dim=1)
        attentions = torch.cat(atts, dim=1)

        return text, attentions, closs

    def forward(self, *input):
        pass

    def forward_classifier(self, inputs, mask):
        pass

    def train_model(self, args, dataset):
        pass

    def train_model_classifier(self, args, dataset):
        """train the whole network"""
        self.train()

        self.classifier_[0].reset_parameters()
        self.classifier_[1].reset_parameters()
        for i in self.classifier_[2]:
            if hasattr(i, 'reset_parameters'):
                i.reset_parameters()
        trainer_C = optim.Adam(filter(lambda p: p.requires_grad, self.classifier_.parameters()), lr=args.lr)
        criterion_cls = nn.CrossEntropyLoss().to(self.device)

        patience = 0
        log_every = 5
        list_loss_cls = []
        best_train_acc, best_val_acc, best_test_acc, best_iter = 0, 0, 0, 0
        train_start_time = time.time()

        train_iter = dataset.build_train_iter(args.eval_batch_size, self.device)

        for epoch in range(1, args.eval_epochs):
            for batch in iter(train_iter):
                inputs, labels, mask = batch.text.t(), batch.label, batch.mask.t()

                """ Update classifier """
                outputs = self.forward_classifier(inputs, mask)
                loss_cls = criterion_cls(outputs, labels)
                list_loss_cls.append(loss_cls.item())

                trainer_C.zero_grad()
                loss_cls.backward()
                trainer_C.step()

            if epoch % log_every == 0:
                duration = time.time() - train_start_time
                avr_loss_cls = np.mean(list_loss_cls)
                list_loss_cls = []

                val_iter, test_iter = dataset.build_test_iter(args.batch_size, self.device)
                val_acc = self.test(dataset, val_iter)
                test_acc = self.test(dataset, test_iter)
                train_acc = self.test(dataset, train_iter)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_train_acc = train_acc
                    best_iter = epoch
                    patience = 0
                else:
                    if epoch > args.eval_minimal_epoch:
                        patience += log_every

                if args.eval_early_stop and patience > args.eval_patience:
                    break

            # if epoch % 10 == 0:
            #     print(f'Epoch-{epoch}; loss_cls: {avr_loss_cls:.4f}; Valid Acc: {val_acc:.4f}; Test Acc: {test_acc:.4f};'
            #           f' duration: {duration:.2f}')

        # print(f'Best Valid Acc: {best_val_acc:.4f}; Best Test Acc: {best_test_acc:.4f} @ iter {best_iter}')
        return best_train_acc, best_val_acc, best_test_acc

    def test(self, data_iter):
        self.eval()
        total_accuracy, l_loss_cls = [], []
        criterion_cls = nn.CrossEntropyLoss().to(self.device)
        with torch.no_grad():
            for batch in iter(data_iter):
                inputs, labels, mask = batch.text.t(), batch.label, batch.mask.t()
                outputs = self.forward_classifier(inputs, mask)
                loss_cls = criterion_cls(outputs, labels)
                l_loss_cls.append(loss_cls.item())
                accuracy = (outputs.argmax(1) == labels).float().mean().item()
                total_accuracy.append(accuracy)
        acc = sum(total_accuracy) / len(total_accuracy)
        loss = np.mean(l_loss_cls)
        self.train()
        return acc, loss


class NetGen(AbstractNetGen):
    def __init__(self, n_node, n_vocab, n_labels, embed_dim, h_dim, z_dim, p_word_dropout=0.5, unk_idx=0, pad_idx=1, start_idx=2,
                 eos_idx=3, pretrained_embeddings=None, freeze_embeddings=False, teacher_forcing=True, device=False):
        super(NetGen, self).__init__(n_node, n_vocab, n_labels, embed_dim, p_word_dropout, unk_idx, pad_idx, start_idx, eos_idx,
                                     pretrained_embeddings, freeze_embeddings, teacher_forcing, device)

        self.node_dim = 2 * h_dim

        """
        Encoder is GRU with FC layers connected to last hidden unit
        """
        self.encoder = nn.LSTM(self.emb_dim, h_dim, bidirectional=True, batch_first=True)
        self.encoder_fc_c = nn.Linear(self.node_dim, h_dim)
        self.encoder_fc_h = nn.Linear(self.node_dim, h_dim)
        self.encoder_ = nn.ModuleList([
            self.encoder, self.encoder_fc_c, self.encoder_fc_h
        ])

        """
        Decoder is GRU with `z` appended at its inputs
        """
        self.decoder_h_linear = nn.Linear(self.n_node * self.node_dim, z_dim)
        self.decoder_c_linear = nn.Linear(self.n_node * self.node_dim, z_dim)
        self.decoder_d_linear = nn.Linear(self.n_node * self.node_dim, z_dim)
        self.decoder = nn.LSTM(self.emb_dim + z_dim, z_dim, num_layers=1, dropout=0.3, batch_first=True)
        # self.decoder_fc = nn.Linear(n_node * self.emb_dim, n_vocab)
        self.decoder_fc = nn.Linear(z_dim, self.n_vocab)

        """
        Generator 
        """
        self.generate_lstm = nn.LSTM(self.node_dim, h_dim, batch_first=True)
        self.attention_network = Attention(self.node_dim)
        self.generate_adj = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, self.n_node * self.n_node),
            nn.Sigmoid()
        )

        """
        Graph Encoder is GCN with pooling layer
        """
        self.graph_encoder = GraphConvolution(self.node_dim, self.node_dim, bias=False)
        self.graph_encoder_fc = nn.Linear(2 * self.node_dim, self.node_dim, bias=False)
        self.decoder_ = nn.ModuleList([
            self.decoder, self.decoder_fc, self.decoder_c_linear, self.decoder_h_linear, self.decoder_d_linear,
            self.attention_network,
            self.generate_lstm, self.generate_adj,
            self.graph_encoder, self.graph_encoder_fc
        ])

        """
        Classifier is DNN
        """
        self.graph_encoder_cls = GraphConvolution(self.node_dim, self.node_dim, bias=False)
        self.graph_encoder_cls_fc = nn.Linear(2 * self.node_dim, self.node_dim, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(self.node_dim, self.node_dim// 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.node_dim // 4, self.n_labels)
        )
        self.classifier_ = nn.ModuleList([
            self.graph_encoder_cls, self.graph_encoder_cls_fc, self.classifier
        ])

        """
        Grouping the model's parameters: separating encoder, decoder, and discriminator
        """
        self.encoder_params = filter(lambda p: p.requires_grad, self.encoder_.parameters())

        self.decoder_params = filter(lambda p: p.requires_grad, self.decoder_.parameters())

        self.classifier_params = filter(lambda p: p.requires_grad, self.classifier_.parameters())

        self.ae_params = chain(
            self.word_emb.parameters(), self.encoder_.parameters(), self.decoder_.parameters()
        )
        self.ae_params = filter(lambda p: p.requires_grad, self.ae_params)

        """
        Use GPU if set
        """

        # for m in self.modules():
        #     self.weights_init(m)

        self.to(self.device)

    def forward_graph_encoder(self, inputs, adj):
        # x = F.relu(self.graph_encoder(input, adj))
        x = self.graph_encoder(inputs, adj)
        x = F.relu(self.graph_encoder_fc(torch.cat((x, inputs), -1)))
        x = x.view(-1, self.n_node * self.node_dim)
        # x = self.graph_encoder_trans(x)

        return x

    def forward(self, sentence, mask, classifier):
        """
        Params:
        -------
        sentence: sequence of word indices.
        use_c_prior: whether to sample `c` from prior or from `discriminator`.

        Returns:
        --------
        recon_loss: reconstruction loss of VAE.
        kl_loss: KL-div loss of VAE.
        """

        mbsize = sentence.size(0)

        enc_inputs = sentence
        dec_inputs = sentence

        pad_words = torch.LongTensor([self.PAD_IDX]).repeat(mbsize, 1).to(self.device)
        dec_targets = torch.cat([sentence[:, 1:], pad_words], dim=1)

        # Encoder: sentence -> z
        z, h = self.forward_encoder(enc_inputs)

        # Generator: z -> node
        text, attentions, c_loss = self.generate_node(z, h, mask)

        # Generator: z -> adj
        adj = self.generate_adjacency_matrix(h)

        # Graph Encoder: graph -> z'
        z1 = self.forward_graph_encoder(text, adj)
        # z1 = text.view(-1, self.n_node * self.node_dim)

        # Decoder: sentence -> y
        y = self.forward_decoder(dec_inputs, z1)

        recon_loss = F.cross_entropy(y.view(-1, self.n_vocab), dec_targets.view(-1), size_average=True, ignore_index=self.PAD_IDX)

        penal1 = torch.distributions.Categorical(probs=attentions).entropy().mean()

        if classifier:
            # classifier: graph -> prediction
            x = self.graph_encoder_cls(text, adj)
            x = F.relu(self.graph_encoder_cls_fc(torch.cat((x, text), -1)))
            # x = x.view(-1, self.node_dim * self.n_node)
            x = x.sum(dim=1)
            outputs = self.classifier(x)

            return recon_loss, outputs, penal1, c_loss
        else:
            return recon_loss, penal1, c_loss

    def forward_classifier(self, inputs, mask):

        # Encoder: sentence -> z
        z, h = self.forward_encoder(inputs)

        # Generator: z -> node
        text, _, _ = self.generate_node(z, h, mask)

        # Generator: z -> adj
        adj = self.generate_adjacency_matrix(h)

        # classifier: graph -> prediction
        x = self.graph_encoder_cls(text, adj)
        x = F.relu(self.graph_encoder_cls_fc(torch.cat((x, text), -1)))
        # x = x.view(-1, self.node_dim * self.n_node)
        x = x.sum(dim=1)
        outputs = self.classifier(x)
        return outputs

    def generate_adjacency_matrix(self, hidden_state):
        adj = self.generate_adj(torch.cat(hidden_state, dim=-1))
        b_size = adj.size()[1]
        adj = adj.view(b_size, self.n_node, self.n_node)

        return adj

    def generate_graph(self, z, hidden_state, mask):
        node, attentions, _ = self.generate_node(z, hidden_state, mask)
        adj = self.generate_adjacency_matrix(hidden_state)
        return node, adj, attentions

    def train_model(self, args, dataset, logger):
        """train the whole network"""
        self.train()
        self.dataset = dataset

        alpha = args.alpha
        beta = args.beta
        lam = args.cls_weight
        gam = args.recon_weight

        trainer = optim.Adam(self.parameters(), lr=args.lr)
        criterion_cls = nn.CrossEntropyLoss().to(self.device)

        patience = 0
        best_val_acc, best_test_acc, best_iter = 0, 0, 0
        list_loss_recon, list_loss_penal1, list_loss_closs = [], [], []
        list_loss_cls = []
        train_start_time = time.time()
        train_iter = dataset.build_train_iter(args.batch_size, self.device)
        val_iter, test_iter = dataset.build_test_iter(args.batch_size, self.device, shuffle=False)

        for epoch in range(1, args.epochs):

            for i, subdata in enumerate([train_iter]):
                for batch in iter(subdata):
                    inputs, labels, mask = batch.text.t(), batch.label, batch.mask.t()
                    # mask = dataset.create_mask(inputs, device=self.device)

                    recon_loss, outputs, penal1, closs = self.forward(inputs, mask=mask, classifier=True)
                    loss_cls = criterion_cls(outputs, labels)
                    list_loss_cls.append(loss_cls.item())
                    loss_vae = gam * recon_loss + lam * loss_cls + alpha * penal1 + beta * closs

                    list_loss_recon.append(recon_loss.item())
                    list_loss_penal1.append(penal1.item())
                    list_loss_closs.append(closs.item())

                    trainer.zero_grad()
                    loss_vae.backward()
                    # grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
                    trainer.step()

            if epoch % args.log_every == 0:
                duration = time.time() - train_start_time
                avr_loss_recon = np.mean(list_loss_recon)
                avr_loss_penal1 = np.mean(list_loss_penal1)
                avr_closs = np.mean(list_loss_closs)
                avr_cls = np.mean(list_loss_cls)
                list_loss_recon, list_loss_penal1, list_loss_penal2, list_loss_closs = [], [], [], []
                list_loss_cls = []
                logger.info(f'Epoch-{epoch}; loss_recon: {avr_loss_recon:.4f}; loss_cls: {avr_cls:.4f}; penal1: {avr_loss_penal1:.4f}; '
                            f'avr_closs: {avr_closs:.4f}; duration: {round(duration)}')

                train_acc, train_loss = self.test(train_iter)
                val_acc, val_loss = self.test(val_iter)
                test_acc, test_loss = self.test(test_iter)
                loss_ratio = test_loss/train_loss

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_iter = epoch
                    best_loss_ratio = loss_ratio
                    patience = 0
                    graph = self.output_graph(dataset, test_iter, os.path.join(args.log_dir, f'epoch_{epoch}'), save_name='test')
                    graph = self.output_graph(dataset, val_iter, os.path.join(args.log_dir, f'epoch_{epoch}'), save_name='val')
                    graph = self.output_graph(dataset, train_iter, os.path.join(args.log_dir, f'epoch_{epoch}'), save_name='train')
                else:
                    if epoch > args.minimal_epoch:
                        patience += args.log_every

                logger.info(f'loss ratio: {loss_ratio:.4f} Train Acc: {train_acc:.4f}; '
                            f'Valid Acc: {val_acc:.4f}; Test Acc: {test_acc:.4f}; '
                            f'Best Acc: {best_test_acc:.4f} Best loss ratio: {best_loss_ratio:.4f} @ {best_iter}')

                if args.early_stop and patience > args.patience:
                    break

            if epoch % args.save_every == 0:
                graph = self.output_graph(dataset, test_iter, os.path.join(args.log_dir, f'epoch_{epoch}'), save_name='test')
                graph = self.output_graph(dataset, val_iter, os.path.join(args.log_dir, f'epoch_{epoch}'), save_name='val')
                graph = self.output_graph(dataset, train_iter, os.path.join(args.log_dir, f'epoch_{epoch}'), save_name='train')

        logger.info(f'Best Valid Acc: {best_val_acc:.4f}; Best Test Acc: {best_test_acc:.4f} @ epoch {best_iter}')
        return best_val_acc, best_test_acc

    def output_graph(self, dataset, data_iter, save_path=None, save_name=''):
        self.eval()
        input_list, adj_list, word_list = [], [], []
        with torch.no_grad():
            for batch in iter(data_iter):
                inputs, labels, mask = batch.text.t(), batch.label, batch.mask.t()

                z, h = self.forward_encoder(inputs)
                text, adj, attentions = self.generate_graph(z, h, mask)
                chosen = attentions.argmax(dim=-1).cpu().numpy()

                for ins, ch in zip(inputs.cpu().numpy(), chosen):
                    ins = [dataset.idx2word(i) for i in ins if i != self.PAD_IDX]
                    words = [ins[i] for i in ch]
                    input_list.append(' '.join(ins))
                    word_list.append(words)
                adj_list.append(adj.cpu().numpy())
        adj_list = np.concatenate(adj_list, axis=0)
        self.train()

        if save_path:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open(os.path.join(save_path, f'{save_name}.pickle'), 'wb') as handle:
                graph = {
                    'adj_list': adj_list,
                    'word_list': word_list,
                    'input_list': input_list
                }
                pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(save_path, f'{save_name}.json'), 'w') as handle:
                graph = {}
                for idx, (ins, words) in enumerate(zip(input_list, word_list)):
                    graph[idx] = {'doc':ins, 'keywords':words}
                json.dump(graph, handle, indent=4)

        return input_list, word_list, adj_list


class NetGenLink(AbstractNetGen):
    def __init__(self, n_node, n_vocab, n_labels, embed_dim, h_dim, z_dim, p_word_dropout=0.5, unk_idx=0, pad_idx=1, start_idx=2,
                 eos_idx=3, pretrained_embeddings=None, freeze_embeddings=False, teacher_forcing=True, device=False):
        super(NetGenLink, self).__init__(n_node, n_vocab, n_labels, embed_dim, p_word_dropout, unk_idx, pad_idx, start_idx, eos_idx,
                                     pretrained_embeddings, freeze_embeddings, teacher_forcing, device)

        self.node_dim = 2 * h_dim

        """
        Encoder is GRU with FC layers connected to last hidden unit
        """
        self.encoder = nn.LSTM(self.emb_dim, h_dim, bidirectional=True, batch_first=True)
        self.encoder_fc_c = nn.Linear(self.node_dim, h_dim)
        self.encoder_fc_h = nn.Linear(self.node_dim, h_dim)
        self.encoder_ = nn.ModuleList([
            self.encoder, self.encoder_fc_c, self.encoder_fc_h
        ])

        """
        Decoder is GRU with `z` appended at its inputs
        """
        self.decoder_h_linear = nn.Linear(self.n_node * self.node_dim, z_dim)
        self.decoder_c_linear = nn.Linear(self.n_node * self.node_dim, z_dim)
        self.decoder_d_linear = nn.Linear(self.n_node * self.node_dim, z_dim)
        self.decoder = nn.LSTM(self.emb_dim + z_dim, z_dim, num_layers=1, dropout=0.3, batch_first=True)
        # self.decoder_fc = nn.Linear(n_node * self.emb_dim, n_vocab)
        self.decoder_fc = nn.Linear(z_dim, self.n_vocab)

        """
        Generator 
        """
        self.generate_adj = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, self.n_node * self.n_node),
            nn.Sigmoid()
        )

        """
        Graph Encoder is GCN with pooling layer
        """
        self.graph_encoder = GraphConvolution(self.node_dim, self.node_dim, bias=False)
        self.graph_encoder_fc = nn.Linear(2 * self.node_dim, self.node_dim, bias=False)
        self.decoder_ = nn.ModuleList([
            self.decoder, self.decoder_fc, self.decoder_c_linear, self.decoder_h_linear, self.decoder_d_linear,
            self.generate_adj, self.graph_encoder, self.graph_encoder_fc
        ])

        """
        Classifier is DNN
        """
        self.graph_encoder_cls = GraphConvolution(self.node_dim, self.node_dim, bias=False)
        self.graph_encoder_cls_fc = nn.Linear(2 * self.node_dim, self.node_dim, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(self.node_dim, self.node_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.node_dim // 4, self.n_labels)
        )
        self.classifier_ = nn.ModuleList([
            self.graph_encoder_cls, self.graph_encoder_cls_fc, self.classifier
        ])

        """
        Grouping the model's parameters: separating encoder, decoder, and discriminator
        """
        self.encoder_params = filter(lambda p: p.requires_grad, self.encoder_.parameters())

        self.decoder_params = filter(lambda p: p.requires_grad, self.decoder_.parameters())

        self.classifier_params = filter(lambda p: p.requires_grad, self.classifier_.parameters())

        self.ae_params = chain(
            self.word_emb.parameters(), self.encoder_.parameters(), self.decoder_.parameters()
        )
        self.ae_params = filter(lambda p: p.requires_grad, self.ae_params)

        """
        Use GPU if set
        """

        # for m in self.modules():
        #     self.weights_init(m)

        self.to(self.device)

    def generate_node(self, z):
        bsize, n, d_k = z.size()
        text = torch.rand((bsize, self.n_node, self.node_dim)).to(self.device)
        # Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1)[0,1)
        return text

    def forward_graph_encoder(self, inputs, adj):
        # x = F.relu(self.graph_encoder(input, adj))
        x = self.graph_encoder(inputs, adj)
        x = F.relu(self.graph_encoder_fc(torch.cat((x, inputs), -1)))
        x = x.view(-1, self.n_node * self.node_dim)
        # x = self.graph_encoder_trans(x)

        return x

    def forward(self, sentence, mask, classifier):
        """
        Params:
        -------
        sentence: sequence of word indices.
        use_c_prior: whether to sample `c` from prior or from `discriminator`.

        Returns:
        --------
        recon_loss: reconstruction loss of VAE.
        kl_loss: KL-div loss of VAE.
        """

        mbsize = sentence.size(0)

        enc_inputs = sentence
        dec_inputs = sentence

        pad_words = torch.LongTensor([self.PAD_IDX]).repeat(mbsize, 1).to(self.device)
        dec_targets = torch.cat([sentence[:, 1:], pad_words], dim=1)

        # Encoder: sentence -> z
        z, h = self.forward_encoder(enc_inputs)

        # Generator: z -> node
        text = self.generate_node(z)

        # Generator: z -> adj
        adj = self.generate_adjacency_matrix(h)

        # Graph Encoder: graph -> z'
        z1 = self.forward_graph_encoder(text, adj)
        # z1 = text.view(-1, self.n_node * self.node_dim)

        # Decoder: sentence -> y
        y = self.forward_decoder(dec_inputs, z1)

        recon_loss = F.cross_entropy(y.view(-1, self.n_vocab), dec_targets.view(-1), size_average=True, ignore_index=self.PAD_IDX)

        if classifier:
            # classifier: graph -> prediction
            x = self.graph_encoder_cls(text, adj)
            x = F.relu(self.graph_encoder_cls_fc(torch.cat((x, text), -1)))
            # x = x.view(-1, self.node_dim * self.n_node)
            x = x.sum(dim=1)
            outputs = self.classifier(x)

            return recon_loss, outputs
        else:
            return recon_loss

    def forward_classifier(self, inputs, mask):

        # Encoder: sentence -> z
        z, h = self.forward_encoder(inputs)

        # Generator: z -> node
        text = self.generate_node(z)

        # Generator: z -> adj
        adj = self.generate_adjacency_matrix(h)

        # classifier: graph -> prediction
        x = self.graph_encoder_cls(text, adj)
        x = F.relu(self.graph_encoder_cls_fc(torch.cat((x, text), -1)))
        # x = x.view(-1, self.node_dim * self.n_node)
        x = x.sum(dim=1)
        outputs = self.classifier(x)
        return outputs

    def generate_adjacency_matrix(self, hidden_state):
        adj = self.generate_adj(torch.cat(hidden_state, dim=-1))
        b_size = adj.size()[1]
        adj = adj.view(b_size, self.n_node, self.n_node)

        return adj

    def generate_graph(self, z, hidden_state):
        node = self.generate_node(z)
        adj = self.generate_adjacency_matrix(hidden_state)
        return node, adj

    def train_model(self, args, dataset, logger):
        """train the whole network"""
        self.train()
        self.dataset = dataset

        alpha = args.alpha
        beta = args.beta
        lam = args.cls_weight
        gam = args.recon_weight

        trainer = optim.Adam(self.parameters(), lr=args.lr)
        criterion_cls = nn.CrossEntropyLoss().to(self.device)

        patience = 0
        best_val_acc, best_test_acc, best_iter = 0, 0, 0
        list_loss_recon = []
        list_loss_cls = []
        train_start_time = time.time()
        train_iter = dataset.build_train_iter(args.batch_size, self.device)
        val_iter, test_iter = dataset.build_test_iter(args.batch_size, self.device, shuffle=False)

        for epoch in range(1, args.epochs):

            for i, subdata in enumerate([train_iter]):
                for batch in iter(subdata):
                    inputs, labels, mask = batch.text.t(), batch.label, batch.mask.t()
                    # mask = dataset.create_mask(inputs, device=self.device)

                    recon_loss, outputs = self.forward(inputs, mask=mask, classifier=True)
                    loss_cls = criterion_cls(outputs, labels)
                    list_loss_cls.append(loss_cls.item())
                    loss_vae = gam * recon_loss + lam * loss_cls

                    if torch.isnan(loss_vae):
                        a=1
                    list_loss_recon.append(recon_loss.item())

                    trainer.zero_grad()
                    loss_vae.backward()
                    # grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
                    trainer.step()

            if epoch % args.log_every == 0:
                duration = time.time() - train_start_time
                avr_loss_recon = np.mean(list_loss_recon)
                list_loss_recon, list_loss_penal1, list_loss_penal2, list_loss_closs = [], [], [], []
                list_loss_cls = []
                logger.info(f'Epoch-{epoch}; loss_recon: {avr_loss_recon:.4f}; loss_cls: duration: {round(duration)}')

                train_acc, train_loss = self.test(train_iter)
                val_acc, val_loss = self.test(val_iter)
                test_acc, test_loss = self.test(test_iter)
                loss_ratio = test_loss / train_loss

                if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_test_acc = test_acc
                        best_iter = epoch
                        best_loss_ratio = loss_ratio
                        patience = 0
                else:
                    if epoch > args.minimal_epoch:
                        patience += args.log_every

                logger.info(f'loss ratio: {loss_ratio:.4f} Train Acc: {train_acc:.4f}; '
                            f'Valid Acc: {val_acc:.4f}; Test Acc: {test_acc:.4f}; '
                            f'Best Acc: {best_test_acc:.4f} Best loss ratio: {best_loss_ratio:.4f} @ {best_iter}')

                if args.early_stop and patience > args.patience:
                    break

        logger.info(f'Best Valid Acc: {best_val_acc:.4f}; Best Test Acc: {best_test_acc:.4f} @ epoch {best_iter}')
        return best_val_acc, best_test_acc


class NetGenWord(AbstractNetGen):
    def __init__(self, n_node, n_vocab, n_labels, embed_dim, h_dim, z_dim, p_word_dropout=0.5, unk_idx=0, pad_idx=1,
                 start_idx=2, eos_idx=3, pretrained_embeddings=None, freeze_embeddings=False, teacher_forcing=True, device=False):
        super(NetGenWord, self).__init__(n_node, n_vocab, n_labels, embed_dim, p_word_dropout, unk_idx, pad_idx,
                                     start_idx, eos_idx, pretrained_embeddings, freeze_embeddings, teacher_forcing, device)

        self.node_dim = 2 * h_dim

        """
        Encoder is GRU with FC layers connected to last hidden unit
        """
        self.encoder = nn.LSTM(self.emb_dim, h_dim, bidirectional=True, batch_first=True)
        self.encoder_fc_c = nn.Linear(self.node_dim, h_dim)
        self.encoder_fc_h = nn.Linear(self.node_dim, h_dim)
        self.encoder_ = nn.ModuleList([
            self.encoder, self.encoder_fc_c, self.encoder_fc_h
        ])

        """
        Decoder is GRU with `z` appended at its inputs
        """
        self.decoder_h_linear = nn.Linear(self.n_node * self.node_dim, z_dim)
        self.decoder_c_linear = nn.Linear(self.n_node * self.node_dim, z_dim)
        self.decoder_d_linear = nn.Linear(self.n_node * self.node_dim, z_dim)
        self.decoder = nn.LSTM(self.emb_dim + z_dim, z_dim, num_layers=1, dropout=0.3, batch_first=True)
        # self.decoder_fc = nn.Linear(n_node * self.emb_dim, n_vocab)
        self.decoder_fc = nn.Linear(z_dim, self.n_vocab)

        """
        Generator 
        """
        self.generate_lstm = nn.LSTM(self.node_dim, h_dim, batch_first=True)
        self.attention_network = Attention(self.node_dim)

        """
        Node Encoder is GCN with pooling layer
        """
        self.node_encoder_fc = nn.Linear(self.node_dim, self.node_dim, bias=False)
        self.decoder_ = nn.ModuleList([
            self.decoder, self.decoder_fc, self.decoder_c_linear, self.decoder_h_linear, self.decoder_d_linear,
            self.attention_network, self.generate_lstm, self.node_encoder_fc
        ])

        """
        Classifier is DNN
        """
        self.node_encoder_cls_fc = nn.Linear(self.node_dim, self.node_dim, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(self.node_dim, self.node_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.node_dim // 4, self.n_labels)
        )
        self.classifier_ = nn.ModuleList([
            self.node_encoder_cls_fc, self.classifier
        ])

        """
        Grouping the model's parameters: separating encoder, decoder, and discriminator
        """
        self.encoder_params = filter(lambda p: p.requires_grad, self.encoder_.parameters())

        self.decoder_params = filter(lambda p: p.requires_grad, self.decoder_.parameters())

        self.classifier_params = filter(lambda p: p.requires_grad, self.classifier_.parameters())

        self.ae_params = chain(
            self.word_emb.parameters(), self.encoder_.parameters(), self.decoder_.parameters()
        )
        self.ae_params = filter(lambda p: p.requires_grad, self.ae_params)

        """
        Use GPU if set
        """

        # for m in self.modules():
        #     self.weights_init(m)

        self.to(self.device)

    def forward_node_encoder(self, inputs):
        x = F.relu(self.node_encoder_fc(inputs))
        x = x.view(-1, self.n_node * self.node_dim)
        return x

    def forward(self, sentence, mask, classifier):

        mbsize = sentence.size(0)

        enc_inputs = sentence
        dec_inputs = sentence

        pad_words = torch.LongTensor([self.PAD_IDX]).repeat(mbsize, 1).to(self.device)
        dec_targets = torch.cat([sentence[:, 1:], pad_words], dim=1)

        # Encoder: sentence -> z
        z, h = self.forward_encoder(enc_inputs)

        # Generator: z -> node
        text, attentions, c_loss = self.generate_node(z, h, mask)

        # Node Encoder: node -> z'
        z1 = self.forward_node_encoder(text)
        # z1 = text.view(-1, self.n_node * self.node_dim)

        # Decoder: sentence -> y
        y = self.forward_decoder(dec_inputs, z1)

        recon_loss = F.cross_entropy(y.view(-1, self.n_vocab), dec_targets.view(-1), size_average=True,
                                     ignore_index=self.PAD_IDX)

        penal1 = torch.distributions.Categorical(probs=attentions).entropy().mean()

        if classifier:
            # classifier: graph -> prediction
            x = F.relu(self.node_encoder_cls_fc(text))
            # x = x.view(-1, self.n_node * self.node_dim)
            x = x.sum(dim=1)
            outputs = self.classifier(x)

            return recon_loss, outputs, penal1, c_loss
        else:
            return recon_loss, penal1, c_loss

    def forward_classifier(self, inputs, mask):

        # Encoder: sentence -> z
        z, h = self.forward_encoder(inputs)

        # Generator: z -> node
        text, _, _ = self.generate_node(z, h, mask)

        # classifier: node -> prediction
        x = F.relu(self.node_encoder_cls_fc(text))
        # x = x.view(-1, self.n_node * self.node_dim)
        x = x.sum(dim=1)
        outputs = self.classifier(x)
        return outputs

    def train_model(self, args, dataset, logger):
        """train the whole network"""
        self.train()
        self.dataset = dataset

        alpha = args.alpha
        beta = args.beta
        lam = args.cls_weight
        gam = args.recon_weight

        trainer = optim.Adam(self.parameters(), lr=args.lr)
        criterion_cls = nn.CrossEntropyLoss().to(self.device)

        patience = 0
        best_val_acc, best_test_acc, best_iter = 0, 0, 0
        list_loss_recon, list_loss_penal1, list_loss_closs = [], [], []
        list_loss_cls = []
        train_start_time = time.time()
        train_iter = dataset.build_train_iter(args.batch_size, self.device)
        val_iter, test_iter = dataset.build_test_iter(args.batch_size, self.device, shuffle=False)

        for epoch in range(1, args.epochs):

            for i, subdata in enumerate([train_iter]):
                for batch in iter(subdata):
                    inputs, labels, mask = batch.text.t(), batch.label, batch.mask.t()
                    # mask = dataset.create_mask(inputs, device=self.device)

                    recon_loss, outputs, penal1, closs = self.forward(inputs, mask=mask, classifier=True)
                    loss_cls = criterion_cls(outputs, labels)
                    list_loss_cls.append(loss_cls.item())
                    loss_vae = gam * recon_loss + lam * loss_cls + alpha * penal1 + beta * closs

                    if torch.isnan(loss_vae):
                        a = 1
                    list_loss_recon.append(recon_loss.item())
                    list_loss_penal1.append(penal1.item())
                    list_loss_closs.append(closs.item())

                    trainer.zero_grad()
                    loss_vae.backward()
                    # grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
                    trainer.step()

            if epoch % args.log_every == 0:
                duration = time.time() - train_start_time
                avr_loss_recon = np.mean(list_loss_recon)
                avr_loss_penal1 = np.mean(list_loss_penal1)
                avr_closs = np.mean(list_loss_closs)
                avr_cls = np.mean(list_loss_cls)
                list_loss_recon, list_loss_penal1, list_loss_penal2, list_loss_closs = [], [], [], []
                list_loss_cls = []
                logger.info(
                    f'Epoch-{epoch}; loss_recon: {avr_loss_recon:.4f}; loss_cls: {avr_cls:.4f}; penal1: {avr_loss_penal1:.4f}; '
                    f'avr_closs: {avr_closs:.4f}; duration: {round(duration)}')

                train_acc, train_loss = self.test(train_iter)
                val_acc, val_loss = self.test(val_iter)
                test_acc, test_loss = self.test(test_iter)
                loss_ratio = test_loss / train_loss

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_iter = epoch
                    best_loss_ratio = loss_ratio
                    patience = 0
                    graph = self.output_graph(dataset, test_iter, os.path.join(args.log_dir, f'epoch_{epoch}'), save_name='test')
                    graph = self.output_graph(dataset, val_iter, os.path.join(args.log_dir, f'epoch_{epoch}'), save_name='val')
                    graph = self.output_graph(dataset, train_iter, os.path.join(args.log_dir, f'epoch_{epoch}'), save_name='train')
                else:
                    if epoch > args.minimal_epoch:
                        patience += args.log_every

                logger.info(f'loss ratio: {loss_ratio:.4f} Train Acc: {train_acc:.4f}; '
                            f'Valid Acc: {val_acc:.4f}; Test Acc: {test_acc:.4f}; '
                            f'Best Acc: {best_test_acc:.4f} Best loss ratio: {best_loss_ratio:.4f} @ {best_iter}')

                if args.early_stop and patience > args.patience:
                    break

            if epoch % args.save_every == 0:
                graph = self.output_graph(dataset, test_iter, os.path.join(args.log_dir, f'epoch_{epoch}'), save_name='test')
                graph = self.output_graph(dataset, val_iter, os.path.join(args.log_dir, f'epoch_{epoch}'), save_name='val')
                graph = self.output_graph(dataset, train_iter, os.path.join(args.log_dir, f'epoch_{epoch}'), save_name='train')

        logger.info(f'Best Valid Acc: {best_val_acc:.4f}; Best Test Acc: {best_test_acc:.4f} @ epoch {best_iter}')
        return best_val_acc, best_test_acc

    def output_graph(self, dataset, data_iter, save_path=None, save_name=''):
        self.eval()
        input_list, word_list = [], []
        with torch.no_grad():
            for batch in iter(data_iter):
                inputs, labels, mask = batch.text.t(), batch.label, batch.mask.t()

                z, h = self.forward_encoder(inputs)
                text, attentions, _ = self.generate_node(z, h, mask)
                chosen = attentions.argmax(dim=-1).cpu().numpy()

                for ins, ch in zip(inputs.cpu().numpy(), chosen):
                    ins = [dataset.idx2word(i) for i in ins if i != self.PAD_IDX]
                    words = [ins[i] for i in ch]
                    input_list.append(' '.join(ins))
                    word_list.append(words)
        self.train()

        if save_path:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open(os.path.join(save_path, f'{save_name}.pickle'), 'wb') as handle:
                graph = {
                    'word_list': word_list,
                    'input_list': input_list
                }
                pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(save_path, f'{save_name}.json'), 'w') as handle:
                graph = {}
                for idx, (ins, words) in enumerate(zip(input_list, word_list)):
                    graph[idx] = {'doc':ins, 'keywords':words}
                json.dump(graph, handle, indent=4)

        return input_list, word_list


class LSTMClassifer(nn.Module):
    def __init__(self, n_node, n_vocab, n_labels, embed_dim, h_dim, z_dim, p_word_dropout=0.5, unk_idx=0, pad_idx=1,
                 start_idx=2, eos_idx=3, pretrained_embeddings=None, freeze_embeddings=False, teacher_forcing=True, device=False):
        super(LSTMClassifer, self).__init__()

        self.UNK_IDX = unk_idx
        self.PAD_IDX = pad_idx
        self.START_IDX = start_idx
        self.EOS_IDX = eos_idx

        self.n_vocab = n_vocab
        self.n_labels = n_labels
        self.h_dim = h_dim
        self.p_word_dropout = p_word_dropout

        self.device = device

        """
        Word embeddings layer
        """
        if pretrained_embeddings is None:
            self.emb_dim = embed_dim
            self.word_emb = nn.Embedding(n_vocab, embed_dim, self.PAD_IDX)
        else:
            print("use random init emb.")
            sys.exit(1)
            self.emb_dim = pretrained_embeddings.size(1)
            self.word_emb = nn.Embedding(n_vocab, self.emb_dim, self.PAD_IDX)

            # Set pretrained embeddings
            self.word_emb.weight.data.copy_(pretrained_embeddings)

            if freeze_embeddings:
                self.word_emb.weight.requires_grad = False

        self.lstm = nn.LSTM(embed_dim, h_dim, batch_first=True, bidirectional=False)
        self.classifier = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(h_dim, self.n_labels)
        )

        self.to(self.device)

    def forward(self, sentence):
        inputs = self.word_emb(sentence)
        _, (hn, cn) = self.lstm(inputs)
        bsize = hn.size()[1]
        # hn = hn.view(-1, bsize, 2*self.h_dim)
        # cn = cn.view(-1, bsize, 2*self.h_dim)
        final_output = self.classifier(torch.cat((hn, cn), dim=-1).squeeze(0))
        return final_output

    def train_model(self, args, dataset, logger):
        self.train()
        self.dataset = dataset

        patience = 0
        list_loss_cls = []
        best_train_acc, best_val_acc, best_test_acc, best_iter = 0, 0, 0, 0
        trainer = optim.Adam(self.parameters(), lr=args.lr)
        criterion_cls = nn.CrossEntropyLoss().to(self.device)

        train_iter = dataset.build_train_iter(args.batch_size, self.device)
        val_iter, test_iter = dataset.build_test_iter(args.batch_size, self.device, shuffle=False)

        for epoch in range(1, args.epochs):
            for i, subdata in enumerate([train_iter]):
                for batch in iter(subdata):
                    inputs, labels = batch.text.t(), batch.label
                    pred = self.forward(inputs)
                    loss = criterion_cls(pred, labels)
                    list_loss_cls.append(loss.item())
                    trainer.zero_grad()
                    loss.backward()
                    trainer.step()

            if epoch % args.log_every == 0:
                train_acc, train_loss = self.test(train_iter)
                val_acc, val_loss = self.test(val_iter)
                test_acc, test_loss = self.test(test_iter)
                loss_ratio = test_loss / train_loss

                avr_cls = np.mean(list_loss_cls)
                list_loss_cls = []

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_iter = epoch
                    best_loss_ratio = loss_ratio
                    patience = 0
                else:
                    if epoch > args.minimal_epoch:
                        patience += args.log_every

                logger.info(f'loss ratio: {loss_ratio:.4f} Train Acc: {train_acc:.4f}; '
                            f'Valid Acc: {val_acc:.4f}; Test Acc: {test_acc:.4f}; '
                            f'Best Acc: {best_test_acc:.4f} Best loss ratio: {best_loss_ratio:.4f} @ {best_iter}')

                if args.early_stop and patience > args.patience:
                    break

        logger.info(f'Best Valid Acc: {best_val_acc:.4f}; Best Test Acc: {best_test_acc:.4f} @ epoch {best_iter}')
        return best_val_acc, best_test_acc

    def test(self, data_iter):
        self.eval()
        total_accuracy, l_loss_cls = [], []
        criterion_cls = nn.CrossEntropyLoss().to(self.device)
        with torch.no_grad():
            for batch in iter(data_iter):
                inputs, labels = batch.text.t(), batch.label
                outputs = self.forward(inputs)
                loss_cls = criterion_cls(outputs, labels)
                l_loss_cls.append(loss_cls.item())
                accuracy = (outputs.argmax(1) == labels).float().mean().item()
                total_accuracy.append(accuracy)
        acc = sum(total_accuracy) / len(total_accuracy)
        loss = np.mean(l_loss_cls)
        self.train()
        return acc, loss

