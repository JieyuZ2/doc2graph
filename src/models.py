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


class AbstractVAE(nn.Module):
    def __init__(self, n_vocab, n_labels, embed_dim, h_dim, p_word_dropout=0.3, unk_idx=0, pad_idx=1, start_idx=2, eos_idx=3,
                 pretrained_embeddings=None, freeze_embeddings=False, device=False):
        super(AbstractVAE, self).__init__()
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
            self.emb_dim = pretrained_embeddings.size(1)
            self.word_emb = nn.Embedding(n_vocab, self.emb_dim, self.PAD_IDX)

            # Set pretrained embeddings
            self.word_emb.weight.data.copy_(pretrained_embeddings)

            if freeze_embeddings:
                self.word_emb.weight.requires_grad = False

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
        _, h = self.encoder(inputs, None)

        # Forward to latent
        h = h.view(-1, self.h_dim)

        mu = self.q_mu(h)
        logvar = self.q_logvar(h)

        return mu, logvar

    def sample_z(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std*eps; eps ~ N(0, I)
        """
        eps = Variable(torch.randn_like(mu))
        eps = eps.to(self.device)
        return mu + torch.exp(logvar / 2) * eps

    def sample_z_prior(self, mbsize):
        """
        Sample z ~ p(z) = N(0, I)
        """
        z = Variable(torch.randn(mbsize, self.z_dim))
        z = z.to(self.device)
        return z

    def forward_decoder(self, inputs, z):
        """
        Inputs must be embeddings: seq_len x mbsize
        """
        dec_inputs = self.word_dropout(inputs)

        # Forward
        seq_len = dec_inputs.size(0)

        # 1 x mbsize x z_dim
        init_h = z.unsqueeze(0)
        inputs_emb = self.word_emb(dec_inputs)  # seq_len x mbsize x emb_dim
        inputs_emb = torch.cat([inputs_emb, init_h.repeat(seq_len, 1, 1)], 2)

        outputs, _ = self.decoder(inputs_emb, init_h)
        seq_len, mbsize, _ = outputs.size()

        outputs = outputs.view(seq_len * mbsize, -1)
        y = self.decoder_fc(outputs)
        y = y.view(seq_len, mbsize, self.n_vocab)

        return y

    def generate_sentences(self, batch_size, length, temp=1.0, z=None):
        """
        Generate sentences and corresponding z of (batch_size x max_sent_len)
        """
        self.eval()

        if z is None:
            z = self.sample_z_prior(batch_size)

        X_gen = self.sample_sentence(z, mask, length, raw=True, temp=temp)

        # Back to default state: train
        self.train()

        return X_gen

    def sample_sentence(self, z, length, raw=False, temp=1.0):
        """
        Sample single sentence from p(x|z,c) according to given temperature.
        `raw = True` means this returns sentence as in dataset which is useful
        to train discriminator. `False` means that this will return list of
        `word_idx` which is useful for evaluation.
        """

        word = torch.LongTensor([self.START_IDX])
        word = Variable(word)  # '<start>'
        word = word.to(self.device)

        if not isinstance(z, Variable):
            z = Variable(z)

        z = z.view(1, 1, -1)
        h = z

        outputs = []

        if raw:
            outputs.append(self.START_IDX)

        for i in range(length - 1):
            emb = self.word_emb(word).view(1, 1, -1)
            emb = torch.cat([emb, z], 2)

            output, h = self.decoder(emb, h)
            y = self.decoder_fc(output).view(-1)
            y = F.softmax(y / temp, dim=0)

            idx = torch.multinomial(y, num_samples=1)

            word = Variable(torch.LongTensor([int(idx)]))
            word = word.to(self.device)

            idx = int(idx)

            if not raw and idx == self.EOS_IDX:
                break

            outputs.append(idx)

        if raw:
            outputs = Variable(torch.LongTensor(outputs)).unsqueeze(0)
            return outputs.to(self.device)
        else:
            return outputs

    def word_dropout(self, inputs, padding_mask):
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

        mask = mask.to(self.device)
        mask = mask * padding_mask.byte()

        # Set to <unk>
        data[mask] = self.UNK_IDX

        return Variable(data)

    def forward(self, *input):
        pass

    def train_model(self, args, dataset):
        pass


class RNNVAE(AbstractVAE):
    def __init__(self, n_vocab, n_labels, h_dim, z_dim, p_word_dropout=0.3, unk_idx=0, pad_idx=1, start_idx=2, eos_idx=3,
                 pretrained_embeddings=None, freeze_embeddings=False, device=False):
        super(RNNVAE, self).__init__(n_vocab, n_labels, h_dim, p_word_dropout, unk_idx, pad_idx, start_idx, eos_idx,
                                     pretrained_embeddings, freeze_embeddings, device)

        self.z_dim = z_dim

        """
        Encoder is GRU with FC layers connected to last hidden unit
        """
        self.encoder = nn.GRU(self.emb_dim, h_dim)
        self.q_mu = nn.Linear(h_dim, z_dim)
        self.q_logvar = nn.Linear(h_dim, z_dim)
        self.encoder_ = nn.ModuleList([
            self.encoder, self.q_mu, self.q_logvar
        ])

        """
        Decoder is GRU with `z` appended at its inputs
        """
        self.decoder = nn.GRU(self.emb_dim + z_dim, z_dim, dropout=0.3)
        self.decoder_fc = nn.Linear(z_dim, n_vocab)
        self.decoder_ = nn.ModuleList([
            self.decoder, self.decoder_fc
        ])

        """
        Classifier is DNN
        """
        self.classifier = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, self.n_labels)
        )

        """
        Grouping the model's parameters: separating encoder, decoder, and discriminator
        """
        self.encoder_params = filter(lambda p: p.requires_grad, self.encoder_.parameters())

        self.decoder_params = filter(lambda p: p.requires_grad, self.decoder_.parameters())

        self.classifier_params = filter(lambda p: p.requires_grad, self.classifier.parameters())

        self.vae_params = chain(
            self.word_emb.parameters(), self.encoder_.parameters(), self.decoder_.parameters(),
            self.classifier.parameters()
        )
        self.vae_params = filter(lambda p: p.requires_grad, self.vae_params)

        """
        Use GPU if set
        """
        self.to(device)

    def forward(self, sentence):
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

        mbsize = sentence.size(1)

        # sentence: '<start> I want to fly <eos>'
        # enc_inputs: '<start> I want to fly <eos>'
        # dec_inputs: '<start> I want to fly <eos>'
        # dec_targets: 'I want to fly <eos> <pad>'
        pad_words = Variable(torch.LongTensor([self.PAD_IDX])).repeat(1, mbsize)
        pad_words = pad_words.to(self.device)

        enc_inputs = sentence
        dec_inputs = sentence
        dec_targets = torch.cat([sentence[1:], pad_words], dim=0)

        # Encoder: sentence -> z
        mu, logvar = self.forward_encoder(enc_inputs)
        z = self.sample_z(mu, logvar)

        # Decoder: sentence -> y
        y = self.forward_decoder(dec_inputs, z)

        recon_loss = F.cross_entropy(
            y.view(-1, self.n_vocab), dec_targets.view(-1), size_average=True
        )
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1 - logvar, 1))

        return recon_loss, kl_loss, z

    def forward_classifier(self, sentence):
        enc_inputs = sentence

        # Encoder: sentence -> z
        mu, logvar = self.forward_encoder(enc_inputs)

        return self.classifier(mu)

    def train_model(self, args, dataset):
        """train the whole graph gan network"""
        self.train()
        trainer_VAE = optim.Adam(self.vae_params, lr=args.lr)
        # criterion_cls = nn.CrossEntropyLoss().to(self.device)

        train_iter = dataset.build_train_iter(args.batch_size, self.device)

        kl_weight = args.kl_weight
        # cls_weight = args.cls_weight

        list_loss_recon, list_loss_kl, list_loss_cls = [], [], []
        train_start_time = time.time()

        val_acc, test_acc = self.train_model_classifier(args, dataset)
        print(f'Initial Eval: Valid Acc: {val_acc:.4f}; Test Acc: {test_acc:.4f}')

        for epoch in range(1, args.epochs):
            for batch in iter(train_iter):
                inputs, labels = batch.text, batch.label
                recon_loss, kl_loss, z = self.forward(inputs)
                loss_vae = recon_loss + kl_weight * kl_loss
                list_loss_recon.append(recon_loss.item())
                list_loss_kl.append(kl_loss.item())

                trainer_VAE.zero_grad()
                loss_vae.backward(retain_graph=True)
                trainer_VAE.step()

                # outputs = self.forward_classifier(z)
                # loss_cls = criterion_cls(outputs, labels)
                # list_loss_cls.append(loss_cls.item())
                #
                # loss = cls_weight * loss_cls + loss_vae
                #
                # trainer_VAE.zero_grad()
                # loss.backward(retain_graph=True)
                # trainer_VAE.step()

            if epoch % args.log_every == 0:
                duration = time.time() - train_start_time
                avr_loss_recon = np.mean(list_loss_recon)
                avr_loss_kl = np.mean(list_loss_kl)
                list_loss_recon, list_loss_kl = [], []
                print(f'Epoch-{epoch}; loss_recon: {avr_loss_recon:.4f}; loss_kl: {avr_loss_kl:.4f}; duration: {duration:.2f}')

                val_acc, test_acc = self.train_model_classifier(args, dataset)

                # avr_loss_cls = np.mean(list_loss_cls)
                # list_loss_recon, list_loss_kl, list_loss_cls = [], [], []
                # print(f'Iter-{epoch}; loss_recon: {avr_loss_recon:.4f}; loss_kl: {avr_loss_kl:.4f}; '
                #       f'loss_cls: {avr_loss_cls:.4f}; duration: {duration:.2f}')

                # val_iter, test_iter = dataset.build_test_iter()
                # val_acc = self.test(val_iter)
                # test_acc = self.test(test_iter)

                z = self.sample_z_prior(1)
                sample_idxs = self.sample_sentence(z, length=50)
                sample_sent = dataset.idxs2sentence(sample_idxs)
                print(f'Valid Acc: {val_acc:.4f}; Test Acc: {test_acc:.4f}; Sample: "{sample_sent}"')

            if epoch % args.save_every == 0:
                flag = None
                if args.early_stop and epoch > args.minimal_epoch and flag:
                    break

            if epoch > args.epochs:
                break

    def train_model_classifier(self, args, dataset):
        """train the whole graph gan network"""

        self.train()
        for i in self.classifier:
            if hasattr(i, 'reset_parameters'):
                i.reset_parameters()
        classifier_params = filter(lambda p: p.requires_grad, self.classifier.parameters())
        trainer_C = optim.Adam(classifier_params, lr=args.lr)
        criterion_cls = nn.CrossEntropyLoss().to(self.device)

        train_iter = dataset.build_train_iter(args.batch_size, self.device)

        list_loss_cls = []
        best_val_acc, best_test_acc, best_iter = 0, 0, 0
        patience = 0

        train_start_time = time.time()

        for epoch in range(1, args.epochs):
            for batch in iter(train_iter):

                inputs, labels = batch.text, batch.label
                outputs = self.forward_classifier(inputs)

                loss_cls = criterion_cls(outputs, labels)
                list_loss_cls.append(loss_cls.item())

                trainer_C.zero_grad()
                loss_cls.backward()
                trainer_C.step()

            if epoch % args.log_every == 0:

                val_iter, test_iter = dataset.build_test_iter(args.batch_size, self.device)
                val_acc = self.test(val_iter)
                test_acc = self.test(test_iter)

                duration = time.time() - train_start_time
                avr_loss_cls = np.mean(list_loss_cls)
                list_loss_cls = []

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_iter = epoch
                    patience = 0
                else:
                    if epoch > args.minimal_epoch:
                        patience += args.log_every

                if args.early_stop and patience > args.eval_patience:
                    break

            # if epoch % 200 == 0:
            #     print(f'Epoch-{epoch}; loss_cls: {avr_loss_cls:.4f}; Valid Acc: {val_acc:.4f}; Test Acc: {test_acc:.4f};'
            #           f' duration: {duration:.2f}')

        # print(f'Best Valid Acc: {best_val_acc:.4f}; Best Test Acc: {best_test_acc:.4f} @ iter {best_iter}')
        return best_val_acc, best_test_acc

    def test(self, data_iter):
        total_accuracy = []
        self.eval()
        with torch.no_grad():
            for batch in iter(data_iter):
                inputs, labels = batch.text, batch.label
                outputs = self.forward_classifier(inputs)
                accuracy = (outputs.argmax(1) == labels).float().mean().item()
                total_accuracy.append(accuracy)
        acc = sum(total_accuracy) / len(total_accuracy)
        self.train()
        return acc


class RNNAE(AbstractVAE):
    def __init__(self, n_vocab, n_labels, h_dim, z_dim, p_word_dropout=0.3, unk_idx=0, pad_idx=1, start_idx=2, eos_idx=3,
                 pretrained_embeddings=None, freeze_embeddings=False, device=False):
        super(RNNAE, self).__init__(n_vocab, n_labels, h_dim, p_word_dropout, unk_idx, pad_idx, start_idx, eos_idx,
                                     pretrained_embeddings, freeze_embeddings, device)

        self.z_dim = z_dim

        """
        Encoder is GRU with FC layers connected to last hidden unit
        """
        self.encoder = nn.GRU(self.emb_dim, h_dim)
        self.q_mu = nn.Linear(h_dim, z_dim)
        self.encoder_ = nn.ModuleList([
            self.encoder, self.q_mu
        ])

        """
        Decoder is GRU with `z` appended at its inputs
        """
        self.decoder = nn.GRU(self.emb_dim + z_dim, z_dim, dropout=0.3)
        self.decoder_fc = nn.Linear(z_dim, n_vocab)
        self.decoder_ = nn.ModuleList([
            self.decoder, self.decoder_fc
        ])

        """
        Classifier is DNN
        """
        self.classifier = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, self.n_labels)
        )

        """
        Grouping the model's parameters: separating encoder, decoder, and discriminator
        """
        self.encoder_params = filter(lambda p: p.requires_grad, self.encoder_.parameters())

        self.decoder_params = filter(lambda p: p.requires_grad, self.decoder_.parameters())

        self.classifier_params = filter(lambda p: p.requires_grad, self.classifier.parameters())

        self.ae_params = chain(
            self.word_emb.parameters(), self.encoder_.parameters(), self.decoder_.parameters()
        )
        self.ae_params = filter(lambda p: p.requires_grad, self.ae_params)

        """
        Use GPU if set
        """
        self.to(device)

    def forward_encoder_embed(self, inputs):
        """
        Inputs is embeddings of: seq_len x mbsize x emb_dim
        """
        _, h = self.encoder(inputs, None)

        # Forward to latent
        h = h.view(-1, self.h_dim)

        mu = self.q_mu(h)

        return mu

    def forward(self, sentence):
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

        mbsize = sentence.size(1)

        # sentence: '<start> I want to fly <eos>'
        # enc_inputs: '<start> I want to fly <eos>'
        # dec_inputs: '<start> I want to fly <eos>'
        # dec_targets: 'I want to fly <eos> <pad>'
        pad_words = Variable(torch.LongTensor([self.PAD_IDX])).repeat(1, mbsize)
        pad_words = pad_words.to(self.device)

        enc_inputs = sentence
        dec_inputs = sentence
        dec_targets = torch.cat([sentence[1:], pad_words], dim=0)

        # Encoder: sentence -> z
        z = self.forward_encoder(enc_inputs)

        # Decoder: sentence -> y
        y = self.forward_decoder(dec_inputs, z)

        recon_loss = F.cross_entropy(
            y.view(-1, self.n_vocab), dec_targets.view(-1), size_average=True
        )

        return recon_loss, z

    def forward_classifier(self, sentence):
        enc_inputs = sentence

        # Encoder: sentence -> z
        mu = self.forward_encoder(enc_inputs)

        return self.classifier(mu)

    def train_model(self, args, dataset):
        """train the whole graph gan network"""
        self.train()
        trainer_AE = optim.Adam(self.ae_params, lr=args.lr)
        # criterion_cls = nn.CrossEntropyLoss().to(self.device)

        train_iter = dataset.build_train_iter(args.batch_size, self.device)

        # cls_weight = args.cls_weight

        list_loss_recon, list_loss_cls = [], []
        train_start_time = time.time()

        val_acc, test_acc = self.train_model_classifier(args, dataset)
        print(f'Initial Eval: Valid Acc: {val_acc:.4f}; Test Acc: {test_acc:.4f}')

        for epoch in range(1, args.epochs):
            for batch in iter(train_iter):
                inputs, labels = batch.text, batch.label
                recon_loss,  z = self.forward(inputs)
                loss_ae = recon_loss
                list_loss_recon.append(recon_loss.item())

                trainer_AE.zero_grad()
                loss_ae.backward()
                trainer_AE.step()

                # outputs = self.forward_classifier(z)
                # loss_cls = criterion_cls(outputs, labels)
                # list_loss_cls.append(loss_cls.item())
                #
                # loss = cls_weight * loss_cls + loss_vae
                #
                # trainer_VAE.zero_grad()
                # loss.backward(retain_graph=True)
                # trainer_VAE.step()

            if epoch % args.log_every == 0:
                duration = time.time() - train_start_time
                avr_loss_recon = np.mean(list_loss_recon)
                list_loss_recon = []
                print(f'Epoch-{epoch}; loss_recon: {avr_loss_recon:.4f}; duration: {duration:.2f}')

                val_acc, test_acc = self.train_model_classifier(args, dataset)
                print(f'Valid Acc: {val_acc:.4f}; Test Acc: {test_acc:.4f}')

            if epoch % args.save_every == 0:
                flag = None
                if args.early_stop and epoch > args.minimal_epoch and flag:
                    break

            if epoch > args.epochs:
                break

    def train_model_classifier(self, args, dataset):
        """train the whole graph gan network"""

        self.train()
        for i in self.classifier:
            if hasattr(i, 'reset_parameters'):
                i.reset_parameters()
        classifier_params = filter(lambda p: p.requires_grad, self.classifier.parameters())
        trainer_C = optim.Adam(classifier_params, lr=args.lr)
        criterion_cls = nn.CrossEntropyLoss().to(self.device)

        train_iter = dataset.build_train_iter(args.batch_size, self.device)

        list_loss_cls = []
        best_val_acc, best_test_acc, best_iter = 0, 0, 0
        patience = 0

        train_start_time = time.time()

        for epoch in range(1, args.epochs):
            for batch in iter(train_iter):

                inputs, labels = batch.text, batch.label
                outputs = self.forward_classifier(inputs)

                loss_cls = criterion_cls(outputs, labels)
                list_loss_cls.append(loss_cls.item())

                trainer_C.zero_grad()
                loss_cls.backward()
                trainer_C.step()

            if epoch % args.log_every == 0:

                val_iter, test_iter = dataset.build_test_iter(args.batch_size, self.device)
                val_acc = self.test(val_iter)
                test_acc = self.test(test_iter)

                duration = time.time() - train_start_time
                avr_loss_cls = np.mean(list_loss_cls)
                list_loss_cls = []

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_iter = epoch
                    patience = 0
                else:
                    if epoch > args.minimal_epoch:
                        patience += args.log_every

                if args.early_stop and patience > args.eval_patience:
                    break

            # if epoch % 200 == 0:
            #     print(f'Epoch-{epoch}; loss_cls: {avr_loss_cls:.4f}; Valid Acc: {val_acc:.4f}; Test Acc: {test_acc:.4f};'
            #           f' duration: {duration:.2f}')

        # print(f'Best Valid Acc: {best_val_acc:.4f}; Best Test Acc: {best_test_acc:.4f} @ iter {best_iter}')
        return best_val_acc, best_test_acc

    def test(self, data_iter):
        total_accuracy = []
        self.eval()
        with torch.no_grad():
            for batch in iter(data_iter):
                inputs, labels = batch.text, batch.label
                outputs = self.forward_classifier(inputs)
                accuracy = (outputs.argmax(1) == labels).float().mean().item()
                total_accuracy.append(accuracy)
        acc = sum(total_accuracy) / len(total_accuracy)
        self.train()
        return acc


class NetGen(AbstractVAE):
    def __init__(self, n_node, n_vocab, n_labels, embed_dim, h_dim, z_dim, p_word_dropout=0.5, unk_idx=0, pad_idx=1, start_idx=2,
                 eos_idx=3, pretrained_embeddings=None, freeze_embeddings=False, device=False):
        super(NetGen, self).__init__(n_vocab, n_labels, embed_dim, h_dim, p_word_dropout, unk_idx, pad_idx, start_idx, eos_idx,
                                     pretrained_embeddings, freeze_embeddings, device)

        # self.z_dim = n_node * z_dim
        self.node_dim = 2 * h_dim
        self.n_node = n_node
        self.sampler = list(combinations(range(n_node), 2))

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
            nn.Linear(self.node_dim, self.node_dim // 2),
            nn.ReLU(),
            nn.Linear(self.node_dim // 2, self.node_dim // 4)
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
            nn.Linear(self.node_dim * self.n_node, self.emb_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.emb_dim, self.n_labels)
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
        self.to(self.device)

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

    def forward_graph_encoder(self, inputs, adj):
        # x = F.relu(self.graph_encoder(input, adj))
        x = self.graph_encoder(inputs, adj)
        x = F.relu(self.graph_encoder_fc(torch.cat((x, inputs), -1)))
        x = x.view(-1, self.n_node * self.node_dim)
        # x = self.graph_encoder_trans(x)

        return x

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

    def forward_decoder(self, inputs, z, teacher_force=False):
        """
        Inputs must be embeddings: seq_len x mbsize
        """
        dec_inputs = self.word_dropout(inputs)
        bsize, length = dec_inputs.size()

        # 1 x mbsize x z_dim
        init_h = z.unsqueeze(0)
        c = self.decoder_c_linear(init_h).relu()
        h = self.decoder_h_linear(init_h).relu()
        d = self.decoder_d_linear(init_h).relu().transpose(1, 0)

        if teacher_force:
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

        # Generator: z -> graph
        # text, adj = self.generate_graph(z, temp=temp)
        # text, adj, entropy, cost = self.generate_graph(z, mask, return_entropy=True)
        text, adj, attentions, penal2, c_loss = self.generate_graph(z, h, enc_inputs, mask)

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
            # x = text
            x = x.view(-1, self.node_dim * self.n_node)
            outputs = self.classifier(x)

            return recon_loss, outputs, penal1, penal2, c_loss
        else:
            return recon_loss, penal1, penal2, c_loss

    def forward_classifier(self, inputs, mask):

        # Encoder: sentence -> z
        z, h = self.forward_encoder(inputs)

        # Generator: z -> graph
        # text, adj = self.generate_graph(z, temp)
        # text, adj = self.generate_graph(z, mask, return_entropy=False)
        text, adj, _, _, _ = self.generate_graph(z, h, inputs, mask)

        # classifier: graph -> prediction
        x = self.graph_encoder_cls(text, adj)
        x = F.relu(self.graph_encoder_cls_fc(torch.cat((x, text), -1)))
        # x = text
        x = x.view(-1, self.node_dim * self.n_node)
        outputs = self.classifier(x)
        return outputs

    def generate_graph(self, z, hidden_state, inputs, mask):
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

            i_t = torch.bmm(attn_dist, self.word_emb(inputs))
            g_emb.append(i_t.data)


        text = torch.cat(outputs, dim=1)
        attentions = torch.cat(atts, dim=1)
        S = torch.cat(states, dim=1)
        g_emb = torch.cat(g_emb, dim=1)
        adj, penal = self.generate_adjacency_matrix(S, g_emb)

        return text, adj, attentions, penal, closs

    def generate_adjacency_matrix(self, z, text=None):
        # z = torch.cat((z, text), -1)
        z = self.generate_adj(z)

        # adj = F.sigmoid(torch.matmul(z, z.transpose(2, 1)))

        # GLoMo
        # I = torch.eye(self.n_node).to(self.device)
        # z = self.generate_adj(z)
        # z = z / z.norm(p=2, dim=-1, keepdim=True)
        # adj = F.relu(torch.matmul(z, z.transpose(2, 1))) ** 2
        # # adj = adj - I
        # adj = adj / adj.sum(dim=-1, keepdim=True)

        z = z / torch.norm(z, p=2, dim=-1, keepdim=True).data
        adj = torch.matmul(z, z.transpose(2, 1))
        adj = F.relu(adj)
        # # adj = torch.matmul(z, z.transpose(2, 1))
        # mask = (adj > 0.5).float()
        # adj = adj * mask
        #
        # # I = (torch.ones(self.n_node, self.n_node) - torch.eye(self.n_node)).to(self.device)
        I = torch.eye(self.n_node).to(self.device)
        # # penal = l2_matrix_norm(adj @ adj.transpose(1, 2) - I)
        penal = l2_matrix_norm(adj - I)

        # adj = adj / adj.sum(dim=-1, keepdim=True)

        return adj, penal

    def train_model(self, args, dataset, logger):
        """train the whole network"""
        self.train()
        self.dataset = dataset

        alpha = 0
        beta = 0
        gamma = 0
        lam = args.cls_weight

        trainer = optim.Adam(self.parameters(), lr=args.lr)
        criterion_cls = nn.CrossEntropyLoss().to(self.device)

        patience = 0
        best_val_acc, best_test_acc, best_iter = 0, 0, 0
        list_loss_recon, list_loss_penal1, list_loss_penal2, list_loss_closs = [], [], [], []
        list_loss_cls = []
        train_start_time = time.time()
        train_iter = dataset.build_train_iter(args.batch_size, self.device)
        val_iter, test_iter = dataset.build_test_iter(args.batch_size, self.device, shuffle=False)

        for epoch in range(1, args.epochs):

            if epoch == 1:
                alpha = args.alpha
                beta = args.beta

            for i, subdata in enumerate([train_iter]):
                for batch in iter(subdata):
                    inputs, labels, mask = batch.text.t(), batch.label, batch.mask.t()
                    # mask = dataset.create_mask(inputs, device=self.device)

                    if lam != 0 and i == 0:
                        recon_loss, outputs, penal1, penal2, closs = self.forward(inputs, mask=mask, classifier=True)
                        loss_cls = criterion_cls(outputs, labels)
                        list_loss_cls.append(loss_cls.item())
                        loss_vae = recon_loss + lam * loss_cls + alpha * penal1 + gamma * penal2 + beta * closs
                    else:

                        recon_loss, penal1, penal2, closs = self.forward(inputs, mask=mask, classifier=False)
                        loss_vae = recon_loss + alpha * penal1 + gamma * penal2 + beta * closs

                    if torch.isnan(loss_vae):
                        a=1
                    list_loss_recon.append(recon_loss.item())
                    list_loss_penal1.append(penal1.item())
                    list_loss_penal2.append(penal2.item())
                    list_loss_closs.append(closs.item())

                    trainer.zero_grad()
                    loss_vae.backward()
                    # grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
                    trainer.step()

            if epoch % args.log_every == 0:
                duration = time.time() - train_start_time
                avr_loss_recon = np.mean(list_loss_recon)
                avr_loss_penal1 = np.mean(list_loss_penal1)
                avr_loss_penal2 = np.mean(list_loss_penal2)
                avr_closs = np.mean(list_loss_closs)
                avr_cls = np.mean(list_loss_cls)
                list_loss_recon, list_loss_penal1, list_loss_penal2, list_loss_closs = [], [], [], []
                list_loss_cls = []
                logger.info(f'Epoch-{epoch}; loss_recon: {avr_loss_recon:.4f}; loss_cls: {avr_cls:.4f}; penal1: {avr_loss_penal1:.4f}; '
                      f' penal2: {avr_loss_penal2:.4f}; avr_closs: {avr_closs:.4f}; duration: {round(duration)}')

                if lam == 0:
                    train_acc, val_acc, test_acc = self.train_model_classifier(args, dataset)
                else:
                    train_acc = self.test(dataset, train_iter)
                    val_acc = self.test(dataset, val_iter)
                    test_acc = self.test(dataset, test_iter)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_iter = epoch
                    patience = 0
                    graph = self.output_graph(dataset, test_iter, os.path.join(args.log_dir, f'epoch_{epoch}'), save_name='test')
                    graph = self.output_graph(dataset, val_iter, os.path.join(args.log_dir, f'epoch_{epoch}'), save_name='val')
                    graph = self.output_graph(dataset, train_iter, os.path.join(args.log_dir, f'epoch_{epoch}'), save_name='train')
                else:
                    if epoch > args.minimal_epoch:
                        patience += args.log_every

                logger.info(f'Train Acc: {train_acc:.4f}; Valid Acc: {val_acc:.4f}; Test Acc: {test_acc:.4f}; Best Acc: {best_test_acc:.4f} @ {best_iter}')

                if args.early_stop and patience > args.patience:
                    break

            if epoch % args.save_every == 0:
                graph = self.output_graph(dataset, test_iter, os.path.join(args.log_dir, f'epoch_{epoch}'),
                                          save_name='test')
                graph = self.output_graph(dataset, val_iter, os.path.join(args.log_dir, f'epoch_{epoch}'),
                                          save_name='val')
                graph = self.output_graph(dataset, train_iter, os.path.join(args.log_dir, f'epoch_{epoch}'),
                                          save_name='train')

        logger.info(f'Best Valid Acc: {best_val_acc:.4f}; Best Test Acc: {best_test_acc:.4f} @ epoch {best_iter}')
        return best_val_acc, best_test_acc

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

    def test(self, dataset, data_iter):
        self.eval()
        total_accuracy = []
        with torch.no_grad():
            for batch in iter(data_iter):
                inputs, labels, mask = batch.text.t(), batch.label, batch.mask.t()
                outputs = self.forward_classifier(inputs, mask)
                accuracy = (outputs.argmax(1) == labels).float().mean().item()
                total_accuracy.append(accuracy)
        acc = sum(total_accuracy) / len(total_accuracy)
        self.train()
        return acc

    def output_graph(self, dataset, data_iter, save_path=None, save_name=''):
        self.eval()
        input_list, adj_list, word_list = [], [], []
        with torch.no_grad():
            for batch in iter(data_iter):
                inputs, labels, mask = batch.text.t(), batch.label, batch.mask.t()

                z, h = self.forward_encoder(inputs)
                text, adj, attentions, _, _ = self.generate_graph(z, h, inputs, mask)
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


class NetGen1(AbstractVAE):
    def __init__(self, n_node, n_vocab, n_labels, h_dim, z_dim, p_word_dropout=0.3, unk_idx=0, pad_idx=1, start_idx=2,
                 eos_idx=3,
                 pretrained_embeddings=None, freeze_embeddings=False, device=False):
        super(NetGen, self).__init__(n_vocab, n_labels, h_dim, p_word_dropout, unk_idx, pad_idx, start_idx, eos_idx,
                                     pretrained_embeddings, freeze_embeddings, device)
        """
        no graph
        """
        self.z_dim = n_node * z_dim
        self.node_dim = 2 * h_dim
        self.n_node = n_node
        self.sampler = list(combinations(range(n_node), 2))

        """
        Encoder is GRU with FC layers connected to last hidden unit
        """
        self.encoder = nn.LSTM(self.emb_dim, h_dim, bidirectional=True, batch_first=True)
        self.encoder_ = nn.ModuleList([
            self.encoder
        ])

        """
        Decoder is GRU with `z` appended at its inputs
        """
        self.decoder_h_linear = nn.Linear(self.n_node * self.node_dim, self.node_dim)
        self.decoder_c_linear = nn.Linear(self.n_node * self.node_dim, self.node_dim)
        self.decoder = nn.LSTM(self.emb_dim, self.node_dim, dropout=0.3)
        # self.decoder_fc = nn.Linear(n_node * self.emb_dim, n_vocab)
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.node_dim, self.emb_dim),
            nn.LeakyReLU(0.2),
            # nn.ELU(),
            nn.Linear(self.emb_dim, self.emb_dim)
        )

        """
        Generator 
        """
        self.generate_lstm = nn.LSTM(self.node_dim, self.node_dim, batch_first=True)
        self.k_linear = nn.Linear(self.node_dim, self.node_dim)
        self.q_linear = nn.Linear(self.node_dim, self.node_dim)
        self.v_linear = nn.Linear(self.node_dim, self.node_dim)
        self.generate_adj = nn.Sequential(
            nn.Linear(z_dim, z_dim, bias=False),
            nn.LeakyReLU(0.2),
            # nn.ELU(),
            nn.Linear(z_dim, z_dim, bias=False)
        )

        """
        Graph Encoder is GCN with pooling layer
        """
        self.graph_encoder = GraphConvolution(self.node_dim, self.node_dim)
        self.graph_encoder_trans = nn.Sequential(
            nn.Linear(self.n_node * self.node_dim, self.n_node * self.node_dim),
            nn.LeakyReLU(0.2),
            # nn.ELU(),
            nn.Linear(self.n_node * self.node_dim, self.n_node * self.node_dim),
        )
        self.decoder_ = nn.ModuleList([
            self.decoder, self.decoder_fc, self.decoder_c_linear, self.decoder_h_linear,
            self.k_linear, self.q_linear, self.v_linear,
            self.generate_lstm, self.generate_adj,
            self.graph_encoder, self.graph_encoder_trans
        ])

        """
        Discriminator is CNN as in Kim, 2014
        """
        # self.conv3 = nn.Conv2d(1, 100, (3, self.emb_dim))
        # self.conv4 = nn.Conv2d(1, 100, (4, self.emb_dim))
        # self.conv5 = nn.Conv2d(1, 100, (5, self.emb_dim))
        #
        # self.disc_fc = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(300, 1)
        # )
        #
        # self.discriminator_ = nn.ModuleList([
        #     self.conv3, self.conv4, self.conv5, self.disc_fc
        # ])

        self.discriminator = nn.LSTM(self.emb_dim, h_dim, bidirectional=True)
        self.disc_fc = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            # nn.ELU(),
            nn.Linear(h_dim, 1)
        )
        self.discriminator_ = nn.ModuleList([
            self.discriminator, self.disc_fc
        ])

        """
        Classifier is DNN
        """
        self.graph_encoder_cls = GraphConvolution(self.node_dim, self.node_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.node_dim * self.n_node, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.n_labels)
        )
        self.classifier_ = nn.ModuleList([
            self.graph_encoder_cls, self.classifier
        ])

        """
        Grouping the model's parameters: separating encoder, decoder, and discriminator
        """
        self.encoder_params = filter(lambda p: p.requires_grad, self.encoder_.parameters())

        self.decoder_params = filter(lambda p: p.requires_grad, self.decoder_.parameters())

        self.discriminator_params = filter(lambda p: p.requires_grad, self.discriminator_.parameters())

        self.classifier_params = filter(lambda p: p.requires_grad, self.classifier_.parameters())

        self.vae_params = chain(
            self.word_emb.parameters(), self.encoder_.parameters(), self.decoder_.parameters()
        )
        self.vae_params = filter(lambda p: p.requires_grad, self.vae_params)

        """
        Use GPU if set
        """
        self.to(self.device)

    def forward_graph_encoder(self, input, adj):
        x = F.relu(self.graph_encoder(input, adj))
        # x = self.graph_encoder(input, adj)
        x = x.view(-1, self.n_node * self.node_dim)
        x = self.graph_encoder_trans(x)

        return x

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
        outputs, h = self.encoder(inputs.transpose(0, 1))
        return outputs, h

    def forward_decoder(self, inputs, z):
        """
        Inputs must be embeddings: seq_len x mbsize
        """
        dec_inputs = self.word_dropout(inputs)

        # Forward
        seq_len = dec_inputs.size(0)

        # 1 x mbsize x z_dim
        init_h = z.unsqueeze(0)
        c = self.decoder_c_linear(init_h)
        h = self.decoder_h_linear(init_h)
        inputs_emb = self.word_emb(dec_inputs)  # seq_len x mbsize x emb_dim
        # inputs_emb = torch.cat([inputs_emb, init_h.repeat(seq_len, 1, 1)], 2)
        outputs, _ = self.decoder(inputs_emb, (h, c))
        seq_len, mbsize, _ = outputs.size()

        outputs = outputs.view(seq_len * mbsize, -1)
        y = self.decoder_fc(outputs)
        y = y @ self.word_emb.weight.t()
        y = y.view(seq_len, mbsize, self.n_vocab)

        return y

    def forward_discriminator(self, inputs):
        """
        Inputs is batch of sentences: mbsize x seq_len
        """
        inputs = self.word_emb(inputs)
        return self.forward_discriminator_embed(inputs)

    def forward_discriminator_embed(self, inputs):
        """
        Inputs must be embeddings: mbsize x seq_len x emb_dim
        """
        # inputs = inputs.unsqueeze(1)  # mbsize x 1 x seq_len x emb_dim
        #
        # x3 = F.relu(self.conv3(inputs)).squeeze(3)
        # x4 = F.relu(self.conv4(inputs)).squeeze(3)
        # x5 = F.relu(self.conv5(inputs)).squeeze(3)
        #
        # # Max-over-time-pool
        # x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)
        # x4 = F.max_pool1d(x4, x4.size(2)).squeeze(2)
        # x5 = F.max_pool1d(x5, x5.size(2)).squeeze(2)
        #
        # x = torch.cat([x3, x4, x5], dim=1)
        #
        # y = self.disc_fc(x)

        _, h = self.discriminator(inputs.transpose(1, 0), None)

        # Forward to latent
        h = h.view(-1, self.h_dim * 2)
        y = self.disc_fc(h)

        return y

    def forward(self, sentence, mask, use_c_prior=True, temp=1):
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

        mbsize = sentence.size(1)

        # sentence: '<start> I want to fly <eos>'
        # enc_inputs: '<start> I want to fly <eos>'
        # dec_inputs: '<start> I want to fly <eos>'
        # dec_targets: 'I want to fly <eos> <pad>'
        pad_words = Variable(torch.LongTensor([self.PAD_IDX])).repeat(1, mbsize)
        pad_words = pad_words.to(self.device)

        enc_inputs = sentence
        dec_inputs = sentence
        dec_targets = torch.cat([sentence[1:], pad_words], dim=0)

        # Encoder: sentence -> z
        z, h = self.forward_encoder(enc_inputs)

        # Generator: z -> graph
        # text, adj = self.generate_graph(z, temp=temp)
        # text, adj, entropy, cost = self.generate_graph(z, mask, return_entropy=True)
        text, adj, attentions = self.generate_graph(z, h, mask)

        # Graph Encoder: graph -> z'
        # z1 = self.forward_graph_encoder(text, adj)
        z1 = text.view(-1, self.n_node * self.node_dim)

        # Decoder: sentence -> y
        y = self.forward_decoder(dec_inputs, z1)

        recon_loss = F.cross_entropy(
            y.view(-1, self.n_vocab), dec_targets.view(-1), size_average=True
        )

        # I = torch.eye(attention.size()[1]).to(self.device)
        # penal = l2_matrix_norm(attention @ attention.transpose(1, 2) - I)
        penal = torch.distributions.Categorical(probs=attentions).entropy().mean()

        c = attentions[:, 0, :]
        current_attention = attentions[:, 1, :]
        c_loss = torch.min(c, current_attention)
        for i in range(2, self.n_node):
            c = c + current_attention
            current_attention = attentions[:, i, :]
            c_loss = c_loss + torch.min(c, current_attention)
        c_loss = c_loss.sum(dim=1).mean()

        return recon_loss, penal, c_loss

    def forward_pretrain(self, sentence, mask, use_c_prior=True, temp=1):
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

        mbsize = sentence.size(1)

        # sentence: '<start> I want to fly <eos>'
        # enc_inputs: '<start> I want to fly <eos>'
        # dec_inputs: '<start> I want to fly <eos>'
        # dec_targets: 'I want to fly <eos> <pad>'
        pad_words = Variable(torch.LongTensor([self.PAD_IDX])).repeat(1, mbsize)
        pad_words = pad_words.to(self.device)

        enc_inputs = sentence
        dec_inputs = sentence
        dec_targets = torch.cat([sentence[1:], pad_words], dim=0)

        # Encoder: sentence -> z
        z, h = self.forward_encoder(enc_inputs)

        # Decoder: sentence -> y
        dec_inputs = self.word_dropout(dec_inputs)

        # 1 x mbsize x z_dim
        inputs_emb = self.word_emb(dec_inputs)  # seq_len x mbsize x emb_dim
        # inputs_emb = torch.cat([inputs_emb, init_h.repeat(seq_len, 1, 1)], 2)
        outputs, _ = self.decoder(inputs_emb, h)
        seq_len, mbsize, _ = outputs.size()

        outputs = outputs.view(seq_len * mbsize, -1)
        y = self.decoder_fc(outputs)
        y = y @ self.word_emb.weight.t()
        y = y.view(seq_len, mbsize, self.n_vocab)

        recon_loss = F.cross_entropy(
            y.view(-1, self.n_vocab), dec_targets.view(-1), size_average=True
        )

        return recon_loss

    def forward_classifier(self, inputs, mask, temp=1, take_mean=True):

        # Encoder: sentence -> z
        z, h = self.forward_encoder(inputs)

        # Generator: z -> graph
        # text, adj = self.generate_graph(z, temp)
        # text, adj = self.generate_graph(z, mask, return_entropy=False)
        text, adj, _ = self.generate_graph(z, h, mask)

        # classifier: graph -> prediction
        # x = F.relu(self.graph_encoder_cls(text, adj))
        x = text
        x = x.view(-1, self.node_dim * self.n_node)
        outputs = self.classifier(x)
        return outputs

    def generate_sentences(self, batch_size, length, temp=1.0, z=None, mask=None):
        """
        Generate sentences and corresponding z of (batch_size x max_sent_len)
        """
        self.eval()

        if z is None:
            z = self.sample_z_prior(batch_size)

        X_gen = self.sample_sentence(z, mask, length, raw=True, temp=temp)

        # Back to default state: train
        self.train()

        return X_gen

    def sample_sentence(self, z, mask, length, raw=False, temp=1):
        """
        Sample single sentence from p(x|z,c) according to given temperature.
        `raw = True` means this returns sentence as in dataset which is useful
        to train discriminator. `False` means that this will return list of
        `word_idx` which is useful for evaluation.
        """
        mbsize = z.size(0)

        start_word = torch.LongTensor([self.START_IDX]).repeat(mbsize).view(1, -1)
        start_word = start_word.to(self.device)
        word = Variable(start_word)  # '<start>'

        if not isinstance(z, Variable):
            z = Variable(z)

        z = z.view(-1, self.n_node, self.node_dim)

        # Generator: z -> graph
        # text, adj = self.generate_graph(z, temp=temp)
        text, adj = self.generate_graph(z, mask, return_entropy=False)

        # Graph Encoder: graph -> z'
        z1 = self.forward_graph_encoder(text, adj)
        z1 = z1.unsqueeze(0)
        h = z1

        outputs = []

        if raw:
            outputs.append(start_word)

        for i in range(length - 1):
            emb = self.word_emb(word)
            emb = torch.cat([emb, z1], 2)

            output, h = self.decoder(emb, h)
            y = self.decoder_fc(output)
            y = y @ self.word_emb.weight.t()
            y = F.softmax(y / temp, dim=2)

            idx = torch.multinomial(y.squeeze(0), num_samples=1)

            word = idx.transpose(1, 0)

            # idx = int(idx)
            #
            # if not raw and idx == self.EOS_IDX:
            #     break

            outputs.append(word)

        outputs = torch.cat(outputs, dim=0).transpose(1, 0)

        for i in range(mbsize):
            for j, idx in enumerate(outputs[i]):
                if idx == self.EOS_IDX:
                    outputs[i][j:] = self.PAD_IDX

        if raw:
            return outputs
        else:
            return outputs.cpu().int()

    def generate_soft_embed(self, batch_size, length, temp=1, z=None, mask=None):
        """
        Generate soft embeddings of (mbsize x emb_dim) along with target z
        and c for each row (mbsize x {z_dim, c_dim})
        """
        if z is None:
            z = self.sample_z_prior(batch_size)

        X_gen = self.sample_soft_embed(z, mask, length, temp=temp)

        return X_gen

    def sample_soft_embed(self, z, mask, length, temp=1):
        """
        Sample single soft embedded sentence from p(x|z,c) and temperature.
        Soft embeddings are calculated as weighted average of word_emb
        according to p(x|z,c).
        """
        mbsize = z.size(0)

        start_word = torch.LongTensor([self.START_IDX]).repeat(mbsize).view(1, -1)
        start_word = start_word.to(self.device)
        word = Variable(start_word)  # '<start>'

        if not isinstance(z, Variable):
            z = Variable(z)

        z = z.view(-1, self.n_node, self.node_dim)

        # Generator: z -> graph
        # text, adj = self.generate_graph(z, temp=temp)
        # text, adj = self.generate_graph(z, mask, return_entropy=False)
        text = z
        adj = self.generate_adjacency_matrix(z)

        # Graph Encoder: graph -> z'
        z1 = self.forward_graph_encoder(text, adj)
        z1 = z1.unsqueeze(0)
        h = z1

        emb = self.word_emb(word)
        emb = torch.cat([emb, z1], 2)

        outputs = [self.word_emb(word)]

        for i in range(length - 1):
            output, h = self.decoder(emb, h)
            o = self.decoder_fc(output)

            # o = o @ self.word_emb.weight.t()
            #
            # # Sample softmax with temperature
            # y = F.softmax(o / temp, dim=2)
            #
            # # Take expectation of embedding given output prob -> soft embedding
            # # <y, w> = 1 x n_vocab * n_vocab x emb_dim
            # emb = y @ self.word_emb.weight

            emb = o

            # Save resulting soft embedding
            outputs.append(emb)

            # Append with z for the next input
            emb = torch.cat([emb, z1], 2)

        # mbsize x length x emb_dim
        outputs = torch.cat(outputs, dim=0).transpose(1, 0)

        return outputs

    def output_graph(self, z):
        adj = self.generate_adjacency_matrix(z).cpu().float()
        word_idx = self.generate_word(z).cpu().int()
        return adj, word_idx

    def output_graph_from_text(self, text):
        flag = 0
        if self.training:
            self.eval()
            flag = 1

        z, = self.forward_encoder(text)

        if flag:
            self.train()

        return self.output_graph(z)

    def generate_graph(self, z, hidden_state, mask):
        bsize, d_k = z.size()[0], z.size()[2]
        s = torch.zeros(bsize, 1, d_k).to(self.device)
        h, c = hidden_state
        h = h.view(1, bsize, -1)
        c = c.view(1, bsize, -1)
        outputs, atts = [], []
        for i in range(self.n_node):
            s, (h, c) = self.generate_lstm(s, (h, c))
            # q = s
            q = self.q_linear(s)
            k = self.k_linear(z)
            v = self.v_linear(z)
            s, att = self_attention(q, k, v, d_k, mask)
            outputs.append(s)
            atts.append(att)
        text = torch.cat(outputs, dim=1)
        attentions = torch.cat(atts, dim=1)
        adj = self.generate_adjacency_matrix(text)

        return text, adj, attentions

    def generate_adjacency_matrix(self, z):
        # z = self.generate_adj(z)
        # adj = F.sigmoid(torch.matmul(z, z.transpose(2, 1)))
        # return adj

        # GLoMo
        I = torch.eye(z.size()[1]).to(self.device)
        # z = self.generate_adj(z)
        z = z / z.norm(p=2, dim=2, keepdim=True)
        adj = F.relu(torch.matmul(z, z.transpose(2, 1))) ** 2
        adj = (adj - I) / adj.sum(dim=1).unsqueeze(1)
        return adj

    def generate_word(self, z):
        outputs = self.generate_text(z).view(-1)
        word_idx = torch.argmax(F.softmax(outputs, dim=0), dim=0)
        return word_idx

    def generate_soft_node_embed(self, z, mask=None, temp=1.0, return_entropy=True):
        """
        Sample single soft embedded sentence from p(x|z) and temperature.
        Soft embeddings are calculated as weighted average of word_emb
        according to p(x|z).
        """

        outputs = self.generate_text(z)
        outputs = outputs @ self.word_emb.weight.t()
        if mask is not None:
            outputs = outputs.masked_fill(mask.unsqueeze(1) == 0, -float('inf'))

        # Sample softmax with temperature
        y = F.softmax(outputs / temp, dim=2)

        # Take expectation of embedding given output prob -> soft embedding
        emb = torch.matmul(y, self.word_emb.weight)

        if return_entropy:
            # Calculate entropy for penalty
            entropy = torch.distributions.Categorical(probs=y).entropy().mean()
            # cost = self.estimate_cost(y)
            cost = torch.FloatTensor([0.0])
            return emb, entropy, cost
        else:
            return emb

    def estimate_cost(self, probs, sample_size=10):
        sampled = random.sample(self.sampler, sample_size)
        cost = torch.FloatTensor([0.0]).to(self.device)
        for x, y in sampled:
            n1 = probs[:, x, :]
            n2 = probs[:, y, :]
            kl = F.kl_div(n1.log(), n2, reduce=False).sum(dim=1)
            kl = F.hardtanh(kl, max_val=10, min_val=-10).sum() / kl.size(0)
            cost -= kl
            # if not torch.isinf(kl):
            #     cost -= kl
        return cost / sample_size

    def train_model(self, args, dataset):
        """train the whole network"""
        self.train()

        kl_weight = args.kl_weight
        alpha = 0
        beta = 0
        gamma = args.gamma

        trainer_D = optim.Adam(self.discriminator_params, lr=args.lr)
        trainer_G = optim.Adam(self.decoder_params, lr=args.lr)
        trainer_E = optim.Adam(self.encoder_params, lr=args.lr)

        criterion_bce = nn.BCEWithLogitsLoss().to(self.device)
        ones_label = Variable(torch.ones((args.batch_size, 1))).to(self.device)
        zeros_label = Variable(torch.zeros((args.batch_size, 1))).to(self.device)

        patience = 0
        best_val_acc, best_test_acc, best_iter = 0, 0, 0
        list_loss_D, list_loss_G, list_loss_E, list_loss_recon, list_loss_penal, list_loss_closs, list_entropy, \
        list_cost = [], [], [], [], [], [], [], []
        train_start_time = time.time()
        train_iter = dataset.build_train_iter(args.batch_size, self.device)
        val_iter, test_iter = dataset.build_test_iter(args.batch_size, self.device, shuffle=False)

        # val_acc, test_acc = self.train_model_classifier(args, dataset)
        # print(f'Initial Eval: Valid Acc: {val_acc:.4f}; Test Acc: {test_acc:.4f}')

        for epoch in range(1, args.epochs):

            if epoch == 200:
                alpha = args.alpha
                beta = args.beta

            current_temp = temp(epoch)
            for subdata in [train_iter, val_iter, test_iter]:
                for batch in iter(subdata):
                    inputs, labels = batch.text, batch.label
                    padding_mask = dataset.create_padding_mask(inputs.transpose(1, 0), device=self.device)
                    length, batch_size = inputs.size()
                    recon_loss, penal, closs = self.forward(inputs, mask=padding_mask, temp=current_temp)
                    loss_vae = recon_loss + alpha * penal + beta * closs
                    list_loss_recon.append(recon_loss.item())
                    list_loss_penal.append(penal.item())
                    list_loss_closs.append(closs.item())

                    """ Update discriminator """
                    # y_disc_real = self.forward_discriminator(inputs.transpose(1, 0))
                    # loss_D_real = criterion_bce(y_disc_real, ones_label[:batch_size])
                    #
                    # # x_recon = self.generate_sentences(batch_size, length, temp=current_temp, z=z0, mask=mask)
                    # # y_disc_fake_recon = self.forward_discriminator(x_recon)
                    # x_recon_embed = self.generate_soft_embed(batch_size, length, temp=current_temp, z=z0, mask=mask)
                    # y_disc_fake_recon = self.forward_discriminator_embed(x_recon_embed)
                    # loss_D_recon = criterion_bce(y_disc_fake_recon, zeros_label[:batch_size])
                    #
                    # # x_gen = self.generate_sentences(batch_size, length, temp=current_temp)
                    # # y_disc_fake_sample = self.forward_discriminator(x_gen)
                    # x_gen_embed = self.generate_soft_embed(batch_size, length, temp=current_temp)
                    # y_disc_fake_sample = self.forward_discriminator_embed(x_gen_embed)
                    # loss_D_sample = criterion_bce(y_disc_fake_sample, zeros_label[:batch_size])
                    #
                    # loss_D = loss_D_real + loss_D_recon + loss_D_sample
                    # list_loss_D.append(loss_D.item())
                    # trainer_D.zero_grad()
                    # loss_D.backward(retain_graph=True)
                    # # grad_norm = torch.nn.utils.clip_grad_norm_(self.discriminator_params, 5)
                    # trainer_D.step()

                    """ Update generator  """
                    # y_disc_real = self.forward_discriminator(inputs.transpose(1, 0))
                    # loss_D_real = criterion_bce(y_disc_real, zeros_label[:batch_size])
                    #
                    # x_recon_embed = self.generate_soft_embed(batch_size, length, temp=current_temp, z=z0, mask=mask)
                    # y_disc_fake_recon = self.forward_discriminator_embed(x_recon_embed)
                    # loss_D_recon = criterion_bce(y_disc_fake_recon, ones_label[:batch_size])
                    #
                    # x_gen_embed = self.generate_soft_embed(batch_size, length, temp=current_temp)
                    # y_disc_fake_sample = self.forward_discriminator_embed(x_gen_embed)
                    # loss_D_sample = criterion_bce(y_disc_fake_sample, ones_label[:batch_size])
                    # loss_D = (loss_D_real + loss_D_recon + loss_D_sample)

                    # loss_G = alpha * loss_D + loss_vae
                    loss_G = loss_vae
                    # list_loss_G.append(loss_D.item())
                    trainer_G.zero_grad()
                    loss_G.backward(retain_graph=True)
                    # grad_norm = torch.nn.utils.clip_grad_norm_(self.decoder_params, 5)
                    trainer_G.step()

                    """ Update encoder  """
                    # loss_E = beta * loss_D + loss_vae
                    loss_E = loss_vae
                    # list_loss_E.append(-loss_D.item())
                    trainer_E.zero_grad()
                    loss_E.backward(retain_graph=True)
                    # grad_norm = torch.nn.utils.clip_grad_norm_(self.encoder_params, 5)
                    trainer_E.step()

            if epoch % args.log_every == 0:
                duration = time.time() - train_start_time
                avr_loss_D = np.mean(list_loss_D)
                avr_loss_G = np.mean(list_loss_G)
                # avr_loss_E = np.mean(list_loss_E)
                avr_loss_recon = np.mean(list_loss_recon)
                avr_loss_penal = np.mean(list_loss_penal)
                avr_closs = np.mean(list_loss_closs)
                # avr_cost = np.mean(list_cost)
                list_loss_D, list_loss_G, list_loss_E, list_loss_recon, list_loss_penal, list_loss_closs, list_entropy, \
                list_cost = [], [], [], [], [], [], [], []
                print(f'Epoch-{epoch}; temp: {current_temp:.4f}; loss_D: {avr_loss_D:.4f}; loss_G: {avr_loss_G:.4f}; '
                      f'loss_recon: {avr_loss_recon:.4f}; penal: {avr_loss_penal:.4f}; avr_closs: {avr_closs:.4f}; '
                      f'duration: {duration:.2f}')

                val_acc, test_acc = self.train_model_classifier(args, dataset, current_temp)

                # z = self.sample_z_prior(1)
                # sample_idxs = [i for i in self.sample_sentence(z, None, length=50)[0] if i != self.PAD_IDX]
                # sample_sent = dataset.idxs2sentence(sample_idxs)
                # print(f'Valid Acc: {val_acc:.4f}; Test Acc: {test_acc:.4f}; Sample: "{sample_sent}"')
                print(f'Valid Acc: {val_acc:.4f}; Test Acc: {test_acc:.4f}')
                # print(f'Sample: "{sample_sent}"')

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_iter = epoch
                    patience = 0
                else:
                    if epoch > args.minimal_epoch:
                        patience += args.log_every

                if args.early_stop and patience > args.patience:
                    break

        print(f'Best Valid Acc: {best_val_acc:.4f}; Best Test Acc: {best_test_acc:.4f} @ epoch {best_iter}')
        return best_val_acc, best_test_acc

    def train_model_classifier(self, args, dataset, temp=1.0):
        """train the whole network"""
        self.train()

        self.classifier_[0].reset_parameters()
        for i in self.classifier_[1]:
            if hasattr(i, 'reset_parameters'):
                i.reset_parameters()
        trainer_C = optim.Adam(filter(lambda p: p.requires_grad, self.classifier_.parameters()), lr=args.lr)
        criterion_cls = nn.CrossEntropyLoss().to(self.device)

        patience = 0
        log_every = 5
        list_loss_cls = []
        best_val_acc, best_test_acc, best_iter = 0, 0, 0
        train_start_time = time.time()

        train_iter = dataset.build_train_iter(args.eval_batch_size, self.device)

        for epoch in range(1, args.eval_epochs):
            for batch in iter(train_iter):
                inputs, labels = batch.text, batch.label
                mask = dataset.create_padding_mask(inputs.transpose(1, 0), device=self.device)

                """ Update classifier """
                outputs = self.forward_classifier(inputs, mask, temp)
                loss_cls = criterion_cls(outputs, labels)
                list_loss_cls.append(loss_cls.item())

                trainer_C.zero_grad()
                loss_cls.backward(retain_graph=True)
                trainer_C.step()

            if epoch % log_every == 0:
                duration = time.time() - train_start_time
                avr_loss_cls = np.mean(list_loss_cls)
                list_loss_cls = []

                val_iter, test_iter = dataset.build_test_iter(args.batch_size, self.device)
                val_acc = self.test(dataset, val_iter)
                test_acc = self.test(dataset, test_iter)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
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
        return best_val_acc, best_test_acc

    def test(self, dataset, data_iter):
        self.eval()
        total_accuracy = []
        with torch.no_grad():
            for batch in iter(data_iter):
                inputs, labels = batch.text, batch.label
                mask = dataset.create_padding_mask(inputs.transpose(1, 0), device=self.device)
                outputs = self.forward_classifier(inputs, mask)
                accuracy = (outputs.argmax(1) == labels).float().mean().item()
                total_accuracy.append(accuracy)
        acc = sum(total_accuracy) / len(total_accuracy)
        self.train()
        return acc


# class NetGen(AbstractVAE):
#     def __init__(self, n_node, n_vocab, n_labels, h_dim, z_dim, p_word_dropout=0.3, unk_idx=0, pad_idx=1, start_idx=2, eos_idx=3,
#                  pretrained_embeddings=None, freeze_embeddings=False, device=False):
#         super(NetGen, self).__init__(n_vocab, n_labels, h_dim, p_word_dropout, unk_idx, pad_idx, start_idx, eos_idx,
#                                           pretrained_embeddings, freeze_embeddings, device)
#
#         """
#             self attention, generate node at the same time
#         """
#
#         self.z_dim = n_node * z_dim
#         self.node_dim = 2 * h_dim
#         self.n_node = n_node
#         self.sampler = list(combinations(range(n_node), 2))
#
#         """
#         Encoder is GRU with FC layers connected to last hidden unit
#         """
#         self.encoder = nn.LSTM(self.emb_dim, h_dim, bidirectional=True, batch_first=True)
#         # self.compress_layer = nn.Linear(2 * h_dim, z_dim)
#         self.self_attention = nn.Sequential(
#             nn.Linear(2 * h_dim, z_dim, bias=False),
#             nn.Tanh(),
#             nn.Linear(z_dim, n_node, bias=False)
#         )
#         self.encoder_ = nn.ModuleList([
#             self.encoder, self.self_attention
#         ])
#
#         """
#         Decoder is GRU with `z` appended at its inputs
#         """
#         self.decoder = nn.GRU(self.emb_dim + n_node * self.node_dim, n_node * self.node_dim, dropout=0.3)
#         # self.decoder_fc = nn.Linear(n_node * self.emb_dim, n_vocab)
#         self.decoder_fc = nn.Sequential(
#             nn.Linear(n_node * self.node_dim, self.emb_dim),
#             nn.LeakyReLU(0.2),
#             # nn.ELU(),
#             nn.Linear(self.emb_dim, self.emb_dim)
#         )
#
#         """
#         Generator
#         """
#         self.generate_text = nn.Sequential(
#             nn.Linear(z_dim, self.emb_dim, bias=False),
#             nn.LeakyReLU(0.2),
#             # nn.ELU(),
#             nn.Linear(self.emb_dim, self.emb_dim, bias=False)
#         )
#         self.generate_adj = nn.Sequential(
#             nn.Linear(z_dim, z_dim, bias=False),
#             nn.LeakyReLU(0.2),
#             # nn.ELU(),
#             nn.Linear(z_dim, z_dim, bias=False)
#         )
#
#         """
#         Graph Encoder is GCN with pooling layer
#         """
#         self.graph_encoder = GraphConvolution(self.node_dim, self.node_dim)
#         self.graph_encoder_trans = nn.Sequential(
#             nn.Linear(self.n_node * self.node_dim, self.n_node * self.node_dim),
#             nn.LeakyReLU(0.2),
#             # nn.ELU(),
#             nn.Linear(self.n_node * self.node_dim, self.n_node * self.node_dim),
#         )
#         self.decoder_ = nn.ModuleList([
#             self.decoder, self.decoder_fc,
#             self.generate_text, self.generate_text,
#             self.graph_encoder, self.graph_encoder_trans
#         ])
#
#         """
#         Discriminator is CNN as in Kim, 2014
#         """
#         # self.conv3 = nn.Conv2d(1, 100, (3, self.emb_dim))
#         # self.conv4 = nn.Conv2d(1, 100, (4, self.emb_dim))
#         # self.conv5 = nn.Conv2d(1, 100, (5, self.emb_dim))
#         #
#         # self.disc_fc = nn.Sequential(
#         #     nn.Dropout(0.5),
#         #     nn.Linear(300, 1)
#         # )
#         #
#         # self.discriminator_ = nn.ModuleList([
#         #     self.conv3, self.conv4, self.conv5, self.disc_fc
#         # ])
#
#         self.discriminator = nn.LSTM(self.emb_dim, h_dim, bidirectional=True)
#         self.disc_fc = nn.Sequential(
#             nn.Linear(2 * h_dim, h_dim),
#             nn.BatchNorm1d(h_dim),
#             nn.ReLU(),
#             # nn.ELU(),
#             nn.Linear(h_dim, 1)
#         )
#         self.discriminator_ = nn.ModuleList([
#             self.discriminator, self.disc_fc
#         ])
#
#         """
#         Classifier is DNN
#         """
#         self.graph_encoder_cls = GraphConvolution(self.node_dim, self.node_dim)
#         self.classifier = nn.Sequential(
#             nn.Linear(self.node_dim * self.n_node, self.emb_dim),
#             nn.ReLU(),
#             nn.Linear(self.emb_dim, self.n_labels)
#         )
#         self.classifier_ = nn.ModuleList([
#             self.graph_encoder_cls, self.classifier
#         ])
#
#         """
#         Grouping the model's parameters: separating encoder, decoder, and discriminator
#         """
#         self.encoder_params = filter(lambda p: p.requires_grad, self.encoder_.parameters())
#
#         self.decoder_params = filter(lambda p: p.requires_grad, self.decoder_.parameters())
#
#         self.discriminator_params = filter(lambda p: p.requires_grad, self.discriminator_.parameters())
#
#         self.classifier_params = filter(lambda p: p.requires_grad, self.classifier_.parameters())
#
#         self.vae_params = chain(
#             self.word_emb.parameters(), self.encoder_.parameters(), self.decoder_.parameters()
#         )
#         self.vae_params = filter(lambda p: p.requires_grad, self.vae_params)
#
#         """
#         Use GPU if set
#         """
#         self.to(self.device)
#
#     def forward_graph_encoder(self, input, adj):
#         x = F.relu(self.graph_encoder(input, adj))
#         # x = self.graph_encoder(input, adj)
#         x = x.view(-1, self.n_node * self.node_dim)
#         x = self.graph_encoder_trans(x)
#
#         return x
#
#     def forward_encoder(self, inputs, mask):
#         """
#         Inputs is batch of sentences: seq_len x mbsize
#         """
#         inputs = self.word_emb(inputs)
#         return self.forward_encoder_embed(inputs, mask)
#
#     def forward_encoder_embed(self, inputs, mask):
#         """
#         Inputs is embeddings of: seq_len x mbsize x emb_dim
#         """
#         outputs, h = self.encoder(inputs.transpose(0, 1))
#
#         # Forward to latent
#         attention = self.self_attention(outputs).softmax(dim=1).transpose(1, 2)
#         attention = attention * mask.unsqueeze(1)
#         normalization_factor = attention.sum(2, keepdim=True)
#         attention = attention / normalization_factor
#
#         outputs = attention @ outputs
#         return outputs, attention
#
#     def forward_decoder(self, inputs, z):
#         """
#         Inputs must be embeddings: seq_len x mbsize
#         """
#         dec_inputs = self.word_dropout(inputs)
#
#         # Forward
#         seq_len = dec_inputs.size(0)
#
#         # 1 x mbsize x z_dim
#         init_h = z.unsqueeze(0)
#         inputs_emb = self.word_emb(dec_inputs)  # seq_len x mbsize x emb_dim
#         inputs_emb = torch.cat([inputs_emb, init_h.repeat(seq_len, 1, 1)], 2)
#
#         outputs, _ = self.decoder(inputs_emb, init_h)
#         seq_len, mbsize, _ = outputs.size()
#
#         outputs = outputs.view(seq_len * mbsize, -1)
#         y = self.decoder_fc(outputs)
#         y = y @ self.word_emb.weight.t()
#         y = y.view(seq_len, mbsize, self.n_vocab)
#
#         return y
#
#     def forward_discriminator(self, inputs):
#         """
#         Inputs is batch of sentences: mbsize x seq_len
#         """
#         inputs = self.word_emb(inputs)
#         return self.forward_discriminator_embed(inputs)
#
#     def forward_discriminator_embed(self, inputs):
#         """
#         Inputs must be embeddings: mbsize x seq_len x emb_dim
#         """
#         # inputs = inputs.unsqueeze(1)  # mbsize x 1 x seq_len x emb_dim
#         #
#         # x3 = F.relu(self.conv3(inputs)).squeeze(3)
#         # x4 = F.relu(self.conv4(inputs)).squeeze(3)
#         # x5 = F.relu(self.conv5(inputs)).squeeze(3)
#         #
#         # # Max-over-time-pool
#         # x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)
#         # x4 = F.max_pool1d(x4, x4.size(2)).squeeze(2)
#         # x5 = F.max_pool1d(x5, x5.size(2)).squeeze(2)
#         #
#         # x = torch.cat([x3, x4, x5], dim=1)
#         #
#         # y = self.disc_fc(x)
#
#         _, h = self.discriminator(inputs.transpose(1, 0), None)
#
#         # Forward to latent
#         h = h.view(-1, self.h_dim * 2)
#         y = self.disc_fc(h)
#
#         return y
#
#     def forward(self, sentence, mask, use_c_prior=True, temp=1):
#         """
#         Params:
#         -------
#         sentence: sequence of word indices.
#         use_c_prior: whether to sample `c` from prior or from `discriminator`.
#
#         Returns:
#         --------
#         recon_loss: reconstruction loss of VAE.
#         kl_loss: KL-div loss of VAE.
#         """
#
#         mbsize = sentence.size(1)
#
#         # sentence: '<start> I want to fly <eos>'
#         # enc_inputs: '<start> I want to fly <eos>'
#         # dec_inputs: '<start> I want to fly <eos>'
#         # dec_targets: 'I want to fly <eos> <pad>'
#         pad_words = Variable(torch.LongTensor([self.PAD_IDX])).repeat(1, mbsize)
#         pad_words = pad_words.to(self.device)
#
#         enc_inputs = sentence
#         dec_inputs = sentence
#         dec_targets = torch.cat([sentence[1:], pad_words], dim=0)
#
#         # Encoder: sentence -> z
#         z, attention = self.forward_encoder(enc_inputs, mask)
#
#         # Generator: z -> graph
#         # text, adj = self.generate_graph(z, temp=temp)
#         # text, adj, entropy, cost = self.generate_graph(z, mask, return_entropy=True)
#         text = z
#         adj = self.generate_adjacency_matrix(z)
#
#         # Graph Encoder: graph -> z'
#         z1 = self.forward_graph_encoder(text, adj)
#
#         # Decoder: sentence -> y
#         y = self.forward_decoder(dec_inputs, z1)
#
#         recon_loss = F.cross_entropy(
#             y.view(-1, self.n_vocab), dec_targets.view(-1), size_average=True
#         )
#
#         # I = torch.eye(attention.size()[1]).to(self.device)
#         # penal = l2_matrix_norm(attention @ attention.transpose(1, 2) - I)
#         penal = torch.distributions.Categorical(probs=attention).entropy().mean()
#
#         c = attention[:, 0, :]
#         current_attention = attention[:, 1, :]
#         c_loss = torch.min(c, current_attention)
#         for i in range(2, self.n_node):
#             c = c + current_attention
#             current_attention = attention[:, i, :]
#             c_loss = c_loss + torch.min(c, current_attention)
#         c_loss = c_loss.sum(dim=1).mean()
#
#         return recon_loss, penal, c_loss
#
#     def forward_classifier(self, inputs, mask, temp=1, take_mean=True):
#
#         # Encoder: sentence -> z
#         z, _ = self.forward_encoder(inputs, mask)
#
#         # Generator: z -> graph
#         # text, adj = self.generate_graph(z, temp)
#         # text, adj = self.generate_graph(z, mask, return_entropy=False)
#         text = z
#         adj = self.generate_adjacency_matrix(z)
#
#         # classifier: graph -> prediction
#         x = F.relu(self.graph_encoder_cls(text, adj))
#         x = x.view(-1, self.node_dim * self.n_node)
#         outputs = self.classifier(x)
#         return outputs
#
#     def generate_sentences(self, batch_size, length, temp=1.0, z=None, mask=None):
#         """
#         Generate sentences and corresponding z of (batch_size x max_sent_len)
#         """
#         self.eval()
#
#         if z is None:
#             z = self.sample_z_prior(batch_size)
#
#         X_gen = self.sample_sentence(z, mask, length, raw=True, temp=temp)
#
#         # Back to default state: train
#         self.train()
#
#         return X_gen
#
#     def sample_sentence(self, z, mask, length, raw=False, temp=1):
#         """
#         Sample single sentence from p(x|z,c) according to given temperature.
#         `raw = True` means this returns sentence as in dataset which is useful
#         to train discriminator. `False` means that this will return list of
#         `word_idx` which is useful for evaluation.
#         """
#         mbsize = z.size(0)
#
#         start_word = torch.LongTensor([self.START_IDX]).repeat(mbsize).view(1, -1)
#         start_word = start_word.to(self.device)
#         word = Variable(start_word)  # '<start>'
#
#         if not isinstance(z, Variable):
#             z = Variable(z)
#
#         z = z.view(-1, self.n_node, self.node_dim)
#
#         # Generator: z -> graph
#         # text, adj = self.generate_graph(z, temp=temp)
#         text, adj = self.generate_graph(z, mask, return_entropy=False)
#
#         # Graph Encoder: graph -> z'
#         z1 = self.forward_graph_encoder(text, adj)
#         z1 = z1.unsqueeze(0)
#         h = z1
#
#         outputs = []
#
#         if raw:
#             outputs.append(start_word)
#
#         for i in range(length - 1):
#             emb = self.word_emb(word)
#             emb = torch.cat([emb, z1], 2)
#
#             output, h = self.decoder(emb, h)
#             y = self.decoder_fc(output)
#             y = y @ self.word_emb.weight.t()
#             y = F.softmax(y / temp, dim=2)
#
#             idx = torch.multinomial(y.squeeze(0), num_samples=1)
#
#             word = idx.transpose(1, 0)
#
#             # idx = int(idx)
#             #
#             # if not raw and idx == self.EOS_IDX:
#             #     break
#
#             outputs.append(word)
#
#         outputs = torch.cat(outputs, dim=0).transpose(1, 0)
#
#         for i in range(mbsize):
#             for j, idx in enumerate(outputs[i]):
#                 if idx == self.EOS_IDX:
#                     outputs[i][j:] = self.PAD_IDX
#
#         if raw:
#             return outputs
#         else:
#             return outputs.cpu().int()
#
#     def generate_soft_embed(self, batch_size, length, temp=1, z=None, mask=None):
#         """
#         Generate soft embeddings of (mbsize x emb_dim) along with target z
#         and c for each row (mbsize x {z_dim, c_dim})
#         """
#         if z is None:
#             z = self.sample_z_prior(batch_size)
#
#         X_gen = self.sample_soft_embed(z, mask, length, temp=temp)
#
#         return X_gen
#
#     def sample_soft_embed(self, z, mask, length, temp=1):
#         """
#         Sample single soft embedded sentence from p(x|z,c) and temperature.
#         Soft embeddings are calculated as weighted average of word_emb
#         according to p(x|z,c).
#         """
#         mbsize = z.size(0)
#
#         start_word = torch.LongTensor([self.START_IDX]).repeat(mbsize).view(1, -1)
#         start_word = start_word.to(self.device)
#         word = Variable(start_word)  # '<start>'
#
#         if not isinstance(z, Variable):
#             z = Variable(z)
#
#         z = z.view(-1, self.n_node, self.node_dim)
#
#         # Generator: z -> graph
#         # text, adj = self.generate_graph(z, temp=temp)
#         # text, adj = self.generate_graph(z, mask, return_entropy=False)
#         text = z
#         adj = self.generate_adjacency_matrix(z)
#
#         # Graph Encoder: graph -> z'
#         z1 = self.forward_graph_encoder(text, adj)
#         z1 = z1.unsqueeze(0)
#         h = z1
#
#         emb = self.word_emb(word)
#         emb = torch.cat([emb, z1], 2)
#
#         outputs = [self.word_emb(word)]
#
#         for i in range(length-1):
#             output, h = self.decoder(emb, h)
#             o = self.decoder_fc(output)
#
#             # o = o @ self.word_emb.weight.t()
#             #
#             # # Sample softmax with temperature
#             # y = F.softmax(o / temp, dim=2)
#             #
#             # # Take expectation of embedding given output prob -> soft embedding
#             # # <y, w> = 1 x n_vocab * n_vocab x emb_dim
#             # emb = y @ self.word_emb.weight
#
#             emb = o
#
#             # Save resulting soft embedding
#             outputs.append(emb)
#
#             # Append with z for the next input
#             emb = torch.cat([emb, z1], 2)
#
#         # mbsize x length x emb_dim
#         outputs = torch.cat(outputs, dim=0).transpose(1, 0)
#
#         return outputs
#
#     def output_graph(self, z):
#         adj = self.generate_adjacency_matrix(z).cpu().float()
#         word_idx = self.generate_word(z).cpu().int()
#         return adj, word_idx
#
#     def output_graph_from_text(self, text):
#         flag = 0
#         if self.training:
#             self.eval()
#             flag = 1
#
#         z,  = self.forward_encoder(text)
#
#         if flag:
#             self.train()
#
#         return self.output_graph(z)
#
#     def generate_graph(self, z, mask, temp=1.0, return_entropy=False):
#         adj = self.generate_adjacency_matrix(z)
#         if return_entropy:
#             text, entropy, cost = self.generate_soft_node_embed(z, mask, temp, True)
#             return text, adj, entropy, cost
#         else:
#             text = self.generate_soft_node_embed(z, mask, temp, False)
#             return text, adj
#
#     def generate_adjacency_matrix(self, z):
#         # z = self.generate_adj(z)
#         # adj = F.sigmoid(torch.matmul(z, z.transpose(2, 1)))
#         # return adj
#
#         # GLoMo
#         I = torch.eye(z.size()[1]).to(self.device)
#         # z = self.generate_adj(z)
#         z = z / z.norm(p=2, dim=2, keepdim=True)
#         adj = F.relu(torch.matmul(z, z.transpose(2, 1))) ** 2
#         adj = (adj - I) / adj.sum(dim=1).unsqueeze(1)
#         return adj
#
#     def generate_word(self, z):
#         outputs = self.generate_text(z).view(-1)
#         word_idx = torch.argmax(F.softmax(outputs, dim=0), dim=0)
#         return word_idx
#
#     def generate_soft_node_embed(self, z, mask=None, temp=1.0, return_entropy=True):
#         """
#         Sample single soft embedded sentence from p(x|z) and temperature.
#         Soft embeddings are calculated as weighted average of word_emb
#         according to p(x|z).
#         """
#
#         outputs = self.generate_text(z)
#         outputs = outputs @ self.word_emb.weight.t()
#         if mask is not None:
#             outputs = outputs.masked_fill(mask.unsqueeze(1) == 0, -float('inf'))
#
#         # Sample softmax with temperature
#         y = F.softmax(outputs / temp, dim=2)
#
#         # Take expectation of embedding given output prob -> soft embedding
#         emb = torch.matmul(y, self.word_emb.weight)
#
#         if return_entropy:
#             # Calculate entropy for penalty
#             entropy = torch.distributions.Categorical(probs=y).entropy().mean()
#             # cost = self.estimate_cost(y)
#             cost = torch.FloatTensor([0.0])
#             return emb, entropy, cost
#         else:
#             return emb
#
#     def estimate_cost(self, probs, sample_size=10):
#         sampled = random.sample(self.sampler, sample_size)
#         cost = torch.FloatTensor([0.0]).to(self.device)
#         for x, y in sampled:
#             n1 = probs[:, x, :]
#             n2 = probs[:, y, :]
#             kl = F.kl_div(n1.log(), n2, reduce=False).sum(dim=1)
#             kl = F.hardtanh(kl, max_val=10, min_val=-10).sum() / kl.size(0)
#             cost -= kl
#             # if not torch.isinf(kl):
#             #     cost -= kl
#         return cost / sample_size
#
#     def train_model(self, args, dataset):
#         """train the whole network"""
#         self.train()
#
#         kl_weight = args.kl_weight
#         alpha = args.alpha
#         beta = args.beta
#         gamma = args.gamma
#
#         trainer_D = optim.Adam(self.discriminator_params, lr=args.lr)
#         trainer_G = optim.Adam(self.decoder_params, lr=args.lr)
#         trainer_E = optim.Adam(self.encoder_params, lr=args.lr)
#
#         criterion_bce = nn.BCEWithLogitsLoss().to(self.device)
#         ones_label = Variable(torch.ones((args.batch_size, 1))).to(self.device)
#         zeros_label = Variable(torch.zeros((args.batch_size, 1))).to(self.device)
#
#         patience = 0
#         best_val_acc, best_test_acc, best_iter = 0, 0, 0
#         list_loss_D, list_loss_G, list_loss_E, list_loss_recon, list_loss_penal, list_loss_closs, list_entropy, \
#         list_cost = [], [], [], [], [], [], [], []
#         train_start_time = time.time()
#         train_iter = dataset.build_train_iter(args.batch_size, self.device)
#         val_iter, test_iter = dataset.build_test_iter(args.batch_size, self.device, shuffle=False)
#
#         val_acc, test_acc = self.train_model_classifier(args, dataset)
#         print(f'Initial Eval: Valid Acc: {val_acc:.4f}; Test Acc: {test_acc:.4f}')
#
#         for epoch in range(1, args.epochs):
#             current_temp = temp(epoch)
#             for subdata in [train_iter, val_iter, test_iter]:
#                 for batch in iter(subdata):
#                     inputs, labels = batch.text, batch.label
#                     padding_mask = dataset.create_padding_mask(inputs.transpose(1, 0), device=self.device)
#                     length, batch_size = inputs.size()
#                     recon_loss, penal, closs = self.forward(inputs, mask=padding_mask, temp=current_temp)
#                     loss_vae = recon_loss + alpha * penal + beta * closs
#                     list_loss_recon.append(recon_loss.item())
#                     list_loss_penal.append(penal.item())
#                     list_loss_closs.append(closs.item())
#
#                     """ Update discriminator """
#                     # y_disc_real = self.forward_discriminator(inputs.transpose(1, 0))
#                     # loss_D_real = criterion_bce(y_disc_real, ones_label[:batch_size])
#                     #
#                     # # x_recon = self.generate_sentences(batch_size, length, temp=current_temp, z=z0, mask=mask)
#                     # # y_disc_fake_recon = self.forward_discriminator(x_recon)
#                     # x_recon_embed = self.generate_soft_embed(batch_size, length, temp=current_temp, z=z0, mask=mask)
#                     # y_disc_fake_recon = self.forward_discriminator_embed(x_recon_embed)
#                     # loss_D_recon = criterion_bce(y_disc_fake_recon, zeros_label[:batch_size])
#                     #
#                     # # x_gen = self.generate_sentences(batch_size, length, temp=current_temp)
#                     # # y_disc_fake_sample = self.forward_discriminator(x_gen)
#                     # x_gen_embed = self.generate_soft_embed(batch_size, length, temp=current_temp)
#                     # y_disc_fake_sample = self.forward_discriminator_embed(x_gen_embed)
#                     # loss_D_sample = criterion_bce(y_disc_fake_sample, zeros_label[:batch_size])
#                     #
#                     # loss_D = loss_D_real + loss_D_recon + loss_D_sample
#                     # list_loss_D.append(loss_D.item())
#                     # trainer_D.zero_grad()
#                     # loss_D.backward(retain_graph=True)
#                     # # grad_norm = torch.nn.utils.clip_grad_norm_(self.discriminator_params, 5)
#                     # trainer_D.step()
#
#                     """ Update generator  """
#                     # y_disc_real = self.forward_discriminator(inputs.transpose(1, 0))
#                     # loss_D_real = criterion_bce(y_disc_real, zeros_label[:batch_size])
#                     #
#                     # x_recon_embed = self.generate_soft_embed(batch_size, length, temp=current_temp, z=z0, mask=mask)
#                     # y_disc_fake_recon = self.forward_discriminator_embed(x_recon_embed)
#                     # loss_D_recon = criterion_bce(y_disc_fake_recon, ones_label[:batch_size])
#                     #
#                     # x_gen_embed = self.generate_soft_embed(batch_size, length, temp=current_temp)
#                     # y_disc_fake_sample = self.forward_discriminator_embed(x_gen_embed)
#                     # loss_D_sample = criterion_bce(y_disc_fake_sample, ones_label[:batch_size])
#                     # loss_D = (loss_D_real + loss_D_recon + loss_D_sample)
#
#                     # loss_G = alpha * loss_D + loss_vae
#                     loss_G = loss_vae
#                     # list_loss_G.append(loss_D.item())
#                     trainer_G.zero_grad()
#                     loss_G.backward(retain_graph=True)
#                     # grad_norm = torch.nn.utils.clip_grad_norm_(self.decoder_params, 5)
#                     trainer_G.step()
#
#                     """ Update encoder  """
#                     # loss_E = beta * loss_D + loss_vae
#                     loss_E = loss_vae
#                     # list_loss_E.append(-loss_D.item())
#                     trainer_E.zero_grad()
#                     loss_E.backward(retain_graph=True)
#                     # grad_norm = torch.nn.utils.clip_grad_norm_(self.encoder_params, 5)
#                     trainer_E.step()
#
#             if epoch % args.log_every == 0:
#                 duration = time.time() - train_start_time
#                 avr_loss_D = np.mean(list_loss_D)
#                 avr_loss_G = np.mean(list_loss_G)
#                 # avr_loss_E = np.mean(list_loss_E)
#                 avr_loss_recon = np.mean(list_loss_recon)
#                 avr_loss_penal = np.mean(list_loss_penal)
#                 avr_closs = np.mean(list_loss_closs)
#                 # avr_cost = np.mean(list_cost)
#                 list_loss_D, list_loss_G, list_loss_E, list_loss_recon, list_loss_penal, list_loss_closs, list_entropy, \
#                 list_cost = [], [], [], [], [], [], [], []
#                 print(f'Epoch-{epoch}; temp: {current_temp:.4f}; loss_D: {avr_loss_D:.4f}; loss_G: {avr_loss_G:.4f}; '
#                       f'loss_recon: {avr_loss_recon:.4f}; penal: {avr_loss_penal:.4f}; avr_closs: {avr_closs:.4f}; '
#                       f'duration: {duration:.2f}')
#
#                 val_acc, test_acc = self.train_model_classifier(args, dataset, current_temp)
#
#                 # z = self.sample_z_prior(1)
#                 # sample_idxs = [i for i in self.sample_sentence(z, None, length=50)[0] if i != self.PAD_IDX]
#                 # sample_sent = dataset.idxs2sentence(sample_idxs)
#                 # print(f'Valid Acc: {val_acc:.4f}; Test Acc: {test_acc:.4f}; Sample: "{sample_sent}"')
#                 print(f'Valid Acc: {val_acc:.4f}; Test Acc: {test_acc:.4f}')
#                 # print(f'Sample: "{sample_sent}"')
#
#                 if val_acc > best_val_acc:
#                     best_val_acc = val_acc
#                     best_test_acc = test_acc
#                     best_iter = epoch
#                     patience = 0
#                 else:
#                     if epoch > args.minimal_epoch:
#                         patience += args.log_every
#
#                 if args.early_stop and patience > args.patience:
#                     break
#
#         print(f'Best Valid Acc: {best_val_acc:.4f}; Best Test Acc: {best_test_acc:.4f} @ epoch {best_iter}')
#         return best_val_acc, best_test_acc
#
#     def train_model_classifier(self, args, dataset, temp=1.0):
#         """train the whole network"""
#         self.train()
#
#         self.classifier_[0].reset_parameters()
#         for i in self.classifier_[1]:
#             if hasattr(i, 'reset_parameters'):
#                 i.reset_parameters()
#         trainer_C = optim.Adam(filter(lambda p: p.requires_grad, self.classifier_.parameters()), lr=args.lr)
#         criterion_cls = nn.CrossEntropyLoss().to(self.device)
#
#         patience = 0
#         log_every = 5
#         list_loss_cls = []
#         best_val_acc, best_test_acc, best_iter = 0, 0, 0
#         train_start_time = time.time()
#
#         train_iter = dataset.build_train_iter(args.eval_batch_size, self.device)
#
#         for epoch in range(1, args.eval_epochs):
#             for batch in iter(train_iter):
#
#                 inputs, labels = batch.text, batch.label
#                 mask = dataset.create_padding_mask(inputs.transpose(1, 0), device=self.device)
#
#                 """ Update classifier """
#                 outputs = self.forward_classifier(inputs, mask, temp)
#                 loss_cls = criterion_cls(outputs, labels)
#                 list_loss_cls.append(loss_cls.item())
#
#                 trainer_C.zero_grad()
#                 loss_cls.backward(retain_graph=True)
#                 trainer_C.step()
#
#             if epoch % log_every == 0:
#                 duration = time.time() - train_start_time
#                 avr_loss_cls = np.mean(list_loss_cls)
#                 list_loss_cls = []
#
#                 val_iter, test_iter = dataset.build_test_iter(args.batch_size, self.device)
#                 val_acc = self.test(dataset, val_iter)
#                 test_acc = self.test(dataset, test_iter)
#
#                 if val_acc > best_val_acc:
#                     best_val_acc = val_acc
#                     best_test_acc = test_acc
#                     best_iter = epoch
#                     patience = 0
#                 else:
#                     if epoch > args.eval_minimal_epoch:
#                         patience += log_every
#
#                 if args.eval_early_stop and patience > args.eval_patience:
#                     break
#
#             # if epoch % 10 == 0:
#             #     print(f'Epoch-{epoch}; loss_cls: {avr_loss_cls:.4f}; Valid Acc: {val_acc:.4f}; Test Acc: {test_acc:.4f};'
#             #           f' duration: {duration:.2f}')
#
#         # print(f'Best Valid Acc: {best_val_acc:.4f}; Best Test Acc: {best_test_acc:.4f} @ iter {best_iter}')
#         return best_val_acc, best_test_acc
#
#     def test(self, dataset, data_iter):
#         self.eval()
#         total_accuracy = []
#         with torch.no_grad():
#             for batch in iter(data_iter):
#                 inputs, labels = batch.text, batch.label
#                 mask = dataset.create_padding_mask(inputs.transpose(1, 0), device=self.device)
#                 outputs = self.forward_classifier(inputs, mask)
#                 accuracy = (outputs.argmax(1) == labels).float().mean().item()
#                 total_accuracy.append(accuracy)
#         acc = sum(total_accuracy) / len(total_accuracy)
#         self.train()
#         return acc


# class NetGen(AbstractVAE):
#     def __init__(self, n_node, n_vocab, n_labels, h_dim, z_dim, p_word_dropout=0.3, unk_idx=0, pad_idx=1, start_idx=2, eos_idx=3,
#                  pretrained_embeddings=None, freeze_embeddings=False, device=False):
#         super(NetGen, self).__init__(n_vocab, n_labels, h_dim, p_word_dropout, unk_idx, pad_idx, start_idx, eos_idx,
#                                           pretrained_embeddings, freeze_embeddings, device)
#
#         self.z_dim = n_node * z_dim
#         self.node_dim = z_dim
#         self.n_node = n_node
#         self.sampler = list(combinations(range(n_node), 2))
#
#         """
#         Encoder is GRU with FC layers connected to last hidden unit
#         """
#         self.encoder = nn.GRU(self.emb_dim, h_dim)
#         self.q_mu = nn.Sequential(
#             nn.Linear(h_dim, h_dim),
#             nn.BatchNorm1d(h_dim),
#             nn.LeakyReLU(0.2),
#             # nn.ELU(),
#             nn.Linear(h_dim, self.z_dim)
#         )
#         self.q_logvar = nn.Sequential(
#             nn.Linear(h_dim, h_dim),
#             nn.BatchNorm1d(h_dim),
#             nn.LeakyReLU(0.2),
#             # nn.ELU(),
#             nn.Linear(h_dim, self.z_dim)
#         )
#         self.encoder_ = nn.ModuleList([
#             self.encoder, self.q_mu, self.q_logvar
#         ])
#
#         """
#         Decoder is GRU with `z` appended at its inputs
#         """
#         self.decoder = nn.GRU(self.emb_dim + n_node * self.emb_dim, n_node * self.emb_dim, dropout=0.3)
#         # self.decoder_fc = nn.Linear(n_node * self.emb_dim, n_vocab)
#         self.decoder_fc = nn.Sequential(
#             nn.Linear(n_node * self.emb_dim, self.emb_dim),
#             nn.LeakyReLU(0.2),
#             # nn.ELU(),
#             nn.Linear(self.emb_dim, self.emb_dim)
#         )
#
#         """
#         Generator
#         """
#         self.generate_text = nn.Sequential(
#             nn.Linear(z_dim, self.emb_dim, bias=False),
#             nn.LeakyReLU(0.2),
#             # nn.ELU(),
#             nn.Linear(self.emb_dim, self.emb_dim, bias=False)
#         )
#         self.generate_adj = nn.Sequential(
#             nn.Linear(z_dim, z_dim, bias=False),
#             nn.LeakyReLU(0.2),
#             # nn.ELU(),
#             nn.Linear(z_dim, z_dim, bias=False)
#         )
#
#         """
#         Graph Encoder is GCN with pooling layer
#         """
#         self.graph_encoder = GraphConvolution(self.emb_dim, self.emb_dim)
#         self.graph_encoder_trans = nn.Sequential(
#             nn.Linear(self.n_node * self.emb_dim, self.n_node * self.emb_dim),
#             nn.LeakyReLU(0.2),
#             # nn.ELU(),
#             nn.Linear(self.n_node * self.emb_dim, self.n_node * self.emb_dim),
#         )
#         self.decoder_ = nn.ModuleList([
#             self.decoder, self.decoder_fc,
#             self.generate_text, self.generate_text,
#             self.graph_encoder, self.graph_encoder_trans
#         ])
#
#         """
#         Discriminator is CNN as in Kim, 2014
#         """
#         # self.conv3 = nn.Conv2d(1, 100, (3, self.emb_dim))
#         # self.conv4 = nn.Conv2d(1, 100, (4, self.emb_dim))
#         # self.conv5 = nn.Conv2d(1, 100, (5, self.emb_dim))
#         #
#         # self.disc_fc = nn.Sequential(
#         #     nn.Dropout(0.5),
#         #     nn.Linear(300, 1)
#         # )
#         #
#         # self.discriminator_ = nn.ModuleList([
#         #     self.conv3, self.conv4, self.conv5, self.disc_fc
#         # ])
#
#         self.discriminator = nn.GRU(self.emb_dim, h_dim, bidirectional=True)
#         self.disc_fc = nn.Sequential(
#             nn.Linear(2 * h_dim, h_dim),
#             nn.BatchNorm1d(h_dim),
#             nn.ReLU(),
#             # nn.ELU(),
#             nn.Linear(h_dim, 1)
#         )
#         self.discriminator_ = nn.ModuleList([
#             self.discriminator, self.disc_fc
#         ])
#
#         """
#         Classifier is DNN
#         """
#         self.graph_encoder_cls = GraphConvolution(self.emb_dim, self.emb_dim)
#         self.classifier = nn.Sequential(
#             nn.Linear(self.emb_dim * self.n_node, self.emb_dim),
#             nn.ReLU(),
#             nn.Linear(self.emb_dim, self.n_labels)
#         )
#         self.classifier_ = nn.ModuleList([
#             self.graph_encoder_cls, self.classifier
#         ])
#
#         """
#         Grouping the model's parameters: separating encoder, decoder, and discriminator
#         """
#         self.encoder_params = filter(lambda p: p.requires_grad, self.encoder_.parameters())
#
#         self.decoder_params = filter(lambda p: p.requires_grad, self.decoder_.parameters())
#
#         self.discriminator_params = filter(lambda p: p.requires_grad, self.discriminator_.parameters())
#
#         self.classifier_params = filter(lambda p: p.requires_grad, self.classifier_.parameters())
#
#         self.vae_params = chain(
#             self.word_emb.parameters(), self.encoder_.parameters(), self.decoder_.parameters()
#         )
#         self.vae_params = filter(lambda p: p.requires_grad, self.vae_params)
#
#         """
#         Use GPU if set
#         """
#         self.to(self.device)
#
#     def forward_graph_encoder(self, input, adj):
#         x = F.relu(self.graph_encoder(input, adj))
#         # x = self.graph_encoder(input, adj)
#         x = x.view(-1, self.n_node * self.emb_dim)
#         x = self.graph_encoder_trans(x)
#
#         return x
#
#     def forward_encoder_embed(self, inputs):
#         """
#         Inputs is embeddings of: seq_len x mbsize x emb_dim
#         """
#         _, h = self.encoder(inputs, None)
#
#         # Forward to latent
#         h = h.view(-1, self.h_dim)
#
#         mu = self.q_mu(h)
#         logvar = self.q_logvar(h)
#
#         return mu, logvar
#
#     def forward_decoder(self, inputs, z):
#         """
#         Inputs must be embeddings: seq_len x mbsize
#         """
#         dec_inputs = self.word_dropout(inputs)
#
#         # Forward
#         seq_len = dec_inputs.size(0)
#
#         # 1 x mbsize x z_dim
#         init_h = z.unsqueeze(0)
#         inputs_emb = self.word_emb(dec_inputs)  # seq_len x mbsize x emb_dim
#         inputs_emb = torch.cat([inputs_emb, init_h.repeat(seq_len, 1, 1)], 2)
#
#         outputs, _ = self.decoder(inputs_emb, init_h)
#         seq_len, mbsize, _ = outputs.size()
#
#         outputs = outputs.view(seq_len * mbsize, -1)
#         y = self.decoder_fc(outputs)
#         y = y @ self.word_emb.weight.t()
#         y = y.view(seq_len, mbsize, self.n_vocab)
#
#         return y
#
#     def forward_discriminator(self, inputs):
#         """
#         Inputs is batch of sentences: mbsize x seq_len
#         """
#         inputs = self.word_emb(inputs)
#         return self.forward_discriminator_embed(inputs)
#
#     def forward_discriminator_embed(self, inputs):
#         """
#         Inputs must be embeddings: mbsize x seq_len x emb_dim
#         """
#         # inputs = inputs.unsqueeze(1)  # mbsize x 1 x seq_len x emb_dim
#         #
#         # x3 = F.relu(self.conv3(inputs)).squeeze(3)
#         # x4 = F.relu(self.conv4(inputs)).squeeze(3)
#         # x5 = F.relu(self.conv5(inputs)).squeeze(3)
#         #
#         # # Max-over-time-pool
#         # x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)
#         # x4 = F.max_pool1d(x4, x4.size(2)).squeeze(2)
#         # x5 = F.max_pool1d(x5, x5.size(2)).squeeze(2)
#         #
#         # x = torch.cat([x3, x4, x5], dim=1)
#         #
#         # y = self.disc_fc(x)
#
#         _, h = self.discriminator(inputs.transpose(1, 0), None)
#
#         # Forward to latent
#         h = h.view(-1, self.h_dim * 2)
#         y = self.disc_fc(h)
#
#         return y
#
#     def forward(self, sentence, mask, use_c_prior=True, temp=1):
#         """
#         Params:
#         -------
#         sentence: sequence of word indices.
#         use_c_prior: whether to sample `c` from prior or from `discriminator`.
#
#         Returns:
#         --------
#         recon_loss: reconstruction loss of VAE.
#         kl_loss: KL-div loss of VAE.
#         """
#
#         mbsize = sentence.size(1)
#
#         # sentence: '<start> I want to fly <eos>'
#         # enc_inputs: '<start> I want to fly <eos>'
#         # dec_inputs: '<start> I want to fly <eos>'
#         # dec_targets: 'I want to fly <eos> <pad>'
#         pad_words = Variable(torch.LongTensor([self.PAD_IDX])).repeat(1, mbsize)
#         pad_words = pad_words.to(self.device)
#
#         enc_inputs = sentence
#         dec_inputs = sentence
#         dec_targets = torch.cat([sentence[1:], pad_words], dim=0)
#
#         # Encoder: sentence -> z
#         mu, logvar = self.forward_encoder(enc_inputs)
#         z0 = self.sample_z(mu, logvar)
#         # z0, _ = self.forward_encoder(enc_inputs)
#         z = z0.view(-1, self.n_node, self.node_dim)
#
#         # Generator: z -> graph
#         # text, adj = self.generate_graph(z, temp=temp)
#         text, adj, entropy, cost = self.generate_graph(z, mask, return_entropy=True)
#
#         # Graph Encoder: graph -> z'
#         z1 = self.forward_graph_encoder(text, adj)
#
#         # Decoder: sentence -> y
#         y = self.forward_decoder(dec_inputs, z1)
#
#         recon_loss = F.cross_entropy(
#             y.view(-1, self.n_vocab), dec_targets.view(-1), size_average=True
#         )
#         kl_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1 - logvar, 1))
#         # kl_loss = torch.FloatTensor([0]).to(self.device)
#
#         return recon_loss, kl_loss, entropy, cost, z0
#
#     def forward_classifier(self, inputs, mask, temp=1, take_mean=True):
#
#         # Encoder: sentence -> z
#         mu, logvar = self.forward_encoder(inputs)
#         if take_mean:
#             z0 = mu
#         else:
#             z0 = self.sample_z(mu, logvar)
#         z = z0.view(-1, self.n_node, self.node_dim)
#
#         # Generator: z -> graph
#         # text, adj = self.generate_graph(z, temp)
#         text, adj = self.generate_graph(z, mask, return_entropy=False)
#
#         # classifier: graph -> prediction
#         x = F.relu(self.graph_encoder_cls(text, adj))
#         x = x.view(-1, self.emb_dim * self.n_node)
#         outputs = self.classifier(x)
#         return outputs
#
#     def generate_sentences(self, batch_size, length, temp=1.0, z=None, mask=None):
#         """
#         Generate sentences and corresponding z of (batch_size x max_sent_len)
#         """
#         self.eval()
#
#         if z is None:
#             z = self.sample_z_prior(batch_size)
#
#         X_gen = self.sample_sentence(z, mask, length, raw=True, temp=temp)
#
#         # Back to default state: train
#         self.train()
#
#         return X_gen
#
#     def sample_sentence(self, z, mask, length, raw=False, temp=1):
#         """
#         Sample single sentence from p(x|z,c) according to given temperature.
#         `raw = True` means this returns sentence as in dataset which is useful
#         to train discriminator. `False` means that this will return list of
#         `word_idx` which is useful for evaluation.
#         """
#         mbsize = z.size(0)
#
#         start_word = torch.LongTensor([self.START_IDX]).repeat(mbsize).view(1, -1)
#         start_word = start_word.to(self.device)
#         word = Variable(start_word)  # '<start>'
#
#         if not isinstance(z, Variable):
#             z = Variable(z)
#
#         z = z.view(-1, self.n_node, self.node_dim)
#
#         # Generator: z -> graph
#         # text, adj = self.generate_graph(z, temp=temp)
#         text, adj = self.generate_graph(z, mask, return_entropy=False)
#
#         # Graph Encoder: graph -> z'
#         z1 = self.forward_graph_encoder(text, adj)
#         z1 = z1.unsqueeze(0)
#         h = z1
#
#         outputs = []
#
#         if raw:
#             outputs.append(start_word)
#
#         for i in range(length - 1):
#             emb = self.word_emb(word)
#             emb = torch.cat([emb, z1], 2)
#
#             output, h = self.decoder(emb, h)
#             y = self.decoder_fc(output)
#             y = y @ self.word_emb.weight.t()
#             y = F.softmax(y / temp, dim=2)
#
#             idx = torch.multinomial(y.squeeze(0), num_samples=1)
#
#             word = idx.transpose(1, 0)
#
#             # idx = int(idx)
#             #
#             # if not raw and idx == self.EOS_IDX:
#             #     break
#
#             outputs.append(word)
#
#         outputs = torch.cat(outputs, dim=0).transpose(1, 0)
#
#         for i in range(mbsize):
#             for j, idx in enumerate(outputs[i]):
#                 if idx == self.EOS_IDX:
#                     outputs[i][j:] = self.PAD_IDX
#
#         if raw:
#             return outputs
#         else:
#             return outputs.cpu().int()
#
#     def generate_soft_embed(self, batch_size, length, temp=1, z=None, mask=None):
#         """
#         Generate soft embeddings of (mbsize x emb_dim) along with target z
#         and c for each row (mbsize x {z_dim, c_dim})
#         """
#         if z is None:
#             z = self.sample_z_prior(batch_size)
#
#         X_gen = self.sample_soft_embed(z, mask, length, temp=temp)
#
#         return X_gen
#
#     def sample_soft_embed(self, z, mask, length, temp=1):
#         """
#         Sample single soft embedded sentence from p(x|z,c) and temperature.
#         Soft embeddings are calculated as weighted average of word_emb
#         according to p(x|z,c).
#         """
#         mbsize = z.size(0)
#
#         start_word = torch.LongTensor([self.START_IDX]).repeat(mbsize).view(1, -1)
#         start_word = start_word.to(self.device)
#         word = Variable(start_word)  # '<start>'
#
#         if not isinstance(z, Variable):
#             z = Variable(z)
#
#         z = z.view(-1, self.n_node, self.node_dim)
#
#         # Generator: z -> graph
#         # text, adj = self.generate_graph(z, temp=temp)
#         text, adj = self.generate_graph(z, mask, return_entropy=False)
#
#         # Graph Encoder: graph -> z'
#         z1 = self.forward_graph_encoder(text, adj)
#         z1 = z1.unsqueeze(0)
#         h = z1
#
#         emb = self.word_emb(word)
#         emb = torch.cat([emb, z1], 2)
#
#         outputs = [self.word_emb(word)]
#
#         for i in range(length-1):
#             output, h = self.decoder(emb, h)
#             o = self.decoder_fc(output)
#
#             # o = o @ self.word_emb.weight.t()
#             #
#             # # Sample softmax with temperature
#             # y = F.softmax(o / temp, dim=2)
#             #
#             # # Take expectation of embedding given output prob -> soft embedding
#             # # <y, w> = 1 x n_vocab * n_vocab x emb_dim
#             # emb = y @ self.word_emb.weight
#
#             emb = o
#
#             # Save resulting soft embedding
#             outputs.append(emb)
#
#             # Append with z for the next input
#             emb = torch.cat([emb, z1], 2)
#
#         # mbsize x length x emb_dim
#         outputs = torch.cat(outputs, dim=0).transpose(1, 0)
#
#         return outputs
#
#     def output_graph(self, z):
#         adj = self.generate_adjacency_matrix(z).cpu().float()
#         word_idx = self.generate_word(z).cpu().int()
#         return adj, word_idx
#
#     def output_graph_from_text(self, text):
#         flag = 0
#         if self.training:
#             self.eval()
#             flag = 1
#
#         mu, logvar = self.forward_encoder(text)
#         z = self.sample_z(mu, logvar)
#         z = z.view(-1, self.n_node, self.node_dim)
#
#         if flag:
#             self.train()
#
#         return self.output_graph(z)
#
#     def generate_graph(self, z, mask, temp=1.0, return_entropy=False):
#         adj = self.generate_adjacency_matrix(z)
#         if return_entropy:
#             text, entropy, cost = self.generate_soft_node_embed(z, mask, temp, True)
#             return text, adj, entropy, cost
#         else:
#             text = self.generate_soft_node_embed(z, mask, temp, False)
#             return text, adj
#
#     def generate_adjacency_matrix(self, z):
#         # z = self.generate_adj(z)
#         # adj = F.sigmoid(torch.matmul(z, z.transpose(2, 1)))
#         # return adj
#
#         # GLoMo
#         I = torch.eye(z.size()[1]).to(self.device)
#         z = self.generate_adj(z)
#         z = z / z.norm(p=2, dim=2, keepdim=True)
#         adj = F.relu(torch.matmul(z, z.transpose(2, 1))) ** 2
#         adj = (adj - I) / adj.sum(dim=1).unsqueeze(1)
#         return adj
#
#     def generate_word(self, z):
#         outputs = self.generate_text(z).view(-1)
#         word_idx = torch.argmax(F.softmax(outputs, dim=0), dim=0)
#         return word_idx
#
#     def generate_soft_node_embed(self, z, mask=None, temp=1.0, return_entropy=True):
#         """
#         Sample single soft embedded sentence from p(x|z) and temperature.
#         Soft embeddings are calculated as weighted average of word_emb
#         according to p(x|z).
#         """
#
#         outputs = self.generate_text(z)
#         outputs = outputs @ self.word_emb.weight.t()
#         if mask is not None:
#             outputs = outputs.masked_fill(mask.unsqueeze(1) == 0, -float('inf'))
#
#         # Sample softmax with temperature
#         y = F.softmax(outputs / temp, dim=2)
#
#         # Take expectation of embedding given output prob -> soft embedding
#         emb = torch.matmul(y, self.word_emb.weight)
#
#         if return_entropy:
#             # Calculate entropy for penalty
#             entropy = torch.distributions.Categorical(probs=y).entropy().mean()
#             # cost = self.estimate_cost(y)
#             cost = torch.FloatTensor([0.0])
#             return emb, entropy, cost
#         else:
#             return emb
#
#     def estimate_cost(self, probs, sample_size=10):
#         sampled = random.sample(self.sampler, sample_size)
#         cost = torch.FloatTensor([0.0]).to(self.device)
#         for x, y in sampled:
#             n1 = probs[:, x, :]
#             n2 = probs[:, y, :]
#             kl = F.kl_div(n1.log(), n2, reduce=False).sum(dim=1)
#             kl = F.hardtanh(kl, max_val=10, min_val=-10).sum() / kl.size(0)
#             cost -= kl
#             # if not torch.isinf(kl):
#             #     cost -= kl
#         return cost / sample_size
#
#     def train_model(self, args, dataset):
#         """train the whole network"""
#         self.train()
#
#         kl_weight = args.kl_weight
#         alpha = args.alpha
#         beta = args.beta
#         gamma = args.gamma
#
#         trainer_D = optim.Adam(self.discriminator_params, lr=args.lr)
#         trainer_G = optim.Adam(self.decoder_params, lr=args.lr)
#         trainer_E = optim.Adam(self.encoder_params, lr=args.lr)
#
#         criterion_bce = nn.BCEWithLogitsLoss().to(self.device)
#         ones_label = Variable(torch.ones((args.batch_size, 1))).to(self.device)
#         zeros_label = Variable(torch.zeros((args.batch_size, 1))).to(self.device)
#
#         patience = 0
#         best_val_acc, best_test_acc, best_iter = 0, 0, 0
#         list_loss_D, list_loss_G, list_loss_E, list_loss_recon, list_loss_kl, list_loss_cls, list_entropy, \
#         list_cost = [], [], [], [], [], [], [], []
#         train_start_time = time.time()
#         train_iter = dataset.build_train_iter(args.batch_size, self.device)
#         val_iter, test_iter = dataset.build_test_iter(args.batch_size, self.device, shuffle=False)
#
#         val_acc, test_acc = self.train_model_classifier(args, dataset)
#         print(f'Initial Eval: Valid Acc: {val_acc:.4f}; Test Acc: {test_acc:.4f}')
#
#         for epoch in range(1, args.epochs):
#             current_temp = temp(epoch)
#             for subdata in [train_iter, val_iter, test_iter]:
#                 for batch in iter(subdata):
#                     inputs, labels = batch.text, batch.label
#                     mask = dataset.create_mask(inputs.transpose(1, 0), device=self.device)
#                     length, batch_size = inputs.size()
#                     recon_loss, kl_loss, entropy, cost, z0 = self.forward(inputs, mask=mask, temp=current_temp)
#                     loss_vae = recon_loss + kl_weight * kl_loss + gamma * entropy #+ cost
#                     list_loss_recon.append(recon_loss.item())
#                     list_loss_kl.append(kl_loss.item())
#                     list_entropy.append(entropy.item())
#                     list_cost.append(cost.item())
#
#                     """ Update discriminator """
#                     # y_disc_real = self.forward_discriminator(inputs.transpose(1, 0))
#                     # loss_D_real = criterion_bce(y_disc_real, ones_label[:batch_size])
#                     #
#                     # # x_recon = self.generate_sentences(batch_size, length, temp=current_temp, z=z0, mask=mask)
#                     # # y_disc_fake_recon = self.forward_discriminator(x_recon)
#                     # x_recon_embed = self.generate_soft_embed(batch_size, length, temp=current_temp, z=z0, mask=mask)
#                     # y_disc_fake_recon = self.forward_discriminator_embed(x_recon_embed)
#                     # loss_D_recon = criterion_bce(y_disc_fake_recon, zeros_label[:batch_size])
#                     #
#                     # # x_gen = self.generate_sentences(batch_size, length, temp=current_temp)
#                     # # y_disc_fake_sample = self.forward_discriminator(x_gen)
#                     # x_gen_embed = self.generate_soft_embed(batch_size, length, temp=current_temp)
#                     # y_disc_fake_sample = self.forward_discriminator_embed(x_gen_embed)
#                     # loss_D_sample = criterion_bce(y_disc_fake_sample, zeros_label[:batch_size])
#                     #
#                     # loss_D = loss_D_real + loss_D_recon + loss_D_sample
#                     # list_loss_D.append(loss_D.item())
#                     # trainer_D.zero_grad()
#                     # loss_D.backward(retain_graph=True)
#                     # # grad_norm = torch.nn.utils.clip_grad_norm_(self.discriminator_params, 5)
#                     # trainer_D.step()
#
#                     """ Update generator  """
#                     # y_disc_real = self.forward_discriminator(inputs.transpose(1, 0))
#                     # loss_D_real = criterion_bce(y_disc_real, zeros_label[:batch_size])
#                     #
#                     # x_recon_embed = self.generate_soft_embed(batch_size, length, temp=current_temp, z=z0, mask=mask)
#                     # y_disc_fake_recon = self.forward_discriminator_embed(x_recon_embed)
#                     # loss_D_recon = criterion_bce(y_disc_fake_recon, ones_label[:batch_size])
#                     #
#                     # x_gen_embed = self.generate_soft_embed(batch_size, length, temp=current_temp)
#                     # y_disc_fake_sample = self.forward_discriminator_embed(x_gen_embed)
#                     # loss_D_sample = criterion_bce(y_disc_fake_sample, ones_label[:batch_size])
#                     # loss_D = (loss_D_real + loss_D_recon + loss_D_sample)
#
#                     # loss_G = alpha * loss_D + loss_vae
#                     loss_G = loss_vae
#                     # list_loss_G.append(loss_D.item())
#                     trainer_G.zero_grad()
#                     loss_G.backward(retain_graph=True)
#                     # grad_norm = torch.nn.utils.clip_grad_norm_(self.decoder_params, 5)
#                     trainer_G.step()
#
#                     """ Update encoder  """
#                     # loss_E = beta * loss_D + loss_vae
#                     loss_E = loss_vae
#                     # list_loss_E.append(-loss_D.item())
#                     trainer_E.zero_grad()
#                     loss_E.backward(retain_graph=True)
#                     # grad_norm = torch.nn.utils.clip_grad_norm_(self.encoder_params, 5)
#                     trainer_E.step()
#
#             if epoch % args.log_every == 0:
#                 duration = time.time() - train_start_time
#                 avr_loss_D = np.mean(list_loss_D)
#                 avr_loss_G = np.mean(list_loss_G)
#                 # avr_loss_E = np.mean(list_loss_E)
#                 avr_loss_recon = np.mean(list_loss_recon)
#                 avr_loss_kl = np.mean(list_loss_kl)
#                 avr_entropy = np.mean(list_entropy)
#                 avr_cost = np.mean(list_cost)
#                 list_loss_D, list_loss_G, list_loss_E, list_loss_recon, list_loss_kl, list_loss_cls, list_entropy, \
#                 list_cost = [], [], [], [], [], [], [], []
#                 print(f'Epoch-{epoch}; temp: {current_temp:.4f}; loss_D: {avr_loss_D:.4f}; loss_G: {avr_loss_G:.4f}; '
#                       f'loss_recon: {avr_loss_recon:.4f}; loss_kl: {avr_loss_kl:.4f}; cost: {avr_cost:.4f}; '
#                       f'entropy: {avr_entropy:.4f}; duration: {duration:.2f}')
#
#                 val_acc, test_acc = self.train_model_classifier(args, dataset, current_temp)
#
#                 # z = self.sample_z_prior(1)
#                 # sample_idxs = [i for i in self.sample_sentence(z, None, length=50)[0] if i != self.PAD_IDX]
#                 # sample_sent = dataset.idxs2sentence(sample_idxs)
#                 # print(f'Valid Acc: {val_acc:.4f}; Test Acc: {test_acc:.4f}; Sample: "{sample_sent}"')
#                 print(f'Valid Acc: {val_acc:.4f}; Test Acc: {test_acc:.4f}')
#                 # print(f'Sample: "{sample_sent}"')
#
#                 if val_acc > best_val_acc:
#                     best_val_acc = val_acc
#                     best_test_acc = test_acc
#                     best_iter = epoch
#                     patience = 0
#                 else:
#                     if epoch > args.minimal_epoch:
#                         patience += args.log_every
#
#                 if args.early_stop and patience > args.patience:
#                     break
#
#         print(f'Best Valid Acc: {best_val_acc:.4f}; Best Test Acc: {best_test_acc:.4f} @ epoch {best_iter}')
#         return best_val_acc, best_test_acc
#
#     def train_model_classifier(self, args, dataset, temp=1.0):
#         """train the whole network"""
#         self.train()
#
#         self.classifier_[0].reset_parameters()
#         for i in self.classifier_[1]:
#             if hasattr(i, 'reset_parameters'):
#                 i.reset_parameters()
#         trainer_C = optim.Adam(filter(lambda p: p.requires_grad, self.classifier_.parameters()), lr=args.lr)
#         criterion_cls = nn.CrossEntropyLoss().to(self.device)
#
#         patience = 0
#         log_every = 5
#         list_loss_cls = []
#         best_val_acc, best_test_acc, best_iter = 0, 0, 0
#         train_start_time = time.time()
#
#         train_iter = dataset.build_train_iter(args.eval_batch_size, self.device)
#
#         for epoch in range(1, args.eval_epochs):
#             for batch in iter(train_iter):
#
#                 inputs, labels = batch.text, batch.label
#                 mask = dataset.create_mask(inputs.transpose(1, 0), device=self.device)
#
#                 """ Update classifier """
#                 outputs = self.forward_classifier(inputs, mask, temp)
#                 loss_cls = criterion_cls(outputs, labels)
#                 list_loss_cls.append(loss_cls.item())
#
#                 trainer_C.zero_grad()
#                 loss_cls.backward(retain_graph=True)
#                 trainer_C.step()
#
#             if epoch % log_every == 0:
#                 duration = time.time() - train_start_time
#                 avr_loss_cls = np.mean(list_loss_cls)
#                 list_loss_cls = []
#
#                 val_iter, test_iter = dataset.build_test_iter(args.batch_size, self.device)
#                 val_acc = self.test(dataset, val_iter)
#                 test_acc = self.test(dataset, test_iter)
#
#                 if val_acc > best_val_acc:
#                     best_val_acc = val_acc
#                     best_test_acc = test_acc
#                     best_iter = epoch
#                     patience = 0
#                 else:
#                     if epoch > args.eval_minimal_epoch:
#                         patience += log_every
#
#                 if args.eval_early_stop and patience > args.eval_patience:
#                     break
#
#             # if epoch % 10 == 0:
#             #     print(f'Epoch-{epoch}; loss_cls: {avr_loss_cls:.4f}; Valid Acc: {val_acc:.4f}; Test Acc: {test_acc:.4f};'
#             #           f' duration: {duration:.2f}')
#
#         # print(f'Best Valid Acc: {best_val_acc:.4f}; Best Test Acc: {best_test_acc:.4f} @ iter {best_iter}')
#         return best_val_acc, best_test_acc
#
#     def test(self, dataset, data_iter):
#         self.eval()
#         total_accuracy = []
#         with torch.no_grad():
#             for batch in iter(data_iter):
#                 inputs, labels = batch.text, batch.label
#                 mask = dataset.create_mask(inputs.transpose(1, 0), device=self.device)
#                 outputs = self.forward_classifier(inputs, mask)
#                 accuracy = (outputs.argmax(1) == labels).float().mean().item()
#                 total_accuracy.append(accuracy)
#         acc = sum(total_accuracy) / len(total_accuracy)
#         self.train()
#         return acc
