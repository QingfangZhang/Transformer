import gc
import torch
from torch import nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import collections
import os
import re
from operator import itemgetter
import jieba
import json
from torchtext.data.metrics import bleu_score
import matplotlib


matplotlib.use('TkAgg')


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_token=None):
        if tokens is None:
            tokens = []
        else:
            if isinstance(tokens[0], list):
                tokens = [token for s in tokens for token in s]
        sorted_token = sorted(collections.Counter(tokens).items(), key=itemgetter(1), reverse=True)
        if reserved_token is None:
            reserved_token = []
        if 'UNK' not in reserved_token:
            reserved_token = ['UNK'] + reserved_token
        self.token_to_idx = {token: idx for idx, token in enumerate(reserved_token)}
        self.idx_to_token = {idx: token for idx, token in enumerate(reserved_token)}
        for token, freq in sorted_token:
            if freq < min_freq:
                break
            else:
                if token not in self.token_to_idx.keys():
                    self.token_to_idx[token] = len(self.token_to_idx)
                    self.idx_to_token[len(self.idx_to_token)] = token

    def __len__(self):
        return len(self.token_to_idx)

    def to_idx(self, token):
        """input can only be a word or a sentence"""
        if isinstance(token, list):
            return [self.token_to_idx.get(i, self.token_to_idx['UNK']) for i in token]
        else:
            return self.token_to_idx.get(token, self.token_to_idx['UNK'])

    def to_token(self, idx):
        """input can only be a number or a list of numbers"""
        if isinstance(idx, list):  # for a sentence
            return [self.idx_to_token[i] for i in idx]
        elif isinstance(idx, torch.Tensor):
            return [self.idx_to_token[i.item()] for i in idx]
        else:  # for a single word
            return self.idx_to_token[idx]


def get_vocab():
    filepath = 'D:\\Deeplearning\\\dataset\En-Ch_translation_dataset\\ai_challenger_translation_train_20170904\\' \
               'translation_train_data_20170904\\'

    if os.path.exists('en_vocab'):
        en_vocab = Vocab()
        with open('en_vocab', mode='r', encoding='utf-8') as f:
            en_vocab.token_to_idx = json.load(f)
        en_vocab.idx_to_token = {idx: token for idx, token in enumerate(en_vocab.token_to_idx)}
    else:
        if os.path.exists('en_tokens'):
            en_tokens = torch.load('en_tokens')
        else:
            with open(filepath + 'train.en', mode='r', encoding='utf-8') as f:
                en = f.readlines()
            en_tokens = [en_tokenize(line) for line in en]
            torch.save(en_tokens, 'en_tokens')
        en_tokens = remove_zero_and_en_line(en_tokens)
        en_vocab = Vocab(en_tokens, min_freq=10, reserved_token=['BOS', 'PAD', 'UNK', 'EOS'])
        with open('en_vocab', mode='w', encoding='utf-8') as f:
            json.dump(en_vocab.token_to_idx, f)
        del en_tokens
        gc.collect()

    if os.path.exists('ch_vocab'):
        ch_vocab = Vocab()
        with open('ch_vocab', mode='r', encoding='utf-8') as f:
            ch_vocab.token_to_idx = json.load(f)
        ch_vocab.idx_to_token = {idx: token for idx, token in enumerate(ch_vocab.token_to_idx)}
    else:
        if os.path.exists('ch_tokens'):
            ch_tokens = torch.load('ch_tokens')
        else:
            with open(filepath + 'train.zh', mode='r', encoding='utf-8') as f:
                ch = f.readlines()
            ch_tokens = [ch_tokenize(line) for line in ch]
            torch.save(ch_tokens, 'ch_tokens')
        ch_tokens = remove_zero_and_en_line(ch_tokens)
        ch_vocab = Vocab(ch_tokens, min_freq=10, reserved_token=['BOS', 'PAD', 'UNK', 'EOS'])
        with open('ch_vocab', mode='w', encoding='utf-8') as f:
            json.dump(ch_vocab.token_to_idx, f)
        del ch_tokens
        gc.collect()

    return en_vocab, ch_vocab


def add_space_around_punctuation(text):
    punctuation = ''',./<>?;:"[{]}\\|`~!@#$%^&*()-_=+'''
    for p in punctuation:
        text = text.replace(p, ' ' + p + ' ')
    return text


def has_chinese(text):
    # 使用正则表达式匹配是否存在中文字符
    return bool(re.search('[\u4e00-\u9fa5]', text))


def has_english(text):
    # 使用正则表达式匹配是否存在英文字母
    return bool(re.search('[a-zA-Z]', text))


def remove_zero_and_en_line(data):
    # len_en = [len(line) for line in en]
    # len_ch = [len(line) for line in ch]
    # en_zero_idx = [i for i, x in enumerate(len_en) if x == 0]
    # ch_zero_idx = [i for i, x in enumerate(len_ch) if x == 0]
    en_zero_idx = [5338753, 5338754, 5338755, 5338756, 5338757, 5338758, 5338759, 5338760, 5338761]
    ch_zero_idx = [4848496, 5338753, 9417189]
    # ch_idx = [i for i, s in enumerate(en) if has_chinese(s)]
    # en_idx = [i for i, s in enumerate(ch) if has_english(s)]
    en_idx = torch.load('ch_has_en_idx')
    idx = set(en_zero_idx + ch_zero_idx + en_idx)  # set is faster
    # unique_indexes = set(idx)
    # sorted_indexes = sorted(unique_indexes, reverse=True)
    # for i in sorted_indexes:
    #     removed_element = data.pop(i)
    data = [x for i, x in enumerate(data) if i not in idx]
    return data


def en_tokenize(data):
    return add_space_around_punctuation(data).lower().split()


def ch_tokenize(data):
    return jieba.lcut(data.strip('\n').strip())


def truncate_or_pad(tokens, vocab, num_steps):
    if len(tokens) >= num_steps:
        tokens_idx = vocab.to_idx(tokens[:num_steps])
    else:
        tokens_idx = vocab.to_idx(tokens) + vocab.to_idx(['PAD']) * (num_steps - len(tokens))
    return tokens_idx


class EnChTranslationDataSet(torch.utils.data.Dataset):
    def __init__(self, en_vocab, ch_vocab, num_steps):
        super(EnChTranslationDataSet, self).__init__()
        filepath = 'D:\\Deeplearning\\\dataset\En-Ch_translation_dataset\\ai_challenger_translation_train_20170904\\' \
                   'translation_train_data_20170904\\'
        with open(filepath + 'train.en', mode='r', encoding='utf-8') as f:
            self.en = remove_zero_and_en_line(f.readlines())
        with open(filepath + 'train.zh', mode='r', encoding='utf-8') as f:
            self.ch = remove_zero_and_en_line(f.readlines())
        self.en_vocab = en_vocab
        self.ch_vocab = ch_vocab
        self.num_steps = num_steps

    def __getitem__(self, item):
        en_tokens = en_tokenize(self.en[item]) + ['EOS']
        en_valid_len = torch.tensor(len(en_tokens), dtype=torch.long)
        en_tokens_idx = torch.tensor(truncate_or_pad(en_tokens, self.en_vocab, self.num_steps), dtype=torch.long)
        ch_tokens = ch_tokenize(self.ch[item]) + ['EOS']
        ch_valid_len = torch.tensor(len(ch_tokens), dtype=torch.long)
        ch_tokens_idx = torch.tensor(truncate_or_pad(ch_tokens, self.ch_vocab, self.num_steps), dtype=torch.long)
        return en_tokens_idx, en_valid_len, ch_tokens_idx, ch_valid_len

    def __len__(self):
        return len(self.en)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, queries, keys, values, mask):
        """
        calculate the scaled dot-product attention
        :param queries: shape(batch_size*head_num, query_num, query_dim/head_num)
        :param keys: shape(batch_size*head_num, key_num, key_dim/head_num)
        :param values: shape(batch_size*head_num, value_num, value_dim/head_num)
        :param mask: mask the 'PAD' in the input
        :return: scaled dot product attention, shape(batch_size*head_num, query_num, value_dim/head_num)
        """
        assert queries.shape[-1] == keys.shape[-1]
        d = queries.shape[-1]
        sdp = torch.matmul(queries, torch.transpose(keys, -2, -1)) / math.sqrt(d)
        if mask is not None:
            mask = mask.to(queries.device)
            assert sdp.shape == mask.shape
            sdp = sdp.masked_fill(mask, -1e9)
        attention_weight = F.softmax(sdp, dim=-1)
        return torch.matmul(attention_weight, values)


def transpose_in(a, head_num):
    b = a.reshape(a.shape[0], a.shape[1], head_num, -1).permute(0, 2, 1, 3)
    return b.reshape(a.shape[0] * head_num, a.shape[1], -1)


def transpose_out(a, head_num):
    b = a.reshape(-1, head_num, a.shape[1], a.shape[2]).permute(0, 2, 1, 3)
    return b.reshape(-1, a.shape[1], head_num * a.shape[2])


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, query_dim, key_dim, value_dim, head_num):
        super().__init__()
        self.head_num = head_num
        self.Wq = nn.Linear(hidden_dim, query_dim)
        self.Wk = nn.Linear(hidden_dim, key_dim)
        self.Wv = nn.Linear(hidden_dim, value_dim)
        self.atten = ScaledDotProductAttention()
        self.Wo = nn.Linear(value_dim, hidden_dim)

    def forward(self, queries, keys, values, mask):
        q = transpose_in(self.Wq(queries), self.head_num)
        k = transpose_in(self.Wk(keys), self.head_num)
        v = transpose_in(self.Wv(values), self.head_num)
        head_out = transpose_out(self.atten(q, k, v, mask), self.head_num)
        return self.Wo(head_out)


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_in_number, ffn_hidden_number, ffn_out_number):
        super(PositionWiseFFN, self).__init__()
        self.linear1 = nn.Linear(ffn_in_number, ffn_hidden_number)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ffn_hidden_number, ffn_out_number)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout_prob):
        super(AddNorm, self).__init__()
        self.layernorm = nn.LayerNorm(norm_shape)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, y):
        return self.layernorm(x + self.dropout(y))


class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim, query_dim, key_dim, value_dim, head_num, ffn_in_number, ffn_hidden_number,
                 ffn_out_number, dropout_prob):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(hidden_dim, query_dim, key_dim, value_dim, head_num)
        # self.addnorm1 = AddNorm(hidden_dim, dropout_prob)   # norm_shape = hidden_dim
        self.ffn = PositionWiseFFN(ffn_in_number, ffn_hidden_number, ffn_out_number)
        # self.addnorm2 = AddNorm(ffn_out_number, dropout_prob)   # norm_shape = ffn_out_number
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_prob)

    def forward(self, x, mask):
        # mid = self.addnorm1(x, self.attention(x, x, x, mask))
        # return self.addnorm2(mid, self.ffn(mid))
        x1 = self.layernorm1(x)
        mid = x + self.dropout1(self.attention(x1, x1, x1, mask))
        return mid + self.dropout2(self.ffn(self.layernorm2(mid)))


def position_encoding(max_len, hidden_dim):
    PE = torch.zeros(max_len, hidden_dim)
    i = torch.arange(0, hidden_dim, 2)
    e = torch.pow(10000, i / hidden_dim).reshape(1, -1)
    f = torch.arange(max_len).reshape(-1, 1)
    PE[:, i] = torch.sin(f / e)
    PE[:, i + 1] = torch.cos(f / e)
    return PE.unsqueeze(dim=0)


class TransformerEncoder(nn.Module):
    def __init__(self, vocab, hidden_dim, encoder_num, query_dim, key_dim, value_dim, head_num, ffn_in_number,
                 ffn_hidden_number, ffn_out_number, max_len, dropout_prob):
        super(TransformerEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.embedding = nn.Embedding(len(vocab), hidden_dim)
        self.pos_encoding = position_encoding(max_len, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.encoder_blocks = nn.Sequential()
        for i in range(encoder_num):
            self.encoder_blocks.add_module(f'EncoderBlock{i}', EncoderBlock(hidden_dim, query_dim, key_dim, value_dim,
                                                                            head_num, ffn_in_number, ffn_hidden_number,
                                                                            ffn_out_number, dropout_prob))

    def forward(self, x, enc_valid_len):
        x = self.embedding(x) * math.sqrt(self.hidden_dim) + self.pos_encoding[:, :x.shape[1], :].to(x.device)
        x = self.dropout(x)
        enc_mask = get_enc_mask(enc_valid_len, self.head_num, x.shape[0], x.shape[1], x.shape[1])
        for encoder_blk in self.encoder_blocks:
            encoder_output = encoder_blk(x, enc_mask)
        return encoder_output


class EncoderDecoder(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, hidden_dim, query_dim, key_dim, value_dim, head_num, ffn_in_number,
                 ffn_hidden_number, ffn_out_number, max_len, dropout_prob):
        super(EncoderDecoder, self).__init__()
        encoder_num = 6
        decoder_num = 6
        self.encoder = TransformerEncoder(src_vocab, hidden_dim, encoder_num, query_dim, key_dim, value_dim, head_num,
                                          ffn_in_number, ffn_hidden_number, ffn_out_number, max_len, dropout_prob)
        self.decoder = TransformerDecoder(tgt_vocab, hidden_dim, decoder_num, query_dim, key_dim, value_dim, head_num,
                                          ffn_in_number, ffn_hidden_number, ffn_out_number, max_len, dropout_prob)

    def forward(self, x, dec_x, enc_valid_len):
        enc_valid_len = enc_valid_len.reshape(-1, 1, 1)
        encoder_output = self.encoder(x, enc_valid_len)
        state = self.decoder.init_state()
        return self.decoder(encoder_output, dec_x, enc_valid_len, state)


class TransformerDecoder(nn.Module):
    def __init__(self, vocab, hidden_dim, decoder_num, query_dim, key_dim, value_dim, head_num, ffn_in_number,
                 ffn_hidden_number, ffn_out_number, max_len, dropout_prob):
        super(TransformerDecoder, self).__init__()
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.decoder_num = decoder_num
        self.embedding = nn.Embedding(len(vocab), hidden_dim)
        self.pos_encoding = position_encoding(max_len, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.decoder_blocks = nn.Sequential()
        for i in range(decoder_num):
            self.decoder_blocks.add_module(f'DecoderBlock{i}', DecoderBlock(hidden_dim, query_dim, key_dim, value_dim,
                                                                            head_num, ffn_in_number, ffn_hidden_number,
                                                                            ffn_out_number, i, dropout_prob))
        self.linear = nn.Linear(hidden_dim, len(vocab))

    def init_state(self):
        state = [None] * self.decoder_num
        return state

    def forward(self, encoder_output, dec_x, enc_valid_len, state):
        x = self.embedding(dec_x) * math.sqrt(self.hidden_dim) + self.pos_encoding[:, :dec_x.shape[1], :].to(
            dec_x.device)
        x = self.dropout(x)
        memory_mask = get_enc_mask(enc_valid_len, self.head_num, dec_x.shape[0], dec_x.shape[1],
                                   encoder_output.shape[1])
        dec_mask = get_dec_mask(self.head_num, x.shape[0], x.shape[1], training=self.training)
        for decoder_blk in self.decoder_blocks:
            x, state = decoder_blk(encoder_output, x, memory_mask, dec_mask, state)
        return self.linear(x), state


def get_enc_mask(enc_valid_len, head_num, batch_size, query_num, key_num):
    """

    :param enc_valid_len:
    :param head_num:
    :param batch_size:
    :param query_num:  for encoder self_attention, query_num = num_steps, for encoder-decoder memory attention, query_num = decoder input number
    :param key_num: for encoder self attention, key_num = num_steps, for encoder-decoder memory attention, key_num = encoder input number
    :return:
    """
    if enc_valid_len.dim() != 3:
        enc_valid_len = enc_valid_len.reshape(-1, 1, 1)
    enc_valid_len = torch.repeat_interleave(enc_valid_len, repeats=head_num, dim=0)
    enc_mask = torch.arange(1, key_num + 1).reshape(1, -1).repeat(batch_size * head_num, query_num, 1).to(
        enc_valid_len.device) \
               > torch.repeat_interleave(torch.repeat_interleave(enc_valid_len, key_num, dim=-1), query_num, dim=1)
    return enc_mask


def get_dec_mask(head_num, batch_size, num_steps, training):
    if training:
        dec_mask = torch.triu(torch.ones(num_steps, num_steps), diagonal=1).type(torch.bool).repeat(
            batch_size * head_num, 1, 1)
    else:
        dec_mask = None
    return dec_mask


class DecoderBlock(nn.Module):
    def __init__(self, hidden_dim, query_dim, key_dim, value_dim, head_num, ffn_in_number, ffn_hidden_number,
                 ffn_out_number, i, dropout_prob):
        super(DecoderBlock, self).__init__()
        self.attention1 = MultiHeadAttention(hidden_dim, query_dim, key_dim, value_dim, head_num)
        # self.addnorm1 = AddNorm(hidden_dim, dropout_prob)
        self.attention2 = MultiHeadAttention(hidden_dim, query_dim, key_dim, value_dim, head_num)
        # self.addnorm2 = AddNorm(hidden_dim, dropout_prob)
        self.ffn = PositionWiseFFN(ffn_in_number, ffn_hidden_number, ffn_out_number)
        # self.addnorm3 = AddNorm(ffn_out_number, dropout_prob)
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.layernorm3 = nn.LayerNorm(hidden_dim)
        self.layernorm4 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.i = i

    def forward(self, encoder_output, x, memory_mask, dec_mask, state):
        if self.training:
            state[self.i] = x
        else:
            if state[self.i] is None:
                state[self.i] = x
            else:
                state[self.i] = torch.cat((state[self.i], x), dim=1)

        # mid1 = self.addnorm1(x, self.attention1(x, state[self.i], state[self.i], dec_mask))
        # mid2 = self.addnorm2(mid1, self.attention2(mid1, encoder_output, encoder_output, memory_mask))
        # mid3 = self.addnorm3(mid2, self.ffn(mid2))
        x1 = self.layernorm1(x)
        mid1 = x + self.dropout1(self.attention1(x1, x1, x1, dec_mask))  # 注意这里有误，应该是state
        enc = self.layernorm4(encoder_output)
        mid2 = mid1 + self.dropout2(self.attention2(self.layernorm2(mid1), enc, enc, memory_mask))
        mid3 = mid2 + self.dropout3(self.ffn(self.layernorm3(mid2)))
        return mid3, state


def MaskedCrossEntropyLoss(y_hat, y, dec_valid_len):
    mask = torch.stack([F.pad(torch.ones(L), (0, y_hat.shape[1] - L)) for L in dec_valid_len]).to(y.device)
    loss = F.cross_entropy(y_hat.transpose(1, 2), y, reduction='none') * mask
    return loss.sum() / torch.sum(dec_valid_len)


def get_device():
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
    else:
        devices = [torch.device('cpu')]
    return devices


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def SaveCheckpoint(checkpoint_path, batch_id, net, train_loss, bleu_list, lr_list):
    checkpoint = {
        'batch_id': batch_id,
        'model_state_dict': net.state_dict(),
        'train_loss': train_loss,
        'bleu_list': bleu_list,
        'lr_list': lr_list,
    }
    torch.save(checkpoint, checkpoint_path)


def train_transformer(train_iter, net, model_dim, batch_num, en_vocab, ch_vocab, devices, resume):
    checkpoint_path = 'checkpoint'
    if resume:
        checkpoint = torch.load(checkpoint_path)
        batch_id = checkpoint['batch_id'] + 1
        net.load_state_dict(checkpoint['model_state_dict'])
        train_loss = checkpoint['train_loss']
        bleu_list = checkpoint['bleu_list']
        lr_list = checkpoint['lr_list']
    else:
        batch_id = 1
        net.apply(init_weights)
        train_loss, bleu_list, lr_list = [], [], []

    if len(devices) > 1:
        nn.DataParallel(net.cuda(), device_ids=devices)
    else:
        net.to(devices[0])
    warmup_steps = 4000
    lr_schedule = lambda batch_id: model_dim ** (-0.5) * min(batch_id ** (-0.5), batch_id * (warmup_steps ** (-1.5)))
    # lr_schedule = lambda batch_id: (5e-4 / (warmup_steps ** (-1.5))) * min(batch_id ** (-0.5), batch_id * (warmup_steps ** (-1.5)))
    optimizer = torch.optim.Adam(net.parameters(), lr=lr_schedule(batch_id), betas=(0.9, 0.98), eps=1e-09)
    lr_list.append(optimizer.param_groups[0]['lr'])
    keep_training = True
    net.train()
    while batch_id < batch_num and keep_training:
        for en_tokens_idx, en_valid_len, ch_tokens_idx, ch_valid_len in train_iter:
            bos = torch.ones((ch_tokens_idx.shape[0], 1), dtype=torch.long) * ch_vocab.to_idx('BOS')
            dec_x = torch.cat((bos, ch_tokens_idx[:, :-1]), dim=1).to(devices[0])
            enc_x, enc_valid_len = en_tokens_idx.to(devices[0]), en_valid_len.to(devices[0])
            dec_valid_len, ch_tokens_idx = ch_valid_len.to(devices[0]), ch_tokens_idx.to(devices[0])
            optimizer.zero_grad()
            y_hat, _ = net(enc_x, dec_x, enc_valid_len)
            l = MaskedCrossEntropyLoss(y_hat, ch_tokens_idx, dec_valid_len)
            l.backward()
            optimizer.step()
            train_loss.append(l.detach().cpu())
            pred = torch.argmax(y_hat, dim=2).squeeze()
            #             bleu_list.append(calculate_bleu(pred, ch_tokens_idx, ch_vocab))
            if batch_id % 100 == 0:
                show_examples(enc_x, pred, ch_tokens_idx, en_vocab, ch_vocab, batch_id)
                print(f'train_loss: {train_loss[-1]}')
            del enc_x, dec_x, dec_valid_len, enc_valid_len, y_hat, l, pred
            torch.cuda.empty_cache()
            if batch_id % 1000 == 0:
                SaveCheckpoint(checkpoint_path, batch_id, net, train_loss, bleu_list, lr_list)
            if batch_id >= batch_num:
                keep_training = False
                break
            batch_id += 1
            optimizer.param_groups[0]['lr'] = lr_schedule(batch_id)
            lr_list.append(optimizer.param_groups[0]['lr'])

    plt.figure()
    plt.plot(train_loss, label='train_loss')
    plt.legend()
    #     plt.figure()
    #     plt.plot(bleu_list, label='bleu_score')
    #     plt.legend()
    plt.figure()
    plt.plot(lr_list, label='learning rate')
    plt.legend()
    plt.show()


def calculate_bleu(pred, true_y, ch_vocab):
    if len(pred) != 0:
        if isinstance(pred[0], str):  # use in show_examples, compare two sentences
            return bleu_score([pred], [[true_y]])
        if isinstance(pred, torch.Tensor):  # use in train_transformer, compare two large corpus
            ch_pred = [ch_vocab.to_token(pred[row_i, :]) for row_i in range(pred.shape[0])]
            ch_pred = [[item for item in j if item != 'PAD' and item != 'EOS'] for j in ch_pred]
            ch = [ch_vocab.to_token(true_y[row_i, :]) for row_i in range(true_y.shape[0])]
            ch = [[item for item in j if item != 'PAD' and item != 'EOS'] for j in ch]
            ch = [[i] for i in ch]
            return bleu_score(ch_pred, ch)  # reference 必须有三个括号, candidate 必须有两个括号
    else:
        return 0.0


def predict(en_vocab, ch_vocab, num_steps, net, devices):
    while True:
        en = input('English sentence: ')
        if en != 'stop':
            en_tokens = en_tokenize(en) + ['EOS']
            en_valid_len = torch.tensor(len(en_tokens), dtype=torch.long).reshape(-1, 1, 1).to(devices[0])
            en_tokens_idx = torch.unsqueeze(
                torch.tensor(truncate_or_pad(en_tokens, en_vocab, num_steps), dtype=torch.long), dim=0).to(devices[0])
            dec_x = torch.unsqueeze(torch.tensor([ch_vocab.to_idx('BOS')], dtype=torch.long, device=devices[0]), dim=0)
            net.eval()
            pred = []
            encoder_output = net.encoder(en_tokens_idx, en_valid_len)
            state = net.decoder.init_state()
            for i in range(num_steps):
                y_hat, state = net.decoder(encoder_output, dec_x, en_valid_len, state)
                dec_x = torch.argmax(y_hat, dim=2)
                if ch_vocab.to_token(dec_x.squeeze().cpu().item()) == 'EOS':
                    break
                pred.append(dec_x.squeeze().cpu().item())
            ch_pred = ch_vocab.to_token(pred)
            ch_pred = [item for item in ch_pred if item != 'PAD' and item != 'EOS']
            print(f"model input: {en}")
            print(f"model output: {''.join(ch_pred)}")
        else:
            break


def show_examples(enc, pred, raw_y, en_vocab, ch_vocab, batch_id, example_num=[0, 1]):
    print('=' * 30)
    print(f'batch {batch_id}')
    for i in example_num:
        en = en_vocab.to_token(enc[i])
        ch_pred = ch_vocab.to_token(pred[i])
        ch = ch_vocab.to_token(raw_y[i])
        en = [item for item in en if item != 'PAD' and item != 'EOS']
        # ch_pred = [item for item in ch_pred if item != 'PAD' and item != 'EOS']
        ch = [item for item in ch if item != 'PAD' and item != 'EOS']
        print(f"model input: {' '.join(en)}")
        print(f"model output: {''.join(ch_pred)}")
        print(f"true translation: {''.join(ch)}")
        # print(f"bleu score: {calculate_bleu(ch_pred, ch, ch_vocab)}")


if __name__ == '__main__':
    batch_size = 32
    num_steps = 48
    model_dim = 512
    batch_num = 100000
    resume = True

    en_vocab, ch_vocab = get_vocab()
    train_set = EnChTranslationDataSet(en_vocab, ch_vocab, num_steps=num_steps)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    net = EncoderDecoder(src_vocab=en_vocab, tgt_vocab=ch_vocab, hidden_dim=model_dim, query_dim=model_dim,
                         key_dim=model_dim, value_dim=model_dim, head_num=8, ffn_in_number=model_dim,
                         ffn_hidden_number=2048, ffn_out_number=model_dim, max_len=128, dropout_prob=0.1)

    devices = get_device()
    train_transformer(train_iter, net, model_dim, batch_num, en_vocab, ch_vocab, devices, resume)

    predict(en_vocab, ch_vocab, num_steps, net, devices)

    # for name, param in net.state_dict().items():
    #     print(name, ": ", param.data)

