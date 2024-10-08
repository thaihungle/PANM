import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class Encoder(nn.Module):
	def __init__(self, input_size, embed_size, hidden_size,
				 n_layers=1, dropout=0.5, embedded=False):
		super(Encoder, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.embed_size = embed_size
		self.embedded = embedded
		self.embed = nn.Embedding(input_size, embed_size)
		self.gru = nn.GRU(embed_size, hidden_size, n_layers,
						  dropout=dropout, bidirectional=True)

	def forward(self, src, hidden=None):
		if not self.embedded:
			embedded = self.embed(src)
		else:
			embedded = src
		outputs, hidden = self.gru(embedded, hidden)
		# sum bidirectional outputs
		outputs = (outputs[:, :, :self.hidden_size] +
				   outputs[:, :, self.hidden_size:])
		return outputs, hidden


class Attention(nn.Module):
	def __init__(self, hidden_size):
		super(Attention, self).__init__()
		self.hidden_size = hidden_size
		self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
		self.v = nn.Parameter(torch.rand(hidden_size))
		stdv = 1. / math.sqrt(self.v.size(0))
		self.v.data.uniform_(-stdv, stdv)

	def forward(self, hidden, encoder_outputs):
		timestep = encoder_outputs.size(0)
		h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
		encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
		attn_energies = self.score(h, encoder_outputs)
		return F.softmax(attn_energies, dim=1).unsqueeze(1)

	def score(self, hidden, encoder_outputs):
		# [B*T*2H]->[B*T*H]
		energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
		energy = energy.transpose(1, 2)  # [B*H*T]
		v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
		energy = torch.bmm(v, energy)  # [B*1*T]
		return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
	def __init__(self, embed_size, hidden_size, output_size,
				 n_layers=1, dropout=0.2, embedded=False):
		super(Decoder, self).__init__()
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.embedded = embedded
		self.embed = nn.Embedding(output_size, embed_size)
		self.dropout = nn.Dropout(dropout, inplace=True)
		self.attention = Attention(hidden_size)
		self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
						  n_layers, dropout=dropout)
		self.out = nn.Linear(hidden_size * 2, output_size)

	def forward(self, input, last_hidden, encoder_outputs):
		# Get the embedding of the current input word (last output word)
		if not self.embedded:
			embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
			embedded = self.dropout(embedded)
		else:
			embedded = input
		# Calculate attention weights and apply to encoder outputs
		attn_weights = self.attention(last_hidden[-1], encoder_outputs)
		context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
		context = context.transpose(0, 1)  # (1,B,N)
		# Combine embedded input word and attended context, run through RNN
		rnn_input = torch.cat([embedded, context], 2)
		output, hidden = self.gru(rnn_input, last_hidden)
		output = output.squeeze(0)  # (1,B,N) -> (B,N)
		context = context.squeeze(0)
		output = self.out(torch.cat([output, context], 1))
		return output, hidden, attn_weights


class Seq2Seq(nn.Module):
	def __init__(self, in_size, out_size, embed_size, hidden_size, embedded=False, emb_layer=None):
		super(Seq2Seq, self).__init__()
		encoder = Encoder(in_size, embed_size, hidden_size,
					  n_layers=1, dropout=0.5, embedded=embedded)
		decoder = Decoder(embed_size, hidden_size, out_size,
						n_layers=1, dropout=0.5, embedded=embedded)
		self.embedded = embedded
		self.embed_size = embed_size
		self.encoder = encoder
		self.decoder = decoder
		self.emb_layer = emb_layer

	def forward(self, src, target_length, trg=None, teacher_forcing_ratio=-1):
		if self.emb_layer:
			src = self.emb_layer(src)
		max_len = target_length
		batch_size = src.size(1)
		if trg is not None:
			max_len = trg.size(0)
		else:
			if self.embedded:
				trg = Variable(torch.zeros(max_len, batch_size, self.embed_size)).to(src.device)
			else:
				trg = Variable(torch.zeros(max_len, batch_size).long()).to(src.device)
		outputs = []
		encoder_output, hidden = self.encoder(src)
		hidden = hidden[:self.decoder.n_layers]
		output = Variable(trg.data[0, :])  # sos
		for t in range(0, max_len):
			if self.embedded:
				output = output.unsqueeze(0)
			output, hidden, attn_weights = self.decoder(
					output, hidden, encoder_output)
			outputs.append(output)
			if teacher_forcing_ratio!=-1:
				is_teacher = random.random() < teacher_forcing_ratio
			else:
				is_teacher = True
			top1 = output.data.max(1)[1]
			if not is_teacher and self.embedded:
				top1 = self.tgt_tok_emb(top1)

			output = Variable(trg.data[t] if is_teacher else top1).to(src.device)
		return torch.stack(outputs), None

	def init_sequence(self, batch_size):
		pass
	
	def calculate_num_params(self):
		"""Returns the total number of parameters."""
		num_params = 0
		for p in self.parameters():
			num_params += p.data.view(-1).size(0)
		return num_params