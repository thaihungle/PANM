from json import encoder
import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import MultiheadAttention


class PoiterUnit(nn.Module):
	def __init__(self, in_dim, hid_dim, emb_size, nheads=1, dropout=0, add_space=10):
		super(PoiterUnit, self).__init__()
		self.hid_dim = hid_dim
		self.emb_size = emb_size
		self.controller = nn.GRU(in_dim, hid_dim, batch_first=False)
		self.add_att = MultiheadAttention(embed_dim=emb_size, num_heads=nheads, dropout=dropout, kdim=add_space, vdim=emb_size)
		self.q = nn.Linear(hid_dim, emb_size)
		self.is_train = True
		self.A_map = nn.Linear(add_space, add_space)

	def forward(self, inputs, cur_dp, hidden, A):
		output, hidden = self.controller(cur_dp, hidden)
		cur_val, aweights = self.add_att(self.q(hidden),self.A_map(A), inputs)
		cur_add = torch.matmul(aweights, A.view(A.shape[1],A.shape[0],-1)).squeeze(1)
		return cur_val, cur_add, hidden, aweights



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
		outputs = (outputs[:, :, :self.hidden_size] +
				   outputs[:, :, self.hidden_size:])
		return outputs, hidden


class Attention(nn.Module):
	def __init__(self, hidden_size, memory_size):
		super(Attention, self).__init__()
		self.hidden_size = hidden_size
		self.attn = nn.Linear(self.hidden_size + memory_size, memory_size)
		self.v = nn.Parameter(torch.rand(memory_size))
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
	def __init__(self, embed_size, hidden_size, memory_size, output_size,
				 n_layers=1, dropout=0.2, embedded=False, dis_att=False):
		super(Decoder, self).__init__()
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.embedded = embedded
		self.embed = nn.Embedding(output_size, embed_size)
		self.dropout = nn.Dropout(dropout, inplace=True)
		self.dis_att = dis_att
		if not self.dis_att:
			self.attention = Attention(hidden_size, memory_size)
		else:
			memory_size = 0
		self.out = nn.Linear(hidden_size + memory_size , output_size)

		self.gru = nn.GRU(memory_size + embed_size , hidden_size,
						  n_layers, dropout=dropout)
		
	def forward(self, input, last_hidden, encoder_outputs):
		# Get the embedding of the current input word (last output word)
		if not self.embedded:
			embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
			embedded = self.dropout(embedded)
		else:
			embedded = input
		# Calculate attention weights and apply to encoder outputs
		if not self.dis_att:
			attn_weights = self.attention(last_hidden[-1], encoder_outputs)
			context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
			context = context.transpose(0, 1)  # (1,B,N)
			# Combine embedded input word and attended context, run through RNN
			rnn_input = torch.cat([embedded, context], 2)
			output, hidden = self.gru(rnn_input, last_hidden)
			output = output.squeeze(0)  # (1,B,N) -> (B,N)
			context = context.squeeze(0)
			output = self.out(torch.cat([output, context], 1))
		else:
			output, hidden = self.gru(embedded, last_hidden)
			output = self.out(output).squeeze(0)
			attn_weights = None

		return output, hidden, attn_weights


class PANM(nn.Module):
	def __init__(self, in_size, out_size, embed_size, controller_size = 512,
		layers = 1, nheads = 1,hidden_dim=512,dropout=0,
	 embedded=True, add_space=8, Hc_num_pointers=1, Ha_num_pointers=2):
		super(PANM, self).__init__()
		hidden_size = hidden_dim
		self.Ha_num_pointers=Ha_num_pointers
		self.Hc_num_pointers=Hc_num_pointers
		self.hidden_size = hidden_dim
		self.embedded = embedded
		self.embed_size = embed_size
		self.encoder = Encoder(in_size, embed_size, hidden_size,
					n_layers=1, dropout=0.5, embedded=embedded)
		decoder = Decoder(embed_size+(hidden_size)*(self.Ha_num_pointers), controller_size, hidden_size, hidden_size,
						n_layers=1, dropout=0, embedded=embedded)
		self.decoder = decoder
		self.address_space = add_space
		self.num_address = 2**self.address_space
		self.A, _ = self.gen_address(self.num_address, 1,is_random=False)
		self.A=self.A.squeeze(1)
		self.dropout = nn.Dropout(dropout)
		self.out = nn.Linear(hidden_size*(self.Ha_num_pointers+1+self.Hc_num_pointers), out_size)
		punits = []
		for _ in range(self.Ha_num_pointers):
			addgen = PoiterUnit(add_space, controller_size, hidden_size, add_space=add_space)
			punits.append(addgen)
		self.punits = nn.ModuleList(punits)
		self.add_att = MultiheadAttention(embed_dim=hidden_size, num_heads=1,dropout=0.5, kdim=hidden_size, vdim=hidden_size)
		tohs = []
		for _ in range(self.Hc_num_pointers):
			toh = nn.Linear(hidden_size*self.Ha_num_pointers, hidden_size)	
			tohs.append(toh)
		self.tohs = nn.ModuleList(tohs)
		

	def binary(self, x):
		mask = 2**torch.arange(self.address_space)
		return torch.tensor(x).unsqueeze(-1).bitwise_and(mask).ne(0).byte()

	def gen_address(self, seq_len, bs, is_random=True):
		A = torch.zeros(seq_len, bs, self.address_space)
		start_a=torch.zeros(bs, self.address_space)
		end_a=torch.zeros(bs, self.address_space)
		content_a=torch.zeros(bs, self.address_space)
		for b in range(bs):
			start_p=0
			if is_random:
				start_p=random.randint(0, self.num_address)
			for i in range(seq_len):
				j=(i+start_p)%self.num_address
				A[i,b,:]=self.binary(j)
				if i==0:
					start_a[b] = A[i,b,:]
				if i==seq_len-1:
					end_a[b] = A[i,b,:]
		return A, torch.cat([start_a, end_a, content_a],dim=-1)

	def fast_gen_address(self, seq_len, bs, is_fix=False):
		A = torch.zeros(seq_len, bs, self.address_space)
		start_a=torch.zeros(bs, self.address_space)
		end_a=torch.zeros(bs, self.address_space)
		content_a=torch.zeros(bs, self.address_space)
		for b in range(bs):
			if not is_fix:
				start_p = random.randint(0, self.num_address)
			else:
				start_p = 0
			A[:,b,:] = torch.roll(self.A, start_p, dims=0)[:seq_len]
			start_a[b] = A[0,b,:]
			end_a[b] = A[seq_len-1,b,:]
		return A, start_a, end_a, content_a
		

	def forward(self, src, target_length, tgt=None):
		trg=tgt
		encoder_output, hidden = self.encoder(src)
		
		max_len = target_length
		batch_size = src.size(1)
		if trg is not None:
			max_len = trg.size(0)
		else:
			trg = torch.zeros(max_len, batch_size, self.decoder.output_size).to(src.device)
			if self.embedded:
				trg = Variable(torch.zeros(max_len, batch_size, self.embed_size)).to(src.device)
			else:
				trg = Variable(torch.zeros(max_len, batch_size).long()).to(src.device)
		outputs = []
		

		hidden = hidden[:self.decoder.n_layers]
		output = Variable(trg.data[0, :])  # sos
		A, cur_ptr1, cur_ptr2, cur_ptr3 = self.fast_gen_address(encoder_output.shape[0], batch_size)
		A = A.to(encoder_output.device)
		cur_ptr1 = cur_ptr1.to(encoder_output.device)
		cur_ptr2 = cur_ptr2.to(encoder_output.device)
		cur_ptr3 = cur_ptr3.to(encoder_output.device)
		hiddena = []
		
		
		
		cur_ptrs_mode1 = []
		for i in range(self.Ha_num_pointers):
			if i%2==0:
				cur_ptrs_mode1.append(cur_ptr1)
			else:
				cur_ptrs_mode1.append(cur_ptr2)
			hiddena.append(hidden.clone().detach().zero_())
		
		cur_vals_mode2 = []
		cur_ptrs_mode2 = []
		
		for _ in range(self.Hc_num_pointers):
			cur_vals_mode2.append(torch.zeros(1, batch_size, self.hidden_size).to(encoder_output.device))
			cur_ptrs_mode2.append(cur_ptr3.clone())
		cur_val_mode2 = torch.cat(cur_vals_mode2, dim=-1)
		cur_ptr_mode2 = torch.cat(cur_ptrs_mode2, dim=-1)

		for t in range(0, max_len):
			cur_vals_mode1 = []
			
			for i in range(self.Ha_num_pointers):
				cur_val_mode1, cur_add, hiddena[i], aweights1 = self.punits[i](encoder_output, cur_ptrs_mode1[i].unsqueeze(0), hiddena[i], A)
				cur_vals_mode1.append(cur_val_mode1)
				cur_ptrs_mode1[i] = cur_add
				
			cur_val_mode1 = torch.cat(cur_vals_mode1,dim=-1)
			
	
			output, hidden, attn_weights = self.decoder(
				torch.cat([output.unsqueeze(0), cur_val_mode1], dim=-1),
				hidden, encoder_output)
		

			cur_vals_mode2 = []
			cur_ptrs_mode2 = []
			for i in range(self.Hc_num_pointers):
		
				cur_val_mode2, aweights2 = self.add_att(self.tohs[i](cur_val_mode1.squeeze(0)).unsqueeze(0), encoder_output, encoder_output)
				cur_ptr_mode2 = torch.matmul(aweights2, A.view(A.shape[1],A.shape[0],-1)).squeeze(1)
				cur_vals_mode2.append(cur_val_mode2)
				cur_ptrs_mode2.append(cur_ptr_mode2)
		

			cur_val_mode2 = torch.cat(cur_vals_mode2, dim=-1)
			cur_ptr_mode2 = torch.cat(cur_ptrs_mode2, dim=-1)
			
			fout = torch.cat([output, cur_val_mode1.squeeze(0), cur_val_mode2.squeeze(0)],dim=-1)
			output = self.out(self.dropout(fout))


			outputs.append(output)
			output = trg[t] 

		return torch.stack(outputs), None


	def calculate_num_params(self):
		"""Returns the total number of parameters."""
		num_params = 0
		for p in self.parameters():
			num_params += p.data.view(-1).size(0)
		return num_params

	def init_sequence(self, batch_size):
		pass