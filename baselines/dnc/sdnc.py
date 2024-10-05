#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
import time as timep
from .util import *
from .sp_memory import *

from torch.nn.init import orthogonal, xavier_uniform

class DNC(nn.Module):

    def __init__(
            self,
            input_size,
            final_output_size,
            hidden_size,
            rnn_type='lstm',
            num_layers=1,
            num_hidden_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
            nr_cells=5,
            key_size=0,
            program_key_size=0,
            read_heads=2,
            cell_size=10,
            nonlinearity='tanh',
            gpu_id=-1,
            independent_linears=False,
            share_memory=True,
            debug=False,
            clip=0,
            pass_through_memory=True,
            write_interval = 1,
            program_size=0
    ):
        super(DNC, self).__init__()
        # todo: separate weights and RNNs for the interface and output vectors

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.num_hidden_layers = num_hidden_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.nr_cells = nr_cells
        self.key_size = key_size
        self.program_key_size = program_key_size
        self.read_heads = read_heads
        self.cell_size = cell_size
        self.nonlinearity = nonlinearity
        self.gpu_id = gpu_id
        self.independent_linears = independent_linears
        self.share_memory = share_memory
        self.debug = debug
        self.clip = clip
        self.pass_through_memory = pass_through_memory
        self.write_interval = write_interval
        self.program_size = program_size
        self.emb =None


        self.w = self.cell_size
        self.r = self.read_heads

        self.read_vectors_size = self.r * self.w

        self.nn_input_size = self.input_size
        if self.pass_through_memory:
            self.nn_input_size += self.read_vectors_size
        self.nn_output_size = self.hidden_size

        if self.pass_through_memory:
            self.nn_output_size += self.read_vectors_size

        self.emb_size = self.hidden_size


        if self.bidirectional:
            self.nn_output_size+=self.hidden_size


        self.final_output_size = final_output_size

        self.rnns = []
        self.memories = []

        mem_input_size = self.hidden_size
        if self.bidirectional:
            mem_input_size += self.hidden_size
            self.brnns = []

        self.embs = []
        for layer in range(self.num_layers):
            if layer<self.num_layers-1:
                self.embs.append(nn.Linear(self.nn_output_size, self.emb_size))
            else:
                self.embs.append(nn.Linear(self.nn_output_size, self.final_output_size))
            orthogonal(self.embs[layer].weight)

            if self.rnn_type.lower() == 'rnn':

                self.rnns.append(nn.RNN((self.nn_input_size if layer == 0 else self.emb_size), self.hidden_size,
                                        bias=self.bias, nonlinearity=self.nonlinearity, batch_first=True,
                                        dropout=self.dropout, num_layers=self.num_hidden_layers))
                if self.bidirectional:
                    self.brnns.append(
                        nn.RNN((self.nn_input_size if layer == 0 else self.emb_size), self.hidden_size, bias=self.bias, nonlinearity=self.nonlinearity, batch_first=True,
                                        dropout=self.dropout, num_layers=self.num_hidden_layers))

            elif self.rnn_type.lower() == 'gru':
                self.rnns.append(nn.GRU((self.nn_input_size if layer == 0 else self.emb_size),
                                        self.hidden_size, bias=self.bias, batch_first=True, dropout=self.dropout,
                                        num_layers=self.num_hidden_layers))
                if self.bidirectional:
                    self.brnns.append(
                        nn.GRU((self.nn_input_size if layer == 0 else self.emb_size),
                               self.hidden_size, bias=self.bias, batch_first=True, dropout=self.dropout,
                               num_layers=self.num_hidden_layers))

            if self.rnn_type.lower() == 'lstm':
                self.rnns.append(nn.LSTM((self.nn_input_size if layer == 0 else self.emb_size),
                                         self.hidden_size, bias=self.bias, batch_first=True, dropout=self.dropout,
                                         num_layers=self.num_hidden_layers))
                if self.bidirectional:
                    self.brnns.append(
                        nn.LSTM((self.nn_input_size if layer == 0 else self.emb_size),
                                self.hidden_size, bias=self.bias, batch_first=True, dropout=self.dropout,
                                num_layers=self.num_hidden_layers))

            setattr(self, self.rnn_type.lower() + '_layer_' + str(layer), self.rnns[layer])
            if self.bidirectional:
                setattr(self, self.rnn_type.lower() + '_blayer_' + str(layer), self.brnns[layer])
            setattr(self, self.rnn_type.lower() + '_elayer_' + str(layer), self.embs[layer])

            # memories for each layer
            if not self.share_memory:

                self.memories.append(
                    Memory(
                        input_size=mem_input_size,
                        mem_size=self.nr_cells,
                        cell_size=self.w,
                        key_size=self.key_size,
                        key_program_size=self.program_key_size,
                        read_heads=self.r,
                        gpu_id=self.gpu_id,
                        program_size=self.program_size
                    )
                )
                setattr(self, 'rnn_layer_memory_' + str(layer), self.memories[layer])

        # only one memory shared by all layers
        if self.share_memory:
            self.memories.append(
                Memory(
                    input_size=mem_input_size,
                    mem_size=self.nr_cells,
                    cell_size=self.w,
                    key_size = self.key_size,
                    key_program_size=self.program_key_size,
                    read_heads=self.r,
                    gpu_id=self.gpu_id,
                    program_size=self.program_size
                )
            )
            setattr(self, 'rnn_layer_memory_shared', self.memories[0])


        # final output layer
        # self.output = nn.Linear(self.nn_output_size, self.final_output_size)
        # orthogonal(self.output.weight)

        if self.gpu_id != -1:
            [x.cuda(self.gpu_id) for x in self.rnns]
            [x.cuda(self.gpu_id) for x in self.memories]
            if self.bidirectional:
                [x.cuda(self.gpu_id) for x in self.brnns]
            [x.cuda(self.gpu_id) for x in self.embs]

    def _init_hidden(self, hx, batch_size, reset_experience):
        # create empty hidden states if not provided
        if hx is None:
            hx = (None, None, None)
        (chx, mhx, last_read) = hx

        # initialize hidden state of the controller RNN
        if chx is None:
            h = cuda(T.zeros(self.num_hidden_layers, batch_size, self.hidden_size), gpu_id=self.gpu_id)
            xavier_uniform(h)

            chx = [(h, h) if self.rnn_type.lower() == 'lstm' else h for x in range(self.num_layers)]

        # Last read vectors
        if last_read is None:
            last_read = cuda(T.zeros(batch_size, self.w * self.r), gpu_id=self.gpu_id)

        # memory states
        if mhx is None:
            if self.share_memory:
                mhx = self.memories[0].reset(batch_size, erase=reset_experience)
            else:
                mhx = [m.reset(batch_size, erase=reset_experience) for m in self.memories]
        else:
            if self.share_memory:
                mhx = self.memories[0].reset(batch_size, mhx, erase=reset_experience)
            else:
                mhx = [m.reset(batch_size, h, erase=reset_experience) for m, h in zip(self.memories, mhx)]

        return chx, mhx, last_read

    def _debug(self, mhx, debug_obj):
        if not debug_obj:
            debug_obj = {
                'memory': [],
                'read_weights': [],
                'write_weights': [],
                'usage_vector': [],
            }

        debug_obj['memory'].append(mhx['memory'][0].data.cpu().numpy())
        debug_obj['read_weights'].append(mhx['read_weights'][0].data.cpu().numpy())
        debug_obj['write_weights'].append(mhx['write_weights'][0].data.cpu().numpy())
        debug_obj['usage_vector'].append(mhx['usage_vector'][0].unsqueeze(0).data.cpu().numpy())
        return debug_obj

    def _layer_forward(self, input, layer, hx=(None, None), bwhs = None):
        (chx, mhx) = hx
        # pass through the controller layer
        input, chx = self.rnns[layer](input, chx)

        # clip the controller output
        if self.clip != 0:
            output = T.clamp(input, -self.clip, self.clip)
        else:
            output = input

        if self.bidirectional:
            output = T.cat([output, bwhs], 2)

        # the interface vector
        ξ = output[:,-1,:]



        # pass through memory
        if self.pass_through_memory:
            if self.share_memory:
                read_vecs, mhx = self.memories[0](ξ, mhx)
            else:
                read_vecs, mhx = self.memories[layer](ξ, mhx)
            # the read vectors
            read_vectors = read_vecs.view(-1, read_vecs.shape[2] * self.r)
        else:
            read_vectors = None

        return output, (chx, mhx, read_vectors)

    def forward(self, inputs, target_length, constant_decode_input=True):
        decoder_outputs, self.previous_state = self.forward_encode(inputs, self.previous_state)

        if target_length==0:
            return decoder_outputs, self.previous_state
        decoder_input = torch.zeros(target_length, inputs.shape[1], self.input_size)

        if self.gpu_id != -1:
            decoder_input = decoder_input.cuda(self.gpu_id)

        if constant_decode_input:
            decoder_outputs, self.previous_state = self.forward_encode(decoder_input, self.previous_state)

        return decoder_outputs, self.previous_state

    def forward_encode(self, input, hx=(None, None, None), pass_through_memory=True):
        self.pass_through_memory = pass_through_memory
        # handle packed data
        # start_time = timep.time()
        is_packed = type(input) is PackedSequence
        if is_packed:
            input, lengths = pad(input)
            max_length = lengths[0]
        else:
            max_length = input.size(1) if self.batch_first else input.size(0)
            lengths = [input.size(1)] * max_length if self.batch_first else [input.size(0)] * max_length

        batch_size = input.size(0) if self.batch_first else input.size(1)

        if not self.batch_first:
            input = input.transpose(0, 1)

        if self.emb is not None:
            input = self.emb(input.long())
        # make the data time-first

        controller_hidden, mem_hidden, last_read = hx

        # concat input with last read (or padding) vectors
        if self.pass_through_memory:
            inputs = [T.cat([input[:, x, :], last_read], 1) for x in range(max_length)]
        else:
            inputs = [input[:, x, :] for x in range(max_length)]

        # batched forward pass per element / word / etc
        if self.debug:
            viz = None

        read_vectors = None
        # pass thorugh layers
        for layer in range(self.num_layers):
            inputs_r = T.stack(inputs, 1)
            if self.bidirectional:
                bwhiddens, _ = self.brnns[layer](T.flip(inputs_r,[1]))
                bwhiddens = T.flip(bwhiddens,[1])
            # pass through time
            time_begin=0
            while time_begin<max_length:
                time_end = min(time_begin+self.write_interval, max_length)
                bwhs = None
                if self.bidirectional:
                    bwhs = bwhiddens[:,time_begin:time_end,:]
                # this layer's hidden states
                chx = controller_hidden[layer]
                m = mem_hidden if self.share_memory else mem_hidden[layer]
                # pass through controller
                subouts, (chx, m, read_vectors) = \
                    self._layer_forward(inputs_r[:,time_begin:time_end,:], layer, (chx, m), bwhs)

                # debug memory
                if self.debug:
                    viz = self._debug(m, viz)
                # print(m['read_weights_program'][0])

                # store the memory back (per layer or shared)
                if self.share_memory:
                    mem_hidden = m
                else:
                    mem_hidden[layer] = m
                controller_hidden[layer] = chx

                if read_vectors is not None:
                    # the controller output + read vectors go into next layer
                    subouts = T.cat([subouts,
                                     read_vectors.unsqueeze(1).repeat(1,time_end-time_begin,1)], 2)
                    sb, sT, sh = subouts.size()
                    subouts = self.embs[layer](subouts.view(sb * sT, sh)).view(sb,sT,-1)

                for ii in range(time_begin, time_end):
                    inputs[ii] = subouts[:,ii-time_begin,:]

                if  read_vectors is not None and time_end<max_length-1 and layer<self.num_layers-1:
                    inputs[time_end][:,-read_vectors.shape[1]:] = read_vectors

                time_begin += self.write_interval
        # print('--')
        if self.debug:
            viz = {k: np.array(v) for k, v in viz.items()}
            viz = {k: v.reshape(v.shape[0], v.shape[1] * v.shape[2]) for k, v in viz.items()}

        # pass through final output layer
        # inputs = [self.output(i) for i in inputs]
        outputs = T.stack(inputs, 1 if self.batch_first else 0)




        if is_packed:
            outputs = pack(outputs, lengths)

        if self.debug:
            return outputs, (controller_hidden, mem_hidden, read_vectors), viz
        else:
            return outputs, (controller_hidden, mem_hidden, read_vectors)

    def program_loss_pl1(self):
        ploss = 0
        count = 0
        for m in self.memories:
            for i in range(self.program_size):
                for j in range(i + 1, self.program_size):
                    ploss += F.cosine_similarity \
                        (m.instruction_weight[i, :self.key_size],
                         m.instruction_weight[j, :self.key_size],
                         dim=0)
                    count += 1
        return ploss / count

    def init_sequence(self, batch_size):
        self.previous_state = self._init_hidden(None, batch_size, reset_experience=True)


    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params

    def __repr__(self):
        s = "\n----------------------------------------\n"
        s += '{name}({input_size}, {hidden_size}'
        if self.rnn_type != 'lstm':
            s += ', rnn_type={rnn_type}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.num_hidden_layers != 2:
            s += ', num_hidden_layers={num_hidden_layers}'
        if self.bias != True:
            s += ', bias={bias}'
        if self.batch_first != True:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional != False:
            s += ', bidirectional={bidirectional}'
        if self.nr_cells != 5:
            s += ', nr_cells={nr_cells}'
        if self.read_heads != 2:
            s += ', read_heads={read_heads}'
        if self.cell_size != 10:
            s += ', cell_size={cell_size}'
        if self.nonlinearity != 'tanh':
            s += ', nonlinearity={nonlinearity}'
        if self.gpu_id != -1:
            s += ', gpu_id={gpu_id}'
        if self.independent_linears != False:
            s += ', independent_linears={independent_linears}'
        if self.share_memory != True:
            s += ', share_memory={share_memory}'
        if self.debug != False:
            s += ', debug={debug}'
        if self.clip != 20:
            s += ', clip={clip}'

        s += ")\n" + super(DNC, self).__repr__() + \
             "\n----------------------------------------\n"
        return s.format(name=self.__class__.__name__, **self.__dict__)