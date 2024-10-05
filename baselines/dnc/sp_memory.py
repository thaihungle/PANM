#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import numpy as np
import math
from .util import *


class Memory(nn.Module):

    def __init__(self, input_size, mem_size=512, cell_size=32,
                 key_size=0, read_heads=4, gpu_id=-1, key_program_size=0,
                 program_size=0,
                 deallocate=False):
        super(Memory, self).__init__()

        self.mem_size = mem_size
        self.cell_size = cell_size
        self.read_heads = read_heads
        self.gpu_id = gpu_id
        self.input_size = input_size
        self.deallocate=deallocate
        self.key_size = key_size
        self.key_program_size = key_program_size
        self.program_size = program_size



        if self.key_size==0:
            key_size=self.cell_size

        m = self.mem_size
        w = self.cell_size
        r = self.read_heads

        self.interface_size = (key_size * r) + key_size + 2 * (key_size + w) + (2 * r) + 5
        if self.program_size==0:
            self.interface_weights = nn.Linear(self.input_size, self.interface_size, bias=False)
        else:
            self.interface_size += key_program_size + 1
            self.instruction_weight = nn.Parameter(cuda(T.zeros(self.program_size,
                                                self.key_program_size+self.input_size*self.interface_size)
                                                        .fill_(0), gpu_id=self.gpu_id),requires_grad = True)
            stdv = 1. / math.sqrt(self.input_size)
            nn.init.uniform_(self.instruction_weight,-stdv, stdv)
        # self.I = cuda(1 - T.eye(m).unsqueeze(0), gpu_id=self.gpu_id)  # (1 * n * n)
        self.layernorm = nn.GroupNorm(1, self.interface_size)

    def reset(self, batch_size=1, hidden=None, erase=True):
        m = self.mem_size
        w = self.cell_size
        r = self.read_heads
        b = batch_size

        if hidden is None:
            hidden = {
                'memory': cuda(T.zeros(b, m, self.key_size + w).fill_(0), gpu_id=self.gpu_id),
                'read_weights': cuda(T.zeros(b, r, m).fill_(0), gpu_id=self.gpu_id),
                'write_weights': cuda(T.zeros(b, 1, m).fill_(0), gpu_id=self.gpu_id),
                'usage_vector': cuda(T.zeros(b, m), gpu_id=self.gpu_id),
                'read_keys': [],
                'write_keys': [],
               # 'write_key': cuda(T.zeros(b, 1, self.key_size).fill_(0), gpu_id=self.gpu_id),
            }
            if self.program_size>0:
                hidden['read_weights_program']=cuda(T.zeros(b, 1, self.program_size).fill_(0), gpu_id=self.gpu_id)
                hidden['pmemory']=self.instruction_weight[0,self.key_program_size:].repeat(b,1,1).\
                    view(b, self.input_size, self.interface_size)

        else:
            for k,v in hidden.items():
                if isinstance(v, list):
                    for i2,v2 in enumerate(v):
                        hidden[k][i2]=v2.clone()
                else:
                    hidden[k] = v.clone()
            # hidden['memory'] = hidden['memory'].clone()
            # hidden['read_weights'] = hidden['read_weights'].clone()
            # hidden['write_weights'] = hidden['write_weights'].clone()
            # hidden['usage_vector'] = hidden['usage_vector'].clone()

            if erase:
                for k, v in hidden.items():
                    if isinstance(v, list):
                        hidden[k] = []
                    else:
                        v.data.fill_(0)
                # hidden['memory'].data.fill_(0)
                # hidden['read_weights'].data.fill_(0)
                # hidden['write_weights'].data.fill_(0)
                # hidden['usage_vector'].data.zero_()
        return hidden

    def get_usage_vector(self, usage, free_gates, read_weights, write_weights):
        # write_weights = write_weights.detach()  # detach from the computation graph
        usage = usage + (1 - usage) * (1 - T.prod(1 - write_weights, 1))
        ψ = T.prod(1 - free_gates.unsqueeze(2) * read_weights, 1)
        return usage * ψ, ψ

    def allocate(self, usage):
        # ensure values are not too small prior to cumprod.
        usage = δ + (1 - δ) * usage
        batch_size = usage.size(0)
        # free list
        sorted_usage, φ = T.topk(usage, self.mem_size, dim=1, largest=False)

        # cumprod with exclusive=True
        # https://discuss.pytorch.org/t/cumprod-exclusive-true-equivalences/2614/8
        v = var(sorted_usage.data.new(batch_size, 1).fill_(1))
        cat_sorted_usage = T.cat((v, sorted_usage), 1)
        prod_sorted_usage = T.cumprod(cat_sorted_usage, 1)[:, :-1]

        sorted_allocation_weights = (1 - sorted_usage) * prod_sorted_usage.squeeze()

        # construct the reverse sorting index https://stackoverflow.com/questions/2483696/undo-or-reverse-argsort-python
        _, φ_rev = T.topk(φ, k=self.mem_size, dim=1, largest=False)
        allocation_weights = sorted_allocation_weights.gather(1, φ_rev.long())

        return allocation_weights.unsqueeze(1), usage

    def write_weighting(self, memory, write_content_weights, allocation_weights, write_gate, allocation_gate, last_read_weights):
        lastrw = allocation_gate[:, 0].unsqueeze(-1) *T.mean(last_read_weights, dim=1)
        nallow = allocation_gate[:, 1].unsqueeze(-1) * allocation_weights.squeeze(1)
        conw = allocation_gate[:, 2].unsqueeze(-1) * write_content_weights.squeeze(1)

        fw = lastrw + nallow + conw

        return (write_gate * fw).unsqueeze(1)

    def write(self, write_key, write_vector, erase_vector, free_gates, read_strengths, write_strength, write_gate,
              allocation_gate, hidden):
        # get current usage
        last_read_ws = hidden['read_weights']
        hidden['usage_vector'], ψ = self.get_usage_vector(
            hidden['usage_vector'],
            free_gates,
            last_read_ws,
            hidden['write_weights']
        )

        # lookup memory with write_key and write_strength
        write_content_weights = self.content_weightings(hidden['memory'], write_key,
                                                        write_strength, self.key_size)

        # get memory allocation
        alloc, _ = self.allocate(
            hidden['usage_vector'],
        )

        # get write weightings

        hidden['write_weights'] = self.write_weighting(
            hidden['memory'],
            write_content_weights,
            alloc,
            write_gate,
            allocation_gate,
            last_read_ws
        )

        weighted_resets = hidden['write_weights'].unsqueeze(3) * erase_vector.unsqueeze(2)
        reset_gate = T.prod(1 - weighted_resets, 1)

        # Update memory
        if self.deallocate:
            hidden['memory'][:,:,self.key_size:] = hidden['memory'][:,:,self.key_size:] * ψ
        hidden['memory'] = hidden['memory'] * reset_gate

        hidden['memory'] = self.write_mem(hidden['memory'], hidden['write_weights'], write_vector)

        return hidden

    def content_weightings(self, memory, keys, strengths, key_size):
        if key_size>0:
            d = θ(memory[:,:,:key_size], keys[:,:,:key_size])
        else:
            d = θ(memory[:,:,:], keys)
        return σ(d * strengths.unsqueeze(2), 2)

    def read_mem(self, memory, read_weights, key_size):
        return T.bmm(read_weights, memory[:,:,key_size:])

    def write_mem(self, memory, write_weights, write_vector):
        return memory + \
                    T.bmm(write_weights.transpose(1, 2), write_vector)

    def read(self, read_keys, read_strengths, hidden):
        content_weights = self.content_weightings(hidden['memory'], read_keys, read_strengths, self.key_size)
        hidden['read_weights'] = content_weights
        read_vectors = self.read_mem(hidden['memory'], hidden['read_weights'], self.key_size)
        return read_vectors, hidden

    def read_program(self, read_keys, read_strengths, hidden):
        content_weights = self.content_weightings(self.instruction_weight.unsqueeze(0).repeat(read_keys.shape[0],1,1),
                                                  read_keys, read_strengths, self.key_program_size)

        hidden['read_weights_program'] = content_weights
        read_vectors = self.read_mem(self.instruction_weight.unsqueeze(0).repeat(read_keys.shape[0],1,1),
                                     hidden['read_weights_program'], self.key_program_size)
        return read_vectors, hidden

    def forward(self, ξ, hidden):

        # ξ = ξ.detach()
        k = self.cell_size
        if self.key_size>0:
            k = self.key_size
        m = self.mem_size
        w = self.cell_size
        r = self.read_heads
        b = ξ.size()[0]

        if self.program_size>0:
            ξ = T.matmul(ξ.unsqueeze(1), hidden['pmemory']).squeeze(1)
            # print(hidden['pmemory'])
        else:
            ξ = self.interface_weights(ξ)
            # print(self.interface_weights.weight)
        ξ = self.layernorm(ξ)
        counter = 0
        # r read keys (b * w * r)
        read_keys = F.tanh(ξ[:, counter:counter + r * k].contiguous().view(b, r, k))
        counter += r * k
        # r read strengths (b * r)
        read_strengths = F.softplus(ξ[:, counter:counter + r].contiguous().view(b, r))
        counter += r
        # write key (b * k * 1)
        write_key = F.tanh(ξ[:, counter:counter + k].contiguous().view(b, 1, k))
        counter += k
        # write strength (b * 1)
        write_strength = F.softplus(ξ[:, counter].contiguous().view(b, 1))
        counter += 1
        # erase vector (b * w)
        erase_vector = F.sigmoid(ξ[:, counter: counter + self.key_size + w].contiguous().view(b, 1, self.key_size + w))
        counter += self.key_size + w
        # write vector (b * w)
        write_vector = F.tanh(ξ[:, counter: counter + self.key_size + w].contiguous().view(b, 1, self.key_size + w))
        counter += self.key_size + w
        # r free gates (b * r)
        free_gates = F.sigmoid(ξ[:, counter: counter + r].contiguous().view(b, r))
        counter += r
        # allocation gate (b * 3)
        allocation_gate = σ(ξ[:, counter: counter + 3].contiguous().view(b, 3), 1)
        counter += 3
        # write gate (b * 1)
        write_gate = F.sigmoid(ξ[:, counter].contiguous()).unsqueeze(1).view(b, 1)
        if self.program_size>0:
            counter+=1
            # r read keys program (b * w * 1)
            read_keys_program = F.tanh(ξ[:, counter:counter+1*self.key_program_size].contiguous().view(b, 1, self.key_program_size))
            counter+=1*self.key_program_size
            # r read strengths program (b * 1)
            read_strengths_program = F.softplus(ξ[:, counter:counter+1].contiguous().view(b, 1))


        hidden = self.write(write_key, write_vector, erase_vector, free_gates,
                            read_strengths, write_strength, write_gate, allocation_gate, hidden)
        if self.program_size>0:
            read_vectors_program, hidden = self.read_program(read_keys_program, read_strengths_program, hidden)
            hidden['pmemory'] = read_vectors_program[:,0,:].view(read_vectors_program.shape[0],self.input_size, self.interface_size)
            hidden['pkey'] = self.instruction_weight[:,:self.key_program_size]
            hidden['ploss']=0
            count=0
            for i in range(self.program_size):
                for j in range(i+1, self.program_size):
                    hidden['ploss']+= F.cosine_similarity\
                        (self.instruction_weight[i,:self.key_program_size],
                         self.instruction_weight[j, :self.key_program_size],
                          dim = 0   )
                    count+=1
            hidden['ploss']/=count
        read_vectors, hidden = self.read(read_keys, read_strengths, hidden)
        #read_vectors = T.cat((read_vectors, read_keys), 2)
        hidden['read_keys'].append(read_keys)
        hidden['write_keys'].append(write_key)
        #hidden['write_key']=write_key
        return read_vectors, hidden
