import json
from tqdm import tqdm
import numpy as np
import random
import os
import torch
from torch import nn, optim
from tensorboard_logger import configure, log_value
import torch.nn.functional as F

from datasets import CopyDataset
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from args import get_parser
args = get_parser().parse_args()


def get_data(mode):
	if  "copy" == args.task_name: 
		if mode=="train":
			random_length = np.random.randint(task_params['min_seq_len'],
											  task_params['max_seq_len'] + 1)
		else:
			random_length = int(task_params['max_seq_len']*args.genlen) + 1

		data = dataset.get_sample_wlen(random_length, bs=args.batch_size)
	return data

def model_compute(data, is_train=True):
	if torch.cuda.is_available():
		input, target = data['input'].cuda(), data['target'].cuda()
		out = torch.zeros(target.size()).cuda()
	else:
		input, target = data['input'], data['target']
		out = torch.zeros(target.size())


	# -------------------------------------------------------------------------
	# loop for other tasks
	# -------------------------------------------------------------------------
	if "lstm" in args.model_name or "ntm" in args.model_name:
		for i in range(input.size()[0]):
			in_data = input[i]
			sout, _ = model(in_data)
		if torch.cuda.is_available():
			in_data = torch.zeros(input.size()).cuda()
		else:
			in_data = torch.zeros(input.size())
		for i in range(target.size()[0]):
			sout, _ = model(in_data[-1])
			out[i] = sout
	elif "dnc" in args.model_name or "att" in args.model_name or  "transformer" in args.model_name \
	 or "panm" in args.model_name:
		model.is_train = is_train
		out, _, = model(input, target_length=target.shape[0])

	return out, target, input


def get_err(out, target):
	binary_output = out.clone()
	if torch.cuda.is_available():
		binary_output = binary_output.detach().cpu().apply_(lambda x: 0 if x < 0.5 else 1).cuda()
	else:
		binary_output = binary_output.detach().apply_(lambda x: 0 if x < 0.5 else 1)
	if "nfar" in args.task_name or "onehot" in args.mode_toy:
		binary_output = torch.nn.functional.one_hot(torch.argmax(out, dim=-1))
		error = torch.sum(torch.argmax(out, dim=-1) != torch.argmax(target, dim=-1)).float()/(target.shape[1])
	else:
		# sequence prediction error is calculted in bits per sequence
		error = torch.sum(torch.abs(binary_output - target))/args.batch_size
	return error, binary_output

# ----------------------------------------------------------------------------
# -- initialize datasets, model, criterion and optimizer
# ----------------------------------------------------------------------------


task_params = json.load(open(args.task_json))
args.task_name = task_params['task']
if 'iter' in task_params:
	args.num_iters = task_params['iter']
log_dir = os.path.join(args.log_dir,args.task_name+"-"+args.mode_toy)
if not os.path.isdir(log_dir):
	os.mkdir(log_dir)
log_dir = os.path.join(log_dir, args.model_name+f"-s{args.seed}")
if not os.path.isdir(log_dir):
	os.mkdir(log_dir)



save_dir = os.path.join(args.save_dir,args.task_name+args.model_name+"-"+args.mode_toy)
if not os.path.isdir(save_dir):
	os.mkdir(save_dir)

save_dirbest = os.path.join(save_dir,"{}-{}-best.pt".format(args.model_name, args.seed))
save_dirlast = os.path.join(save_dir,"{}-{}-last.pt".format(args.model_name, args.seed))
if args.mode == "test":
	save_dirlast = os.path.join(save_dir,"{}-{}-best.pt".format(args.model_name, args.seed))


task_params["prob"]=args.task_prob
task_params["mode"]=args.mode_toy

if "copy" == args.task_name:
	dataset = CopyDataset(task_params)


in_dim = dataset.in_dim
out_dim = dataset.out_dim






if 'lstm' in args.model_name:
	from baselines.nvm.lstm_baseline import LSTMBaseline
	hidden_dim = task_params['controller_size']*2
	model = LSTMBaseline(in_dim, hidden_dim, out_dim, 1)
elif 'att' in args.model_name:
	from baselines.nvm.con_att import Seq2Seq
	hidden_dim = task_params['controller_size']*2
	emb_layer = torch.nn.Linear(in_dim, hidden_dim)
	model = Seq2Seq(hidden_dim, out_dim, hidden_dim, hidden_dim, embedded=True, emb_layer=emb_layer)

elif 'dnc' in args.model_name:
	from baselines.dnc.dnc import DNC
	gpu_id=-1
	if torch.cuda.is_available():
		gpu_id=0
	model = DNC(
		input_size=in_dim,
		final_output_size=out_dim,
		hidden_size=task_params['controller_size']*2,
		read_heads = task_params['num_heads'],
		nr_cells=task_params['memory_units'],
		cell_size=task_params['memory_unit_size'],
		gpu_id=gpu_id)
elif 'ntm' in args.model_name:
	from baselines.nvm.ntm_warper import EncapsulatedNTM
	model = EncapsulatedNTM(
		num_inputs=in_dim,
		num_outputs=out_dim,
		controller_size=task_params['controller_size']*2,
		controller_layers =1,
		num_heads = task_params['num_heads'],
		N=task_params['memory_units'],
		M=task_params['memory_unit_size'])
elif 'transformer' in args.model_name:
	from baselines.transformer import TransformerModel
	emsize =  task_params['controller_size']  # embedding dimension
	d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
	nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
	nhead = 8  # number of heads in nn.MultiheadAttention
	dropout = 0.2  # dropout probability
	mlp_encoder = nn.Linear(in_dim, emsize)
	model = TransformerModel(out_dim, emsize, nhead, d_hid, nlayers, dropout, encoder=mlp_encoder)
elif 'panm' in args.model_name:
	from baselines.panm import PANM


	model = PANM(
		in_size = in_dim,
		out_size = out_dim,
		embed_size = in_dim,
		controller_size = task_params['controller_size']//2,
		hidden_dim = task_params['controller_size']//2,
	)



print(model)
if torch.cuda.is_available():
	model.cuda()

print("====num params=====")

print(model.calculate_num_params())

print("========")

criterion = nn.CrossEntropyLoss()


# As the learning rate is task specific, the argument can be moved to json file
optimizer = optim.RMSprop(model.parameters(),
						  lr=args.lr,
						  alpha=args.alpha,
						  momentum=args.momentum)





cur_dir = os.getcwd()


# ----------------------------------------------------------------------------
# -- basic training loop
# ----------------------------------------------------------------------------
losses = []
errors = []
rel_errors = []
loss_pls = []

best_loss = 10000

print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if args.mode=="train":
	model.train()
	num_iter = args.num_iters
	print("===training===")
	configure(log_dir)
elif args.mode=="test":
	num_iter = args.num_eval
	print("===testing===")
	print(DEVICE)
	model.load_state_dict(torch.load(save_dirlast,map_location=DEVICE))
	model.eval()
	print(f"load weight {save_dirlast}")
	

for iter in tqdm(range(num_iter)):
	annelr = iter*1.0/num_iter
	model.annelr = annelr
	optimizer.zero_grad()
	model.init_sequence(batch_size=args.batch_size)

	data = get_data(args.mode)
	out, target, input = model_compute(data)

	# -------------------------------------------------------------------------
	loss = criterion(torch.reshape(out, [-1, dataset.out_dim]),
				torch.argmax(torch.reshape(target, [-1, dataset.out_dim]), -1))
	loss = torch.mean(loss)


	losses.append(loss.item())

	if args.mode=="train":
	

		loss.backward()
		if args.clip_grad > 0:
			nn.utils.clip_grad_value_(model.parameters(), args.clip_grad)
		optimizer.step()

	error, binary_output = get_err(out, target)
	errors.append(error.item())
	rel_errors.append((error/target.shape[0]).item())
	

	# ---logging---
	if args.mode=="train" and iter % args.freq_val == 0:
		print('Iteration: %d\tLoss: %.2f\tError in bits per sequence: %.2f' %
			  (iter, np.mean(losses), np.mean(errors)))
		mloss = np.mean(losses)

		
		log_value('Train/loss', mloss, iter)
		log_value('Train/bit_error_per_sequence', np.mean(errors), iter)
		log_value('Train/percentage_error', np.mean(rel_errors), iter)
		
		losses = []
		rel_errors = []
		errors = []
		loss_pls = []

		print("EVAL ...")
		model.eval()
		eval_err = []
		eval_err2 = []
		for i in range(args.num_eval):
			data = get_data("eval")
			out, target, input = model_compute(data, is_train=False)
			error, binary_output = get_err(out, target)
			eval_err.append((error/target.shape[0]).item())
			eval_err2.append(error.item())

		merror = np.mean(eval_err)
		if merror <=best_loss:
			# ---saving the model---
			print("SAVE MODEL BEST TO:\n", save_dirbest)
			torch.save(model.state_dict(), save_dirbest)
			best_loss = merror
		log_value('Test/bit_error_per_sequence', np.mean(eval_err2), iter)
		log_value('Test/percentage_error', merror, iter)
		model.train()

if args.mode=="train":	
	# ---saving the model---
	print("SAVE LAST MODEL TO:\n", save_dirlast)
	torch.save(model.state_dict(), save_dirlast)

if args.mode=="test":
	print('test_loss', np.mean(losses))
	print(f'bit_error_per_sequence {np.mean(errors)} -->{np.mean(rel_errors)} over {len(errors)} samples')

	