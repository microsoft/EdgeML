# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import torch
import numpy as np

from models import RPool_Face_QVGA_monochrome as module

save_dir_model = '../../../../tools/SeeDot/model/rnnpool/face-2/'
save_dir_datasets = '../../../../tools/SeeDot/datasets/rnnpool/face-2/'

if not os.path.exists(save_dir_model):
    os.makedirs(save_dir_model)
if not os.path.exists(save_dir_datasets):
    os.makedirs(save_dir_datasets)

net = module.build_s3fd('test', num_classes = 2)

checkpoint_dict = torch.load('./weights/RPool_Face_QVGA_monochrome_best_state.pth')

model_dict = {}
net = torch.nn.DataParallel(net)
model_dict = net.state_dict()

model_dict.update(checkpoint_dict)
net.load_state_dict(model_dict, strict = False)
net.eval()

a = np.load('trace_inputs.npy')
a = np.squeeze(a, axis=1)
# a = np.squeeze(a, axis=1)
a = a[0].flatten()

b = np.load('trace_outputs.npy')
b = b[0].flatten()
b = np.concatenate((b, a), axis=0)

np.save(save_dir_datasets + 'train.npy', b.reshape(1, b.shape[0]))
np.save(save_dir_datasets + 'test.npy', b.reshape(1, b.shape[0]))

C1 = net.state_dict()['module.conv.0.weight']
C1m = C1.permute(2, 3, 1, 0).detach().numpy().flatten()
w = net.state_dict()['module.conv.1.weight']
b = net.state_dict()['module.conv.1.bias']
m = net.state_dict()['module.conv.1.running_mean']
v = net.state_dict()['module.conv.1.running_var']
BNW = torch.mul(torch.rsqrt(torch.add(v, 0.00001)), w)
BNB = torch.sub(torch.mul(b, torch.reciprocal(BNW)), m)

np.save(save_dir_model + 'CBR1F.npy', C1m.reshape(1, C1m.shape[0]))
np.save(save_dir_model + 'CBR1W.npy', BNW.view(-1).unsqueeze(axis=0).detach().numpy())
np.save(save_dir_model + 'CBR1B.npy', BNB.view(-1).unsqueeze(axis=0).detach().numpy())

W1 = net.state_dict()['module.rnn_model.cell_rnn.cell.W']
W1m = W1.permute(1, 0)
U1 = net.state_dict()['module.rnn_model.cell_rnn.cell.U']
U1m = U1.permute(1, 0)
Bg1 = net.state_dict()['module.rnn_model.cell_rnn.cell.bias_gate']
Bg1m = Bg1.permute(1, 0)
Bh1 = net.state_dict()['module.rnn_model.cell_rnn.cell.bias_update']
Bh1m = Bh1.permute(1, 0)
zeta1 = net.state_dict()['module.rnn_model.cell_rnn.cell.zeta']
nu1 = net.state_dict()['module.rnn_model.cell_rnn.cell.nu']

np.save(save_dir_model + 'W1.npy', W1m.detach().numpy())
np.save(save_dir_model + 'U1.npy', U1m.detach().numpy())
np.save(save_dir_model + 'Bg1.npy', Bg1m.detach().numpy())
np.save(save_dir_model + 'Bh1.npy', Bh1m.detach().numpy())
np.save(save_dir_model + 'zeta1.npy', zeta1.detach().numpy().item())
np.save(save_dir_model + 'nu1.npy', nu1.detach().numpy().item())

W2 = net.state_dict()['module.rnn_model.cell_bidirrnn.cell.W']
W2m = W2.permute(1, 0)
U2 = net.state_dict()['module.rnn_model.cell_bidirrnn.cell.U']
U2m = U2.permute(1, 0)
Bg2 = net.state_dict()['module.rnn_model.cell_bidirrnn.cell.bias_gate']
Bg2m = Bg2.permute(1, 0)
Bh2 = net.state_dict()['module.rnn_model.cell_bidirrnn.cell.bias_update']
Bh2m = Bh2.permute(1, 0)
zeta2 = net.state_dict()['module.rnn_model.cell_bidirrnn.cell.zeta']
nu2 = net.state_dict()['module.rnn_model.cell_bidirrnn.cell.nu']

np.save(save_dir_model + 'W2.npy', W2m.detach().numpy())
np.save(save_dir_model + 'U2.npy', U2m.detach().numpy())
np.save(save_dir_model + 'Bg2.npy', Bg2m.detach().numpy())
np.save(save_dir_model + 'Bh2.npy', Bh2m.detach().numpy())
np.save(save_dir_model + 'zeta2.npy', zeta2.detach().numpy().item())
np.save(save_dir_model + 'nu2.npy', nu2.detach().numpy().item())

for j in range(14):
	F1 = net.state_dict()['module.mob.%d.conv.0.0.weight' % j]
	shaper = F1.shape
	F1m = F1.reshape(1, shaper[0], shaper[1], 1, 1).permute(0, 3, 4, 2, 1)
	w = net.state_dict()['module.mob.%d.conv.0.1.weight' % j]
	b = net.state_dict()['module.mob.%d.conv.0.1.bias' % j]
	m = net.state_dict()['module.mob.%d.conv.0.1.running_mean' % j]
	v = net.state_dict()['module.mob.%d.conv.0.1.running_var' % j]
	BN1W = torch.mul(torch.rsqrt(torch.add(v, 0.00001)), w)
	BN1B = torch.sub(torch.mul(b, torch.reciprocal(BN1W)), m)

	F2 = net.state_dict()['module.mob.%d.conv.1.0.weight' % j]
	shaper = F2.shape
	F2m = F2.reshape(shaper[0], 1, 1, shaper[2], shaper[3]).permute(0, 3, 4, 2, 1)
	w = net.state_dict()['module.mob.%d.conv.1.1.weight' % j]
	b = net.state_dict()['module.mob.%d.conv.1.1.bias' % j]
	m = net.state_dict()['module.mob.%d.conv.1.1.running_mean' % j]
	v = net.state_dict()['module.mob.%d.conv.1.1.running_var' % j]
	BN2W = torch.mul(torch.rsqrt(torch.add(v, 0.00001)), w)
	BN2B = torch.sub(torch.mul(b, torch.reciprocal(BN2W)), m)

	F3 = net.state_dict()['module.mob.%d.conv.2.weight' % j]
	shaper = F3.shape
	F3m = F3.reshape(1, shaper[0], shaper[1], 1, 1).permute(0, 3, 4, 2, 1)
	w = net.state_dict()['module.mob.%d.conv.3.weight' % j]
	b = net.state_dict()['module.mob.%d.conv.3.bias' % j]
	m = net.state_dict()['module.mob.%d.conv.3.running_mean' % j]
	v = net.state_dict()['module.mob.%d.conv.3.running_var' % j]
	BN3W = torch.mul(torch.rsqrt(torch.add(v, 0.00001)), w)
	BN3B = torch.sub(torch.mul(b, torch.reciprocal(BN3W)), m)

	np.save(save_dir_model + 'L%dF1.npy' % j, F1m.detach().numpy().flatten())
	np.save(save_dir_model + 'L%dF2.npy' % j, F2m.detach().numpy().flatten())
	np.save(save_dir_model + 'L%dF3.npy' % j, F3m.detach().numpy().flatten())

	np.save(save_dir_model + 'L%dW1.npy' % j, BN1W.view(-1).unsqueeze(axis=0).detach().numpy())
	np.save(save_dir_model + 'L%dW2.npy' % j, BN2W.view(-1).unsqueeze(axis=0).detach().numpy())
	np.save(save_dir_model + 'L%dW3.npy' % j, BN3W.view(-1).unsqueeze(axis=0).detach().numpy())

	np.save(save_dir_model + 'L%dB1.npy' % j, BN1B.view(-1).unsqueeze(axis=0).detach().numpy())
	np.save(save_dir_model + 'L%dB2.npy' % j, BN2B.view(-1).unsqueeze(axis=0).detach().numpy())
	np.save(save_dir_model + 'L%dB3.npy' % j, BN3B.view(-1).unsqueeze(axis=0).detach().numpy())

k = 0
for j in range(3, 6):
	k += 1
	N = net.state_dict()['module.L2Norm%d_3.weight' % j]
	np.save(save_dir_model + 'normW%d.npy' % k, N.view(-1).unsqueeze(axis=0).detach().numpy())

for j in range(4):
	locw = net.state_dict()['module.loc.%d.weight' % j]
	locwm = locw.permute(2, 3, 1, 0).detach().numpy().flatten()
	locb = net.state_dict()['module.loc.%d.bias' % j]
	confw = net.state_dict()['module.conf.%d.weight' % j]
	confwm = confw.permute(2, 3, 1, 0).detach().numpy().flatten()
	confb = net.state_dict()['module.conf.%d.bias' % j]

	np.save(save_dir_model + 'loc%dw.npy' % j, locwm.reshape(1, locwm.shape[0]))
	np.save(save_dir_model + 'loc%db.npy' % j, locb.view(-1).unsqueeze(axis=0).detach().numpy())
	np.save(save_dir_model + 'conf%dw.npy' % j, confwm.reshape(1, confwm.shape[0]))
	np.save(save_dir_model + 'conf%db.npy' % j, confb.view(-1).unsqueeze(axis=0).detach().numpy())
