import matplotlib.pyplot as plt
import numpy as np
import os

methods = ['d2net','d2net_pyramid','surf','sift', 'rfnet','lfnet','our1060']
names = ['D2-Net','D2-Net_multiscale','SURF','SIFT','RF-Net','LF-Net','Our']
colors = ['yellow','yellow','blue','green','black','orange','red']
linestyles = ['-','--','-','-','-','-','-']

errors = {}

for method in methods:
    output_file = os.path.join('./cache', method + '.npy')
    print(method)
    if os.path.exists(output_file):
        print('Loading precomputed errors...')
        errors[method] = np.load(output_file, allow_pickle=True)
    else:
        print('No file')

plt_lim = [1, 10]
plt_rng = np.arange(plt_lim[0], plt_lim[1] + 1)

plt.rc('axes', titlesize=25)
plt.rc('axes', labelsize=25)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for method, name, color, ls in zip(methods, names, colors, linestyles):
    i_err, v_err, n_i, n_v = errors[method]
    plt.plot(plt_rng, [(i_err[thr] + v_err[thr]) / ((n_i + n_v)) for thr in plt_rng], color=color, ls=ls,
             linewidth=3, label=name)
plt.title('Overall')
plt.xlim(plt_lim)
plt.xticks(plt_rng)
plt.ylabel('MMA')
plt.ylim([0, 1])
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=20)
plt.legend()

plt.subplot(1, 3, 2)
for method, name, color, ls in zip(methods, names, colors, linestyles):
    i_err, v_err, n_i, n_v = errors[method]
    plt.plot(plt_rng, [i_err[thr] / (n_i) for thr in plt_rng], color=color, ls=ls, linewidth=3, label=name)
plt.title('Illumination')
plt.xlabel('threshold [px]')
plt.xlim(plt_lim)
plt.xticks(plt_rng)
plt.ylim([0, 1])
plt.gca().axes.set_yticklabels([])
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=20)

plt.subplot(1, 3, 3)
for method, name, color, ls in zip(methods, names, colors, linestyles):
    i_err, v_err, n_i, n_v = errors[method]
    plt.plot(plt_rng, [v_err[thr] / (n_v) for thr in plt_rng], color=color, ls=ls, linewidth=3, label=name)
plt.title('Viewpoint')
plt.xlim(plt_lim)
plt.xticks(plt_rng)
plt.ylim([0, 1])
plt.gca().axes.set_yticklabels([])
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=20)

plt.savefig('hseq.pdf', bbox_inches='tight', dpi=300)
