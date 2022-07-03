import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
dataset = 'tiny-imagenet-200' # tiny-imagenet-200
dct_size = 8
img_size = 56*dct_size
backbone = 'resnet50_dct' # resnet50_dct, deit_dct_mix
mode = 'resnet_50_dct' # resnet_50_dct,deit_small_dct_mix_8
fold = 'resnet50' # resnet50, resnet10
path = './save/'+fold+'-expr/'+ dataset+'-'+backbone+'-sz'+ str(img_size) + '-bs128-g'+ str(dct_size)+'/'+mode+'_'+str(img_size)+'/default'
data = np.load(path + '/avg_channel_heatmaps_rlt_avgbatch.npy')
data = data.reshape(3*dct_size**2, -1)
data_mean = np.round(np.mean(data, axis = -1), 4).reshape(3,dct_size,dct_size)
np.save(path + '/round.npy', data_mean)
fig = plt.figure(figsize = (16,4))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

new_cmap = plt.cm.Blues

sns.heatmap(data=data_mean[0], ax=ax1, cmap=new_cmap, annot = True, fmt=".3f",  annot_kws={"fontsize":8})
sns.heatmap(data=data_mean[1], ax=ax2, cmap=new_cmap, annot = True, fmt=".3f", annot_kws={"fontsize":8})
sns.heatmap(data=data_mean[2], ax=ax3, cmap=new_cmap, annot = True, fmt = ".3f",  annot_kws={"fontsize":8})

ax1.set(xlabel = "Y")
ax2.set(xlabel = "Cr")
ax3.set(xlabel = "Cb")
# update the desired text annotations
for text in ax1.texts:
    if text.get_text() == '0.000':
        text.set_text('')
for text in ax2.texts:
    if text.get_text() == '0.000':
        text.set_text('')
for text in ax3.texts:
    if text.get_text() == '0.000':
        text.set_text('')

plt.savefig(path + '/sns_'+str(dct_size)+'.png')
