import torch
import matplotlib.pyplot as plt
import sys
import logging
import os

load_name = sys.argv[1]
logging.info("loading checkpoint %s", load_name)       
checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
print(checkpoint.keys())
losses=checkpoint['losses']
t=range(len(losses))
t2=range(0, len(losses),100)
losses2=[losses[i] for i in t2]
fig=plt.figure()
ax=fig.add_subplot(111)
ax.set_title('Training Loss')
ax.set_xlabel('Step')
ax.set_ylabel('Total loss')
#ax.plot(t, losses, color='b')
ax.plot(t2, losses2)

#plt.show()
# Save file
eva_dir = os.path.join('evaluations')
if not os.path.exists(eva_dir):
    os.makedirs(eva_dir)
plt.savefig(eva_dir+'/training_losses_'+load_name.split('/')[-1][:-4]+'.png')