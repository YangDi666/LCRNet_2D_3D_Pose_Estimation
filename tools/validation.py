import torch

import sys, os, pdb
import numpy as np
import pickle
from PIL import Image
from detect_pose import detect_pose
from lcr_net_ppi import LCRNet_PPI
import scene
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
from tqdm import tqdm


NT=5
def get_pose( imagedir, checkpoint, gpuid):

    load_name = checkpoint
    logging.info("loading checkpoint %s", load_name)
    
    # Load checkpoint
    if load_name[-4:]=='.pkl':
        fname = os.path.join(os.path.dirname(__file__), 'models', load_name)
        if not os.path.isfile(fname):
            # Download the files 
            dirname = os.path.dirname(fname)
            if not os.path.isdir(os.path.dirname(fname)):
                os.system('mkdir -p "{:s}"'.format(dirname))
            os.system('wget http://pascal.inrialpes.fr/data2/grogez/LCR-Net/pthmodels/{:s} -P {:s}'.format(load_name, dirname))        
            if not os.path.isfile(fname):
                raise Exception("ERROR: download incomplete")
    
        with open(fname, 'rb') as fid:
            model0 = pickle.load(fid)
            checkpoint=model0
    else:
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        if 'losses' in checkpoint.keys():
            losses=checkpoint['losses']   
    
    # Load anchor configs
    anchorfile='configs/anchors/anchor_cfg.pkl'
    with open(anchorfile, 'rb') as fid:
       anchor_cfg = pickle.load(fid)
    
    model={}   
    model['model']=checkpoint['model'] 
    model['anchor_poses'] = anchor_cfg['anchor_poses'][:,:13*NT]
    model['cfg']=anchor_cfg['cfg']
    model['ppi_params']=anchor_cfg['ppi_params']
    
    K = model['anchor_poses'].shape[0]
    njts = model['anchor_poses'].shape[1]//5 # 5 = 2D + 3D
    # save the model
    model_dir = os.path.join('models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(model_dir+'/'+load_name.split('/')[-1][:-4]+'.pkl', 'wb') as f:
        pickle.dump(model, f)
        print(load_name+'.pkl saved ')
        f.close()
    
    
    imagenames=os.listdir(imagedir)
    img_output_list = [(imagedir+'/'+imagename, None) for imagename in imagenames]
   
    # run lcrnet on a list of images
    res = detect_pose( img_output_list, checkpoint['model'], model['cfg'], model['anchor_poses'], njts, gpuid=gpuid)
    
    projmat = np.load( os.path.join(os.path.dirname(__file__),'standard_projmat.npy') )
    projMat_block_diag, M = scene.get_matrices(projmat, njts)
 
    results={}
    print('postprocessing (PPI) ')
    for i,(imname,_) in tqdm(enumerate(img_output_list)): # for each image
        image = np.asarray(Image.open(imname))
        resolution = image.shape[:2]

        # perform postprocessing

        detections = LCRNet_PPI(res[i], K, resolution, J=njts, **model['ppi_params'])             
        results[imname]=detections
              
    return results

