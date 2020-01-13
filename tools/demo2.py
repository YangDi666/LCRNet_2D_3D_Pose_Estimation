
import sys, os, pdb
import numpy as np
import pickle
from PIL import Image
from detect_pose import detect_pose
from lcr_net_ppi import LCRNet_PPI
import scene
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
#plt.ion()


def display_poses( image, detections, njts, save_name):
    if njts==13:
      left  = [(9,11),(7,9),(1,3),(3,5)] # bones on the left
      right = [(0,2),(2,4),(8,10),(6,8)] # bones on the right
      right += [(4,5),(10,11)] # bones on the torso
      # (manually add bone between middle of 4,5 to middle of 10,11, and middle of 10,11 and 12)
      head = 12
    elif njts==17:
      left  = [(9,11),(7,9),(1,3),(3,5)] # bones on the left
      right = [(0,2),(2,4),(8,10),(6,8)] # bones on the right and the center
      right += [(4,13),(5,13),(13,14),(14,15),(15,16),(12,16),(10,15),(11,15)]  # bones on the torso
      head = 16

    fig = plt.figure()
    
    # 2D 
    ax = fig.add_subplot(211)
    ax.imshow(image)
    for det in detections:
        pose2d = det['pose2d']
        score = det['cumscore']
        lw = 2
        # draw green lines on the left side
        for i,j in left:
            ax.plot( [pose2d[i],pose2d[j]],[pose2d[i+njts],pose2d[j+njts]],'g', scalex=None, scaley=None, lw=lw)
        # draw blue linse on the right side and center
        for i,j in right:
            ax.plot( [pose2d[i],pose2d[j]],[pose2d[i+njts],pose2d[j+njts]],'b', scalex=None, scaley=None, lw=lw)
        if njts==13:   # other bones on torso for 13 jts
            def avgpose2d(a,b,offset=0): # return the coordinate of the middle of joint of index a and b
                return (pose2d[a+offset]+pose2d[b+offset])/2.0         
            ax.plot( [avgpose2d(4,5),  avgpose2d(10,11)], [avgpose2d(4,5,offset=njts),  avgpose2d(10,11,offset=njts)], 'b', scalex=None, scaley=None, lw=lw)
            ax.plot( [avgpose2d(12,12),avgpose2d(10,11)], [avgpose2d(12,12,offset=njts),avgpose2d(10,11,offset=njts)], 'b', scalex=None, scaley=None, lw=lw)        
        # put red markers for all joints
        ax.plot(pose2d[0:njts], pose2d[njts:2*njts], color='r', marker='.', linestyle = 'None', scalex=None, scaley=None)
        # legend and ticks
        ax.text(pose2d[head]-20, pose2d[head+njts]-20, '%.1f'%(score), color='blue')
    ax.set_xticks([])
    ax.set_yticks([])
        
    # 3D
    ax = fig.add_subplot(212, projection='3d')
    for i,det in enumerate(detections):
        pose3d = det['pose3d']
        score = det['cumscore']
        lw = 2
        def get_pair(i,j,offset): 
            return [pose3d[i+offset],pose3d[j+offset]]
        def get_xyz_coord(i,j): 
            return get_pair(i,j,0), get_pair(i,j,njts), get_pair(i,j,njts*2)
        # draw green lines on the left side
        for i,j in left:
            x,y,z = get_xyz_coord(i,j)
            ax.plot( x, y, z, 'g', scalex=None, scaley=None, lw=lw)
        # draw blue linse on the right side and center
        for i,j in right:
            x,y,z = get_xyz_coord(i,j)
            ax.plot( x, y, z, 'b', scalex=None, scaley=None, lw=lw)
        if njts==13: # other bones on torso for 13 jts
            def avgpose3d(a,b,offset=0): 
                return (pose3d[a+offset]+pose3d[b+offset])/2.0
            def get_avgpair(i1,i2,j1,j2,offset):
                return [avgpose3d(i1,i2,offset),avgpose3d(j1,j2,offset)]
            def get_xyz_avgcoord(i1,i2,j1,j2): 
                return get_avgpair(i1,i2,j1,j2,0), get_avgpair(i1,i2,j1,j2,njts), get_avgpair(i1,i2,j1,j2,njts*2)
            x,y,z = get_xyz_avgcoord(4,5,10,11)
            ax.plot( x, y, z, 'b', scalex=None, scaley=None, lw=lw)
            x,y,z = get_xyz_avgcoord(12,12,10,11)
            ax.plot( x, y, z, 'b', scalex=None, scaley=None, lw=lw)
        # put red markers for all joints
        ax.plot( pose3d[0:njts], pose3d[njts:2*njts], pose3d[2*njts:3*njts], color='r', marker='.', linestyle = 'None', scalex=None, scaley=None)
        # score
        ax.text(pose3d[head]+0.1, pose3d[head+njts]+0.1, pose3d[head+2*njts], '%.1f'%(score), color='blue')
    # legend and ticks
    ax.set_aspect('equal')
    ax.elev = -90
    ax.azim = 90
    ax.dist = 8
    ax.set_xlabel('X axis', labelpad=-5)
    ax.set_ylabel('Y axis', labelpad=-5)
    ax.set_zlabel('Z axis', labelpad=-5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    #plt.show()
    #pdb.set_trace()
    
    # Save .png file
    save_name=save_name.split('/')[-1]
    test_dir = os.path.join('tests')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    plt.savefig(test_dir+'/result_'+save_name[:-4]+'.png')

def demo( imagename, modelname, gpuid):

    fname = os.path.join(os.path.dirname(__file__), '../models', modelname+'.pkl')
    if not os.path.isfile(fname):
        # Download the files 
        dirname = os.path.dirname(fname)
        if not os.path.isdir(os.path.dirname(fname)):
            os.system('mkdir -p "{:s}"'.format(dirname))
        os.system('wget http://pascal.inrialpes.fr/data2/grogez/LCR-Net/pthmodels/{:s} -P {:s}'.format(modelname+'.pkl', dirname))        
        if not os.path.isfile(fname):
            raise Exception("ERROR: download incomplete")

    with open(fname, 'rb') as fid:
      model = pickle.load(fid)
            
    anchor_poses = model['anchor_poses']
    K = anchor_poses.shape[0]
    njts = anchor_poses.shape[1]//5 # 5 = 2D + 3D

    if  imagename[-4:]=='.jpg' or imagename[-4:]=='.png':
        img_output_list = [(imagename, None)]
    else:
        imagedir=imagename
        imagenames=os.listdir(imagedir)
        img_output_list = [(imagedir+'/'+imagename, None) for imagename in imagenames]
    

    # run lcrnet on a list of images
    res = detect_pose( img_output_list, model['model'], model['cfg'], anchor_poses, njts, gpuid=gpuid)
    
    projmat = np.load( os.path.join(os.path.dirname(__file__),'standard_projmat.npy') )
    projMat_block_diag, M = scene.get_matrices(projmat, njts)

    rows=[]
    for i,(imname,_) in enumerate(img_output_list): # for each image
        image = np.asarray(Image.open(imname))
        resolution = image.shape[:2]

        # perform postprocessing
        print('postprocessing (PPI) on image ', imname)
        detections = LCRNet_PPI(res[i], K, resolution, J=njts, **model['ppi_params'])             
        
        # save result as annotation
        set_dir = os.path.join('data/new_set')
        if not os.path.exists(set_dir):
            os.makedirs(set_dir)
        img_dir = os.path.join('data/new_set/images')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        print(imname)
        os.system('cp '+imname+' '+'data/new_set/images/'+imname.split('/')[-1])
        #open /data/new_set/images/
       
        for det in detections:
            pose2d=list(det['pose2d'])
            pose3d=list(det['pose3d'])
            box=[int(min(pose2d[:13])-5), int(min(pose2d[13:])-5), int(max(pose2d[:13])+5), int(max(pose2d[13:])+5)]
            rows.append([imname]+box+ pose2d+pose3d+['person'])
                 
        # move 3d pose into scene coordinates
        print('3D scene coordinates regression on image ', imname)
        for detection in detections:
          delta3d = scene.compute_reproj_delta_3d(detection, projMat_block_diag, M, njts)
          detection['pose3d'][      :  njts] += delta3d[0]
          detection['pose3d'][  njts:2*njts] += delta3d[1]
          detection['pose3d'][2*njts:3*njts] -= delta3d[2]
        
        # show results   
        display_poses(image, detections, njts, imname)
        print('result saved to tests/result_'+imname.split('/')[-1][:-4]+'.png')
        
    #/data/new_set/csv    
    with open('data/new_set/train.csv',"a+", newline='') as f: 
        f_csv = csv.writer(f) 
        f_csv.writerows(rows)
    print('annotation saved to data/newset')

if __name__=="__main__":
    if len(sys.argv) not in [3,4]:
        print("Usage: python demo.py <modelname> <imagename> [<gpuid>]")
        sys.exit(1)
    modelname = sys.argv[1]
    imagename = sys.argv[2]
    gpuid = int(sys.argv[3]) if len(sys.argv)>3 else -1

    #if modelname not in ['H36_R50_1M4_K100_fg7
    #    print("ERROR: Unknown modelname {:s}, it should be R50FPN_2M7_K100x2_fg5".format(modelname))
    #    sys.exit(1)
    
    demo(imagename, modelname, gpuid)
