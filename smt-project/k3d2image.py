#
# input param: input directory
#
#


import smt_process.read_component_db as read_component_db
import smt_process.detect_class as detect_class
import argparse
import os
import cv2
import skimage.io
import skimage.transform

DATA_ROOT='/home/junwon/smt-data/Train_All_filtered/'
IMAGE_ROOT='/home/junwon/smt-data/images/images_gray_png/'

def main():
    desc="""
    generate image directories  from k3d files

    """
    parser = argparse.ArgumentParser(desc,formatter_class=argparse.RawTextHelpFormatter)
    group = parser.add_argument_group()
    group.add_argument("-i", "--inputs", default=os.path.expanduser(DATA_ROOT),
            help = "The input directory path containing k3d files")
    args = parser.parse_args()

    if  args.inputs =="":
        print "argument is missing"
        return

    subdirs = [x[0] for x in os.walk(args.inputs)]
    print subdirs
    print  '00000000000'
    for subdir in subdirs:
        print 'subdir:',subdir
        handle_subdir(subdir)
        print '----------------'

def handle_subdir(subdir):
    for filename in os.listdir(subdir):
        if not filename.endswith('k3d'):
            continue
        

        
        f_path = os.path.join(subdir,filename)
        f = open(f_path,'r')
        DC = detect_class.ComponentDetector()
        k3dfile = DC.parse_k3d(f.read())
        f.close()

        comp_id = k3dfile['component_id']
        img_dir = os.path.join(IMAGE_ROOT,comp_id)

        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        k3dnm = filename[:-4]
        img_bgr_path = img_dir + '/' + k3dnm + '_bgr.png'
        img_gray_path = img_dir + '/' + k3dnm + '_gray.png'
        img_3d_path = img_dir + '/' + k3dnm + '_3d.png'

        # resize image to 224 x 224
        #img_gray_resized = skimage.transform.resize(k3dfile['img_gray'],[224,224])
        #cv2.imwrite(img_gray_path,img_gray_resized)
 
        #img_bgr_resized = skimage.transform.resize(k3dfile['img_bgr'],[224,224])
        #cv2.imwrite(img_bgr_path,img_bgr_resized)
          
   #     cv2.imwrite(img_bgr_path,k3dfile['img_bgr'])
        cv2.imwrite(img_gray_path,k3dfile['img_gray'])
   #     cv2.imwrite(img_3d_path,k3dfile['img_3d'].astype(int))
       

main()
