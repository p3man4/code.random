
###########################################################################
# (c) 2017 Koh Young Research America (KYRA)                              #
#                                                                         #
# Proprietary and confidential                                            #
###########################################################################
#
# remove k3d file if its part name is not in pkg file
#
import os
import smt_process.detect_class as detect_class
import argparse
import sys


desc="""
Remove k3d file if its part name is not in pkg file
"""


parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawTextHelpFormatter)
group = parser.add_argument_group()
group.add_argument("-d", "--dpath",default="", help="The k3d directory containing k3d files")
group.add_argument("-p", "--ppath", default="", help="pkg file path")

args = parser.parse_args()

if args.dpath !="" and args.ppath !="":
    if os.path.exists(os.path.expanduser(args.dpath)) and os.path.exists(os.path.expanduser(args.ppath)):
        DATA_HOME= os.path.expanduser(args.dpath)
        PKG_FILE = os.path.expanduser(args.ppath)
        print "Setting DATA_HOME:",DATA_HOME
        print "Setting PKG_FILE:",PKG_FILE
    else:
        print "Invalid path to k3d directory or pkg file"
        sys.exit()
else:
    print "Path to the k3d directory and pkd file"
    sys.exit()

        
#DATA_HOME="/home/junwon/smt-data/Train_All_copied/A2C7437850000-TOP-PKG/"
#DATA_HOME="/home/junwon/smt-data/Train_All_copied/CBC_A2C7384620600/"
#DATA_HOME="/home/junwon/smt-data/Train_All_copied/KY_BOARD/"
#DATA_HOME="/home/junwon/smt-data/Train_All_copied/REM_A2C7311600800/"
#PKG_FILE="/home/junwon/smt-project/SMT/detect_part/training_data/kypkg_db_updated.pkg"

comp_id_list=set()
with open(PKG_FILE,'r') as f:
    for index,line in enumerate(f):
        comp_id = line.split("@")[0]
        comp_id_list.add(comp_id)


print "comp list:",len(comp_id_list)


remove_list=[]

for d in os.listdir(DATA_HOME):
    if not d.endswith('k3d'):
        continue

    f = open(DATA_HOME + d,'r')

    d_name = d.split('.')[0]
    
    DC = detect_class.ComponentDetector()
    k3dfile = DC.parse_k3d(f.read())
    f.close()
    cid = k3dfile['component_id']
    if not cid in comp_id_list:
        print 'no exist:',d
        remove_list.append(d_name)
    else:
        print 'exist:',d


print len(remove_list)
print remove_list

## now remove k3d files
for d2 in remove_list:
    os.remove(DATA_HOME + d2 + ".k3d")
#    print 'succeed to rm ',d2, '.k3d'
    os.remove(DATA_HOME + d2 + ".txt")
#    print 'succeed to rm ',d2, '.txt'
print 'done'


