
###########################################################################
# (c) 2017 Koh Young Research America (KYRA)                              #
#                                                                         #
# Proprietary and confidential                                            #
###########################################################################
#
# remove k3d file if its part name is not in pkg file
#
import os
import argparse
import sys
sys.path.append("/home/junwon/smt-project/SMT-newtest/detect_part")
import smt_process.detect_class as detect_class

desc="""
Remove k3d file if its part name is not in mapping table
"""


#parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawTextHelpFormatter)
#group = parser.add_argument_group()
#group.add_argument("-d", "--dpath",default="", help="The k3d directory containing k3d files")
#group.add_argument("-p", "--ppath", default="", help="pkg file path")
#
#args = parser.parse_args()
#
#if args.dpath !="" and args.ppath !="":
#    if os.path.exists(os.path.expanduser(args.dpath)) and os.path.exists(os.path.expanduser(args.ppath)):
#        DATA_HOME= os.path.expanduser(args.dpath)
#        PKG_FILE = os.path.expanduser(args.ppath)
#        print "Setting DATA_HOME:",DATA_HOME
#        print "Setting PKG_FILE:",PKG_FILE
#    else:
#        print "Invalid path to k3d directory or pkg file"
#        sys.exit()
#else:
#    print "Path to the k3d directory and pkd file"
#    sys.exit()

        
DATA_HOME="/home/junwon/smt-data/Train_0818_extra/Kohyoung_Board2_Measure/"

MAPPING_FILE="/home/junwon/smt-project/SMT-newtest/detect_part/training_data/Template2Part_Map.csv"

part_nm_list=set()
with open(MAPPING_FILE,'r') as f:
    next(f) # skip the first row
    for index,line in enumerate(f):
        part_nm = line.split(",")[2]
        part_nm = part_nm.replace("\"","")
        part_nm = part_nm[0:20]
        part_nm_list.add(part_nm)


print "part_nm_list:",part_nm_list


remove_list=[]

for d in os.listdir(DATA_HOME):
    if d.endswith('txt'): # skip .txt 
        continue

    f = open(DATA_HOME + d,'r')

    d_name = d.split('.')[0]
    
    DC = detect_class.ComponentDetector()
    k3dfile = DC.parse_k3d(f.read())
    f.close()
    cid = k3dfile['component_id']
    if not cid in part_nm_list:
        print 'no exist:',d, ",id:",cid
        print "d_name:",d_name
        remove_list.append(d_name)
    else:
        print 'exist:',d,",id:",cid


print len(remove_list)
print remove_list

## now remove k3d files
for d2 in remove_list:
    os.remove(DATA_HOME + d2 + ".k3d")
    print 'succeed to rm ',d2, '.k3d'
    #os.remove(DATA_HOME + d2 + ".txt")
    #print 'succeed to rm ',d2, '.txt'
print 'done'


