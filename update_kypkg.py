
###################################################################################
# (c) 2017 Koh Young Research America (KYRA)
#
# Proprietary and confidential
####################################################################################
#
# update pkg file according to k3d kypkg file
#


import os
import json
import argparse
import sys

desc="""
Update pkg file according to the k3d kypkg file
"""

parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawTextHelpFormatter)
group = parser.add_argument_group()
group.add_argument("-d", "--dpath", default="", help="kypkg path")
group.add_argument("-p", "--ppath", default="", help="pkg path")
group.add_argument("-f", "--flag", default="a", help="a/w to append or write")

args = parser.parse_args()

#TRAIN_HOME="/home/junwon/smt-data/Train_kyboard/"

#IN_FILE_LIST=["REM_A2C7311600800/REM_A2C7311600800.kypkg",
#        "A2C7437850000-TOP-PKG/A2C7437850000-TOP-PKG.kypkg",
#        "CBC_A2C7384620600/CBC_A2C7384620600.kypkg",
#        "KY_BOARD/KY_BOARD.kypkg"
#        ]
#




IN_FILE_LIST=["KY_BOARD/KY_BOARD.kypkg"]

OUT_FILE="/home/junwon/smt-project/SMT/detect_part/training_data/kypkg_db_kyboard.pkg"

def main():
    cid_list=set()

    # bring out existing content
#    with open(OUT_FILE,'r') as f:
#        for line,row in enumerate(f):
#            cid = row.split('@')[0]
#            cid_list.add(cid)
#
#    print cid_list
#

    for in_file in IN_FILE_LIST:
        infile =os.path.join(TRAIN_HOME,in_file)
        print 'infile:',infile
        parser(infile,cid_list)



def parser(infile,cid_list):
    in_dicts=[]
    with open(infile,'r') as f:
        in_dicts = eval(f.read())

    with open(OUT_FILE,'a+') as g:

        print len(in_dicts['PKGList'])
    
        for pkg in in_dicts['PKGList']:
            packagenm = pkg['PKG']['PackageNm']
            print 'packagenm:',packagenm
            bodythickness = pkg['PKG']['BodyThickness']
            author="KYRA"
            bodyheight=pkg['PKG']['BodyHeight']
            bodylength=pkg['PKG']['BodyLength']
            leadcnt= pkg['PKG']['LeadCnt']
            packagetypecd = pkg['PKG']['PackageTypeCd']

            out_pkg=dict()
            out_pkg["BodyThickness"]=bodythickness
            out_pkg["PackageNm"]=packagenm
            out_pkg["Author"]=author
            out_pkg["BodyHeight"]=bodyheight
            out_pkg["BodyLength"]=bodylength
            out_pkg["LeadCnt"]=leadcnt
            out_pkg["PackageTypeCd"]=packagetypecd

            out_dict=dict()
            out_dict["PKG"]=out_pkg
            out_str = packagenm + "@" + json.dumps(out_dict)
           
            #if not packagenm in cid_list:
                # print out_str
            g.write(out_str + "\n")

main()
