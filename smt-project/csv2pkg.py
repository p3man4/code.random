import json
import csv
IN_FILE='/home/junwon/smt-project/SMT-newtest/detect_part/training_data/Daniel-Post_PkgInfo_201708180740.csv'
#IN_FILE='/home/junwon/smt-project/SMT-newtest/detect_part/training_data/sample.csv'
OUT_FILE='/home/junwon/smt-project/SMT-newtest/detect_part/training_data/Daniel-Post_PkgInfo_201708180740.pkg'


def read_csv(filename,delimiter=",", quotechar="|"):
    ret_val={}

    with open(OUT_FILE,"w") as f:
        with open(filename,"rb") as csvfile:
            next(csvfile) # skip first row
            
            for i, line in enumerate(csvfile):
                tokens = [w.replace("\"","") for w in  line.split(",")]
               # print tokens

                template_nm=tokens[2].strip()
                for token in tokens:
                    key = token.split(":")[0]
                    if "BodyThickness" == key:
                        btn = token.split(":")[1].strip()
                        print token
                        print "btn:",btn
                    if "LeadCnt" == key:
                        lc = token.split(":")[1].strip()
                        print token
                        print "lc:",lc
                    if "PackageNm" == key:
                        pn = token.split(":")[1].strip()
                        print token
                        print "pn:",pn
                    if "Author" == key:
                        author = token.split(":")[1].strip()
                        print token
                        print "author:",author
                    if "PackageTypeCd"  == key:
                        ptc = token.split(":")[1].strip()
                        print token
                        print "ptc:",ptc
                    if "BodyLength" ==  key:
                        bl = token.split(":")[1].strip()
                        print token
                        print "bl:",bl
                    if "BodyHeight" ==  key:
                        bh = token.split(":")[1].strip()
                        print token
                        print "bh:",bh


                f.write(template_nm + '@{"PKG":{"BodyThickness":"' + btn + 
                        '","PackageNm":"' + pn + 
                        '","Author":"KYRA' +  
                        '","BodyHeight":"' + bh + 
                        '","BodyLength":"' + bl + 
                        '","LeadCnt":"' + lc + 
                        '","PackageTypeCd":"' + ptc +
                        '"}}' + '\n')


read_csv(IN_FILE)
