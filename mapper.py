import sys

for line in sys.stdin:
    line = line.strip()
    
    tokens=line.split(",")

    if len(tokens) == 5:
        print line
    else:
        tokens=line.split("\t")
        id = tokens[0]
        feats = tokens[1]
        print (id + "," + feats)

        
	
