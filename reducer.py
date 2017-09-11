
import sys

current_id = None
id = None
feats1 = None
feats2 = None
for line in sys.stdin:
    line = line.strip()

    tokens = line.split(",")

    id = tokens[0]

    if len(tokens) == 5:
        feats1 = ','.join(tokens[1:])
    else:
        feats2 = tokens[1]

    if current_id == id:
        print '%s,%s,%s' % (id,feats1,feats2)
    else:
        current_id = id



