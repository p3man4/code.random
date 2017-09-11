import os
from datetime import datetime

IN_FILE="/home/junwon/smt-data/Train_0818/Template2Part_Map.csv"

def build_table(filename):
        
    part_template=dict()

    with open(filename,'r') as f:
        next(f) # skip first line
        for index, line in enumerate(f):
            template_nm = line.split(",")[1]
            part_nm = line.split(",")[2]
            
            template_nm = template_nm.replace("\"","")
            part_nm = part_nm.replace("\"","")

            write_t = line.split(",")[4]
            modf_t = line.split(",")[5]
           
            year=write_t.split(" ")[0].split("-")[0]
            year = year.replace("\"","")

            month = write_t.split(" ")[0].split("-")[1]
            day = write_t.split(" ")[0].split("-")[2]

            hour = write_t.split(" ")[1].split(":")[0]
            minute = write_t.split(" ")[1].split(":")[1]
            second = write_t.split(" ")[1].split(":")[2]
            second = second.replace("\"","")
            
            dt = datetime(int(year),int(month),int(day),int(hour),int(minute),int(second))

            if part_nm not in part_template.keys():
                part_template[part_nm] = (template_nm,dt)
            else:
                t_template_nm,t_dt = part_template[part_nm]

                if dt > t_dt:
                    part_template[part_nm] = (template_nm,dt)

    return part_template

table = build_table(IN_FILE)
print table
for key in table.keys():
    print 'key:',key + ',value:' + table[key][0]
