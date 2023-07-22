file = open("/home/lohith/te_male/male_line_index.tsv","r")
lines  = file.readlines()
spkrs = []
for i in lines:
    if i.split("\t")[0].split("_")[1] not in spkrs:
        spkrs.append(i.split("\t")[0].split("_")[1])
print(len(spkrs))
print(spkrs)
for i in lines:
    path,trans = i.split("\t")
    new_file = open("/home/lohith/te_male/"+path+".txt","w")
    new_file.write(trans[:-1])
    new_file.close()