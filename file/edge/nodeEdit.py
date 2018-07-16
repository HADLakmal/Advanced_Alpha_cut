import csv


with open("../edgeTestPickme/edges.txt") as f:
    datas = f.read()

data = datas.split('\n')
print(data[0].split(" "))
sValue = ""
array = []
arrayCounting = 0
for k in data:
    if arrayCounting in array:
        arrayCounting+=1
        continue
    splitted = k.split(" ")
    count = 0
    max = 0
    for r in data:
        tempSplit = r.split(" ")
        if(splitted[0]==tempSplit[0])&(splitted[1]==tempSplit[1]):
            if count in array:
                break
            else:
                print(count)
                array.append(count)
                if(max<int(splitted[2])):
                    max = int(splitted[2])
        count+=1
    print("......")
    if max!=0:
        sValue+= splitted[0]+" "+splitted[1]+" "+splitted[2]+"\n"
    arrayCounting+=1

with open("edgeList.txt","w+") as f:
    datas = f.write(sValue)