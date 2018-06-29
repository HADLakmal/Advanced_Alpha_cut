import numpy as np




data = dict()
file = str()
with open("file/edge/supernodes4.txt","r") as raw_data:
    file = raw_data.read()
count = -1
keycount = 0
while(count<len(file)-1):
    count += 1
    if ':' in file[count]:
        start = 0
        if keycount<10:
            key = file[count - 1:count]
        elif keycount<100:
            key = file[count - 2:count]
        else:
            key = file[count - 3:count]
        value = ''
        while (count < len(file)-1):
            count += 1
            if '[' in file[count]:
                start = count
            if ']' in file[count]:
                keycount+=1
                value = file[start+1:count].split(", ")
                data[int(key)] = value
                break

with open("test_4.txt") as f:
    datas = f.read()
# print(data.split(']')[0])
partition_data= datas.split(']')[0].replace('[','').split(', ')
# partition_data2 = datas.split(']')[1].replace('[', '').split(', ')
# partition_data3 = datas.split(']')[2].replace('[', '').split(', ')
# partition_data4 = datas.split(']')[3].replace('[', '').split(', ')

partition = []
parttemp = []
for k in partition_data:
    for c in data[int(k)]:
        parttemp.append(int(c))
partition.append(parttemp)
# parttemp = []
# for k in partition_data2:
#     for c in data[int(k)]:
#         parttemp.append(int(c))
# partition.append(parttemp)
# parttemp = []
# for k in partition_data3:
#     for c in data[int(k)]:
#         parttemp.append(int(c))
# partition.append(parttemp)
# parttemp = []
# for k in partition_data4:
#     for c in data[int(k)]:
#         parttemp.append(int(c))
# partition.append(parttemp)
#np.savetxt('test_1.txt', partition,fmt='%r')
with open("file/edge/edges3.txt") as f:
    datas = f.read()

data = datas.split('\n')
partitions = []

for t in partition:
    temppartition = []
    print(t)
    for f in t:
        for r in data:

            if int(f)==int(r.split(" ")[4]):
                temppartition.append(int(r.split(" ")[5]))
                temppartition.append(int(r.split(" ")[6]))
                break
    partitions.append(temppartition)
temppartition = []
partitions.append(temppartition)

np.savetxt('test_5.txt', partitions,fmt='%r')
# for i in data:
#     print(i.split(" "))

