# process image assistant download file to normal txtfile
# with each line a url to download

f = open('mj.txt','r')
alldata = f.readlines()
urls = []
for line in alldata:
    parse=line.split()
    if len(parse)!=0:
        urls.append(parse[2].replace('"',''))

#保存数据txt文件
with open(r'mdurls.txt',"w",encoding='utf-8') as l:
    for var in urls:
        l.write(str(var) + '\n')
