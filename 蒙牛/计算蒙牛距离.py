import matplotlib
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.metrics import pairwise_distances
from sqlalchemy import create_engine


def trans(m):
    return list(zip(*m))
def computenaifen2naifen(X):


    Y = X
    naifennum = len(X)
    OUT = []

    for j in range(naifennum):
        for i in range(naifennum):
            Z = []
            Z.append(X.iloc[j])
            Z.append(Y.iloc[i])
            if i == j:
                OUT.append(0)
                continue
            OUT.append(pairwise_distances(Z, metric="cosine")[1][0])
    X = [OUT[i:i + 1] for i in range(0, len(OUT), 1)]
    X1 = []
    for i in X:
        i = list(map(float, i))
        X1.append(i)
    X = X1
    Y = []
    temp = []
    cnt = 0
    for i in X:
        cnt = cnt + 1
        if cnt == naifennum:
            temp.append(i)
            temp = trans(temp)
            Y.extend(temp)
            cnt = 0
            temp = []
        else:
            temp.append(i)
    new_list = Y
    temp2 = []
    for i in new_list:
        temp1 = list(i)
        temp2.append(temp1)
    print('naifen2naifen')
    return temp2

def computema2ma(X):


    Y = X
    naifennum = len(X)
    OUT = []

    for j in range(naifennum):
        for i in range(naifennum):
            Z = []
            Z.append(X.iloc[j])
            Z.append(Y.iloc[i])
            if i == j:
                OUT.append(0)
                continue
            OUT.append(pairwise_distances(Z, metric="euclidean")[1][0])
    X = [OUT[i:i + 1] for i in range(0, len(OUT), 1)]
    X1 = []
    for i in X:
        i = list(map(float, i))
        X1.append(i)
    X = X1
    Y = []
    temp = []
    cnt = 0
    for i in X:
        cnt = cnt + 1
        if cnt == naifennum:
            temp.append(i)
            temp = trans(temp)
            Y.extend(temp)
            cnt = 0
            temp = []
        else:
            temp.append(i)
    new_list = Y
    temp2 = []
    for i in new_list:
        temp1 = list(i)
        temp2.append(temp1)
    print('mama2mama')
    return temp2

def ave(X):
    new_list1 = []
    for i in X:
        temp = 0
        ilen = len(i)
        for j in range(len(i)):
            if j == len(i)-1:
                new_list1.append(i[j])
                new_list1.append(temp)
                # new_list1.insert(0, temp)
            else:
                new_list1.append(i[j])
                temp = temp + i[j]
    iilen = ilen + 1
    new_list_t = [new_list1[i:i + iilen] for i in range(0, len(new_list1), iilen)]
    print('ave')
    return new_list_t

def max(X,mamaid):

    averange = []
    delmama = len(mamaid)*0.2
    cnt = 0
    for i in X:
        averange.append(i[len(i) - 1])
        averange.append(mamaid[cnt])
        cnt = cnt + 1

    X = [averange[i:i + 2] for i in range(0, len(averange), 2)]
    X.sort()
    X.reverse()
    l = []
    cnt = 0
    for a in X:
        cnt = cnt + 1
        if cnt <= delmama:
            l.append(a[1])
        else:
            break
    print('max')
    return l
def computema2naifen(X,Y,naifen,filename):
    naifennum = 24
    nummamamam = len(X)
    OUT = []
    for j in range(nummamamam):

        for i in range(naifennum):
            Z = []
            Z.append(X.iloc[j])
            Z.append(Y.iloc[i])
            # print('1欧氏距离')
            # OUT.append(pairwise_distances(Z, metric="euclidean")[1][0])
            OUT.append((pairwise_distances(Z, metric="cosine")[1][0]))
            # print('3马氏')
            # x = np.array(Z)
            # xT=x.T
            # D=np.cov(x)
            # invD=np.linalg.inv(D)
            # tp=xT[0]-xT[1]
            # OUT.append(np.sqrt(dot(dot(tp,invD),tp.T)))
            # print('4曼哈顿')
            # OUT.append(pairwise_distances(Z, metric="manhattan")[1][0])
            # print('5切比雪夫')
            # OUT.append(pairwise_distances(Z, metric="chebyshev")[1][0])
            OUT.append(pairwise_distances(Z, metric="braycurtis")[1][0])
            # print('7canberra')
            # OUT.append(pairwise_distances(Z, metric="canberra")[1][0])
            u = Z[0]
            v = Z[1]
            umu = np.average(u)
            vmu = np.average(v)
            u = u - umu
            v = v - vmu
            uv = np.average(u * v)
            uu = np.average(np.square(u))
            vv = np.average(np.square(v))
            dist = 1.0 - uv / np.sqrt(uu * vv)
            OUT.append(dist)
            # print('9sqeuclidean')
            # OUT.append(pairwise_distances(Z, metric="sqeuclidean")[1][0])
    X = [OUT[i:i + 3] for i in range(0, len(OUT), 3)]
    X1 = []
    for i in X:
        i = list(map(float, i))
        X1.append(i)
    X = X1


    Y = []
    temp = []
    cnt = 0
    for i in X:
        cnt = cnt + 1
        if cnt == nummamamam:
            temp.append(i)
            temp = trans(temp)
            Y.extend(temp)
            cnt = 0
            temp = []
        else:
            temp.append(i)

    new_list = Y
    temp2 = []
    for i in new_list:
        temp1 = list(i)
        temp2.append(temp1)

    new_list1 = []
    ilen = 0

    for i in temp2:
        temp = 0
        ilen = len(i)
        for j in range(len(i)):
            if j == len(i)-1:
                new_list1.append(i[j])
                temp = temp / len(i)
                new_list1.append(temp)
            else:
                new_list1.append(i[j])
                temp = temp + i[j]
    iilen = ilen + 1
    new_list = [new_list1[i:i + iilen] for i in range(0, len(new_list1), iilen)]


    Z = naifen
    Z3 = [
        "余弦距离",
        "braycurtis",
        "correlation",
    ]

    distancenum = len(Z3) - 1
    count = -1
    cnt = 1
    zcnt = 0
    for i in new_list:
        count = count + 1
        if count == distancenum:
                i.insert(0, Z3[count])
                i.insert(0, Z[zcnt])
                zcnt = zcnt + 1
                cnt = cnt + 1
                count = -1

        else:
                i.insert(0, Z3[count])
                i.insert(0, Z[zcnt])


    # data = DataFrame(new_list)
    # num = nummamamam + 2
    # data = pd.concat([data.loc[:,0:1],data.loc[:,num]], axis=1)
    # #data = data.loc[:,0:1]+data.loc[:,num]
    # #data.sort_values(by=[2,1])
    # data.sort_values(by=[2,1], ascending=[False])
    # # idex = np.lexsort([data[:, 3], data[:, 1]])
    # # new_list = data[idex, :]

    data = np.array(new_list)
    num = nummamamam + 2
    data = np.c_[data[:, 0:2], data[:, num]]

    idex = np.lexsort([data[:, 2], data[:, 1]])
    new_list = data[idex, :]

    head1 = ["配方粉名称",
             "距离名称",
             "平均值"]
    # for i in range(nummamamam):
    #     head1.append('A'+str(i+1))
    # head1.append("平均值")

    head1 = np.array(head1)
    new_list = np.insert(new_list, 0, values=head1, axis=0)
    data1 = DataFrame(new_list)
    data1.to_csv('奶粉和妈妈' +str(filename) + '.csv',encoding='utf-8-sig')
    print('mama2naifen')
    return new_list

def linkdatabase(time,num,delmama,delmamid,dim,naifen):
    engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format('zdd', 'root', 'localhost', '3306', 'zhangduoduo_www'))
    delmamid = tuple(delmamid)
    delmamid = str(delmamid)
    if delmama == 1 and naifen == 0 :
        sql_query = 'SELECT mother_id, AVG(lai) AS lai,AVG(su) AS su,AVG(jie) AS jie,AVG(dan) AS dan,AVG(yiliang) AS yiliang,AVG(liang) AS liang,AVG(benbing) AS benbing,AVG(se) AS se,AVG(zu) AS zu,AVG(tiandong) AS tiandong,AVG(si) AS si,AVG(gu) AS gu,AVG(gan) AS gan,AVG(bing) AS bing,AVG(lao) AS lao,AVG(jing) AS jing,AVG(fu) AS fu,AVG(banguang) AS banguang FROM ' + str(time) + ' where miruqi = ' + str(num) + ' and mother_id not in ' + delmamid + ' group by mother_id'
    elif delmama != 1 and naifen == 0 :
        sql_query = 'SELECT mother_id, AVG(lai) AS lai,AVG(su) AS su,AVG(jie) AS jie,AVG(dan) AS dan,AVG(yiliang) AS yiliang,AVG(liang) AS liang,AVG(benbing) AS benbing,AVG(se) AS se,AVG(zu) AS zu,AVG(tiandong) AS tiandong,AVG(si) AS si,AVG(gu) AS gu,AVG(gan) AS gan,AVG(bing) AS bing,AVG(lao) AS lao,AVG(jing) AS jing,AVG(fu) AS fu,AVG(banguang) AS banguang FROM ' + str(time) + ' where miruqi = ' + str(num) + ' group by mother_id'
    else:
        sql_query = 'select * from ' + time + '  ;'
    X = pd.read_sql_query(sql_query, engine)
    mamaid = X.iloc[:, 0]
    if dim == 3:
        X = X.iloc[:, 6:9]
    elif dim == 9:
        X = X.iloc[:, 1:10]
    elif dim == 11:
        X = pd.concat(
            [X['lai'], X['su'], X['jie'], X['dan'], X['yiliang'], X['liang'], X['benbing'], X['se'], X['zu'], X['lao'],
             X['banguang']], axis=1)
    elif dim == 18:
        X = X.iloc[:, 1:19]
    #dan lao jing guang se
    else:
        #X = X['dan'] + X['lao'] + X['jing'] + X['banguang'] + X['se']
        X = pd.concat([X['dan'] , X['lao'] , X['jing'] , X['banguang'] ,X['se']], axis=1)

    print('linkdatabase')
    return X,mamaid

def naifenpaixu(X,id,filename):
    X = np.array(X)
    id = np.array(id)

    data = np.c_[id, X]

    idex = np.lexsort([data[:, 25]])
    new_list = data[idex, :]

    head1 = ["配方粉名称",
            ]
    for i in range(24):
        head1.append('A'+str(i+1))
    head1.append("平均值")

    head1 = np.array(head1)
    new_list = np.insert(new_list, 0, values=head1, axis=0)
    data1 = DataFrame(new_list)
    data1.to_csv('奶粉和奶粉'+str(filename) + '.csv',encoding='utf-8-sig')
    print('naifenpaixu')
    return new_list
def compara(hanliang,time,dim,fielname):
    X,mamaid = linkdatabase(hanliang,time,0,'',dim,0)
    a = computema2ma(X)
    b = ave(a)
    mutmaxid = max(b,mamaid)
    #def linkdatabase(time,num,delmama,delmamid,dim,naifen):
    X, mamaid =linkdatabase(hanliang,time,1,mutmaxid,dim,0)
    Y, naifenid= linkdatabase('nf_'+str(hanliang),'',0,'',dim,1)
    Z = computema2naifen(X,Y,naifenid,fielname)
    M = naifenpaixu(ave(computenaifen2naifen(Y)),naifenid,fielname)


# compara('xiangdui',4,18,'相对含量-晚期成熟乳-18维')
# compara('xiangdui',4,11,'相对含量-晚期成熟乳-11维')
# compara('xiangdui',4,9,'相对含量-晚期成熟乳-9维')
compara('xiangdui',4,5,'相对含量-晚期成熟乳-5维')
compara('xiangdui',4,3,'相对含量-晚期成熟乳-3维')


compara('juedui',4,18,'绝对含量-晚期成熟乳-18维')
compara('juedui',4,11,'绝对含量-晚期成熟乳-11维')
compara('juedui',4,9,'绝对含量-晚期成熟乳-9维')
compara('juedui',4,5,'绝对含量-晚期成熟乳-5维')
compara('juedui',4,3,'绝对含量-晚期成熟乳-3维')


compara('xiangdui',3,18,'相对含量-早期成熟乳-18维')
compara('xiangdui',3,11,'相对含量-早期成熟乳-11维')
compara('xiangdui',3,9,'相对含量-早期成熟乳-9维')
compara('xiangdui',3,5,'相对含量-早期成熟乳-5维')
compara('xiangdui',3,3,'相对含量-早期成熟乳-3维')


compara('juedui',3,18,'绝对含量-早期成熟乳-18维')
compara('juedui',3,11,'绝对含量-早期成熟乳-11维')
compara('juedui',3,9,'绝对含量-早期成熟乳-9维')
compara('juedui',3,5,'绝对含量-早期成熟乳-5维')
compara('juedui',3,3,'绝对含量-早期成熟乳-3维')
