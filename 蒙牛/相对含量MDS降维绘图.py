import matplotlib
import matplotlib.pyplot as plt
from nltk.classify import svm
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
matplotlib.rc("font",family='YouYuan')
def trans(m):
    return list(zip(*m))
def linkdatabase(num,dim):
    engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format('zdd', 'root', 'localhost', '3306', 'zhangduoduo_www'))

    #sql_query = 'select * from xiangdui where miruqi = '+ num +' ;'
    sql_query = 'SELECT mother_id,birth_jd,city, AVG(lai) AS lai,AVG(su) AS su,AVG(jie) AS jie,AVG(dan) AS dan,AVG(yiliang) AS yiliang,AVG(liang) AS liang,AVG(benbing) AS benbing,AVG(se) AS se,AVG(zu) AS zu,AVG(tiandong) AS tiandong,AVG(si) AS si,AVG(gu) AS gu,AVG(gan) AS gan,AVG(bing) AS bing,AVG(lao) AS lao,AVG(jing) AS jing,AVG(fu) AS fu,AVG(banguang) AS banguang FROM xiangdui where miruqi = ' + str(num) + ' group by mother_id'

    X = pd.read_sql_query(sql_query, engine)

    mamaid = X.iloc[:, 1]
    age = X.iloc[:, 7]
    fenmian = X.iloc[:, 14]
    chanci = X.iloc[:, 9]
    season = X['birth_jd']
    city = X['city']

    if dim == 3:
        X = pd.concat([X['benbing'] , X['se'] , X['zu'] ,], axis=1)
    elif dim == 9:
        X = pd.concat([X['lai'] , X['su'] , X['jie'] ,X['dan'] , X['yiliang'] , X['liang'],X['benbing'] , X['se'] , X['zu']], axis=1)
    elif dim == 11:
        X = pd.concat([X['lai'] , X['su'] , X['jie'] ,X['dan'] , X['yiliang'] , X['liang'],X['benbing'] , X['se'] , X['zu'], X['lao'] , X['banguang']], axis=1)
    elif dim == 18:
        X = X.iloc[:, 26:44]
    #dan lao jing guang se
    elif dim == 5:
        X = pd.concat([X['dan'], X['lao'], X['jing'], X['banguang'], X['se']], axis=1)
    elif dim == 8:
        #X = X['dan'] +1         2                3      4          5               6         7             8         9         10       11
        X = pd.concat([ X['su'] , X['jie'] , X['dan']  , X['benbing'] , X['se'] , X['zu'] ,X['lao'] ,X['banguang']], axis=1)
    else:
        X = pd.concat([ X['dan'], X['zu'], X['lao'], X['banguang']], axis=1)

    print('linkdatabase')
    return X,mamaid,chanci,age,fenmian,season,city

def comMDS(X,para):
    Y = X
    naifennum = len(X)

    OUT = []


    for j in range(naifennum):

        for i in range(naifennum):
            if i == 840:
                i = i


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
    print(X)

    Y = []
    temp = []
    cnt = 0
    for i in X:
        cnt = cnt +1
        if cnt == naifennum:
            temp.append(i)
            temp = trans(temp)
            Y.extend(temp)
            cnt = 0
            temp = []
        else:
            temp.append(i)
    new_list = Y
    temp1 = []
    temp2 = []
    for i in new_list:
        temp1 = list(i)
        temp2.append(temp1)
    new_list = temp2


    clf2 = MDS(para)
    data = new_list
    clf2.fit(data)

    iris_t2=clf2.fit_transform(data)
    return iris_t2
def compca(X,para):
    bnum = para
    pca = PCA(n_components=bnum)
    m = pca.fit_transform(X)
    X = pca.inverse_transform(m)
    a = [[] for i in m[0]]
    for i in m:
        for j in range(len(i)):
            a[j].append(i[j])
    principalComponents = a

    print(principalComponents)
    # 写入

    importance = pca.explained_variance_ratio_
    for i in importance:
        print(i)
    #p = principalComponents.reverse()
    p = trans(principalComponents)
    return p
def trans(m):
    return list(zip(*m))
def plot(iris_t2,filename,mamaid,chanci,age,fenmian,season,city):
    influe = ['季节', '南北方', '分娩方式', '产次-年龄-分娩方式']

    filename1 = filename+str(influe[0])
    plt.title(filename1)
    try:
        lnum = iris_t2.shape[0]
    except:
        lnum = len(iris_t2[0])
    for i in range(lnum):
        if season[i] == 1 :
            plt.plot(iris_t2[i][0], iris_t2[i][1], 'o',color='green' )
        else:
            plt.plot(iris_t2[i][0], iris_t2[i][1], '*',color='red' )

    plt.xlabel('绿色-冬春,  红色-夏秋')
    plt.savefig(str(filename1) + '.jpg', dpi=300)
    plt.show()
    #
    filename1 = filename + str(influe[1])
    plt.title(filename1)
    for i in range(lnum):
        if city[i] == 1:
            plt.plot(iris_t2[i][0], iris_t2[i][1], 'o', color='green')
        else:
            plt.plot(iris_t2[i][0], iris_t2[i][1], '*', color='red')

    plt.xlabel('绿色-南方,  红色-北方')
    plt.savefig(str(filename1) + '.jpg', dpi=300)
    plt.show()


def plot_3d(iris_t2,filename,mamaid,chanci,age,fenmian,season,city):
    influe = ['季节', '南北方', '分娩方式', '产次-年龄-分娩方式']
    try:
        lnum = iris_t2.shape[0]
    except:
        lnum = len(iris_t2)
    data = iris_t2


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    filename1 = filename + str(influe[1])
    plt.title(filename1)
    for i in range(lnum):
        if city[i] == 1:
            ax.scatter(data[i][0], data[i][1], data[i][2], marker='o', c='green', )
        else:
            ax.scatter(data[i][0], data[i][1], data[i][2], marker='*', c='red', )
    plt.xlabel('绿色-南方,  红色-北方')
    plt.savefig(str(filename1) + '.jpg', dpi=300)
    plt.show()

def plot_p_2(iris_t2,filename,mamaid,chanci,age,fenmian,num1,num2,num3):
    influe = ['季节', '南北方', '分娩方式', '产次-年龄-分娩方式']

    filename1 = filename + str(influe[0])
    plt.title(filename1)
    numlem = len(iris_t2[0])
    for i in range(numlem):

        if season[i] == 1:
            plt.plot(iris_t2[i][0], iris_t2[i][1], 'o', color='green')
        else:
            plt.plot(iris_t2[i][0], iris_t2[i][1], '*', color='red')

    plt.xlabel('绿色-冬春,  红色-夏秋')
    plt.savefig(str(filename1) + '.jpg', dpi=300)
    plt.show()
    #
    filename1 = filename + str(influe[1])
    plt.title(filename1)
    for i in range(numlem):
        if city[i] == 1:
            plt.plot(iris_t2[i][0], iris_t2[i][1], 'o', color='green')
        else:
            plt.plot(iris_t2[i][0], iris_t2[i][1], '*', color='red')

    plt.xlabel('绿色-南方,  红色-北方')
    plt.savefig(str(filename1) + '.jpg', dpi=300)
    plt.show()




time = ['-PCA-早期-','-MDS-早期-']
dim = ['苯丙氨酸、色氨酸、组氨酸-','苯丙氨酸、色氨酸、组氨酸、酪氨酸、半胱氨酸-',
       '全部必须氨基酸-','全部必须氨基酸+酪氨酸、半胱氨酸-','全部氨基酸-']

#l = [7,8]
l = [7]
for i in l:
    X,mamaid,chanci,age,fenmian,season,city = linkdatabase('3',i)
    m3 = comMDS(X,2)
    filename = str(i)+str(time[1])
    plot(m3,filename,mamaid,chanci,age,fenmian,season,city)
    # p3 = compca(X, 3)
    # filename = str(i)+ str(time[0])
    # plot_3d(p3,filename,mamaid,chanci,age,fenmian,season,city)
