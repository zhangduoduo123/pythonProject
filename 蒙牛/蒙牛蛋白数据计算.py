import copy
import math

import pandas as pd
from django.core.paginator import Paginator
from django.db.models import Q, Avg, Sum, Max, F, Min
from django.http import HttpResponse, request
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from django.shortcuts import render
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import pairwise_distances

matplotlib.rc("font",family='YouYuan')
def charts_vio_box(all_naifen,charts,):


    X = all_naifen
    x = [ "age", "birth_jd","fenmian_way", "chanci","minzu","zaochan"]
    y = ['a','b']
    # x 影响因素 y 氨基酸
    g = sns.PairGrid(X,
                     x_vars=x,
                     y_vars=y, palette='GnBu_d',
                     )
    # 下三角绘多 sns.boxplot plt.scatter sns.boxplot sns.violinplot
    if charts == 'box':
        plotway = sns.boxplot
    elif charts == 'violinplot':
        plotway = sns.violinplot
    else:
        plotway = plt.scatter

    my_dict_x = {"city": "南方1北方2", "age": "年龄28以下1，28以上2", "birth_jd": "冬春2夏秋1",
                 "fenmian_way": "1，阴道分娩，2，阴道手术分娩，3，剖宫产", "chanci": "1胎2胎3胎","minzu":"1汉族2少数民族",
                 "zaochan":"1，足月，2，早产，3，过期产" ,

               }
    xlabels = []
    for i in range(len(x)):
        xlabels.append(my_dict_x[x[i]])
    my_dict_y = {"a": "α-乳白蛋白","b": "β-酪蛋白", }
    ylabels = []
    for i in range(len(y)):
        ylabels.append(my_dict_y[y[i]])
    g.map_diag(plotway,)
    g.map_offdiag(plotway, )
    for i in range(len(x)):
        for j in range(len(y)):
            g.axes[j, i].xaxis.set_label_text(xlabels[i])
            g.axes[j, i].yaxis.set_label_text(ylabels[j])
    plt.show()

    g.savefig(str(charts)+'.jpg', dpi=400)
# inf
def plot_2d(iris_t2,inf):

        numlem = len(iris_t2)
        for i in range(numlem):
            if iris_t2[inf][i] == 1:
                plt.plot(iris_t2.iloc[i][-1], iris_t2.iloc[i][-2], 'o', color='green')
            elif iris_t2[inf][i] == 2:
                plt.plot(iris_t2.iloc[i][-1], iris_t2.iloc[i][-2], '*', color='red')
            else:
                plt.plot(iris_t2.iloc[i][-1], iris_t2.iloc[i][-2], '*', color='blue')
        for i in range(numlem):
            plt.text(iris_t2.iloc[i][-1], iris_t2.iloc[i][-2],iris_t2.iloc[i][0], fontsize=5)

        my_dict_x = {"city": "南方-绿色，北方-红色", "age": "年龄28以下-绿色，28以上-红色",
                     "birth_jd": "冬春-红色，夏秋-绿色",
                     "fenmian_way": "绿色-阴道分娩，红色-阴道手术分娩，蓝色-剖宫产", "chanci": "绿色-1胎，红色-2胎，蓝色-3胎次",
                     "minzu": "绿色-汉族，红色-少数民族",
                     "zaochan": "绿色-足月，红色-早产，蓝色-过期产",

                     }
        xl = my_dict_x[inf]
        plt.xlabel(xl)
        plt.savefig(str(xl) + '.jpg', dpi=300)
        plt.show()

def distanceCompute(all_naifen,all_anjisuan):
    OUT = []
    Z2 = all_anjisuan['NO']


    al = len(all_anjisuan)

    for anjisuan in range(al):
        Y1 = list(all_anjisuan.iloc[anjisuan])
        for anjisuan in range(al):
            X = list(all_anjisuan.iloc[anjisuan])

            Z = []
            Z.append(X[-2:])
            Z.append(Y1[-2:])

            OUT.append((pairwise_distances(Z, metric="euclidean")[1][0]))
            OUT.append((pairwise_distances(Z, metric="cosine")[1][0]))
            OUT.append(pairwise_distances(Z, metric="braycurtis")[1][0])
            OUT.append((pairwise_distances(Z, metric="correlation")[1][0]))


    X = [OUT[i:i + 4] for i in range(0, len(OUT), 4)]
    Y = []
    temp = []
    cnt = 0
    for i in X:
        cnt = cnt + 1
        # 妈妈类型个数
        if cnt == len(all_anjisuan):
            temp.append(i)
            temp = list(zip(*temp))
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
    new_list = temp2
    new_list1 = []
    for i in new_list:
        temp = 0
        ilen = len(i)
        for j in range(len(i)):
            if j == len(i) - 1:
                new_list1.append(i[j])
                new_list1.append(temp)
            else:
                new_list1.append(i[j])
                temp = temp + i[j]
    iilen = ilen + 1
    new_list_t = [new_list1[i:i + iilen] for i in range(0, len(new_list1), iilen)]

    Z1 = [
        "欧氏距离",
        "余弦距离",
        "braycurtis",
        "correlation",

    ]
    count = -1
    zcnt = 0

    for i in new_list_t:
        count = count + 1
        # 12 距离种类个数
        if count == 3:
            i.insert(0, Z1[count])
            i.insert(0, Z2[zcnt])
            zcnt = zcnt + 1
            count = -1
        else:
            i.insert(0, Z1[count])
            i.insert(0, Z2[zcnt])


    df_data1 = pd.DataFrame(new_list_t)
    df_data1.to_csv('距离计算.csv',encoding='utf-8-sig')



df = pd.read_csv('mama.csv',encoding='utf-8-sig')
distanceCompute(df,df)

# charts_vio_box(df,'box')
# charts_vio_box(df,'violinplot')
# charts_vio_box(df,'sca')
#
#
# infl=[ "age", "birth_jd","fenmian_way", "chanci","minzu","zaochan"]
# for i in infl:
#     plot_2d(df, i)
