import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.nonparametric.api as smnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine
import seaborn as sns
import pandas as pd
matplotlib.rc("font",family='YouYuan')
def trans(m):
    return list(zip(*m))
def linkdatabase():
    engine = create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format('zdd', 'root', 'localhost', '3306', 'zhangduoduo_www'))

    sql_query = 'select * from xiangdui where miruqi = 3 ;'
    X2 = pd.read_sql_query(sql_query, engine)
    sql_query1 = 'select  mother_id as mother_id1 ,sumaa,baa as baa1 from yuanshi where miruqi = 3 ;'
    X1 = pd.read_sql_query(sql_query1, engine)
    sumX = pd.concat([X2, X1], axis=1)

    print('linkdatabase')
    return sumX


def plot(iris_t2,filename):
    X = iris_t2
    g = sns.PairGrid(X,
                     x_vars=['city', 'age1', 'birth_jd','fenmian_way', 'chanci',  ],
                     y_vars=["lai",
                             "su",
                             "jie",
                             "dan",
                             "yiliang",
                             "liang",
                             "benbing",
                             "se",
                             "zu",
                             "tiandong",
                             "si",
                             "gu",
                             "gan",
                             "bing",
                             "lao",
                             "jing",
                             "fu",
                             "banguang",
                             "baa",
                             "b2fb",
                             "sumaa",
                             "baa1",

                             ], palette='GnBu_d',
                     )

    # 下三角绘多 sns.boxplot plt.scatter sns.boxplot sns.violinplot
    plotway = sns.violinplot
    xlabels = ['南方1北方2', '年龄30以下1，30以上2', '冬春1夏秋2','顺产1，剖腹产2', '1胎2胎',]
    ylabels = ["赖氨酸",
                             "苏氨酸",
                             "缬氨酸",
                             "蛋氨酸",
                             "异亮氨酸",
                             "亮氨酸",
                             "苯丙氨酸",
                             "色氨酸",
                             "组氨酸",
                             "天冬氨酸",
                             "丝氨酸",
                             "谷氨酸",
                             "甘氨酸",
                             "丙氨酸",
                             "酪氨酸",
                             "精氨酸",
                             "脯氨酸",
                             "半胱氨酸",
               "必须aa/总aa",
               "必须aa/非必须aa",
               "总aa",
               "必须aa",]
    g.map_diag(plotway, )
    g.map_offdiag(plotway, )
    #ax = sns.swarmplot(x='species', y='sepal_length', data=df, color="grey")
    # g.map_diag(sns.swarmplot,color="grey" )
    # g.map_offdiag(sns.swarmplot,color="grey" )
    #g = g.add_legend()
    for i in range(5):
        for j in range(22):
            g.axes[j, i].xaxis.set_label_text(xlabels[i])
            g.axes[j, i].yaxis.set_label_text(ylabels[j])
    plt.show()
    #plt.savefig(str(filename) + '.jpg', dpi=300)
    #g.savefig(str(filename)+'.jpg', dpi=400)

    return filename

def plot_2inf(iris_t2,filename):
    X = iris_t2
    X['city'].replace({1: '南方', 2: '北方'}, inplace=True)
    g = sns.PairGrid(X,
                     x_vars=['birth_jd' ],
                     y_vars=["lai",
                             "su",
                             "jie",
                             "dan",
                             "yiliang",
                             "liang",
                             "benbing",
                             "se",
                             "zu",
                             "tiandong",
                             "si",
                             "gu",
                             "gan",
                             "bing",
                             "lao",
                             "jing",
                             "fu",
                             "banguang",
                             "baa",
                             "b2fb",
                             "sumaa",
                             "baa1",

                             ], hue = 'city',palette='Pastel1',
                     )

    # 下三角绘多 sns.boxplot plt.scatter sns.boxplot sns.violinplot
    plotway = sns.violinplot
    #xlabels = ['南方1北方2', '年龄30以下1，30以上2', '冬春1夏秋2','顺产1，剖腹产2', '1胎2胎',]
    xlabels = ['冬春1夏秋2' ]
    ylabels = ["赖氨酸",
                             "苏氨酸",
                             "缬氨酸",
                             "蛋氨酸",
                             "异亮氨酸",
                             "亮氨酸",
                             "苯丙氨酸",
                             "色氨酸",
                             "组氨酸",
                             "天冬氨酸",
                             "丝氨酸",
                             "谷氨酸",
                             "甘氨酸",
                             "丙氨酸",
                             "酪氨酸",
                             "精氨酸",
                             "脯氨酸",
                             "半胱氨酸",
               "必须aa/总aa",
               "必须aa/非必须aa",
               "总aa",
               "必须aa",]
    g.map_diag(plotway, )
    g.map_offdiag(plotway, )
    g = g.add_legend()
    #l = g.fig.legend()
    #l.texts[0].set_text('Gender')
    #g = g.add_legend(handles=l.legendHandles, labels=['南方', '北方'])
    #ax = sns.swarmplot(x='species', y='sepal_length', data=df, color="grey")
    # g.map_diag(sns.swarmplot,color="grey" )
    # g.map_offdiag(sns.swarmplot,color="grey" )
    #g = g.add_legend()
    xlnum = len(xlabels)
    ylnum = len(ylabels)
    for i in range(xlnum):
        for j in range(ylnum):
            g.axes[j, i].xaxis.set_label_text(xlabels[i])
            g.axes[j, i].yaxis.set_label_text(ylabels[j])
    #plt.legend()
    plt.show()
    #plt.savefig(str(filename) + '.jpg', dpi=300)
    #g.savefig(str(filename)+'.jpg', dpi=400)

    return filename

def plot_q(iris_t2,filename):
    engine = create_engine( "mysql+pymysql://{}:{}@{}:{}/{}".format('zdd', 'root', 'localhost', '3306', 'zhangduoduo_www'))
    res = []
    ajsl = ["lai",
                             "su",
                             "jie",
                             "dan",
                             "yiliang",
                             "liang",
                             "benbing",
                             "se",
                             "zu",
                             "tiandong",
                             "si",
                             "gu",
                             "gan",
                             "bing",
                             "lao",
                             "jing",
                             "fu",
                             "banguang",
                             "baa",
                             "b2fb",


                             ]
    for ajs in ajsl:
                res.append(ajs)
                sql_query = 'select '+str(ajs)+' from xiangdui where miruqi = 3 ;'
                X = pd.read_sql_query(sql_query, engine)
                des = pd.DataFrame(X).describe()
                # 得到每列的平均值,是一维数组
                mean = des.iat[1,0]
                # 得到每列的标准差,是一维数组
                std = des.iat[2,0]
                numll = des.iat[0,0]
                numll = math.sqrt(numll)
                q0 = mean - 1.96 * std / numll
                q5 = mean + 1.96 * std / numll
                # q1 = des.iat[4,0]
                q2 = des.iat[5,0]
                # q3 = des.iat[6,0]
                # IQR = q3-q1
                # q0 = q1-1.5*IQR
                # q5 = q3+1.5*IQR
                res.append(q0)
                # res.append(q1)
                res.append(q2)
                # res.append(q3)
                res.append(q5)
                # sql_query1 = 'select distinct mother_id from xiangdui where miruqi = 3 and '+str(inf)+' = '+str(num)+' and '+str(ajs)+' <= '+str(q0)+';'
                # X1 = pd.read_sql_query(sql_query1, engine)
                # X1 = np.array(X1)  # 先将数据框转换为数组
                # X1 = X1.tolist()  # 其次转换为列表
                # res.append(X1)
                # sql_query2 = 'select distinct mother_id from xiangdui where miruqi = 3 and '+str(inf)+' = '+str(num)+' and '+str(ajs)+' >= '+str(q5)+';'
                # X2 = pd.read_sql_query(sql_query2, engine)
                # X2 = np.array(X2)  # 先将数据框转换为数组
                # X2 = X2.tolist()
                # res.append(X2)


    X = [res[i:i + 4] for i in range(0, len(res), 4)]
    return X
def plot_city_season(iris_t2,filename):
    engine = create_engine( "mysql+pymysql://{}:{}@{}:{}/{}".format('zdd', 'root', 'localhost', '3306', 'zhangduoduo_www'))

    res = []
    sql_query1 = 'select distinct mother_id from xiangdui where miruqi = 3 and city = 1 and birth_jd =1 ;'
    X1 = pd.read_sql_query(sql_query1, engine)
    X1 = np.array(X1)  # 先将数据框转换为数组
    X1 = X1.tolist()  # 其次转换为列表
    res.append('南方-冬春')
    res.append(X1)
    sql_query2 = 'select distinct mother_id from xiangdui where miruqi = 3 and city = 1 and birth_jd =2 ;'
    X2 = pd.read_sql_query(sql_query2, engine)
    X2 = np.array(X2)  # 先将数据框转换为数组
    X2 = X2.tolist()
    res.append('南方-夏秋')
    res.append(X2)
    sql_query3 = 'select distinct mother_id from xiangdui where miruqi = 3 and city = 2 and birth_jd =1 ;'
    X1 = pd.read_sql_query(sql_query3, engine)
    X1 = np.array(X1)  # 先将数据框转换为数组
    X1 = X1.tolist()  # 其次转换为列表
    res.append('北方-冬春')
    res.append(X1)
    sql_query4 = 'select distinct mother_id from xiangdui where miruqi = 3 and city = 2 and birth_jd =2 ;'
    X2 = pd.read_sql_query(sql_query4, engine)
    X2 = np.array(X2)  # 先将数据框转换为数组
    X2 = X2.tolist()
    res.append('北方-夏秋')
    res.append(X2)


    X = [res[i:i + 2] for i in range(0, len(res), 2)]
    return X
def tocsv(X,filename):
    dataframe = pd.DataFrame(X)
    dataframe.to_csv(str(filename)+".csv", index=False, sep=',',encoding='utf-8-sig')

def plot_vil(ww,filename):

    for i in range(ww.shape[0]):
        if ww['city'][i] == 2:
            ww['city'][i] = 1

    g = sns.PairGrid(ww,
                     x_vars=['city','city'],
                     y_vars=["lai",
                             "su",
                             "jie",
                             "dan",
                             "yiliang",
                             "liang",
                             "benbing",
                             "se",
                             "zu",
                             "tiandong",
                             "si",
                             "gu",
                             "gan",
                             "bing",
                             "lao",
                             "jing",
                             "fu",
                             "banguang",
                             "baa",
                             "b2fb",
                             "sumaa",
                             "baa1",

                             ], palette='GnBu_d',
                     )

    # 下三角绘多 sns.boxplot plt.scatter sns.boxplot sns.violinplot distplot
    plotway = sns.violinplot
    xlabels = ['南方北方','南方北方']
    ylabels = ["赖氨酸",
                             "苏氨酸",
                             "缬氨酸",
                             "蛋氨酸",
                             "异亮氨酸",
                             "亮氨酸",
                             "苯丙氨酸",
                             "色氨酸",
                             "组氨酸",
                             "天冬氨酸",
                             "丝氨酸",
                             "谷氨酸",
                             "甘氨酸",
                             "丙氨酸",
                             "酪氨酸",
                             "精氨酸",
                             "脯氨酸",
                             "半胱氨酸",
               "必须aa/总aa",
               "必须aa/非必须aa",
               "总aa",
               "必须aa",]
    g.map_diag(plotway,inner="quartile" ,scale = 'count',)
    g.map_offdiag(plotway,inner="quartile" ,scale = 'count',)
    # g.map_diag(plotway,  )
    # g.map_offdiag(plotway, )
    #ax = sns.swarmplot(x='species', y='sepal_length', data=df, color="grey")
    # g.map_diag(sns.swarmplot,color="grey" )
    # g.map_offdiag(sns.swarmplot,color="grey" )
    #g = g.add_legend()
    for i in range(2):
        for j in range(22):
            g.axes[j, i].xaxis.set_label_text(xlabels[i])
            g.axes[j, i].yaxis.set_label_text(ylabels[j])
    plt.show()
    #plt.savefig(str(filename) + '.jpg', dpi=300)
    #g.savefig(str(filename)+'.jpg', dpi=400)

    return filename
def plot_kel(ww,filename):



    sns.distplot(ww['lai'])
    plt.show()

    return filename



def kde_test(data, kernel, bw, gridsize, cut):
    """
    :param data:样本数据
    :param kernel:核函数
    :param bw:带宽
    :param gridsize:绘制拟合曲线中的离散点数；可理解为精度，会改变kde曲线的圆滑程度
    :param cut: 源代码说明——Draw the estimate to cut * bw from the extreme data points.
    :return: kde估计曲线的x、y坐标
    """
    fft = kernel == "gau"
    kde = smnp.KDEUnivariate(data)
    kde.fit(kernel, bw, fft, gridsize=gridsize, cut=cut)
    return kde.support, kde.density


class Kdefitplot:
    def __init__(self, data, kernel='gau', bw="scott", legends=None, labels=None, fsize=(10, 6.18), show_point=False, show_info=True):
        """
        :param data:以列表格式存储的数据
        :param kernel:密度估计选用的核函数，可选{'gau'|'cos'|'biw'|'epa'|'tri'|‘triw’}，默认为'gau'
        :param bw:密度估计选用的自适应带宽方法，可选{'scott'|'silverman'|scalar|pair of scalars}，默认为"scott"
        :param legends: 图例名，默认为 "概率密度", "kde曲线", "最大值点"
        :param labels:坐标轴标题名，默认为 "数据x", "概率密度"
        :param show_info:是否显示拟合结果信息，默认为True
        :param show_point:在kde曲线上显示目标点（最大值点），默认为False
        """
        self.data = data
        self.kernel = kernel
        self.bw = bw
        if legends is None:
            legends = ["概率密度", "kde曲线", "最大值点"]
        if labels is None:
            labels = ["样本数据", "概率密度"]
        self.legends = legends
        self.labels = labels
        self.fsize = fsize
        self.show_info = show_info
        self.show_point = show_point
        self.gridsize = 100
        self.cut = 3

    def change_legend(self, new_legends):
        # 更改图例
        self.legends = new_legends

    def change_label(self, new_labels):
        # 更改坐标轴标题
        self.labels = new_labels

    def draw_plot(self):
        # 利用seaborn库对字体大小进行统一设置，为fgsize[1]的0.12倍，即画布纵向大小为1000时，font_scale=1.2
        sns.set_style("darkgrid")
        sns.set_context("talk", font_scale=self.fsize[1] * 0.15)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=self.fsize)

        # 绘制频率直方图
        sns.distplot(self.data, label=self.legends[0], rug=True, kde=True, kde_kws={"color": "g", "lw": 0})

        # 以gau为核函数，scott为带宽估计方法
        sns.kdeplot(self.data,
                    kernel=self.kernel,
                    bw=self.bw,
                    label=self.legends[1],
                    color="r",
                    linewidth=self.fsize[0]*0.2
                    )

        # 计算kde曲线的x、y值
        kdefit_x, kdefit_y = kde_test(self.data,
                                      self.kernel,
                                      self.bw,
                                      gridsize=self.gridsize,
                                      cut=self.cut)

        # point为kde曲线最大值点
        point = np.where(kdefit_y == np.max(kdefit_y))

        # 在kde曲线上显示目标点，格式为黑色实心圆
        if self.show_point:
            plt.plot(kdefit_x[point], kdefit_y[point], "o", color='k', linewidth=self.fsize[0]*0.4, label=self.legends[2])

        # 打印统计信息
        if self.show_info:
            # 显示核密度估计信息:kernel为核函数、bw为自适应带宽方法、point为kde曲线最大值点
            # 基本统计信息：Size为样本数据点个数、Average为平均值、Q25/Q50/Q75分别为25%/50%/75%分位数
            q25, q50, q75 = [round(q, 4) for q in np.percentile(self.data, [25, 50, 75])]
            base_info = f"Size:{len(self.data)}\nAver:{np.mean(self.data)}\nQ25:{q25}; Q50:{q50}; Q75:{q75}\n\n"
            kde_info = f"kernel:{self.kernel}\nbw:{self.bw}\nMax point appear in {kdefit_x[point]}\n"
            print(base_info + kde_info)

        # 设置x、y坐标轴标题
        plt.xlabel(self.labels[0])
        plt.ylabel(self.labels[1])

        plt.legend()
        plt.tight_layout()
        return kdefit_x[point]



time = ['晚期-','早期成熟乳-']
dim = ['18-必须-总-必须/非必须-必须/总']

filename = str(time[1])+str(dim[0])
ww = linkdatabase()

y_vars = ["lai",
          "su",
          "jie",
          "dan",
          "yiliang",
          "liang",
          "benbing",
          "se",
          "zu",
          "tiandong",
          "si",
          "gu",
          "gan",
          "bing",
          "lao",
          "jing",
          "fu",
          "banguang",


          ]
pmax = []

# for i in range(18):
#     pmax.append(y_vars[i])
#     kdeplot1 = Kdefitplot(ww[y_vars[i]], show_point=True)
#     p = kdeplot1.draw_plot()
#     pmax.append(p)
#     plt.show()
#
# X = [pmax[i:i + 2] for i in range(0, len(pmax), 2)]
#tocsv(plot_q(ww,'minmidmax'),'minmidmax')
plot(linkdatabase(),'a')