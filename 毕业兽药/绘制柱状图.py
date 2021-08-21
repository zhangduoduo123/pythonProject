import seaborn as sns
import pandas as pd
from py2neo import Graph,Node,Relationship
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
plt.rcParams['font.sans-serif']=["SimHei"] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负
def mysql2neo4j():
    graph = Graph("http://localhost:7474", auth=("neo4j", "root"))
    #graph.run("MATCH (r) DETACH DELETE r")

    X = pd.read_csv('F:\研究生\研二上\兽药数据\兽药数据重新整理\兽药抽检2.csv', encoding='utf-8-sig')
    for i in range(X.shape[0]):
        foodtype1 = X.iat[i, 0]
        food1 = X.iat[i, 1]
        veterinary1 = X.iat[i, 5]
        vtype1 = X.iat[i, 6]
        results1 = X.iat[i, 7]
        time1 = X.iat[i, 9]
        area1 = X.iat[i, 10]
        tox1 = X.iat[i, 11]

        b = Node('food_event',
                 name=str(food1)+str(veterinary1)+str(results1),
                 results=results1,
                 time=time1,
                )

        g = Node('food', name=food1,
                 foodtype=foodtype1,)
        r2 = Relationship(b, '抽检食物', g)
        graph.create(r2)

        c = Node('veterinary', name=veterinary1,vtype=vtype1,tox=tox1)
        r1 = Relationship(b, '残留药物', c)
        graph.create(r1)

        e = Node('area', name=area1)
        r3 = Relationship(b, '所属地区', e)
        graph.create(r3)

    #graph.run("MATCH (n:food) WITH n.name AS name, COLLECT(n) AS nodelist, COUNT(*) AS count WHERE count > 1 CALL apoc.refactor.mergeNodes(nodelist) YIELD node RETURN node")
    #graph.run("MATCH (n:veterinary) WITH n.name AS name, COLLECT(n) AS nodelist, COUNT(*) AS count WHERE count > 1 CALL apoc.refactor.mergeNodes(nodelist) YIELD node RETURN node")
    #graph.run("MATCH (n:area) WITH n.name AS name, COLLECT(n) AS nodelist, COUNT(*) AS count WHERE count > 1 CALL apoc.refactor.mergeNodes(nodelist) YIELD node RETURN node")

# def searchNeo4j():
#     graph = Graph("http://localhost:7474",auth=("neo4j","root"))
#     f = graph.run("match (a:veterinary{name:'恩诺沙星'})-[r1]-(b:food)-[r2]-(c:area) return c.name,b.foodtype,count(c) as num order by c.name,b.foodtype,num desc")
#     df1 = pd.DataFrame(columns=['name','foodtype','num'])
#     for i in f:
#         df1 = df1.append({'name': i[0],'foodtype': i[1],'num': i[2],}, ignore_index=True)
#     return df1
#
# def plot(df1):
#     plt.figure()
#     g = sns.catplot(
#         data=df1, kind="bar",
#         x="num", y="name", hue="foodtype",
#         ci="sd", palette="dark", alpha=.6, height=6
#     )
#     g.despine(left=True)
#     g.savefig('3.jpg', dpi = 400)
#
# #mysql2neo4j()
# plot(searchNeo4j())
mysql2neo4j()