import re
import time

import lxml
import xlsxwriter
from selenium import webdriver
import pandas as pd


# chromedriver的绝对路径
driver_path = r'F:\googledownload\chromedriver.exe'
# 初始化一个driver，并且指定chromedriver的路径
driver = webdriver.Chrome(executable_path=driver_path)
content = []
vnamel = ["呋喃西林",
"氯霉素",
"沙丁胺醇",
"菌落总数",
"磺胺类",
"多西环素",
"呋喃唑酮",
"氟苯尼考",
"氟苯尼考 ",
"环丙沙星",
"甲硝唑",
"金刚烷胺",
"氧氟沙星",
"果糖和葡萄糖",
"诺氟沙星",
"诺氟沙星 ",
"嗜渗酵母计数",
"氧氟沙星 ",
"3-氨基-2-恶唑烷基酮",
"金霉素",
"呋喃它酮",
"呋喃妥因",
"呋喃西林 ",
"呋喃唑酮 ",
"磺胺类 ",
"挥发性盐基氮",
"尼卡巴嗪",
"尼卡巴嗪 ",
"沙拉沙星",
"四环素",
"替米考星",
"替米考星 ",
"土霉素",
"土霉素 ",
"孔雀石绿",
"过氧化值",
"地塞米松",
"铬",
"克伦特罗",
"林可霉素",
"羟基甲硝唑",
"氨基脲",
"氨基脲 ",
"达氟沙星",
"地西泮",
"地西泮 ",
"二氧化硫残留量",
"镉",
"甲硝唑 ",
"喹乙醇",
"羟基甲硝唑 ",
"硝基呋喃",
"五氯酚酸钠",
"氯丙嗪",
"沙丁胺醇 ",
"亚硝酸盐",
]
for vname in vnamel:
    print(vname)
    content.append(vname)
    try:
        url = 'https://baike.baidu.com/item/'+vname
        driver.get(url)
        allelements = driver.find_elements_by_class_name("para")
        ferdigtxt = []
        for i in allelements:
            if i.text in ferdigtxt:
                pass
            else:
                ferdigtxt.append(i.text)
        print(ferdigtxt)
        content.append(ferdigtxt)
    except:
        content.append('')


    # driver.close()
    # driver.switch_to.window(driver.window_handles[0])



new_list = [content[i:i +2] for i in range(0, len(content), 2)]
dataframe = pd.DataFrame(new_list)
dataframe.to_csv( "百科兽药.csv", index=False, sep=',', encoding='utf-8-sig')



