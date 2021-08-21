import re
import time

import lxml
import xlsxwriter
from selenium import webdriver
import pandas as pd


# chromedriver的绝对路径
driver_path = r'E:\Program Files\ChromeDriver\chromedriver.exe'
# 初始化一个driver，并且指定chromedriver的路径
driver = webdriver.Chrome(executable_path=driver_path)
content = []
vnamel = [
"nan",
"57-56-7",
"112398-08-0",
"50-02-2",
"439-14-5",
"564-25-0",
"93106-60-6",
"139-91-3",
"67-20-9",
"59-87-0",
"67-45-8",
"73231-34-2",
"85721-33-1",
"nan",
"443-48-1",
"768-94-5",
"57-62-5",
"129138-58-5",
"2437-29-8",
"23696-28-8",
"154-21-2",
"50-53-3",
"56-75-7",
"330-95-0",
"70458-96-7",
"4812-40-2",
"18559-94-9",
"nan",
"60-54-8",
"108050-54-0",
"79-57-2",
"nan",
"619-72-7",
"nan",
"82419-36-1",
]
#vnamel = vnamel[0:3]
for vname in vnamel:
    print(vname)
    content.append(vname)
    try:
        url = 'https://chem.nlm.nih.gov/chemidplus/rn/'+vname
        driver.get(url)
        time.sleep(10)
        driver.implicitly_wait(10)
        allelements = driver.find_elements_by_id("toxicity")
        driver.implicitly_wait(10)
        try:
            ferdigtxt = allelements[0].text
        except:
            ferdigtxt = ''

        content.append(ferdigtxt)
    except:
        content.append('')


new_list = [content[i:i +2] for i in range(0, len(content), 2)]
dataframe = pd.DataFrame(new_list)
dataframe.to_csv( "LD50兽药.csv", index=False, sep=',', encoding='utf-8-sig')



