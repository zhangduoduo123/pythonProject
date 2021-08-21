from pyhanlp import HanLP

fw = open('HanLP_save_path.txt', 'w', encoding='utf-8')    # 分词结果保存
with open('origin_path.txt', 'r', encoding='utf-8') as fr:  # 需要分词的文档
    for line in fr:
        line = line.strip()
        word_list = HanLP.segment(line)  # 返回是一个列表[女性/n, ，/w, 88/m, 岁/qt, ，/w, 农民/nnt, ]
        # print(word_list)
        for term in word_list:    # 分词结果格式：单词和词性。term.word, term.nature：获取单词与词性
            # print(term.word)
            fw.write(term.word + str(term.nature) + ' ')
        fw.write('\n')
fw.close()
