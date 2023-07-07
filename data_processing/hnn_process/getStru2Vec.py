'''
程序功能：
并行分词
'''

'''
对本程序代码所进行的优化如下：
1、将源代码中未使用到的导入库注释并至于代码顶部，若存在程序中未完善部分可以解除注释并使用

# 以下为未使用库文件

# import os
# import logging
#FastText库  gensim 3.4.0
# from gensim.models import FastText
# import numpy as np
# 词频率统计
# import collections
# 词云展示库
# import wordcloud
#图像处理库 Pillow 5.1.0
# from PIL import Image
'''

# 多进程
from multiprocessing import Pool as ThreadPool
import pickle

# 在涉及到相关操作时以前目录为根目录
import sys
sys.path.append("..")

# 解析结构，对导入库函数sqlang、python进行选择
# 仅仅引入了使用到的函数
from python_structured import python_query_parse
from python_structured import python_code_parse
from python_structured import python_code_parse
from sqlang_structured import sqlang_query_parse
from sqlang_structured import sqlang_code_parse
from sqlang_structured import sqlang_context_parse


# 以下为对data_list进行python解析
def multipro_python_query(data_list):
    result=[python_query_parse(line) for line in data_list]
    return result

def multipro_python_code(data_list):
    result = [python_code_parse(line) for line in data_list]
    return result

def multipro_python_context(data_list):
    result = []
    for line in data_list:
        if (line == '-10000'):
            result.append(['-10000'])
        else:
            result.append(python_context_parse(line))
    return result


#sql解析
def multipro_sqlang_query(data_list):
    result=[sqlang_query_parse(line) for line in data_list]
    return result

def multipro_sqlang_code(data_list):
    result = [sqlang_code_parse(line) for line in data_list]
    return result

def multipro_sqlang_context(data_list):
    result = []
    for line in data_list:
        if (line == '-10000'):
            result.append(['-10000'])
        else:
            result.append(sqlang_context_parse(line))
    return result

def parse_python(python_list,split_num):
    # 解析acont1
    acont1_data =  [i[1][0][0] for i in python_list]
    acont1_split_list = [acont1_data[i:i + split_num] for i in range(0, len(acont1_data), split_num)]
    pool = ThreadPool(10)
    acont1_list = pool.map(multipro_python_context, acont1_split_list)
    pool.close()
    pool.join()
    acont1_cut = []
    for p in acont1_list:
        acont1_cut += p
    print('acont1条数：%d' % len(acont1_cut))

    # 解析acont2
    acont2_data = [i[1][1][0] for i in python_list]
    acont2_split_list = [acont2_data[i:i + split_num] for i in range(0, len(acont2_data), split_num)]
    pool = ThreadPool(10)
    acont2_list = pool.map(multipro_python_context, acont2_split_list)
    pool.close()
    pool.join()
    acont2_cut = []
    for p in acont2_list:
        acont2_cut += p
    print('acont2条数：%d' % len(acont2_cut))


    # 解析query
    query_data = [i[3][0] for i in python_list]
    query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
    pool = ThreadPool(10)
    query_list = pool.map(multipro_python_query, query_split_list)
    pool.close()
    pool.join()
    query_cut = []
    for p in query_list:
        query_cut += p
    print('query条数：%d' % len(query_cut))

    # 解析code
    code_data = [i[2][0][0] for i in python_list]
    code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
    pool = ThreadPool(10)
    code_list = pool.map(multipro_python_code, code_split_list)
    pool.close()
    pool.join()
    code_cut = []
    for p in code_list:
        code_cut += p
    print('code条数：%d' % len(code_cut))

    # 获取qids
    qids = [i[0] for i in python_list]
    print(qids[0])
    print(len(qids))

    return acont1_cut,acont2_cut,query_cut,code_cut,qids

# 解析SQL数据
def parse_sqlang(sqlang_list, split_num):
    # 解析acont1
    acont1_data = [i[1][0][0] for i in sqlang_list]
    acont1_split_list = [acont1_data[i:i + split_num] for i in range(0, len(acont1_data), split_num)]
    pool = ThreadPool(10)
    acont1_list = pool.map(multipro_sqlang_context, acont1_split_list)
    pool.close()
    pool.join()
    acont1_cut = []
    for p in acont1_list:
        acont1_cut += p
    print('acont1条数：%d' % len(acont1_cut))

    # 解析acont2
    acont2_data = [i[1][1][0] for i in sqlang_list]
    acont2_split_list = [acont2_data[i:i + split_num] for i in range(0, len(acont2_data), split_num)]
    pool = ThreadPool(10)
    acont2_list = pool.map(multipro_sqlang_context, acont2_split_list)
    pool.close()
    pool.join()
    acont2_cut = []
    for p in acont2_list:
        acont2_cut += p
    print('acont2条数：%d' % len(acont2_cut))

    # 解析query
    query_data = [i[3][0] for i in sqlang_list]
    query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
    pool = ThreadPool(10)
    query_list = pool.map(multipro_sqlang_query, query_split_list)
    pool.close()
    pool.join()
    query_cut = []
    for p in query_list:
        query_cut += p
    print('query条数：%d' % len(query_cut))

    # 解析code
    code_data = [i[2][0][0] for i in sqlang_list]
    code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
    pool = ThreadPool(10)
    code_list = pool.map(multipro_sqlang_code, code_split_list)
    pool.close()
    pool.join()
    code_cut = []
    for p in code_list:
        code_cut += p
    print('code条数：%d' % len(code_cut))

    # 获取qids
    qids = [i[0] for i in sqlang_list]

    return acont1_cut, acont2_cut, query_cut, code_cut, qids



# main函数的衍生函数（1/2）
def read_file(source_path):
    file = open(source_path, 'rb')
    file_extension = source_path.split('.')[-1]
    if file_extension == 'pickle':
        corpus_list = pickle.load(file)
        return corpus_list
    elif file_extension == 'txt':
        corpus_list = eval(file.read())
    file.close()
    return corpus_list
# main函数的衍生函数（2/2）
def deal_with_left(lang_type, corpus_list, split_num):
    if lang_type=='python':
        parse_acont1, parse_acont2, parse_query, parse_code,qids  = parse_python(corpus_list,split_num)
    if lang_type == 'sql':
        parse_acont1, parse_acont2, parse_query, parse_code,qids = parse_sqlang(corpus_list, split_num)
    return parse_acont1, parse_acont2, parse_query, parse_code,qids
# 增加了read_file、deal_with_left两个功能函数，以实现原本通过注释部分代码实现的文件读取功能
def main(lang_type,split_num,source_path,save_path):
    total_data = []
    corpus_list = read_file(source_path)
    parse_acont1, parse_acont2, parse_query, parse_code,qids = deal_with_left(lang_type, corpus_list, split_num)
    for i in range(0,len(qids)):
        total_data.append([qids[i],[parse_acont1[i],parse_acont2[i]],[parse_code[i]],parse_query[i]])
    f = open(save_path, "w")
    f.write(str(total_data))
    f.close()


# 将输入给main函数的操作类型存储为全局变量，并修改为大写，方便区分
PYTHON_TYPE= 'python'
SQLANG_TYPE ='sql'
WOEDS_TOP = 100
SQLIT_NUM = 1000


# 重写启动函数，实现了处理数据的自由选取
if __name__ == '__main__':
    # 路径1
    staqc_python_path = '../hnn_process/ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_save ='../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'
    # 路径2
    staqc_sql_path = '../hnn_process/ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_save = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    # 路径3
    large_python_path='../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    large_python_save='../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'
    # 路径4
    large_sql_path='../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    large_sql_save='../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'


    print(r"本程序中一共包含四组路径，分别是“1.staqc_python、2.staqc_sql、3.large_python、4.large_sql”，请选择你需要进行操作的组号：")
    
    while(True):
        choice = input("请输入你的选择：")
        if input == 1:
            type = PYTHON_TYPE
            source_path = staqc_python_path
            save_path = staqc_python_save
        elif input == 2:
            type = SQLANG_TYPE
            source_path = staqc_sql_path
            save_path = staqc_sql_save
        elif input == 3:
            type = PYTHON_TYPE
            source_path = large_python_path
            save_path = large_python_save
        elif input == 4:
            type = SQLANG_TYPE
            source_path = large_sql_path
            save_path = large_sql_save
        else:
            print("抱歉，系统未能识别到对应输入的数据")
            continue
        main(type, SQLIT_NUM, source_path, save_path)
        print("已经成功对对应数据进行操作，键入1对其他组数据进行操作，键入其他任意数据退出")
        choice = input("你的选择：")
        if(choice != 1):
            break

    print("再见")