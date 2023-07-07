
'''
从一个庞大的词典当中抽取适用于特定文本库的子集
并将所有数据变为“标签待定”的状态
'''
# 已经完成优化修改

import numpy as np

import pickle
from gensim.models import KeyedVectors

# 特殊词和词向量，通过全局的角度创建以下变量
SPECIAL_WORDS = ['PAD', 'SOS', 'EOS', 'UNK']
SPECIAL_VECTORS = [np.zeros((1, 300)).squeeze(),
                   np.random.uniform(-0.25, 0.25, size=(1, 300)).squeeze(),
                   np.random.uniform(-0.25, 0.25, size=(1, 300)).squeeze(),
                   np.random.uniform(-0.25, 0.25, size=(1, 300)).squeeze()]

# 加载并返回内容
def LoadFile(filePath):
    try:
        with open(filePath, 'r') as file:
            content = eval(file.read())
    except Exception as e:
        print(f'读取以下文件时发生错误: {filePath}, error: {e}')
        content = None
    return content

# 读取pickle文件
def LoadPickle(filePath):
    try:
        with open(filePath, 'rb') as file:
            tempAll = pickle.load(file)
    except Exception as e:
        print(f'发生以下未知错误：{e}')
    return tempAll

# 将字符转换为python对象
def ChangeToPy(filePath):
    try:
        with open(filePath, 'r') as file:
            tempAll = eval(file.read())
            file.close()
    except Exception as e:
        print(f'遇到了未知的错误：{e}')
        tempAll = None
    return tempAll

# 将数据保存为pickle文件
def SaveToPickle(filePath, data):
    try:
        with open(filePath, 'wb') as file:
            pickle.dump(data, file)
    except Exception as e:
        print(f'读取以下文件时发生错误: {filePath}, error: {e}')

# 读取模型文件
def LoadModel(filePath):
    try: 
        model = KeyedVectors.load(filePath, mmap='r')
    except Exception as e:
        print(f'读取以下文件时发生错误: {filePath}, error: {e}')
        model = None
    return model

# 词向量文件保存成bin文件，从而提高后续的文件加载速度
def TransToBin(inputPath, outputPath):
    # 词向量文件保存成bin文件，从而提高后续的文件加载速度
    wordVectors = KeyedVectors.load_word2vec_format(inputPath, binary=False)
    wordVectors.init_sims(replace=True)
    wordVectors.save(outputPath)

# 判断词向量是否正确
def CheckIfRight(model, wordDict, wordVectors):
    count = 0
    for i in range(4, len(wordDict)):
        if wordVectors[i].all() == model.wv[wordDict[i]].all():
            continue
        else:
            count += 1
    print(f'不匹配词汇数量：{count}')
    return count

# 构建新的词典和词向量矩阵
# 对这个代码中的一些读取、存储功能转为调用子函数进行操作
def ConstructNewDictAndWordVector(typeVectorPath, wordVectorPath, finalWordPath, finalVectorPath):
    # 加载预转换bin文件
    model = LoadFile(typeVectorPath)
    # 将字符转换为python对象
    totalWord = ChangeToPy(wordVectorPath)
    # 得到全局变量参数的备份
    wordDict = SPECIAL_WORDS.copy()
    wordVectors = SPECIAL_VECTORS.copy()
    # 创建用于存储未找到词汇的空数组
    failWord = []
    print(len(totalWord))

    # 将预先的bin文件中词向量存入表单中
    for word in totalWord:
        try:
            # 尝试从模型中加载词向量
            wordVectors.append(model.wv[word]) # 加载词向量
            wordDict.append(word)  # 将词添加到词典列表中
        except:
            print(f'无法打印这个词: {word}')  # 打印无法加载词向量的词
            failWord.append(word)  # 将无法加载的词添加到失败列表中
    # 关于有多少个词，以及多少个词没有找到
    print("总词数：", len(wordDict))  # 打印词典中的词数
    print("找到的词向量数：", len(wordVectors))  # 打印成功加载词向量的词数
    print("未找到的词数：", len(failWord))  # 打印无法加载词向量的词数

    #判断词向量是否正确
    count = CheckIfRight(model, wordDict, wordVectors)

    wordVectors = np.array(wordVectors)
    wordDict = dict(map(reversed, enumerate(wordDict)))
    
    SaveToPickle(finalVectorPath, wordDict)
    SaveToPickle(finalWordPath, wordDict)

    v = pickle.load(open(finalVectorPath, 'rb'), encoding='iso-8859-1')
    wordDict = LoadPickle(finalWordPath)
    count = 0

    print("全部操作已完成")

# 根据提供的文本和词典，返回一个列表。该列表包含文本中每个词在词典中的位置。
# 如果词不在词典中，则返回‘UNK’的位置
def GetWordIndices(text, wordDict):
    return [wordDict.get(t, wordDict.get('UNK')) for t in text]

# 得到词在词典中的位置
def GetIndex(type, text, word_dict):
    location = []
    if type == 'code':  # 如果类型是'code'
        location.append(1)  # 在位置列表开始添加1
        len_c = len(text)  # 获取文本长度
        length_limit = 348 if len_c >= 350 else len_c  # 限制文本长度，确保不超过350
        location.extend(GetWordIndices(text[:length_limit], word_dict))
        location.append(2) if len_c+1 < 350 else location.extend([2] * (350 - len_c))
    else:  # 如果类型不是'code'
        if len(text) == 0 or text[0] == '-10000':  # 如果文本长度为0或者文本的第一个元素是'-10000'
            location.append(0)  # 在位置列表中添加0
        else:  # 否则
            location.extend(GetWordIndices(text, word_dict))  # 获取文本中所有词的位置列表并添加到位置列表中
    return location  # 返回位置列表

# 将一个列表填充或者截断到目标长度
# 参数分别是：输入的列表、目标长度、填充的值（默认为0）
def PadList(inputList, targetLength, padValue=0):
    if len(inputList) > targetLength:
        # 如果列表长度大于目标长度，截断列表
        return inputList[: targetLength]
    else:
        # 如果列表长度小于目标长度，填充列表
        return inputList + [padValue] * (targetLength - len(inputList))

# 用于序列化corpus
# 参数分别是词典文件路径、类型文件路径、最终类型文件路径
def Serialization(wordDictPath, typePath, finalTypePath):
    # 加载词典文件
    wordDict = LoadPickle(wordDictPath)
    # 加载corpus
    corpus = LoadFile(typePath)
    # 初始化总数据列表
    totalData = []

    # 遍历corpus中的每一个元素
    for i in range(len(corpus)):
        qid = corpus[i][0]

        # 获取词位置，并进行填充或截断操作，使得列表长度符合要求
        SiWordList = PadList(GetIndex('text', corpus[i][1][0], wordDict), 100)
        Si1WordList = PadList(GetIndex('text', corpus[i][1][1], wordDict), 100)
        tokenized_code = PadList(GetIndex('code', corpus[i][2][0], wordDict), 350)
        queryWordList = PadList(GetIndex('text', corpus[i][3], wordDict), 25)

        blockLength = 4
        label = 0

        oneData = [qid, [SiWordList, Si1WordList], [tokenized_code], queryWordList, blockLength, label]
        totalData.append(oneData)  # 往总数据列表中添加数据

    # 把总数据列表保存为pickle文件
    SaveToPickle(finalTypePath, totalData)

# 加载之前的词典和词向量
def LoadPreviousData(previousDict, previousVec, appendWordPath):
    preWordDict = LoadPickle(previousDict)
    preWordVec = LoadPickle(previousVec)
    appendWord = LoadPickle(appendWordPath)
    return preWordDict, preWordVec, appendWord

# 生成新的词向量
def GenerateNewWordVec(model, appendWord):
    rng = np.random.RandomState(None)
    wordVectors = []
    failWord = []
    for word in appendWord:
        try:
            wordVectors.append(model.wv[word])
        except:
            failWord.append(word)
    return wordVectors, failWord

# 保存新的词向量
def SaveNewDictVec(wordDict, wordVectors, finalVecPath, finalWordPath):
    wordVectors = np.array(wordVectors)
    wordDict = dict(map(reversed, enumerate(wordDict)))
    SaveToPickle(finalVecPath, wordVectors)
    SaveToPickle(finalWordPath, wordDict)

# 获取新增词语后的词典和词向量
def GetNewDictAppend(typeVecPath, previousDict, previousVec, appendWordPath, finalVecPath, finalWordPath):
    model = KeyedVectors.load(typeVecPath, mmap='r')
    preWordDict, preWordVec, appendWord = LoadPreviousData(previousDict, previousVec, appendWordPath)
    wordDict =  list(preWordDict.keys())
    wordVectors = preWordVec.tolist()
    newWordVectors, failWord = GenerateNewWordVec(model, appendWord)
    wordVectors += newWordVectors
    wordDict += appendWord
    print(f'添加后的词典大小：{len(wordDict)}，词向量大小：{len(wordVectors)}，未找到的词向量数量{len(failWord)}')
    SaveNewDictVec(wordDict, wordVectors, finalVecPath, finalWordPath)
    print("Completed")

#-------------------------参数配置----------------------------------
#python 词典 ：1121543 300
# 使用if __name__ == '__main__': 来让python执行main测试
if __name__ == "__main__":
    # 用户输入需要执行的任务编号
    task = int(input("请输入需要执行的项目编号(1-6):"))

    # 以下是需要用到的文件路径
    ps_path = '../hnn_process/embeddings/10_10/python_struc2vec1/data/python_struc2vec.txt'  #239s
    ps_path_bin = '../hnn_process/embeddings/10_10/python_struc2vec.bin'  #2s
    sql_path = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.txt'
    sql_path_bin = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.bin'
    python_word_path = '../hnn_process/data/word_dict/python_word_vocab_dict.txt'
    python_word_vec_path = '../hnn_process/embeddings/python/python_word_vocab_final.pkl'
    python_word_dict_path = '../hnn_process/embeddings/python/python_word_dict_final.pkl'
    sql_word_path = '../hnn_process/data/word_dict/sql_word_vocab_dict.txt'
    sql_word_vec_path = '../hnn_process/embeddings/sql/sql_word_vocab_final.pkl'
    sql_word_dict_path = '../hnn_process/embeddings/sql/sql_word_dict_final.pkl'
    new_sql_staqc = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    large_word_dict_sql = '../hnn_process/ulabel_data/sql_word_dict.txt'
    sql_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/sql_word_vocab_final.pkl'
    sql_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/sql_word_dict_final.pkl'
    staqc_sql_f = '../hnn_process/ulabel_data/staqc/seri_sql_staqc_unlabled_data.pkl'
    large_sql_f = '../hnn_process/ulabel_data/large_corpus/multiple/seri_ql_large_multiple_unlable.pkl'
    new_python_staqc = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'
    new_python_large = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'
    final_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'
    large_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'
    python_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/python_word_vocab_final.pkl'
    python_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/python_word_dict_final.pkl'
    staqc_python_f ='../hnn_process/ulabel_data/staqc/seri_python_staqc_unlabled_data.pkl'
    large_python_f ='../hnn_process/ulabel_data/large_corpus/multiple/seri_python_large_multiple_unlable.pkl' 
    
    # 根据用户输入的任务编号执行相应的任务
    if task == 1:
        TransToBin(sql_path,sql_path_bin)
    elif task == 2:
        TransToBin(ps_path, ps_path_bin)
    elif task == 3:
        ConstructNewDictAndWordVector(sql_path_bin,sql_word_path,sql_word_vec_path,sql_word_dict_path)
        ConstructNewDictAndWordVector(ps_path_bin,python_word_path,python_word_vec_path,python_word_dict_path)
    elif task == 4:
        ConstructNewDictAndWordVector(sql_path_bin, sql_final_word_dict_path, sql_final_word_vec_path, sql_final_word_dict_path)
        GetNewDictAppend(sql_path_bin, sql_word_dict_path, sql_word_vec_path, large_word_dict_sql, sql_final_word_vec_path,sql_final_word_dict_path)
        Serialization(sql_final_word_dict_path, new_sql_staqc, staqc_sql_f)
        Serialization(sql_final_word_dict_path, new_sql_large, large_sql_f)
    elif task == 5:
        ConstructNewDictAndWordVector(ps_path_bin, final_word_dict_python, python_final_word_vec_path, python_final_word_dict_path)
        GetNewDictAppend(ps_path_bin, python_word_dict_path, python_word_vec_path, large_word_dict_python, python_final_word_vec_path,python_final_word_dict_path)
        Serialization(python_final_word_dict_path, new_python_staqc, staqc_python_f)
    elif task == 6:
        Serialization(python_final_word_dict_path, new_python_large, large_python_f)
    else:
        print("输入的项目编号错误，请输入1-6之间的数字。")
    
    print('项目完成')
