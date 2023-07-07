import pickle

# 用于从corpus中提取词汇
def getVocabulary(corpus1, corpus2):
    word_vocab = set()
    
    for corpus in [corpus1, corpus2]:
        for row in corpus:
            for column in row[1:4]:
                for word in column[0]:
                    word_vocab.add(word)
   
    print(len(word_vocab))
    return word_vocab

def LoadTxt(filePath):
    file = open(filePath, 'r')
    bakInfo = eval(file.read())
    file.close()
    return bakInfo

# 处理最终的词汇表
def finalVocabPrpcessing(filepath1, filepath2, save_path):
    word_set = set()
    
    total_data1 = set(LoadTxt(filepath1))
    total_data1 = list(total_data1)
    total_data2 = LoadTxt(filepath2)

    vocab = getVocabulary(total_data1, total_data2)
    
    for word in vocab:
        if word not in total_data1:
            word_set.add(word)

    print(len(total_data1))
    print(len(word_set))

    with open(save_path, "w") as f:
        f.write(str(word_set))
        f.close()

if __name__ == "__main__":
    # 文件路径定义
    sql_word_dict = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/word_dict/sql_word_vocab_dict.txt'
    new_sql_large = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    large_word_dict_sql = '../hnn_process/ulabel_data/sql_word_dict.txt'
    # 生成词汇表
    finalVocabPrpcessing(sql_word_dict, new_sql_large, large_word_dict_sql)
