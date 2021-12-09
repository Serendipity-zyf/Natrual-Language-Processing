# 以语句“研究生命的起源”为例子
# 构建分词器MM
class MM(object):
    '''
    Args:
        dic:指机器词典，人工构建的词典
        window_size:词典中的最长词的长度
        text:被分词的文本
    '''
    def __init__(self, dic):
        self.window_size = max([len(item) for item in dic])
    
    def cut(self, dic, text):
        result = [] # 用于保存分词的结果
        index = 0 # 表示匹配的开始位置，匹配结束意味着text_length == index
        text_length = len(text) # 被分词文本长度
        while text_length > index:
            for size in range(self.window_size + index, index, -1): 
            # 从最长字串长度开始匹配， 匹配成功则从词的末尾开始继续匹配； 
            # 匹配失败就删除匹配字段的一个字，重新扫描
                piece = text[index:size] # 匹配字段
                if piece in dic:
                    index = size - 1
                    break
            index = index + 1
            result.append(piece + "----")
        return result
                
        
if __name__ == "__main__" :   
    #人工构建的词典    
    dic = ["研究", "研究生", "生命", "命", "的", "起源", "源"] 
    text = "研究生命的起源"
    tokenizer = MM(dic)
    print(tokenizer.cut(dic, text))