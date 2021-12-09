class BidM(object):
    '''
    Args:
        dic:指机器词典，人工构建的词典
        window_size:词典中的最长词的长度
        text:被分词的文本
        num_single_word_MM:正向最大匹配法最终分词单字个数
        num_single_word_RMM:正向最大匹配法最终分词单字个数
    '''
    def __init__(self, dic):
        self.dic = dic
        self.window_size = max([len(item) for item in dic])
        self.num_single_word_MM = 0
        self.num_single_word_RMM = 0
    
    def MM(self, text):
        result = [] # 用于保存分词的结果
        index = 0 # 表示匹配的开始位置，匹配结束意味着text_length == index
        text_length = len(text) # 被分词文本长度
        while text_length > index:
            for size in range(self.window_size + index, index, -1): 
            # 从最长字串长度开始匹配， 匹配成功则从词的末尾开始继续匹配； 匹配失败就删除匹配字段的一个字，重新扫描
                piece = text[index:size] # 匹配字段
                if piece in self.dic:
                    index = size - 1
                    if len(piece) == 1:
                        self.num_single_word_MM += 1
                    break
            index = index + 1
            result.append(piece + "----")
        return result
    
    def RMM(self, text):
        result = [] # 用于保存分词的结果
        text_length = len(text) # 被分词文本长度
        index = text_length # 表示匹配的开始位置，匹配结束意味着index == 0
        while index > 0:
            for size in range(index - self.window_size, index): 
            # 从最长字串长度开始匹配， 匹配成功则从词的前端开始继续匹配
            # 匹配失败就删除匹配字段开头的一个字，重新扫描
                piece = text[size:index] # 匹配字段
                if piece in self.dic:
                    index = size + 1
                    if len(piece) == 1:
                        self.num_single_word_RMM += 1
                    break
            index = index - 1
            result.append(piece + "----")
        result.reverse()
        return result

    
if __name__ == "__main__":
    dic = ["研究", "研究生", "生命", "命", "的", "起源", "源"]
    text = "研究生命的起源"
    tokenizer = BidM(dic)
    MM_result = tokenizer.MM(text)
    RMM_result = tokenizer.RMM(text)
    if tokenizer.num_single_word_MM <= tokenizer.num_single_word_RMM:
        result = MM_result
    else:
        result = RMM_result
    print(result)