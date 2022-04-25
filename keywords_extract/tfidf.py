import math
import os
from jieba import posseg

class TFIDF(object):
    """
    用法：
    1. 离线基于分好词的大规模预料，训练idf；
    2. 加载idf；
    3. 在线计算tf-idf；
    """

    def __init__(self):
        self.idf = {}  # 存放词及其idf权重
        self.idf_median = 0  # idf中位数，防止在线时遇到一些（训练时未见过的）新词

        # 加载停用词
        self._stop_words = set()
        with open('stop_words.txt', 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if not word:
                    continue
                self._stop_words.add(word)
        # 保留的词性
        self._ALLOW_POS = ('ns', 'n', 'vn', 'v', 'i', 'a', 'ad', 'an')

    def clean(self, text, min_word_len=1):
        """
        清洗
        :param text: str 文本
        :param min_word_len: 最小化word长度
        :return: words list
        """
        words = []
        for word, pos in posseg.cut(text):
            if not word.strip():
                continue
            if word in self._stop_words:
                continue
            if len(word) < min_word_len:
                continue
            if pos not in self._ALLOW_POS:
                continue
            words.append(word)
        return words

    def compute_tfidf(self, words):
        """
        在线计算tfidf
        Args:
            words: 分好词的一短文本，list
        Returns:
            list [('word1', weight1), ('word2', weight2), ...]
        """
        # 确保idf已经被加载：idf不为空，并且idf中位数不为0
        assert self.idf and self.idf_median, "请确保idf被加载！"

        # 统计tf
        tf = {}
        for word in words:
            tf[word] = tf.get(word, 0) + 1

        # 从加载好的idf字典中取idf，计算tfidf
        tfidf = {}
        for word in set(words):
            tfidf[word] = tf[word] / len(words) * self.idf.get(word, self.idf_median)

        # 对所有词的tfidf排序，按照权重从高到低排序，返回
        tfidf = sorted(tfidf.items(), key=lambda item: item[1], reverse=True)
        return tfidf

    def load_idf(self, idf_path, splitter=' '):
        """
        加载idf
        """
        with open(idf_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or len(line.split(splitter)) != 2:  # 跳过为空的文本、不合法的文本
                    continue
                term, idf = line.split(splitter)
                self.idf[term] = float(idf)
        self.idf_median = sorted(self.idf.values())[len(self.idf) // 2]  # 计算idf的中位数

    def train_idf(self, seg_files, output_file_name, splitter=' '):
        """
        离线训练idf
        Args:
            seg_files: list 分过词的训练文件列表，txt格式，文档用\n换行
            output_file_name: 输出的标准idf文本，txt格式
            splitter: term之间分隔符，word和idf的分隔符
        """
        # 总文档数初始化为0
        doc_count = 0

        # 统计df
        for seg_file in seg_files:  # 迭代所有文件
            with open(seg_file, encoding='utf-8') as f:
                for line in f:  # 迭代每一行
                    line = line.strip()
                    if not line:
                        continue
                    doc_count += 1  # 更新总文档数
                    words = set(line.split(splitter))
                    for word in words:
                        self.idf[word] = self.idf.get(word, 0) + 1  # 更新当前word的文档频数

        # 计算idf，保存到文件
        with open(output_file_name, 'w', encoding='utf-8') as f:
            for word, df in self.idf.items():
                self.idf[word] = math.log(doc_count / (df + 1))  # 计算idf
                f.write('{}{}{}\n'.format(word, splitter, self.idf[word]))


if __name__ == '__main__':
    seg_files_dir = 'test_data/seg_data'
    seg_files = ['{}/{}'.format(seg_files_dir, f) for f in os.listdir(seg_files_dir)]
    tfidf = TFIDF()
    # tfidf.train_idf(seg_files, 'idf.txt', ' ')
    tfidf.load_idf('idf.txt', ' ')

    sentence = "今天他给我说，我的放假旅游计划泡汤了，因为要封校"
    # sl = jieba.lcut(sentence)
    sl = tfidf.clean(sentence)
    print(sl)
    result = tfidf.compute_tfidf(sl)
    print(result)
