import jieba
from jieba import posseg


class UndirectedWeightedGraph(object):  # 无向带权图

    def __init__(self, iter_num=15):
        self.iter_num = iter_num  # 设置迭代次数，默认为15次
        self.graph = {}  # 初始化图，格式为{node1: [edge1, edge2, ...], ...}
        self.d = 0.85  # 阻尼系数，设置为0.85

    def add_edge(self, start_node, end_node, weight):
        """
        使用tuple作为边, 即(start_node, end_node, weight)
        这里是无向的，所以同时为start_node和end_node插入边
        """
        if start_node not in self.graph:
            self.graph[start_node] = []
        if end_node not in self.graph:
            self.graph[end_node] = []
        self.graph[start_node].append((start_node, end_node, weight))
        self.graph[end_node].append((end_node, start_node, weight))

    def rank(self):
        node_ranks = {}  # 节点的PR值
        node_edge_sum_weight = {}  # 节点的所有边的总weight

        init_rank = 1.0 / (len(self.graph) or 1.0)  # 设置初始rank值
        for node, edges in self.graph.items():
            node_ranks[node] = init_rank  # 初始化所有node的PR值
            node_edge_sum_weight[node] = sum((edge[2] for edge in edges))  # 计算当前node的所有边的总权重

        nodes = sorted(self.graph.keys())  # 获取所有node，排个序
        for iter_i in range(self.iter_num):  # 迭代15次
            for node in nodes:  # 对于每一个node，计算其PR值
                # 根据公式 (1 - d) + d * _sum，计算
                _sum = 0
                for (_, neighbor_node, weight) in self.graph[node]:  # 迭代node的所有neighbor_node（所有有边相连的节点）
                    # 累加当前neighbor_node贡献的pr值
                    _sum += weight / node_edge_sum_weight[neighbor_node] * node_ranks[neighbor_node]
                node_ranks[node] = (1 - self.d) + self.d * _sum

        # 归一化rank，使用最大最小归一化(x - min) / (max - min)
        min_rank, max_rank = min(node_ranks.values()), max(node_ranks.values())
        for node, rank_val in node_ranks.items():
            node_ranks[node] = (rank_val - min_rank) / (max_rank - min_rank)
        return node_ranks


class TextRank(object):

    def __init__(self):

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

    @staticmethod
    def _calculate_graph(elements_relation):
        """
        根据词语共现，构建page-rank图，并计算pr值
        """
        # 创建空的无向图
        graph = UndirectedWeightedGraph()

        # 根据共现，逐步添加向图中添加边
        for (word1, word2), freq in elements_relation.items():
            graph.add_edge(word1, word2, freq)

        # 运行page-rank迭代
        nodes_rank = graph.rank()

        # 解析结果
        results = sorted(nodes_rank.items(), key=lambda item: item[1], reverse=True)
        return results

    def textrank(self, elements, window_size=3):
        """
        Args:
            elements: text rank计算的元素集合
            window_size: 窗口大小
        Returns:
            list [('院长', 1.0), ('渎职', 0.9980043667706294)]
        """
        # 保存词之间的关系成为dict
        # {
        #   (word1, word2): freq,
        #       ...
        # }
        elements_relation = {}

        # 统计每个词对儿的频数
        for i, ele in enumerate(elements):  # 迭代每个位置，准备针对每个位置滑窗
            for j in range(i + 1, min(i + window_size, len(elements))):  # 滑窗，两两保存关系
                term = (ele, elements[j])  # 当前两个词
                elements_relation[term] = elements_relation.get(term, 0) + 1  # 统计共现次数

        # 构建page-rank图，计算每个节点的权重
        results = self._calculate_graph(elements_relation)

        return results


if __name__ == '__main__':
    content = '报道称，拜登政府对中美经济关系的全面审查已进行了7个月的时间，现在审查应当回答一个核心问题，即如何处理前总统特朗普在2020' \
              '年初签署的协议。在这份协议中，为了保护美国的关键产业，如汽车和飞机制造，特朗普政府威胁对价值3600' \
              '亿美元的中国进口商品开征关税，经过双方协商，中国承诺购买更多美国产品并改变其贸易做法。但拜登政府至今没有提及这笔交易将何去何从，导致价值3600 亿美元的中国进口商品的关税悬而未决。 '
    textrank = TextRank()
    # words = [word for word in jieba.cut(content) if word not in textrank.stop_words]  # 去除停用词
    words = textrank.clean(content)
    keywords = textrank.textrank(words, window_size=3)
    print(keywords)
