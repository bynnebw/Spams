# -*- coding: utf-8 -*-

import re
import math

def getwords(doc):
    splitter = re.compile('\\W*')
    words = [s.lower() for s in splitter.split(doc) if 2 < len(s) < 20]
    return dict([(w, 1) for w in words])

class classifier:
    def __init__(self, getfeatures, filename=None):
        # 统计特征/分类组合的数量
        self.fc = {}
        # 统计每个分类中的文档数量
        self.cc = {}
        self.getfeatures = getfeatures

    '''
    fc is used to record the features like this:
    {'python':{'bad':0, 'good':6}, 'the':{'bad':3, 'good':3}}
    In this dictionary, the word 'python' and 'the' are the features,
    and the 'bad'/'good'are the categories.
    '''

    # 增加对 特征/分类 组合的计数值
    def incf(self, f, cat):
        self.fc.setdefault(f, {})
        self.fc[f].setdefault(cat, 0)
        self.fc[f][cat] += 1

    # 增加对某一分类的计数值
    def incc(self, cat):
        self.cc.setdefault(cat, 0)
        self.cc[cat] += 1

    # 某一特征出现于某一分类中的次数
    def fcount(self, f, cat):
        if f in self.fc and cat in self.fc[f]:
            return float(self.fc[f][cat])
        return 0.0

    # 属于某一分类的内容项数量
    def catcount(self, cat):
        if cat in self.cc:
            return float(self.cc[cat])
        return 0

    # 所有内容项的数量
    def totalcount(self):
        return sum(self.cc.values())

    # 所有分类的列表
    def categories(self):
        return self.cc.keys()

    def train(self, item, cat):
        features = self.getfeatures(item)
        for f in features:
            self.incf(f, cat)
        self.incc(cat)

    def fprob(self, f, cat):
        if self.catcount(cat) == 0:
            return 0
        return self.fcount(f, cat) / self.catcount(cat)

    # 广泛使用函数指针
    def weightedprob(self, f, cat, prf, weight=1.0, ap=0.5):
        # 计算当前的概率值
        basicprob = prf(f, cat)
        # 统计特征在所有分类中出现的次数
        totals = sum([self.fcount(f, c) for c in self.categories()])

        # 计算加权平均
        bp = ((weight*ap) + (totals*basicprob)) / (weight+totals)
        return bp

class naivebayes(classifier):
    # 计算整篇文章的概率
    def docprob(self, item, cat):
        features = self.getfeatures(item)

        p = 1
        for f in features:
            p *= self.weightedprob(f, cat, self.fprob)
        return p



def sampletrain(cl):
    cl.train('Nobody owns the water.', 'good')
    cl.train('the quick rabbit jumps fences', 'good')
    cl.train('buy pharmaceuticals now', 'bad')
    cl.train('make quick money at the online casino', 'bad')
    cl.train('the quick brown fox jumps', 'good')


