# -*- coding: utf-8 -*-
import operator
import jieba
import jieba.analyse
import pymysql
from flask import Flask
from flask import jsonify
from flask import request
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from aip import AipNlp


app = Flask(__name__)

topK = 10
withWeight = 1
executor = ThreadPoolExecutor(10)    # 同时处理的最大线程数

MYSQL_HOST = "192.168.10.121"
MYSQL_USER = 'hzyg'
MYSQL_PASSWORD = '@hzyq20180426..'
MYSQL_DB = "yfhunt"


@app.route('/', methods=['GET'])
def index():
    return "Hello，欢迎访问API！"


@app.route('/keywords', methods=['POST'])
def keywords():
    # jieba.enable_parallel()
    # id = request.form.get('id')
    # mysql = MysqlClient()
    # mysql.open_sql('yfhunt')
    # content = mysql.get_content(id)
    # mysql.close_sql()
    content = request.form.get('content')
    try:
        data_list = jieba.analyse.textrank(content, topK=topK, withWeight=withWeight)
        tags = []
        for data in data_list:
            word = {}
            word['tag'] = data[0]
            word['weight'] = data[1]
            tags.append(word)
        status = 1
    except Exception as e:
        print(e)
        tags = {}
        status = 2
    output = {'keywords': tags, 'status': status}
    # executor.submit(keywords, id)
    return jsonify(output)


@app.route('/summary', methods=['POST'])
def summary():
    # jieba.enable_parallel()
    # id = request.form.get('id')
    # mysql = MysqlClient()
    # mysql.open_sql('yfhunt')
    # content = mysql.get_content(id)
    # mysql.close_sql()
    sum = Summary()
    content = request.form.get('content')
    summary, status = sum.func(content)
    output = {'summary': summary, 'status': status}
    # executor.submit(keywords, id)
    return jsonify(output)

# def func(id):
#     pass


@app.route('/surface', methods=['POST'])
def surface():
    # jieba.enable_parallel()
    # id = request.form.get('id')
    # mysql = MysqlClient()
    # mysql.open_sql('yfhunt')
    # content = mysql.get_content(id)
    # summary = mysql.get_summary(id)
    # mysql.close_sql()
    bc = BdNlp()
    content = request.form.get('content')
    summary = request.form.get('summary')
    result, status = bc.surface(content, summary)
    output = {'surface': result, 'status': status}
    # executor.submit(keywords, id)
    return jsonify(output)


@app.route('/category', methods=['POST'])
def category():
    # jieba.enable_parallel()
    # id = request.form.get('id')
    # mysql = MysqlClient()
    # mysql.open_sql('yfhunt')
    # title = mysql.get_title(id)
    # content = mysql.get_content(id)
    # mysql.close_sql()
    title = request.form.get('title')
    content = request.form.get('content')
    bc = BdNlp()
    category, status = bc.category(title, content)
    output = {'category': category, 'status': status}
    # executor.submit(keywords, id)
    return jsonify(output)


class MysqlClient(object):

    def __init__(self):
        self.mysql_host = "192.168.10.121"
        self.mysql_user = 'hzyg'
        self.mysql_password = '@hzyq20180426..'

    def open_sql(self, ms_db):
        self.link = pymysql.connect(self.mysql_host, self.mysql_user, self.mysql_password, ms_db)
        self.link.set_charset('utf8')
        self.cursor = self.link.cursor()

    def get_title(self, id):
        sql = "select title from rq_news where id='%s'" % id
        self.cursor.execute(sql)
        title = self.cursor.fetchall()
        self.link.commit()
        return title[0][0]

    def get_content(self, id):
        sql = "select content from rq_news where id='%s'" % id
        self.cursor.execute(sql)
        content = self.cursor.fetchall()
        self.link.commit()
        return content[0][0]

    def get_summary(self, id):
        sql = "select new_abstract from rq_news where id='%s'" % id
        self.cursor.execute(sql)
        summary = self.cursor.fetchall()
        self.link.commit()
        return summary[0][0]

    def close_sql(self):
        self.link.close()


class Summary(object):

    def func(self, content):
        texts = content.split(' ')
        sentences = []
        clean = []
        originalSentenceOf = {}
        try:
            # Data cleansing
            for line in texts:
                parts = line.split('。')[:-1]  # 句子拆分
                # print(parts)
                for part in parts:
                    cl = self.cleanData(part)  # 句子切分以及去掉停止词
                    # print(cl)
                    sentences.append(part)  # 原本的句子
                    clean.append(cl)  # 干净有重复的句子
                    originalSentenceOf[cl] = part  # 字典格式
            setClean = set(clean)  # 干净无重复的句子

            # calculate Similarity score each sentence with whole documents
            scores = {}
            for data in clean:
                temp_doc = setClean - set([data])  # 在除了当前句子的剩余所有句子
                score = self.calculateSimilarity(data, list(temp_doc))  # 计算当前句子与剩余所有句子的相似度
                scores[data] = score  # 得到相似度的列表

            # calculate MMR
            n = 25 * len(sentences) / 100  # 摘要的比例大小
            alpha = 0.7
            summarySet = []
            while n > 0:
                mmr = {}
                # kurangkan dengan set summary
                for sentence in scores.keys():
                    if sentence not in summarySet:
                        mmr[sentence] = alpha * scores[sentence] - \
                                        (1 - alpha) * self.calculateSimilarity(sentence, summarySet)  # 公式
                selected = max(mmr.items(), key=operator.itemgetter(1))[0]
                summarySet.append(selected)
                n -= 1

            rows = []
            for sentence in summarySet:
                row = originalSentenceOf[sentence].lstrip(' ')
                rows.append(row)
            summary = '。'.join(rows) + '。'
            status = 1
        except Exception as e:
            print(e)
            summary = ""
            status = 2
        return summary, status

    @staticmethod
    def cleanData(name):
        f = open('stopword.txt', encoding='utf-8')  # 停止词
        stopwords = f.readlines()
        stopwords = [i.replace("\n", "") for i in stopwords]
        setlast = jieba.cut(name, cut_all=False)
        seg_list = [i.lower() for i in setlast if i not in stopwords]
        return " ".join(seg_list)

    @staticmethod
    def calculateSimilarity(sentence, doc):  # 根据句子和句子，句子和文档的余弦相似度
        if doc == []:
            return 0
        vocab = {}
        for word in sentence.split():
            vocab[word] = 0  # 生成所在句子的单词字典，值为0

        docInOneSentence = ''
        for t in doc:
            docInOneSentence += (t + ' ')  # 所有剩余句子合并
            for word in t.split():
                vocab[word] = 0  # 所有剩余句子的单词字典，值为0

        cv = CountVectorizer(vocabulary=vocab.keys())

        docVector = cv.fit_transform([docInOneSentence])
        sentenceVector = cv.fit_transform([sentence])
        return cosine_similarity(docVector, sentenceVector)[0][0]


class BdNlp(object):

    def __init__(self):
        self.id = '14569406'
        self.key = 'ks1rNDsGK09ydjF2dMSXZGB2'
        self.secret = 'oF2ooRt1PMgs4qThtx7PubO6GwZgt5zY'
        self.client = AipNlp(self.id, self.key, self.secret)

    """ 调用情感倾向分析 """
    def surface(self, content, summary):
        try:
            data = self.client.sentimentClassify(content)
            if 'error_code' in data:
                data = self.client.sentimentClassify(summary)
                result = data['items'][0]
                status = 1
            else:
                result = data['items'][0]
                status = 1
        except Exception as e:
            print(e)
            result = {}
            status = 2
        return result, status

    """ 调用文章分类 """
    def category(self, title, content):
        title = title.encode('utf-8').decode('utf-8')
        content = content.encode('utf-8').decode('utf-8')
        try:
            data = self.client.topic(title, content)
            category = data['item']
            status = 1
        except Exception as e:
            print(e)
            category = {}
            status = 2
        return category, status


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
