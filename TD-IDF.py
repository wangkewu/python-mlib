import numpy as np
import pandas as pd

docA = "The cat sat on my bed"
docB = "The dog sat on my kness"

bowA = docA.split(" ")
bowB = docB.split(" ")

#构建词库
wordSet = set(bowA).union(set(bowB))

print(wordSet)

#统计字典来保存词出现的次数

wordDictA = dict.fromkeys(wordSet, 0)
wordDictB = dict.fromkeys(wordSet, 0)
print("wordDictA:", wordDictA)


#遍历文档 统计词数
for word in bowA:
	wordDictA[word] += 1
for word in bowB:
	wordDictB[word] += 1

print(pd.DataFrame([wordDictA, wordDictB]))


#计算词频
def computeTF(wordDict, bow):
	#字典对象记录所有的tf  把所有的词对应在bow文档里的tf都算出来
	tfDict = {}
	nbowCount = len(bow)

	for word, count in wordDict.items():
		tfDict[word] = count/nbowCount

	return tfDict

tfA = computeTF(wordDictA, bowA)
tfB = computeTF(wordDictB, bowB)

#计算逆文档频率IDF
def coumputeIDF(wordDictList):
	#用一个字典对象 保存idf结果 每个词作为key 
	idfDict = dict.fromkeys(wordDictList[0], 0)
	N = len(wordDictList)
	import math

	for wordDict in wordDictList:
		#遍历的是字典中的每个词汇 统计Ni  表示文档集中包含了词语i的文档数
		for word, count in wordDictB.items():
			if count > 0:
				#先把 Ni 增加 1 存入到idfDict
				idfDict[word] += 1
	#已经得到所有词汇i对应的Ni 现在根据公式把它2替换成idf值
	for word, ni in idfDict.items():
		idfDict[word]  = math.log10((N+1)/(ni+1))

	return idfDict
idfs = coumputeIDF([wordDictA, wordDictB])
print(idfs)


#Last 计算 TF-IDF
def computeTFIDF(tf, idfs):
	tfidf = {}
	for word, tfval in tf.items():
		tfidf[word]  = tfval * idfs[word]

	return tfidf
tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)

print(pd.DataFrame([tfidfA, tfidfB]))