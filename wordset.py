import numpy as np
import nltk
import re
nltk.download('brown')
from nltk.corpus import brown

class WordSet:
    def __init__(self, categories=['news', 'fiction'], vecLength = 10, negSampleSize=15):
        self.negSampleSize = negSampleSize
        self.corpus = self.genCorpus(categories)
        self.entries, self.uMatrix = self.genWordVectors(vecLength)
        _, self.vMatrix = self.genWordVectors(vecLength)
        self.dist = self.genDist()

    def vVector(self, entry):
        id = self.entries[entry]
        return self.vMatrix[id]

    def uVector(self, entry):
        id = self.entries[entry]
        return self.uMatrix[id]

    def uMatrixSlice(self,entries):
        ids = list(map(lambda entry: self.entries[entry], entries))
        return self.uMatrix[np.array(ids),:]

    def updateUMatrixSlice(self,entries,d_u_matrix):
        ids = list(map(lambda entry: self.entries[entry], entries))
        self.uMatrix[np.array(ids),:] += d_u_matrix

    def updateVVector(self,entry,v):
        id = self.entries[entry]
        self.vMatrix[id] += v

    def updateUVector(self,entry,u):
        id = self.entries[entry]
        self.uMatrix[id] += u

    def drawNegSample(self, word):
        sample = list(map(lambda i: self.drawNegWord(),list(range(self.negSampleSize))))
        if word in sample:
            return self.drawNegSample(word)
        return list(map(lambda i: self.drawNegWord(),list(range(self.negSampleSize))))

    def drawNegWord(self):
        (_, highest) = self.dist[len(self.dist)-1]
        v1 = np.random.uniform(0.0, highest)
        for i in range(len(self.dist)):
            (w,v2) = self.dist[i]
            if v1 < v2:
                return w

    def genCorpus(self,categories):
        corpus = []
        for cat in ['news', 'fiction']:
            for text_id in brown.fileids(cat):
                raw_text = list(map(lambda x: ' '.join(x), brown.sents(text_id)))
                text = ' '.join(raw_text)
                text = text.lower()
                text.replace('\n', ' ')
                text = re.sub('[^a-z ]+', '', text)
                corpus.append([w for w in text.split() if w != ''])
        return corpus

    def genWordVectors(self,vecLength):
        d = {}
        list(map(lambda c: list(map(lambda w: self.genWordEntry(d, w, vecLength), c)), self.corpus))
        matrix = self.generateWordMatrix(d)
        entriesToIdx = dict(zip(list(d.keys()),range(len(matrix))))
        return entriesToIdx, matrix

    def genWordEntry(self,dict, word, vecLength):
        if not (word in dict):
            dict[word] = 0.01*np.random.rand(vecLength)

    def generateWordMatrix(self,dict):
        keys = list(dict.keys())
        m = len(keys)
        n = len(dict[keys[0]])
        mat = np.zeros((m,n))
        list( map(lambda id: self.setMatrixVector(id,mat,dict,keys[id]), range(m)))
        return mat

    def setMatrixVector(self,row,mat,d,entry):
        mat[row] = d[entry]

    def genDist(self):
        freqs = {}
        list(map(lambda c: list(map(lambda w: self.addToFreq(freqs, w), c)), self.corpus))
        dist = freqs.items()
        dist = list(map(lambda e: (e[0],e[1]**0.75),dist))
        for i in range(1,len(dist)):
            (_,v1)= dist[i-1]
            (w,v2) = dist[i]
            dist[i] = (w,v2+v1)
        return dist

    def addToFreq(self,freqs, entry):
        if entry in freqs:
            freqs[entry] += 1.0
        else:
            freqs[entry] = 1.0
