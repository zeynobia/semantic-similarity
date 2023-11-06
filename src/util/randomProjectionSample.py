from gensim.models import RpModel
from gensim.corpora.textcorpus import TextCorpus
import re
from numba import jit
import numpy as np
import math
import pickle
def cosine_similarity_numpy(arr1,arr2):
    len1=np.sqrt(np.dot(arr1,arr1))
    len2=np.sqrt(np.dot(arr2,arr2))
    cosine = np.dot(arr1,arr2)/len1/len2
    return cosine

@jit(nopython=True,fastmath=True)
def cosine_similarity_numba(arr1,arr2):
    len1=np.sqrt(np.dot(arr1,arr1))
    len2=np.sqrt(np.dot(arr2,arr2))
    cosine = np.dot(arr1,arr2)/len1/len2
    return cosine

def getTopSimDocByCosineSim(trainFeatures,testFeature,topN):

    sims= np.zeros(len(trainFeatures), float) 
    for i in range(0,len(trainFeatures)):
     
        cosineSim=cosine_similarity_numba(trainFeatures[i],testFeature)  #cosine similarity
        sims[i]=-1.0*cosineSim # for reverse sort
        

    idx=np.argsort(sims)
    sims_=np.sort(sims)
    sim_idx=idx[:topN]
    return sim_idx,sims_


def preprocess(text):
    text = re.sub(r"[^\w'\s]"," ",text)
    textLower = text.lower()
    return ' '.join(textLower.strip().split())

num_topics_=100
topN=25


#corpusFile='input/arxiv-sample-abstract.txt'
corpusFile='input/corpusBig.txt'
minCnt=4 #4
#corpus_ = TextCorpus(corpusFile)
#id2word_=corpus_.dictionary
#id2word_.filter_extremes(no_below=minCnt, no_above=1) 


corpus_ = pickle.load(open(modelDir+'/'+'corpus','rb'))
id2word_=pickle.load(open(modelDir+'/'+'dict','rb'))

model = RpModel(corpus=corpus_, id2word=id2word_,num_topics=num_topics_)  # fit model
#model.save('modelRp/rp')
model.save('modelRpSample')


'''



### load model
model =  RpModel.load('modelRp/rp')  #
infp=open('input/arxiv-sample-abstract.txt','r',encoding='utf-8')
features=[]
idx=0
for line in infp:
   text=line.replace('\n','')
   content=preprocess(text)
   doc_bow = model.id2word.doc2bow(content.split())
   docVector= model.__getitem__(doc_bow)
   featureArr =list(zip(*docVector ))
   if(len(featureArr)>1):
      feature=featureArr[1]
   else:
      feature=[]
   feature=np.pad(feature, (0, num_topics_-len(feature)))
   features.append(feature)
   idx=idx+1

print('test')
#infp=open('input/arxiv-sample-abstract-NoStop.txt','r',encoding='utf-8')
infp=open('input/rake.txt','r',encoding='utf-8')
#infp=open('input/textRank.txt','r',encoding='utf-8')
idx=0
correct=0
for line in infp:
   text=line.replace('\n','')
   content=preprocess(text)
   doc_bow = model.id2word.doc2bow(content.split())
   docVector= model.__getitem__(doc_bow)
   featureArr =list(zip(*docVector ))
   if(len(featureArr)>1):
      feature=featureArr[1]
   else:
      feature=[]
   feature=np.pad(feature, (0, num_topics_-len(feature)))

 
   sim_idx,sims_=getTopSimDocByCosineSim(features,feature,topN)
   if(idx in sim_idx):
      correct=correct+1
   idx=idx+1
print(correct)
print(round(100*(correct/idx),2))
'''
