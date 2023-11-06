
import simplejson as json
from flask import Flask,render_template,request, Response
import sys
import os
import gensim
import re
from collections import defaultdict
import numpy as np
import time
import ctypes # for c dynamic library
import math
from numpy.ctypeslib import ndpointer
from ctypes import POINTER
from operator import itemgetter
import json
from datetime import datetime
import random   
import sys
import pickle
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


app = Flask(__name__)


def listToText(sentences):
    strRes=''
    for elem in sentences:
        strRes+=str(elem)+' '
    strRes=preprocess(strRes)
    return strRes
	
	
def addPunctionWindow(text,window=12):
    punc='. '
    words=re.split('\s+',text)
    puncText=''
    wordCnt=0
    for word in words:
        puncText=puncText+word+' '
        if(wordCnt!=0 and wordCnt%window==0):
           puncText+=punc
        wordCnt+=1
    if(puncText[len(puncText)-2]!='.'):
       puncText+=punc
    return puncText

def preprocessExceptDot(text):
    text = re.sub(r"[^\w\s.]"," ",text)
    textLower = text.lower()
    return ' '.join(textLower.strip().split())

def removeStopWords(text,cachedStopWords):
    filteredText = ' '.join([word for word in text.split() if word not in cachedStopWords])
    return filteredText

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text,stemmer):
    tokens = nltk.word_tokenize(text)
    #tokens = [i for i in tokens if i not in string.punctuation]
    stems = stem_tokens(tokens, stemmer)
    return ' '.join(stems)

def preprocess(text):
    text = re.sub(r"[^\w\s]"," ",text)
    textLower = text.lower()
    return ' '.join(textLower.strip().split())

def writeListToFile(outfile,list):
    outfp=open(outfile,'w',encoding='utf-8')
    for elem in list:
        outfp.write(str(elem)+'\n')
    outfp.close()    
 

def split_list(lst, n):
    splitted = []
    for i in reversed(range(1, n + 1)):
        split_point = len(lst)//i
        splitted.append(lst[:split_point])
        lst = lst[split_point:]
    return splitted


def getLoadModel(algorithm,modelFile):

  if((algorithm=="hdp") or (algorithm=="hdpSparse")):
      model=gensim.models.HdpModel.load(modelFile)
  elif(algorithm=="doc2vec-dbow" or algorithm=="doc2vec-dbpv"):
      model =gensim.models.Doc2Vec.load(modelFile)
  return model
 

def  getTopSimDocByCosineSimC(trainFeatures,trainIds,testFeature,len_train,len_dim,topN):

    if(isSingleSearchMultiThread):
       sim_obj=libSim.cosine_similarity_topMTSingle(trainFeatures,testFeature,len_train,len_dim,topN,singleMultiThreadCnt)  #idx,similarity
    else:
       sim_obj=libSim.cosine_similarity_top(trainFeatures,testFeature,len_train,len_dim,topN)  #idx,similarity
    sim_ids=[]
    sim_s=[]
    for t in range(topN):
        sim_ids.append(int(sim_obj[t][0])) #id
        sim_s.append(np.round(sim_obj[t][1],4)) #id,sim

    return sim_ids,sim_s


def getDictIds(inFile):
   idsDict=dict()
   lines=[]
   infp=open(inFile,'r',encoding='utf-8')
   for line in infp:
      text=line.replace("\n","")
      parts=text.split('\t')
      idsDict[int(parts[0])]=parts[1]
   return idsDict


def getBow(dictVocab,text):
   parts=text.split()
   bowArr=[]
   for elem in parts:
     if(elem in dictVocab):
        bowArr.append(dictVocab[elem])
   return bowArr




def loadDict(inFile,rev=True,delim='\t'):
   resDict={}
   infp=open(inFile,'r',encoding='utf-8')
   for line in infp:
      text=line.replace('\n','')
      parts=text.split(delim)
      if(rev==False):
         resDict[(parts[0])]=parts[1]
      else:
         resDict[parts[1]]=(parts[0])
   return resDict


def getFeatureFromText(model,tfidf_model,text,algorithm):
   

       if((algorithm=="hdp") and (isHdpDivide==False)):
         doc_bow = model.id2word.doc2bow(text.split())
         docVector=model[doc_bow]
         docVectorTmp=[]
         for i in range(featuresHdp):
           docVectorTmp.append(0)
         for x in docVector:
           if(len(x)>1):
             docVectorTmp[x[0]]=round(x[1],precision)
         feature=docVectorTmp


       elif((algorithm=="hdp") and (isHdpDivide==True)):

         allParts=text.split('.') #punction dot character
         tmpList=list(range(len(allParts)))
         hdpDivideCntTmp=hdpDivideCnt
         if(len(allParts)<(minHdpDivideSent*hdpDivideCnt)):
            hdpDivideCntTmp=math.ceil(len(allParts)/minHdpDivideSent)
         if(len(allParts)> (maxHdpDivideSent*hdpDivideCnt)):
            hdpDivideCntTmp=math.ceil(len(allParts)/maxHdpDivideSent)
         
         random.shuffle(tmpList)
         allSplitParts=split_list(tmpList,hdpDivideCntTmp)

         paraphs=[]
         for ij in range(len(allSplitParts)):
           tmpArr=allSplitParts[ij]
           strTmp=''
           for jk in tmpArr:
             strTmp+=allParts[jk]+' '
           paraphs.append(strTmp.replace('  ',' '))

         docVectorHdpTmp=[]
         for i in range(featuresHdp):
            docVectorHdpTmp.append(0)
         for contentEle in paraphs:
           contentPre=preprocess(contentEle)
           doc_bow = model.id2word.doc2bow(contentPre.split())
           docVector=model[doc_bow]
           #print(docVector)
           for x in docVector:
             if(len(x)>1):
               docVectorHdpTmp[x[0]]=docVectorHdpTmp[x[0]]+round(x[1]/hdpDivideCntTmp,6)
         feature=docVectorHdpTmp
       else: #doc2vec
         docVector=[x for x in model.infer_vector(text.split(), alpha=start_alpha,min_alpha=min_alpha_, steps=infer_epoch)]
         feature =docVector 
      
       return feature


def getLoadVectorFloatArray(vectFile,delim=';'):
   X_train=[]
   infp=open(vectFile,'r',encoding='utf-8')
   for line in infp:
        #print(line)
        text=line.replace('\n','')
        if(text!='' and text!=';'  and text!=' ; '):
          vectParts=text.split(delim)
          string_array = np.array(vectParts)
          float_array = string_array.astype(np.float32) 
          X_train.append(float_array)
        else:
          X_train.append([])

   return X_train

    
#id\tdocUid
def getOnlyIdFromFile(fileIdsFile):
    fileIds=[]
    infp=open(fileIdsFile,'r',encoding='utf-8')
    for line in infp:
        text=line.replace('\n','')
        id=text.split('\t')[0] 
        fileIds.append(id)
    return fileIds

def getLoadVector(vectFile,lenDim,allSize,delim=';'):

  infp=open(vectFile,'r',encoding='utf-8')
  c_float_p = POINTER(ctypes.c_float)
  cfloatArray = (c_float_p * allSize) ()
  #print('cfloatArr created')
  idx=0
  for line in infp:
        text=line.replace('\n','')
        if(text!='' and text!=';'  and text!=' ; '):
          vectParts=text.split(delim)
          cfloatArray[idx] = (ctypes.c_float * len(vectParts))()
          for j in range(len(vectParts)):
              cfloatArray[idx][j] = ctypes.c_float( float(vectParts[j]))
          #if(idx%10000==0):
             #print(idx)
        else:
           cfloatArray[idx] = (ctypes.c_float * len(vectParts))()
           for j in range(lenDim):
              cfloatArray[idx][j] =0
        idx=idx+1
  return cfloatArray

def getTopNDoc2Vec(text,model,tfidf_model, X_train_C,topN):
   
   docVector=[x for x in model.infer_vector(text.split(), alpha=start_alpha,min_alpha=min_alpha_, steps=infer_epoch)]
   docVectorNp=np.array([ docVector ],'f')
   sim_ids,sim_s=getTopSimDocByCosineSimC(   X_train_C,trainIds,docVectorNp,len_train,len_dim,topN)
   return  sim_ids,sim_s

def getFeatureLineCnt(inferFile,delim=';'):
    lineCnt=0
    infp=open(inferFile,'r',encoding='utf-8')
    for line in infp:
        lineCnt=lineCnt+1
    infp.close()
    first_line = next(open(inferFile))
    featureCnt=len(first_line.split(delim))
    return featureCnt,lineCnt

def getTopNSmartSearch(text,modelSemantic,modelCluster,tfidf_model,X_train_C,topN,algorithmSemantic='doc2vec-dbow'):

     inferVectorSemantic=getFeatureFromText(modelSemantic,tfidf_model,text,algorithmSemantic)
     inferVectorCluster=getFeatureFromText(modelCluster,tfidf_model,text,algorithmCluster)
     testFeature=np.array([list(inferVectorSemantic)],'f')
     npArrCluster=np.array(inferVectorCluster)
     thProb=minProb
     maxTmpProb=np.max(inferVectorCluster)
     if(thProb>maxTmpProb):
         thProb=maxTmpProb
     indices = np.nonzero(npArrCluster>=thProb)[0]
     maxLabels =indices[np.argsort(npArrCluster[indices])[::-1][:maxSearchTopNCluster]]
     #maxProb= inferVectorCluster[maxLabels[0]]
     labelList= np.concatenate([ labelsDict[maxLabels[x]] for x in range(len(maxLabels)) if maxLabels[x] in labelsDict ])      
     sim_obj=libSim.cosine_similarity_topMTSingleCluster(X_train_C,testFeature,len_dim,topN,labelList,len(labelList),10)  #idx,similarity
     sim_ids=[]
     sim_s=[]
     for t in range(topN):
         sim_ids.append(int(sim_obj[t][0])) #id
         sim_s.append(np.round(sim_obj[t][1],4)) #id,sim
    
     return  sim_ids,sim_s

def loadLabelsDictFromFile(labelsFile):
   infp=open(labelsFile,'r',encoding='utf-8')
   labelsDictDef=defaultdict(list)
   lineCnt=0
   for line in infp:
      strLine=line.replace('\n','')
      labelsDictDef[int(strLine)].append(np.array(lineCnt, dtype=np.int32))
      lineCnt=lineCnt+1

   labelsDict=dict()
   for elem in labelsDictDef:
      labelsDict[elem]=np.array(np.sort(labelsDictDef[elem]),dtype=np.int32)
   labelsDictDef.clear()
   return labelsDict



@app.route('/')
def index():
    return render_template('index.html')

@app.route("/getSemanticSimilarity",methods=['POST'])
def getSemanticSimilarity():
    try:
        content=request.form['content']
        if "algorithm" in request.form:
            algorithm=request.form['algorithm']
        else:
            algorithm='fastSmartSearch' #default algorithm is fastSmartSearch setted.

        text=content.replace('\n','')
        if(isStemNoStopWords):
           text=preprocessExceptDot(text) #preprocess except dot char
           text=removeStopWords(text,cachedStopWords)
           text=tokenize(text,stemmer)
        if((isStemNoStopWords==False) and (isOnlyRemoveStopWords)):
            text=preprocessExceptDot(text) #preprocess except dot char
            text=removeStopWords(text,cachedStopWords)

        if(algorithm!='fastSmartSearch'):
           text=preprocess(text)
 
        if(algorithm=='fastSmartSearch'):
           if((len(text)>1000) and (text.find(".")<0 )):
              text=addPunctionWindow(text)
           parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
           document=parser.document
           sentences=sumBasicSummarizer(document,sentCnt)
           textRes=listToText(sentences)
           sim_ids,sim_s=getTopNSmartSearch(textRes,modelDoc2Vecdbow_,modelCluster,tfidf_model,X_train_C_Dbow,topN,algorithmSemantic)
           
        elif(algorithm=='doc2vec-dbow'):
           sim_ids,sim_s=getTopNDoc2Vec(text,modelDoc2Vecdbow_,tfidf_model,  X_train_C_Dbow,topN)

        simFiles=[]
        for elem in  sim_ids:
            if(int(elem) in idsDict):
              simFiles.append(idsDict[int(elem)])

        data = { "content":content, "similarFiles":simFiles, "similarIds":sim_ids, "similarRatios":sim_s }
        json_response = json.dumps(data,ensure_ascii = False, indent=2,sort_keys=False)
        response = Response(json_response,content_type="application/json; charset=utf-8" )
        return response
    except Exception as e:
          print(e)
          data = { "content":content, "similarFiles":[], "similarIds":[], "similarRatios":[]}
          json_response = json.dumps(data,ensure_ascii = False, indent=2,sort_keys=False)
          response = Response(json_response,content_type="application/json; charset=utf-8" )
          return response

if __name__ == "__main__":

  
  confFile=sys.argv[1]
  with open(confFile, 'r') as f:
    config = json.load(f)

 
  host=config["host"]
  port=config["port"]
  modelDir=config['modelDir']
  modelClusterDir=config['modelClusterDir']
  inferDir=config['inferDir']
  topN=config['topN']
  minProb=config['minProb']
  maxSearchTopNCluster=config['maxSearchTopNCluster']
  sentCnt=config['sentCnt']
  isStemNoStopWords=config['isStemNoStopWords']
  isOnlyRemoveStopWords=config['isOnlyRemoveStopWords']


  fileIdsFile=config['fileIdsFile']
  labelsClusterFile=config['labelsClusterFile']
  infer_epoch=config['infer_epoch'] 
  start_alpha=config['start_alpha'] # learning rate
  min_alpha_=config['min_alpha']
 

  algorithmCluster=config["algorithmCluster"]
  algorithmSemantic=config["algorithmSemantic"]
  featuresHdp=config["featuresHdp"]
  isHdpDivide=config["isHdpDivide"]
  hdpDivideCnt=config["hdpDivideCnt"]
  minHdpDivideSent=config["minHdpDivideSent"]
  maxHdpDivideSent=config["maxHdpDivideSent"]
  isFastSmartSearchCluster=config["isFastSmartSearchCluster"]
 
  isSingleSearchMultiThread=config['isSingleSearchMultiThread']
  singleMultiThreadCnt=config['singleMultiThreadCnt']

  isSearchDoc2vecDbow=config['isSearchDoc2vecDbow']


  modelDoc2VecDbow=modelDir+'/'+str(algorithmSemantic)
  hdpClusterAlgModelFile=modelClusterDir+'/'+str(algorithmCluster)

  tfidfFile=modelDir+'/'+'tfidf'
  precision=6
  vectFileDelim=';'

  #trainIds=getLines(fileIdsFile)
  trainIds=getOnlyIdFromFile(fileIdsFile)
  idsDict=getDictIds(fileIdsFile)
  labelsDict=loadLabelsDictFromFile(labelsClusterFile)

  libSim= ctypes.cdll.LoadLibrary('src/bin/sim.so')

  charptr= ctypes.c_char_p
  libSim.loadVectFileC.restype =  POINTER(POINTER(ctypes.c_float))
  libSim.loadVectFileC.argtypes = [ charptr,charptr];

  libSim.cosine_similarity_top.restype =  POINTER(POINTER(ctypes.c_float))
  libSim.cosine_similarity_top.argtypes = [ POINTER(POINTER(ctypes.c_float))  , ndpointer(ctypes.c_float),ctypes.c_int,ctypes.c_int,ctypes.c_int]

  libSim.cosine_similarity_topMTSingle.restype =  POINTER(POINTER(ctypes.c_float))
  libSim.cosine_similarity_topMTSingle.argtypes = [ POINTER(POINTER(ctypes.c_float))  , ndpointer(ctypes.c_float),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]


  libSim.cosine_similarity_topSparse.restype =  POINTER(POINTER(ctypes.c_float))
  #( uint16_t ** trainIndices,uint16_t * testIndice,float** trainValues,float* testValue,uint16_t * trainLens,uint16_t testLen,int dataSize,int dim,int topN) 
  libSim.cosine_similarity_topSparse.argtypes = [  POINTER(POINTER(ctypes.c_uint16)),ndpointer(ctypes.c_uint16), POINTER(POINTER(ctypes.c_float))  , ndpointer(ctypes.c_float), ndpointer(ctypes.c_uint16), ctypes.c_uint16, ctypes.c_int,ctypes.c_int,ctypes.c_int]

  libSim.jaccard_similarity_top.restype =  POINTER(POINTER(ctypes.c_float))
  libSim.jaccard_similarity_top.argtypes = [ POINTER(POINTER(ctypes.c_int))  , ndpointer(ctypes.c_int),ctypes.c_int,ctypes.c_int,ctypes.c_int]


  libSim.cosine_similarity_topMTSingleCluster.restype =  POINTER(POINTER(ctypes.c_float))
  libSim.cosine_similarity_topMTSingleCluster.argtypes = [ POINTER(POINTER(ctypes.c_float))  , ndpointer(ctypes.c_float),ctypes.c_int,ctypes.c_int,ndpointer(ctypes.c_int),ctypes.c_int,ctypes.c_int]

  tfidf_model=gensim.models.TfidfModel.load(tfidfFile)

  LANGUAGE = "english"
  cachedStopWords = stopwords.words(LANGUAGE)
  stemmer= PorterStemmer()
  sumyStemmer = Stemmer(LANGUAGE)
  stopWords=get_stop_words(LANGUAGE)
  sumBasicSummarizer = SumBasicSummarizer(sumyStemmer)
  sumBasicSummarizer.stop_words = stopWords

  if(isFastSmartSearchCluster):
     modelDoc2Vecdbow_=getLoadModel(algorithmSemantic, modelDoc2VecDbow)
     inferFile=inferDir+'/'+str(algorithmSemantic)+'.txt'
     len_dim,len_train=getFeatureLineCnt(inferFile,delim=vectFileDelim)
     print('Doc2Vec dbow Semantic algorithm vector load feature completed...')
     string1=inferFile.encode('utf-8')
     string2=vectFileDelim.encode('utf-8')
     X_train_C_Dbow=libSim.loadVectFileC(string1,string2)  #idx,similarity
     modelCluster=getLoadModel(algorithmCluster,hdpClusterAlgModelFile)

  if((isFastSmartSearchCluster==False) and (isSearchDoc2vecDbow)):
     modelDoc2Vecdbow_=getLoadModel(algorithmSemantic, modelDoc2VecDbow)
     inferFile=inferDir+'/'+str(algorithmSemantic)+'.txt'
     len_dim,len_train=getFeatureLineCnt(inferFile,delim=vectFileDelim)
     print('Doc2Vec dbow Semantic algorithm vector load feature completed...')
     string1=inferFile.encode('utf-8')
     string2=vectFileDelim.encode('utf-8')
     X_train_C_Dbow=libSim.loadVectFileC(string1,string2)  #idx,similarity

  app.run(host=host,port=port)

