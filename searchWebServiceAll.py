
import simplejson as json
from flask import Flask,render_template,request, Response
import sys
import os
import gensim
import re
from gensim.models.nmf import Nmf as GensimNmf
from collections import defaultdict
import numpy as np
import time
import ctypes # for c dynamic library
import math
from numpy.ctypeslib import ndpointer
from ctypes import POINTER
from numba import jit
from operator import itemgetter
import json
from datetime import datetime
import random   
import sys
from datasketch import MinHash
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


def minHash(strText,num_perm=128):

    minhashObj = MinHash()
    arrStr=strText.split()
    for d in arrStr:
      minhashObj.update(d.encode('utf8'))
    return minhashObj

def getMinHash(text,num_perm=128,isSorted=True):
     content=preprocess(text)
     hash=minHash(content,num_perm)
     docVec=hash.hashvalues
     if(isSorted):
        return sorted(docVec)
     else:
       return docVec 

def  getTopSimDocByHashJaccardSimC(trainFeatures,testFeature,len_train,len_dim,topN):

    sim_obj=libSim.jaccard_similarity_top(trainFeatures,testFeature,len_train,len_dim,topN)  #idx,similarity
    sim_ids=[]
    sim_s=[]
    for t in range(topN):
        sim_ids.append(int(sim_obj[t][0])) #id
        sim_s.append(np.round(sim_obj[t][1],4)) #id,sim

    return sim_ids,sim_s

def getLoadModel(algorithm,modelFile):

  if(algorithm=="lsa"):
      model =gensim.models.LsiModel.load(modelFile) 
  elif(algorithm=="rp"):
      model =gensim.models.RpModel.load(modelFile) 
  elif((algorithm=="nmf") or (algorithm=="nmfSparse") ):
      model =GensimNmf.load(modelFile)
  elif((algorithm=="plsa") or (algorithm=="plsaSparse")):
      model = pickle.load(open(modelFile, 'rb'))
  elif((algorithm=="lda") or (algorithm=="ldaSparse") or (algorithm=="ldaTfidf")):
      model =gensim.models.LdaMulticore.load(modelFile)
  elif((algorithm=="hdp") or (algorithm=="hdpSparse")):
      model=gensim.models.HdpModel.load(modelFile)
  elif(algorithm=="doc2vec-dbow" or algorithm=="doc2vec-dbpv"):
      #inference hyper-parameters
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

def  getTopSimDocByCosineSimCSparse(trainIndices,testIndice,trainFeatures,testFeature,trainLens,testLen,len_train,len_dim,topN):

    sim_obj=libSim.cosine_similarity_topSparse(trainIndices,testIndice,trainFeatures,testFeature,trainLens,testLen,len_train,len_dim,topN)  #idx,similarity
    sim_ids=[]
    sim_s=[]
    for t in range(topN):
        sim_ids.append(int(sim_obj[t][0])) #id
        sim_s.append(np.round(sim_obj[t][1],4)) #id,sim

    return sim_ids,sim_s


def c_float_2darr(numpy_arr):
  c_float_p = POINTER(ctypes.c_float)
  arr = (c_float_p * len(numpy_arr) ) ()
  for i in range(len(numpy_arr)):
    arr[i] = (ctypes.c_float * len(numpy_arr[i]))()
    for j in range(len(numpy_arr[i])):
      arr[i][j] = numpy_arr[i][j]
  return arr

def c_int_2darr(numpy_arr):
  c_int_p = POINTER(ctypes.c_int)
  arr = (c_int_p * len(numpy_arr) ) ()
  for i in range(len(numpy_arr)):
    arr[i] = (ctypes.c_int * len(numpy_arr[i]))()
    for j in range(len(numpy_arr[i])):
      arr[i][j] = numpy_arr[i][j]
  return arr


def c_uint16_2darr(numpy_arr,dim,sizes):
  c_uint16_p = POINTER(ctypes.c_uint16)
  arr = (c_uint16_p * len(numpy_arr) ) ()
  for i in range(dim):
    arr[i] = (ctypes.c_uint16 * dim)()
    for j in range(sizes[i]):
      arr[i][j] = numpy_arr[i][j]
  return arr


def cosine_similarity_numpy(arr1,arr2):
    len1=math.sqrt(np.dot(arr1,arr1))
    len2=math.sqrt(np.dot(arr2,arr2))
    cosine = np.dot(arr1,arr2)/len1/len2
    return cosine

def most_frequentElement(list):
    numeral=[[list.count(nb), nb] for nb in list]
    numeral.sort(key=lambda x:x[0], reverse=True)
    return(numeral[0][1])

@jit(nopython=True,fastmath=True)
def cosine_similarity_numba(arr1,arr2):
    len1=math.sqrt(np.dot(arr1,arr1))
    len2=math.sqrt(np.dot(arr2,arr2))
    cosine = np.dot(arr1,arr2)/len1/len2
    return cosine

def getLines(inFile):
   lines=[]
   infp=open(inFile,'r',encoding='utf-8')
   for line in infp:
      text=line.replace("\n","")
      lines.append(text)

   return lines


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


def createDwidFile(corpusFile,dictVocab,dwidFile):
    outfp=open(dwidFile,'w',encoding='utf-8')
    infp=open(corpusFile,'r',encoding='utf-8')
    for line in infp:
       text=line.replace('\n','')
       content=preprocess(text)
       bowArr=getBow(dictVocab,content)
       listToStr = ' '.join([str(elem) for elem in bowArr])
       outfp.write(listToStr+'\n')
    outfp.close()


def getIndiceValue(str,delim=";"):
  indices=[]
  values=[]
  parts=str.split(")"+delim)
  for token in parts:
     elem=(token.replace(")","").replace("(",""))
     partsElem=elem.split(delim)
     if(len(partsElem)>=2):
       indices.append(int(partsElem[0]))  
       values.append(float(partsElem[1]))  

  return indices, values


def loadSparseVectFile(vectFile):

   infp=open(vectFile,'r',encoding="utf-8")
   indices = []
   values=[]
   sizes=[]
   indice=0
   for line in infp:
      text=line.replace("\n","")
      tmpIndices,tmpValues=getIndiceValue(text)
      indices.append(tmpIndices)
      values.append(tmpValues)
      sizes.append(np.uint16(len(tmpIndices)))
      indice=indice+1

   return indices,values,sizes

def getLoadBtmVector(vectFile,delim=' '):
   X_train=[]
   infp=open(vectFile,'r',encoding='utf-8')
   for line in infp:
      text=line.replace('\n','')
      btmVect = [float(x) for x in text.split()]
      X_train.append(np.array(btmVect))
   return X_train


def inferBtm(testFile,dictVocab,outFile,inferType='sum_b'):

    rand=random.randint(0,1000000)
    tagTime = datetime.now().strftime("%Y%m%d-%H%M%S")+'-'+str(rand)
    outfileDwid=inferDir+'dwid-'+tagTime+'.txt'
    createDwidFile(testFile,dictVocab,outfileDwid)
    #print('inferType',inferType)
    if(inferType=='sum_b'):
       os.system('src/bin/btm inf sum_b'+' '+str(btmTopic)+' '+outfileDwid+' '+modelDir+' '+outFile+' > '+'src/log/btmLog.txt')
    elif(inferType=='sum_w'):
       os.system('src/bin/btm inf sum_w'+' '+str(btmTopic)+' '+outfileDwid+' '+modelDir+' '+outFile+' > '+'src/log/btmLog.txt') 
    elif(inferType=='mix'):
       os.system('src/bin/btm inf mix'+' '+str(btmTopic)+' '+outfileDwid+' '+modelDir+' '+outFile+' > '+'src/log/btmLog.txt')

    os.system('rm -f '+ outfileDwid)


def inferBtmForText(testStr,dictVocab,inferType='sum_b'):

   rand=random.randint(0,1000000)
   tagTime = datetime.now().strftime("%Y%m%d-%H%M%S")+'-'+str(rand)
   outfileTmp=inferDir+'/'+'tmp-'+tagTime+'.txt'
   outfp=open(outfileTmp,'w',encoding='utf-8')
   outfp.write(testStr)
   outfp.close()
   tmpInferFile=modelDir+'/'+'k'+str(btmTopic)+'.pz_d'
   inferBtm(outfileTmp,dictVocab,tmpInferFile,inferType)
   btmVectStr=open( tmpInferFile).readline().rstrip()
   os.system('rm -f '+outfileTmp)
   btmVect = [float(x) for x in btmVectStr.split()]
   return btmVect


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

def getPlsaFeatures1Sample(content,plsaModel,thProb=0.01):

    data=countVectorizer.transform([content])
    doc_vectors = plsaModel.transform(data)[0]
    testFeatures=[]
    listTmp=list( map(np.float32, doc_vectors ))
    listNew=[]
    for elemList in listTmp:
      if(elemList<=thProb):
        listNew.append(0)
      else:
        listNew.append(elemList)
    return listNew


def getFeatureFromText(model,tfidf_model,text,algorithm,inferBtmType='sum_b',thProb=0.01,num_perm_=128):
   
       content=preprocess(text)
       if(algorithm=="lsa"):
         doc_bow = model.id2word.doc2bow(content.split())
         corpus_tfidf = tfidf_model[doc_bow]
         docVector= model[corpus_tfidf]
         featureArr =list(zip(*docVector ))
         if(len(featureArr)>1):
           feature=featureArr[1]
         else:
           feature=[]
       elif(algorithm=="nmfSparse"):
         doc_bow = model.id2word.doc2bow(content.split())
         corpus_tfidf = tfidf_model[doc_bow]
         docVector= model[corpus_tfidf]
         feature =docVector

       elif(algorithm=="nmf"):
         doc_bow = model.id2word.doc2bow(content.split())
         corpus_tfidf = tfidf_model[doc_bow]
         docVector= model[corpus_tfidf]
         docVectorTmp=[]
         for i in range(featuresNmf):
           docVectorTmp.append(0)
         for x in  docVector:
           docVectorTmp[x[0]]=round(x[1],precision)
         
         feature=docVectorTmp

       elif(algorithm=="ldaSparse"):
         doc_bow = model.id2word.doc2bow(content.split())
         docVector= model.get_document_topics(doc_bow, minimum_probability=thProb)
         feature =docVector

      
       elif(algorithm=="lda"):
         doc_bow = model.id2word.doc2bow(content.split())
         docVector= model.get_document_topics(doc_bow, minimum_probability=thProb)
         docVectorTmp=[]
         for i in range(featuresLda):
           docVectorTmp.append(0)
         for x in  docVector:
           docVectorTmp[x[0]]=round(x[1],precision)
         
         feature=docVectorTmp

       elif(algorithm=="plsa" or algorithm=="plsaSparse"):
          featureTmp=getPlsaFeatures1Sample(content,model,thProb)
          
          if(algorithm=='plsaSparse'):
            feature=[]
            idx=0
            for elem in featureTmp:
              if elem>=thProb:
                 feature.append((idx,elem))
              idx=idx+1
          else:
              feature=featureTmp
       elif((algorithm=="hdp") and (isHdpDivide==False)):
         doc_bow = model.id2word.doc2bow(content.split())
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

       elif(algorithm=="ldaTfidf"):
         doc_bow = model.id2word.doc2bow(content.split())
         corpus_tfidf = tfidf_model[doc_bow]
         docVector= model.get_document_topics(corpus_tfidf, minimum_probability=thProb)
         docVectorTmp=[]
         for i in range(featuresLda):
           docVectorTmp.append(0)
         for x in  docVector:
           docVectorTmp[x[0]]=round(x[1],precision)
         
         feature=docVectorTmp

       elif(algorithm=='rp'):
          doc_bow = model.id2word.doc2bow(content.split())
          docVector= model.__getitem__(doc_bow)
          featureArr =list(zip(*docVector ))
          if(len(featureArr)>1):
             feature=featureArr[1]
          else:
             feature=[]
          testFeature=np.pad(feature, (0, featuresRp-len(feature)))
          feature=testFeature

       elif(algorithm=='minHash'):
          feature =getMinHash(content,num_perm=num_perm_)

       elif(algorithm=='btm' or algorithm=='btmSparse' ):
            featureTmp=inferBtmForText(content,dictVocab,inferType=inferBtmType)
            if(algorithm=='btmSparse'):
                feature=[]
                idx=0
                for elem in featureTmp:
                   if elem>=thProb:
                      feature.append((idx,elem))
                   idx=idx+1
  
            else:
               feature=featureTmp
                        
       else: #doc2vec
         docVector=[x for x in model.infer_vector(content.split(), alpha=start_alpha,min_alpha=min_alpha_, steps=infer_epoch)]
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


def getFeatureLineCnt(inferFile,delim=';'):
    lineCnt=0
    infp=open(inferFile,'r',encoding='utf-8')
    for line in infp:
        lineCnt=lineCnt+1
    infp.close()
    first_line = next(open(inferFile))
    featureCnt=len(first_line.split(delim))
    return featureCnt,lineCnt
    
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


def getTopNGeneric(text,model,tfidf_model, X_train_C,algorithm,topN):
   feauture=getFeatureFromText(model,tfidf_model,text,algorithm)
   if(algorithm=='minHash'):
       docVectorNp=np.array(feauture,'i')
       sim_ids,sim_s=getTopSimDocByHashJaccardSimC( X_train_C,docVectorNp,len_train,len_dim,topN)
   else:
      docVectorNp=np.array([ list(feauture) ],'f')
      sim_ids,sim_s=getTopSimDocByCosineSimC(X_train_C,trainIds,docVectorNp,len_train,len_dim,topN)

   return  sim_ids,sim_s



def getTopNGenericSparse(text,model,tfidf_model, X_train_C,algorithm,topN,trainIndices,trainLens,inferType='sum_b',th=0.01):
   testFeaturesTuple=getFeatureFromText(model,tfidf_model,text,algorithm,inferType,th)
   sampleIndices= [a_tuple[0] for a_tuple in testFeaturesTuple]
   sampleFeatures=[a_tuple[1] for a_tuple in testFeaturesTuple]
   testLen=len(sampleIndices)
   sampleIndicesNp=np.array(sampleIndices,dtype=np.uint16)
   docVectorNp=np.array(sampleFeatures,'f')
   sim_ids,sim_s=getTopSimDocByCosineSimCSparse(trainIndices,sampleIndicesNp,X_train_C,docVectorNp,trainLens,testLen,len_train,len_dim,topN)
   return  sim_ids,sim_s


def getTopNDoc2Vec(text,model,tfidf_model, X_train_C,topN):
   
   docVector=[x for x in model.infer_vector(text.split(), alpha=start_alpha,min_alpha=min_alpha_, steps=infer_epoch)]
   docVectorNp=np.array([ docVector ],'f')
   sim_ids,sim_s=getTopSimDocByCosineSimC(   X_train_C,trainIds,docVectorNp,len_train,len_dim,topN)
   return  sim_ids,sim_s


def getTopNSmartSearch(text,modelSemantic,modelCluster,tfidf_model,X_train_C,topN,algorithmSemantic='doc2vec-dbow'):

     inferVectorSemantic=getFeatureFromText(modelSemantic,tfidf_model,text,algorithmSemantic,thProb=0.01)
     inferVectorCluster=getFeatureFromText(modelCluster,tfidf_model,text,algorithmCluster,thProb=0.01)
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
    return render_template('indexAll.html')

@app.route("/getSemanticSimilarity",methods=['POST'])
def getSemanticSimilarity():
    try:
        content=request.form['content']
        if "algorithm" in request.form:
            algorithm=request.form['algorithm']
        else:
            algorithm='fastSmartSearch' #default algorithm is fastSmartSearch setted.

        if(isLdaSparse and algorithm=='lda'):
            algorithm='ldaSparse'
        elif(isNmfSparse and algorithm=='nmf'):
            algorithm='nmfSparse'
        elif(isBtmSparse and algorithm=='btm'):
            algorithm='btmSparse'
        elif(isPlsaSparse and algorithm=='plsa'):
            algorithm='plsaSparse'

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
           sim_ids,sim_s=getTopNSmartSearch(textRes,modelDoc2Vecdbow_,modelCluster,tfidf_model,X_train_C_Dbow,topN,algorithmSemantic='doc2vec-dbow')
        
        elif(algorithm=='doc2vec-dbow'):
           sim_ids,sim_s=getTopNDoc2Vec(text,modelDoc2Vecdbow_,tfidf_model,  X_train_C_Dbow,topN)
        elif(algorithm=='doc2vec-dbpv'): 
           sim_ids,sim_s=getTopNDoc2Vec(text,modelDoc2Vecdbpv_,tfidf_model,  X_train_C_Dbpv,topN)
        elif(algorithm=='lsa'):
           sim_ids,sim_s=getTopNGeneric(text,modelLsa_,tfidf_model,  X_train_CLsa,'lsa',topN)
        elif(algorithm=='nmf'):
           sim_ids,sim_s=getTopNGeneric(text,modelNmf_,tfidf_model,  X_train_CNmf,'nmf',topN)
        elif(algorithm=='lda'):
           sim_ids,sim_s=getTopNGeneric(text,modelLda_,tfidf_model,  X_train_CLda,'lda',topN)

        elif(algorithm=='plsa'):
           sim_ids,sim_s=getTopNGeneric(text,modelPlsa_,tfidf_model,  X_train_CPlsa,'plsa',topN)

        elif(algorithm=='rp'):
           sim_ids,sim_s=getTopNGeneric(text,modelRp_,tfidf_model,  X_train_CRp,'rp',topN)
        elif(algorithm=='minHash'):
           sim_ids,sim_s=getTopNGeneric(text,None,tfidf_model,  X_train_CMinHash,'minHash',topN)


        elif(algorithm=='btm'):
           sim_ids,sim_s=getTopNGeneric(text,None,tfidf_model,  X_train_CBtm,'btm',topN)

        elif(algorithm=='ldaSparse'):
           sim_ids,sim_s=getTopNGenericSparse(text,modelLda_,tfidf_model,featuresLdaSparse,'ldaSparse',topN,trainIndicesLda,trainLensLda)
        elif(algorithm=='nmfSparse'):
           sim_ids,sim_s=getTopNGenericSparse(text,modelNmf_,tfidf_model,featuresNmfSparse,'nmfSparse',topN,trainIndicesNmf,trainLensNmf)

        elif(algorithm=='btmSparse'):
           sim_ids,sim_s=getTopNGenericSparse(text,None,tfidf_model,featuresBtmSparse,'btmSparse',topN,trainIndicesBtm,trainLensBtm,th=sparseBtmTh)
        elif(algorithm=='plsaSparse'):
           sim_ids,sim_s=getTopNGenericSparse(text,modelPlsa_,tfidf_model,featuresPlsaSparse,'plsaSparse',topN,trainIndicesPlsa,trainLensPlsa)

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
  featuresLda=config["featuresLda"]
  featuresNmf=config["featuresNmf"]
  featuresRp=config["featuresRp"]
  featuresPlsa=config["featuresPlsa"]
  btmTopic=config["featuresBtm"]
  featuresMinHash=config["featuresMinHash"]
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

  isSearchLda=config['isSearchLda']
  isSearchLsa=config['isSearchLsa']
  isSearchNmf=config['isSearchNmf']
  isSearchDoc2vecDbow=config['isSearchDoc2vecDbow']
  isSearchDoc2vecDbpv=config['isSearchDoc2vecDbpv']
  isSearchRp=config['isSearchRp']
  isSearchBtm=config['isSearchBtm']
  isSearchPlsa=config['isSearchPlsa']
  isSearchMinHash=config['isSearchMinHash']

  isLdaSparse=config['isLdaSparse']
  isNmfSparse=config['isNmfSparse']
  isBtmSparse=config['isBtmSparse']
  isPlsaSparse=config['isPlsaSparse']
  sparseBtmTh=config['sparseBtmTh']
  


  modelLda=modelDir+'/'+'lda'
  modelLsa=modelDir+'/'+'lsa'
  modelNmf=modelDir+'/'+'nmf'
  modelDoc2VecDbow=modelDir+'/'+'doc2vec-dbow'
  modelDoc2VecDbpv=modelDir+'/'+'doc2vec-dbpv'
  hdpClusterAlgModelFile=modelClusterDir+'/'+str(algorithmCluster)
  modelRp=modelDir+'/'+'rp'
  modelPlsa=modelDir+'/'+'plsa'

  tfidfFile=modelDir+'/'+'tfidf'
  vocabFile=modelDir+'/'+'btmVocab.txt'
  precision=6
  
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
     modelDoc2Vecdbow_=getLoadModel('doc2vec-dbow', modelDoc2VecDbow)
     inferFile=inferDir+'/'+'doc2vec-dbow.txt'
     len_dim,len_train=getFeatureLineCnt(inferFile,delim=';')
     print('Doc2Vec dbow load feature completed...')
     string1=inferFile.encode('utf-8')
     string2=';'.encode('utf-8')
     X_train_C_Dbow=libSim.loadVectFileC(string1,string2)  #idx,similarity
     modelCluster=getLoadModel(algorithmCluster,hdpClusterAlgModelFile)

  if((isFastSmartSearchCluster==False) and (isSearchDoc2vecDbow)):
     modelDoc2Vecdbow_=getLoadModel(algorithmSemantic, modelDoc2VecDbow)
     inferFile=inferDir+'/'+str(algorithmSemantic)+'.txt'
     len_dim,len_train=getFeatureLineCnt(inferFile,delim=';')
     print('Doc2Vec dbow load feature completed...')
     string1=inferFile.encode('utf-8')
     string2=';'.encode('utf-8')
     X_train_C_Dbow=libSim.loadVectFileC(string1,string2)  #idx,similarity

  if(isSearchLsa):
     modelLsa_=getLoadModel('lsa',modelLsa)
     inferFile=inferDir+'/'+'lsa.txt'
     len_dim,len_train=getFeatureLineCnt(inferFile,delim=';')
     print('Lsa load feature completed...')
     string1=inferFile.encode('utf-8')
     string2=';'.encode('utf-8')
     X_train_CLsa=libSim.loadVectFileC(string1,string2)  #idx,similarity


  if(isSearchNmf and (isNmfSparse==False)):
     modelNmf_=getLoadModel('nmf',modelNmf)
     featuresNmf_=getLoadVector(inferDir+'/'+'nmf.txt')
     len_train=len(featuresNmf_)
     len_dim=len(featuresNmf_[0])
     print('Nmf load feature completed...')
     X_train_CNmf= c_float_2darr(featuresNmf_)


 
  if(isSearchDoc2vecDbpv):
     modelDoc2Vecdbpv_=getLoadModel('doc2vec-dbpv', modelDoc2VecDbpv)
     inferFile=inferDir+'/'+'doc2vec-dbpv'+'.txt'
     len_dim,len_train=getFeatureLineCnt(inferFile,delim=';')
     print('Doc2Vec dbpv load feature completed...')
     string1=inferFile.encode('utf-8')
     string2=';'.encode('utf-8')
     X_train_C_Dbpv=libSim.loadVectFileC(string1,string2)  #idx,similarity


  if(isSearchLda and (isLdaSparse==False)):
     modelLda_=getLoadModel('lda',modelLda)
     featuresLda_=getLoadVectorFloatArray(inferDir+'/'+'lda.txt')
     len_train=len( featuresLda_)
     len_dim=len(featuresLda_[0])
     print('Lda load feature completed...')
     X_train_CLda= c_float_2darr(featuresLda_)

  if(isSearchPlsa and (isPlsaSparse==False)):
     
     modelPlsa_=getLoadModel('plsa',modelPlsa)
     countVectorizer= pickle.load(open(modelDir+'/'+'countVectorizer', 'rb'))
     featuresPlsa_=getLoadVectorFloatArray(inferDir+'/'+'plsa.txt')
     len_train=len( featuresPlsa_)
     len_dim=len(featuresPlsa_[0])
     print('Plsa load feature completed...')
     X_train_CPlsa= c_float_2darr(featuresPlsa_)


  if(isSearchRp):
     modelRp_=getLoadModel('rp',modelRp)
     inferFile=inferDir+'/'+'rp.txt'
     len_dim,len_train=getFeatureLineCnt(inferFile,delim=';')
     print('Random projection load feature completed...')
     #X_train_CRp= getLoadVector(inferFile,len_dim,len_train)
     string1=inferFile.encode('utf-8')
     string2=';'.encode('utf-8')
     X_train_CRp=libSim.loadVectFileC(string1,string2)  #idx,similarity

  if(isSearchMinHash):
     featuresMinHash_=getLoadVectorFloatArray(inferDir+'/'+'minHash.txt')
     len_train=len( featuresMinHash_)
     len_dim=len(featuresMinHash_[0])

     print('Minhash load feature completed...')
     X_train_CMinHash= c_int_2darr(featuresMinHash_)



  if(isLdaSparse):
     modelLda_=getLoadModel('lda',modelLda)
     indicesLda,valuesLda,sizesLda=loadSparseVectFile(inferDir+'/'+'ldaSparse.txt')
     len_train=len( valuesLda)
     len_dim=featuresLda
     trainIndicesLda= c_uint16_2darr(indicesLda,len(indicesLda),sizesLda)
     featuresLdaSparse= c_float_2darr(valuesLda)
     trainLensLda=np.array(sizesLda,dtype=np.uint16)
     print('Lda Sparse vector load feature completed...')

  if(isPlsaSparse):
     modelPlsa_=getLoadModel('plsa',modelPlsa)
     countVectorizer= pickle.load(open(modelDir+'/'+'countVectorizer', 'rb'))
     indicesPlsa,valuesPlsa,sizesPlsa=loadSparseVectFile(inferDir+'/'+'plsaSparse.txt')
     len_train=len( valuesPlsa)
     len_dim=featuresPlsa
     trainIndicesPlsa= c_uint16_2darr(indicesPlsa,len(indicesPlsa),sizesPlsa)
     featuresPlsaSparse= c_float_2darr(valuesPlsa)
     trainLensPlsa=np.array(sizesPlsa,dtype=np.uint16)
     print('Plsa Sparse vector load feature completed...')


  if(isNmfSparse):
     modelNmf_=getLoadModel('nmf',modelNmf)
     indicesNmf,valuesNmf,sizesNmf=loadSparseVectFile(inferDir+'/'+'nmfSparse.txt')
     len_train=len(valuesNmf)
     len_dim=featuresNmf
     trainIndicesNmf= c_uint16_2darr(indicesNmf,len(indicesNmf),sizesNmf)
     featuresNmfSparse= c_float_2darr(valuesNmf)
     trainLensNmf=np.array(sizesNmf,dtype=np.uint16)
     print('Nmf Sparse vector load feature completed...')


  if(isSearchBtm and (isBtmSparse==False)):
     modelDir=modelDir+'/'
     inferDir=inferDir+'/'
     dictVocab= loadDict(vocabFile)
     featuresBtm_=getLoadBtmVector(inferDir+'/'+'btm.txt',';')
     len_train=len( featuresBtm_)
     len_dim=len(featuresBtm_[0])
     print('Biterm load feature completed...')
     X_train_CBtm= c_float_2darr( featuresBtm_)


  if(isBtmSparse):
     modelDir=modelDir+'/'
     inferDir=inferDir+'/'
     dictVocab= loadDict(vocabFile)
     
     indicesBtm,valuesBtm,sizesBtm=loadSparseVectFile(inferDir+'/'+'btmSparse.txt')
     len_train=len(valuesBtm)
     len_dim=btmTopic
     trainIndicesBtm= c_uint16_2darr(indicesBtm,len(indicesBtm),sizesBtm)
     featuresBtmSparse= c_float_2darr(valuesBtm)
     trainLensBtm=np.array(sizesBtm,dtype=np.uint16)
     print('Btm Sparse vector load feature completed...')


  app.run(host=host,port=port)

