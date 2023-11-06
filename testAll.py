import sys
import os
import gensim
import re
from gensim.models.nmf import Nmf as GensimNmf
from collections import defaultdict
import uuid
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
from collections import defaultdict
from random import randint
from datasketch import MinHash
from datasketch import MinHashLSHForest
import pickle
import multiprocessing
from multiprocessing import Process
from memory_profiler import profile

def preprocess(text):
    text = re.sub(r"[^\w\s]"," ",text)
    textLower = text.lower()
    return ' '.join(textLower.strip().split())

def writeListToFile(outfile,list):
    outfp=open(outfile,'w',encoding='utf-8')
    for elem in list:
        outfp.write(str(elem)+'\n')
    outfp.close()  

def readSpecificLines(inFile,beginLine,endLine):
   infp=open(inFile,"r",encoding="utf-8")
   lineCnt=0
   lines=[]
   for line in  infp:
      if(lineCnt>=beginLine and lineCnt<endLine):
         lines.append(line.replace('\n',''))
      if(lineCnt>=endLine):
         break
      lineCnt=lineCnt+1
      
   return lines


def split_list(lst, n):
    splitted = []
    for i in reversed(range(1, n + 1)):
        split_point = len(lst)//i
        splitted.append(lst[:split_point])
        lst = lst[split_point:]
    return splitted


  
def getLoadModel(algorithm,modelFile):
  if(algorithm=="lsa"):
      model =gensim.models.LsiModel.load(modelFile)
  elif(algorithm=="rp"):
      model =gensim.models.RpModel.load(modelFile) 
  elif((algorithm=="nmf") or (algorithm=="nmfSparse")):
      model =GensimNmf.load(modelFile)
  elif((algorithm=="lda") or (algorithm=="ldaSparse") or (algorithm=="ldaTfidf")):
      model =gensim.models.LdaMulticore.load(modelFile)
  elif(algorithm=="doc2vec-dbow" or algorithm=="doc2vec-dbpv"):
      #inference hyper-parameters
      model =gensim.models.Doc2Vec.load(modelFile)
  return model
 
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

def getTopSimDocByCosineSim(trainFeatures,testFeature,topN):

    sims= np.zeros(len(trainFeatures), float) 
    for i in range(0,len(trainFeatures)):
        cosineSim=cosine_similarity_numba(trainFeatures[i],testFeature)  #cosine similarity
        sims[i]=-1.0*cosineSim # for reverse sort
 
    idx=np.argsort(sims)
    sims_=-1.0*np.sort(sims)
    sim_idx=idx[:topN]
    return sim_idx,sims_

def  getTopSimDocByCosineSimC(trainFeatures,trainIds,testFeature,len_train,len_dim,topN):

    start=time.time()
    if(isSingleSearchMultiThread):
       sim_obj=libSim.cosine_similarity_topMTSingle(trainFeatures,testFeature,len_train,len_dim,topN,singleMultiThreadCnt)  #idx,similarity
    else:
       sim_obj=libSim.cosine_similarity_top(trainFeatures,testFeature,len_train,len_dim,topN)  #idx,similarity
    end=time.time()
    #print(round(((end-start)*1000),2),' ms')
    sim_ids=[]
    sim_s=[]
    for t in range(topN):
        sim_ids.append(int(sim_obj[t][0])) #id
        sim_s.append(np.round(sim_obj[t][1],4)) #id,sim

    return sim_ids,sim_s


def  getTopSimDocByHashJaccardSimC(trainFeatures,testFeature,len_train,len_dim,topN):

    sim_obj=libSim.jaccard_similarity_top(trainFeatures,testFeature,len_train,len_dim,topN)  #idx,similarity
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


def c_uint16_2darr(numpy_arr,dim,sizes):
  c_uint16_p = POINTER(ctypes.c_uint16)
  arr = (c_uint16_p * len(numpy_arr) ) ()
  for i in range(dim):
    arr[i] = (ctypes.c_uint16 * dim)()
    for j in range(sizes[i]):
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


def cosine_similarity_numpy(arr1,arr2):
    len1=math.sqrt(np.dot(arr1,arr1))
    len2=math.sqrt(np.dot(arr2,arr2))
    if(len1==0 or len2==0):
       return 0.0

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
    if(len1==0 or len2==0):
       return 0.0
    cosine = np.dot(arr1,arr2)/len1/len2
    return cosine


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

def inferBtm(testFile,dictVocab,inferFile,modelDir,inferType='sum_b',tagId=''):

   
    rand=random.randint(0,1000000)
    tagTime = datetime.now().strftime("%Y%m%d-%H%M%S")+'-'+str(rand)+'-'+str(tagId)
    outfileDwid=inferDir+'dwid-'+tagTime+'.txt'
    createDwidFile(testFile,dictVocab,outfileDwid)
   
    if(inferType=='sum_b'):
       os.system('src/bin/btm inf sum_b'+' '+str(btmTopic)+' '+outfileDwid+' '+modelDir+' '+inferFile+' > '+'src/log/btmLog.txt')
    elif(inferType=='sum_w'):
       os.system('src/bin/btm inf sum_w'+' '+str(btmTopic)+' '+outfileDwid+' '+modelDir+' '+inferFile+' > '+'src/log/btmLog.txt') 
    elif(inferType=='mix'):
       os.system('src/bin/btm inf mix'+' '+str(btmTopic)+' '+outfileDwid+' '+modelDir+' '+inferFile+' > '+'src/log/btmLog.txt')
    os.system('rm -f '+ outfileDwid)
   

def getLines(inFile):
   lines=[]
   infp=open(inFile,'r',encoding='utf-8')
   for line in infp:
      text=line.replace("\n","")
      lines.append(text)

   return lines


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


def getFeatureFromText(model,tfidf_model,text,algorithm,thProb=0.01,num_perm_=128):
   
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
       else: #doc2vec
         docVector=[x for x in model.infer_vector(content.split(), alpha=start_alpha,min_alpha=min_alpha_, steps=infer_epoch)]
         feature =docVector 
      
       return feature


def getFeatureFromTextList(model,tfidf_model,textList,algorithm,fileIds,dictDocVectors,thProb=0.01):


    featuresArr=[]
    idx=0
    for elem in textList:
        feature=getFeatureFromText(model,tfidf_model,elem,algorithm,thProb=0.01)
        dictDocVectors[fileIds[idx]]=feature
        idx=idx+1

def getSpecificArray(begin,end):
    array=[]
    for i in range(begin,end):
        array.append(i)
    return array

def getFeatureFromTextListMT(model,tfidf_model,corpusFile,algorithm,vec_file,thProb=0.01):
    
  processBatch=2500
  cntThread= multiprocessing.cpu_count()
  manager = multiprocessing.Manager() #multiprocess

  infp=open(corpusFile,"r",encoding="utf-8")
  lineCount=0
  for line in infp:
     lineCount=lineCount+1

  perProcessLine=round(lineCount/cntThread)
  if(perProcessLine <processBatch):
     processBatch=perProcessLine
  batchSize=cntThread*processBatch
  iterCnt=int(  lineCount/batchSize)
  remainCnt=  lineCount-(iterCnt*batchSize)
  outfp = open(vec_file+".tmp", "a",encoding="utf-8")
  for it in range(iterCnt):
    dictDocVectors= manager.dict()
    docs=readSpecificLines(corpusFile,it*batchSize,(it+1)*batchSize)
    docSplit = split_list(docs, cntThread)
    fileIds=getSpecificArray(it*batchSize,(it+1)*batchSize)
    fileIdsSplit=split_list(fileIds, cntThread)

    processes=[]
    for th in range(cntThread):
        processes.append( Process(target= getFeatureFromTextList, args=(model,tfidf_model,docSplit[th],algorithm,fileIdsSplit[th],dictDocVectors,thProb)) )

    for th in range(cntThread):
      processes[th].start()
    for th in range(cntThread):
      processes[th].join()
  
    for elem in dictDocVectors:
      tmpDocVect=dictDocVectors[elem]
      outfp.write(str(elem)+'\t'+";".join(map(str, tmpDocVect))+"\n")
          
  docs=readSpecificLines(corpusFile,iterCnt*batchSize,lineCount)
  fileIds=[]
  for st in range(iterCnt*batchSize,lineCount):
      fileIds.append(st)
  if(len(fileIds)>0):
     processes=[]
     dictDocVectors= manager.dict()
     processes.append( Process(target= getFeatureFromTextList, args=(model,tfidf_model,docs,algorithm,fileIds,dictDocVectors,thProb)) )
     processes[0].start()
     processes[0].join()
     for elemTmp in dictDocVectors:
       tmpDocVect=dictDocVectors[elemTmp]
       outfp.write(str(elemTmp)+'\t'+";".join(map(str, tmpDocVect))+"\n")


  outfp.close()
  del dictDocVectors
  os.system("sort -n "+vec_file+".tmp"+" > "+vec_file+".sort")
  infp=open(vec_file+".sort","r",encoding="utf-8")
  outfp=open(vec_file,"w",encoding="utf-8")
  for line in infp:
       text=line.replace("\n","")
       parts=line.split("\t")
       if(len(parts)>=2):
          outfp.write(parts[1])

  outfp.close()   
  os.system("rm -f "+vec_file+".tmp")
  os.system("rm -f "+vec_file+".sort") #attention control after
  


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


def convertDocVecToSparseVector(infile,outfile,indelim=';',outdelim=';',th=0.01):

  infp=open(infile,'r',encoding='utf-8')
  outfp=open(outfile,'w',encoding='utf-8')
  for line in infp:
    text=line.replace('\n','')
    features=text.split(indelim)
    idx=0
    strFeature=''
    for elem in features:
       if(elem!='' and float(elem)>=th):
          strFeature+='('+str(idx)+outdelim+str(elem)+')'+outdelim
       idx=idx+1
    strFeature=strFeature[:-1]
    outfp.write(strFeature+'\n')

  outfp.close()


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


def getLoadVector(vectFile,lenDim,allSize,delim=';'):

  infp=open(vectFile,'r',encoding='utf-8')
  c_float_p = POINTER(ctypes.c_float)
  cfloatArray = (c_float_p * allSize) ()
  print('cfloatArr created')
  idx=0
  for line in infp:
        text=line.replace('\n','')
        if(text!='' and text!=';'  and text!=' ; '):
          vectParts=text.split(delim)
          cfloatArray[idx] = (ctypes.c_float * len(vectParts))()
          for j in range(len(vectParts)):
              #cfloatArray[idx][j] = ctypes.c_float( float(vectParts[j]))
              cfloatArray[idx][j] =  np.float32(vectParts[j])
          
          if(idx%10000==0):
             print(idx)
        else:
           cfloatArray[idx] = (ctypes.c_float * len(vectParts))()
           for j in range(lenDim):
              cfloatArray[idx][j] =0
        idx=idx+1
  return cfloatArray




def getLoadBtmVector(vectFile,delim=' '):
   X_train=[]
   infp=open(vectFile,'r',encoding='utf-8')
   for line in infp:
      text=line.replace('\n','')
      btmVect = [float(x) for x in text.split()]
      X_train.append(np.array(btmVect))
   return X_train

def  getTopSimDocByCosineSimCSparseFilter(trainIndices,testIndice,trainFeatures,testFeature,trainLens,testLen,len_train,len_dim,topN,filterIndices,filterLen):
    sim_ids=[]
    sim_s=[]
    sim_obj=libSim.cosine_similarity_topSparseFilter(trainIndices,testIndice,trainFeatures,testFeature,trainLens,testLen,len_train,len_dim,topN,filterIndices,filterLen)  #idx,similarity
    iter=topN
    if(filterLen<topN):
       iter=filterLen
    for t in range(iter):
       sim_ids.append(sim_obj[t][0]) #id
       sim_s.append(np.round(sim_obj[t][1],4)) #id,sim
    return sim_ids,sim_s

def getMaxIndice(str,delim=";"):
  indices=[]
  values=[]
  parts=str.split(")"+delim)
  for token in parts:
     elem=(token.replace(")","").replace("(",""))
     partsElem=elem.split(delim)
     indices.append(int(partsElem[0]))  
     values.append(float(partsElem[1])) 
  #print(values) 
  #print(indices)
  maxIndice=values.index(max(values))
  #print(maxIndice)
  #print(indices[maxIndice])
  return indices[maxIndice]


def loadSparseDataToDict(vectFile):
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

   dictResTemp=defaultdict(list)
   dictRes=defaultdict(list)

   infp=open(vectFile,'r',encoding='utf-8')
   idx=0
   for line in infp:
     text=line.replace('\n','')
     maxIndice=getMaxIndice(text,delim=";")
     dictResTemp[maxIndice].append(idx)
     idx=idx+1
   for elem in dictResTemp:
       dictRes[elem]=np.array(dictResTemp[elem],dtype=np.uint32)

   return indices,values,sizes,dictRes

#@profile
def demoGeneric(trainIds,model,tfidf_model,features, X_train_C,testFile,testIds,algorithm,topN,resultFile):

   outfp=open(resultFile,'a',encoding='utf-8')
   len_train=len(features)
   len_dim=len(features[0])
   infp=open(testFile,"r",encoding="utf-8")
   correct=0
   idx=0
  
   for line in infp:
      text=line.replace("\n","")
      feauture=getFeatureFromText(model,tfidf_model,text,algorithm)
      if(algorithm=='minHash'):
          docVectorNp=np.array(feauture,'i')
          sim_ids,sim_s=getTopSimDocByHashJaccardSimC( X_train_C,docVectorNp,len_train,len_dim,topN)
      else:
        docVectorNp=np.array([ list(feauture) ],'f')
        sim_ids,sim_s=getTopSimDocByCosineSimC(   X_train_C,trainIds,docVectorNp,len_train,len_dim,topN)
      topSimId=sim_ids[0] # for top1
      testSimId=testIds[idx].split('\t')[0]
      isCorrect=False
      #if(int(topSimId)==int(testSimId)):
      if(int(testSimId) in sim_ids):
          correct=correct+1
          isCorrect=True
      print("True: ",int(testSimId)," Contain: ",str(isCorrect))
      idx=idx+1
   print('Algorithm: '+str(algorithm)+' Total: ',len(testIds),' Correct: ',correct, ' Acc: ', round(100*correct/len(testIds),2))
   outfp.write('TestFile:'+str(testFile)+' ModelDir: '+modelDir+' Algorithm: '+str(algorithm)+' TopN: '+str(topN)+' Total: '+str(len(testIds))+' Correct: '+str(correct)+ ' Acc: '+str( round(100*correct/len(testIds),2))+'\n')
   outfp.close()


#@profile
def demoGenericSparseVect(model,tfidf_model,trainFeatures,testFile,testIds,algorithm,topN,resultFile,trainIndices,trainLens,len_dim=100):

   outfp=open(resultFile,'a',encoding='utf-8')
   len_train=len(trainFeatures)
   infp=open(testFile,"r",encoding="utf-8")
   correct=0
   idx=0
  
   for line in infp:
      text=line.replace("\n","")
      testFeaturesTuple=getFeatureFromText(model,tfidf_model,text,algorithm)
      testSimId=testIds[idx].split('\t')[0]
      sampleIndices= [a_tuple[0] for a_tuple in testFeaturesTuple]
      sampleFeatures=[a_tuple[1] for a_tuple in testFeaturesTuple]
      testLen=len(sampleIndices)
      sampleIndicesNp=np.array(sampleIndices,dtype=np.uint16)
      docVectorNp=np.array(sampleFeatures,'f')
      sim_ids,sim_s=getTopSimDocByCosineSimCSparse(trainIndices,sampleIndicesNp,trainFeatures,docVectorNp,trainLens,testLen,len_train,len_dim,topN)
      topSimId=sim_ids[0] # for top1
      isCorrect=False
      #if(int(topSimId)==int(testSimId)):
      if(int(testSimId) in sim_ids):
          correct=correct+1
          isCorrect=True
      print("True: ",int(testSimId)," Contain: ",str(isCorrect))
      idx=idx+1
   print('Algorithm: '+str(algorithm)+' Total: ',len(testIds),' Correct: ',correct, ' Acc: ', round(100*correct/len(testIds),2))
   outfp.write('TestFile:'+str(testFile)+' ModelDir: '+modelDir+' Algorithm: '+str(algorithm)+' TopN: '+str(topN)+' Total: '+str(len(testIds))+' Correct: '+str(correct)+ ' Acc: '+str( round(100*correct/len(testIds),2))+'\n')
   outfp.close()

#@profile
def demoBtmSparseVect(trainFeatures,testFile,testIds,topN,resultFile,trainIndices,trainLens,len_dim=100):

   outfp=open(resultFile,'a',encoding='utf-8')
   len_train=len(trainFeatures)

   infp=open(testFile,"r",encoding="utf-8")
   correct=0
   idx=0

   for line in infp:
      text=line.replace("\n","")
      sampleIndices,sampleFeatures=getIndiceValue(text,delim=";")
      testSimId=testIds[idx].split('\t')[0]
      testLen=len(sampleIndices)
      sampleIndicesNp=np.array(sampleIndices,dtype=np.uint16)
      docVectorNp=np.array(sampleFeatures,'f')
      sim_ids,sim_s=getTopSimDocByCosineSimCSparse(trainIndices,sampleIndicesNp,trainFeatures,docVectorNp,trainLens,testLen,len_train,len_dim,topN)

      topSimId=sim_ids[0] # for top1
      isCorrect=False
      #if(int(topSimId)==int(testSimId)):
      if(int(testSimId) in sim_ids):
          correct=correct+1
          isCorrect=True
      print("True: ",int(testSimId)," Contain: ",str(isCorrect))
      idx=idx+1
   print('Algorithm: '+str(algorithm)+' Total: ',len(testIds),' Correct: ',correct, ' Acc: ', round(100*correct/len(testIds),2))
   outfp.write('TestFile:'+str(testFile)+' ModelDir: '+modelDir+' Algorithm: '+str(algorithm)+' TopN: '+str(topN)+' Total: '+str(len(testIds))+' Correct: '+str(correct)+ ' Acc: '+str( round(100*correct/len(testIds),2))+'\n')
   outfp.close()

#@profile
def demoBtm(features,inferFile,testIds,topN,resultFile,inferType_='sum_b'):
   outfp=open(resultFile,'a',encoding='utf-8')
   infp=open(inferFile,'r',encoding='utf-8')
   idx=0
   correct=0
   
   for line in infp:
      text=line.replace('\n','')
      btmVect = [float(x) for x in text.split()]
      isCorrect=False
      if((len(btmVect)<=0) or np.isnan(btmVect[0])):  # len(ar_nan)
         print('content irrelevant or wrong')
      else:
         sim_idx,sims_=getTopSimDocByCosineSim(features,np.array(btmVect),topN)
         if(idx in sim_idx):
            correct=correct+1
            isCorrect=True
         print("True: ",idx," Contain: ",str(isCorrect))
      idx=idx+1
   print('Algorithm: '+str(algorithm)+' Total: ',len(testIds),' Correct: ',correct, ' Acc: ', round(100*correct/len(testIds),2))
   outfp.write('TestFile:'+str(testFile)+' ModelDir: '+modelDir+' Algorithm: '+str(algorithm)+' TopN: '+str(topN)+' Total: '+str(len(testIds))+' Correct: '+str(correct)+ ' Acc: '+str( round(100*correct/len(testIds),2))+'\n')
   outfp.close()

#@profile
def demoGenericForMT(trainIds,features,inferFileArg,testIds,topN,resultFile,algorithm,len_dim=100,delim_=';'):
   start=time.time()
   outfp=open(resultFile,'a',encoding='utf-8')
   idx=0
   correct=0
   inferVectors=getLoadVectorFloatArray(inferFileArg,delim_)
   len_train=len(trainIds)
   sim_idx=[]
   for docVec in  inferVectors:
      isCorrect=False
      if(algorithm=='minHash'):
         docVectorNp=np.array(docVec,'i')
         sim_idx,sims_=getTopSimDocByHashJaccardSimC( features,docVectorNp,len_train,len_dim,topN)
      else:
         if(len(docVec)!=0):
           sim_idx,sims_=getTopSimDocByCosineSimC(features,trainIds,np.array(docVec),len_train,len_dim,topN)
      if(idx in sim_idx):
         correct=correct+1
         isCorrect=True
         print("True: ",idx," Contain: ",str(isCorrect))
      idx=idx+1
   print('Algorithm: '+str(algorithm)+' Total: ',len(testIds),' Correct: ',correct, ' Acc: ', round(100*correct/len(testIds),2))
   outfp.write('TestFile:'+str(testFile)+' ModelDir: '+modelDir+' Algorithm: '+str(algorithm)+' TopN: '+str(topN)+' Total: '+str(len(testIds))+' Correct: '+str(correct)+ ' Acc: '+str( round(100*correct/len(testIds),2))+'\n')
   outfp.close()
   end=time.time()
   print('Algorithm: ',str(algorithm),' Total: ',len(testIds),' Time: ',(end-start),' sn')



#@profile
def demoGenericSparseVectForMT(trainFeatures,testFile,testIds,topN,resultFile,trainIndices,trainLens,len_dim=100):

   start=time.time()
   outfp=open(resultFile,'a',encoding='utf-8')
   len_train=len(trainFeatures)

   infp=open(testFile,"r",encoding="utf-8")
   correct=0
   idx=0

   for line in infp:
      text=line.replace("\n","")
      sampleIndices,sampleFeatures=getIndiceValue(text,delim=";")
      testSimId=testIds[idx].split('\t')[0]
      testLen=len(sampleIndices)
      sampleIndicesNp=np.array(sampleIndices,dtype=np.uint16)
      docVectorNp=np.array(sampleFeatures,'f')
      sim_ids,sim_s=getTopSimDocByCosineSimCSparse(trainIndices,sampleIndicesNp,trainFeatures,docVectorNp,trainLens,testLen,len_train,len_dim,topN)

      topSimId=sim_ids[0] # for top1
      isCorrect=False
      #if(int(topSimId)==int(testSimId)):
      if(int(testSimId) in sim_ids):
          correct=correct+1
          isCorrect=True
      print("True: ",int(testSimId)," Contain: ",str(isCorrect))
      idx=idx+1
   print('Algorithm: '+str(algorithm)+' Total: ',len(testIds),' Correct: ',correct, ' Acc: ', round(100*correct/len(testIds),2))
   outfp.write('TestFile:'+str(testFile)+' ModelDir: '+modelDir+' Algorithm: '+str(algorithm)+' TopN: '+str(topN)+' Total: '+str(len(testIds))+' Correct: '+str(correct)+ ' Acc: '+str( round(100*correct/len(testIds),2))+'\n')
   outfp.close()
   end=time.time()
   print('Algorithm: ',str(algorithm),' Total: ',len(testIds),' Time: ',(end-start),' sn')
   

#@profile
def demoBtmC(trainIds,features,inferFile,testIds,topN,resultFile,len_dim=100,inferType_='sum_b'):
   outfp=open(resultFile,'a',encoding='utf-8')
   infp=open(inferFile,'r',encoding='utf-8')
   idx=0
   correct=0
   len_train=len(features)
   for line in infp:
      text=line.replace('\n','')
      btmVect = [float(x) for x in text.split()]
      isCorrect=False
      if((len(btmVect)<=0) or np.isnan(btmVect[0])):  # len(ar_nan)
         print('content irrelevant or wrong')
         
      else:
         sim_idx,sim_s=getTopSimDocByCosineSimC(features,trainIds,np.array(btmVect,'f'),len_train,len_dim,topN)
         if(idx in sim_idx):
            correct=correct+1
            isCorrect=True
         print("True: ",idx," Contain: ",str(isCorrect))
      idx=idx+1
   print('Algorithm: '+str(algorithm)+' Total: ',len(testIds),' Correct: ',correct, ' Acc: ', round(100*correct/len(testIds),2))
   outfp.write('TestFile:'+str(testFile)+' ModelDir: '+modelDir+' Algorithm: '+str(algorithm)+' TopN: '+str(topN)+' Total: '+str(len(testIds))+' Correct: '+str(correct)+ ' Acc: '+str( round(100*correct/len(testIds),2))+'\n')
   outfp.close()

#@profile
def demoDoc2Vec(trainIds,model,tfidf_model,features, X_train_C,testFile,testIds,algorithm,topN,resultFile):
  
   outfp=open(resultFile,'a',encoding='utf-8')
   len_train=len(features)
   len_dim=len(features[0])
   infp=open(testFile,"r",encoding="utf-8")
   correct=0
   idx=0
  
   for line in infp:
      text=line.replace("\n","")
      #print(text)
      content=preprocess(text)
      docVector=[x for x in model.infer_vector(content.split(), alpha=start_alpha,min_alpha=min_alpha_, steps=infer_epoch)]
      docVectorNp=np.array([ docVector ],'f')
      sim_ids,sim_s=getTopSimDocByCosineSimC(   X_train_C,trainIds,docVectorNp,len_train,len_dim,topN)
      topSimId=sim_ids[0] # for top1
      testSimId=testIds[idx].split('\t')[0]
      isCorrect=False
      #if(int(topSimId)==int(testSimId)):
      if(int(testSimId) in sim_ids):
          correct=correct+1
          isCorrect=True
      print("True: ",int(testSimId)," Contain: ",str(isCorrect))
      idx=idx+1
   print('Algorithm: '+str(algorithm)+' Total: ',len(testIds),' Correct: ',correct, ' Acc: ', round(100*correct/len(testIds),2))
   outfp.write('TestFile:'+str(testFile)+' ModelDir: '+modelDir+' Algorithm: '+str(algorithm)+' TopN: '+str(topN)+' Total: '+str(len(testIds))+' Correct: '+str(correct)+ ' Acc: '+str( round(100*correct/len(testIds),2))+'\n')
   outfp.close()
   


def getLshForestTopN(lshForest,text,topN):
   textPre=preprocess(text)
   minHashObj=minHash(textPre)
   start=time.time()
   result = lshForest.query(minHashObj, topN)
   end=time.time()
   print(round(end-start,4)*1000,' ms')
   return result

#@profile
def getDemoLshForest(lshForest,testFile,fileIdsTest,topN):
   start=time.time()
   algorithm='minHashFastApproximate'
   outfp=open(resultFile,'a',encoding='utf-8')
   infp=open(testFile,'r',encoding='utf-8')
   idx=0
   correct=0
  
   for line in infp:
      text=line.replace('\n','')
      idTest=fileIdsTest[idx].split('\t')[0]
      resultIds=getLshForestTopN(lshForest,text,topN)
      isContain=False
      if(str(idx) in resultIds):
        correct=correct+1
        isContain=True
      print('True: ',str(idx),' Contain: ',isContain)
      idx=idx+1
 
   
   print('Algorithm: ',str(algorithm),' Total: ',len(fileIdsTest),' Correct: ',correct, ' Acc: ', round(100*correct/len(fileIdsTest),2)) 
   outfp.write('TestFile: '+str(testFile)+' Algorithm: '+str(algorithm)+' TopN: '+str(topN)+' Total: '+str(len(fileIdsTest))+' Correct: '+str(correct)+ ' Acc: '+str( round(100*correct/len(fileIdsTest),2))+'\n') 
   outfp.close()
   end=time.time()
   print('Algorithm: ',str(algorithm),' Total: ',len(testIds),' Time: ',(end-start),' sn')


#@profile
def demoTestSparseVectApproximateFast(vectFile,testInferFile,len_dim,testIds,testFile,algorithm,resultFile,resultFileFilter,topN):

   start=time.time()
   outfpResult=open(resultFile,'a',encoding='utf-8')
   indices,values,sizes,dictRes=loadSparseDataToDict(vectFile)
   testIndices,testValues,testSizes,testDictRes=loadSparseDataToDict(testInferFile)
 
   X_train_vec=values
   trainIndices= c_uint16_2darr(indices,len_dim,sizes)
   trainFeatures= c_float_2darr(X_train_vec)
   trainLens=np.array([sizes],dtype=np.uint16)
   len_train=len(X_train_vec)

   X_test_vec=testValues
   test_size=len(testIds)
   testIdxArr=[]
   for idx in range(test_size):
       testIdxArr.append(idx)

   outfp=open(resultFileFilter,'w',encoding='utf-8')
   correct=0
   #begin=time.time()
   for testIdx in testIdxArr:

     testIndice=np.array([testIndices[testIdx]],dtype=np.uint16)
     testFeature=np.array([X_test_vec[testIdx]],'f')
     maxIndice=np.argmax(testFeature)
     maxFeatureIndice=testIndice[0][maxIndice] #2d
     filterIndices=testDictRes[maxFeatureIndice]
     filterLen=len(  filterIndices)
     testLen=testSizes[testIdx]
     
     simIds,sims=getTopSimDocByCosineSimCSparseFilter(trainIndices,testIndice,trainFeatures,testFeature,trainLens,testLen,len_train,len_dim,topN,filterIndices,filterLen)

     newSimIds=[]
     for simElem in simIds:
         newSimIds.append(filterIndices[int(simElem)])
     if (testIdx in newSimIds):
          correct=correct+1
     #print(newSimIds)
     #print(sims)
     outfp.write(str(newSimIds)+'\t'+str(sims)+'\n')


   #outfp.write('Approximate and Fast Execute Time: '+str(round(end-begin,1))+' sn'+'\n')
   outfp.close()

   #print('Execute Time: ',round(end-begin,1),' sn')
   print('InferFileLen: ',str(len(X_train_vec)),' TestFile: ',testFile,' Algorithm: ',str(algorithm),' Total: ',len(testIds),' Correct: ',correct, ' Acc: ', round(100*correct/len(testIds),2))
   outfpResult.write('InferFileLen: '+str(len(X_train_vec))+' TestFile: '+str(testFile)+' Algorithm: '+str(algorithm)+' TopN: '+str(topN)+' Total: '+str(len(testIds))+' Correct: '+str(correct)+ ' Acc: '+str( round(100*correct/len(testIds),2))+'\n')
   outfpResult.close()
   outfp.close()
   end=time.time()
   print('Algorithm: ',str(algorithm),' Total: ',len(testIds),' Time: ',(end-start),' sn')


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

if __name__=="__main__":
    
  modelDir=sys.argv[1]
  inferDir=sys.argv[2]
  configFile=sys.argv[3]
  fileIdsFile=sys.argv[4]
  testFile=sys.argv[5]
  fileIdsTestFile=sys.argv[6]
  resultFile=sys.argv[7]
  topN=int(sys.argv[8])
  

  algorithms=['lsa','nmf','lda','doc2vec-dbow','doc2vec-dbpv','rp','ldaSparse','ldaApproximateFast','nmfSparse','nmfApproximateFast','minHash','minHashLshForest','btm','btmSparse','plsa','plsaSparse']

  #trainIds=getLines(fileIdsFile)
  #testIds=getLines(fileIdsTestFile)
  trainIds=getOnlyIdFromFile(fileIdsFile)
  testIds=getOnlyIdFromFile(fileIdsTestFile)

  with open(configFile, 'r') as f:
    config = json.load(f)

  # configurations
  featuresLda=config["featuresLda"]
  featuresNmf=config["featuresNmf"]
  featuresRp=config["featuresRp"]
  featuresPlsa=config["featuresPlsa"]

  featuresMinHash=config["featuresMinHash"]
  featuresMinHashLshForest=config["featuresMinHashLshForest"]

  btmTopic=config["featuresBtm"]
  infer_epoch=config['infer_epoch'] 
  start_alpha=config['start_alpha'] # learning rate
  min_alpha_=config['min_alpha']
  isMultiProcess=config['isMultiProcess']
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
  isSearchMinHashLshForest=config['isSearchMinHashLshForest']
  isSearchLdaApproximateFast=config['isSearchLdaApproximateFast']
  isSearchNmfApproximateFast=config['isSearchNmfApproximateFast']


  isSearchLdaSparse=config['isSearchLdaSparse']
  isSearchNmfSparse=config['isSearchNmfSparse']
  isSearchBtmSparse=config['isSearchBtmSparse']
  isSearchPlsaSparse=config['isSearchPlsaSparse']
  sparseBtmTh=config['sparseBtmTh']

  modelLda=modelDir+'/'+'lda'
  modelLsa=modelDir+'/'+'lsa'
  modelNmf=modelDir+'/'+'nmf'
  modelDoc2VecDbow=modelDir+'/'+'doc2vec-dbow'
  modelDoc2VecDbpv=modelDir+'/'+'doc2vec-dbpv'
  modelRp=modelDir+'/'+'rp'
  modelPlsa=modelDir+'/'+'plsa'
  vocabFile=modelDir+'/'+'btmVocab.txt'

  tfidfFile=modelDir+'/'+'tfidf'
  inferType_='sum_b'
  precision=6
  
  
  resultDir=os.path.dirname(resultFile)
  if(os.path.exists(resultDir)==False):
     os.system('mkdir '+resultDir)

  libSim= ctypes.cdll.LoadLibrary('src/bin/sim.so')


  charptr= ctypes.c_char_p
  libSim.loadVectFileC.restype =  POINTER(POINTER(ctypes.c_float))
  libSim.loadVectFileC.argtypes = [ charptr,charptr]


  libSim.cosine_similarity_top.restype =  POINTER(POINTER(ctypes.c_float))
  libSim.cosine_similarity_top.argtypes = [ POINTER(POINTER(ctypes.c_float))  , ndpointer(ctypes.c_float),ctypes.c_int,ctypes.c_int,ctypes.c_int]
  
  libSim.cosine_similarity_topMTSingle.restype =  POINTER(POINTER(ctypes.c_float))
  libSim.cosine_similarity_topMTSingle.argtypes = [ POINTER(POINTER(ctypes.c_float))  , ndpointer(ctypes.c_float),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]

 
  libSim.cosine_similarity_topSparse.restype =  POINTER(POINTER(ctypes.c_float))
  #( uint16_t ** trainIndices,uint16_t * testIndice,float** trainValues,float* testValue,uint16_t * trainLens,uint16_t testLen,int dataSize,int dim,int topN) 
  libSim.cosine_similarity_topSparse.argtypes = [  POINTER(POINTER(ctypes.c_uint16)),ndpointer(ctypes.c_uint16), POINTER(POINTER(ctypes.c_float))  , ndpointer(ctypes.c_float), ndpointer(ctypes.c_uint16), ctypes.c_uint16, ctypes.c_int,ctypes.c_int,ctypes.c_int]


  libSim.cosine_similarity_topSparseFilter.restype =  POINTER(POINTER(ctypes.c_float))
   #( uint16_t ** trainIndices,uint16_t * testIndice,float** trainValues,float* testValue,uint16_t * trainLens,uint16_t testLen,int dataSize,int dim,int topN) 
  libSim.cosine_similarity_topSparseFilter.argtypes = [  POINTER(POINTER(ctypes.c_uint16)),ndpointer(ctypes.c_uint16), POINTER(POINTER(ctypes.c_float))  , ndpointer(ctypes.c_float), ndpointer(ctypes.c_uint16), ctypes.c_uint16, ctypes.c_int,ctypes.c_int,ctypes.c_int,ndpointer(ctypes.c_uint32),ctypes.c_int]
 


  libSim.jaccard_similarity_top.restype =  POINTER(POINTER(ctypes.c_float))
  libSim.jaccard_similarity_top.argtypes = [ POINTER(POINTER(ctypes.c_int))  , ndpointer(ctypes.c_int),ctypes.c_int,ctypes.c_int,ctypes.c_int]

  tfidf_model=None
  for algorithm in algorithms:

   if(algorithm=='ldaTfidf'):
     tfidf_model=gensim.models.TfidfModel.load(tfidfFile)

   if(algorithm=='lsa' and (isSearchLsa)):
     tfidf_model=gensim.models.TfidfModel.load(tfidfFile)
     inferFile=inferDir+'/'+'lsa.txt'
     len_dim,lenInfer=getFeatureLineCnt(inferFile,delim=';')
     string1=inferFile.encode('utf-8')
     string2=';'.encode('utf-8')
     featuresLsa_=libSim.loadVectFileC(string1,string2)  #idx,similarity

     print('Lsa load feature completed...')
     modelLsa_=getLoadModel('lsa', modelLsa)
     if(isMultiProcess):
       inferFileTmp=inferDir+'/'+'lsa.txt.tmp'
       getFeatureFromTextListMT(modelLsa_,tfidf_model,testFile,'lsa',inferFileTmp,thProb=0.01)
       demoGenericForMT( trainIds, featuresLsa_,inferFileTmp,testIds,topN,resultFile,'lsa',len_dim)
       os.system('rm -f '+inferFileTmp)
     else:
       demoGeneric(trainIds,modelLsa_,tfidf_model,featuresLsa_, X_train_CLsa,testFile,testIds,'lsa',topN,resultFile)
   elif(algorithm=='nmf'  and (isSearchNmf)):
     tfidf_model=gensim.models.TfidfModel.load(tfidfFile)
     modelNmf_=getLoadModel('nmf',modelNmf)
     featuresNmf_=getLoadVectorFloatArray(inferDir+'/'+'nmf.txt')
     print('Nmf load feature completed...')
     X_train_CNmf= c_float_2darr(featuresNmf_)
     if(isMultiProcess):
       inferFileTmp=inferDir+'/'+'nmf.txt.tmp'
       getFeatureFromTextListMT(modelNmf_,tfidf_model,testFile,'nmf',inferFileTmp,thProb=0.01)
       len_dim=featuresNmf
       demoGenericForMT( trainIds,X_train_CNmf,inferFileTmp,testIds,topN,resultFile,'nmf',len_dim)
       os.system('rm -f '+inferFileTmp)
     else:
       demoGeneric(trainIds,modelNmf_,tfidf_model,featuresNmf_, X_train_CNmf,testFile,testIds,'nmf',topN,resultFile)

   elif(algorithm=='doc2vec-dbow' and (isSearchDoc2vecDbow)):
     inferFile=inferDir+'/'+'doc2vec-dbow.txt'
     len_dim,lenInfer=getFeatureLineCnt(inferFile,delim=';')
     string1=inferFile.encode('utf-8')
     string2=';'.encode('utf-8')
     featuresDoc2VecDbow_=libSim.loadVectFileC(string1,string2)  #idx,similarity

     print('Doc2Vec dbow load feature completed...')
     modelDoc2Vecdbow_=getLoadModel('doc2vec-dbow', modelDoc2VecDbow)
     if(isMultiProcess):
       inferFileTmp=inferDir+'/'+'doc2vec-dbow.txt.tmp'
       getFeatureFromTextListMT(modelDoc2Vecdbow_,tfidf_model,testFile,'doc2vec-dbow',inferFileTmp,thProb=0.01)
       demoGenericForMT( trainIds, featuresDoc2VecDbow_,inferFileTmp,testIds,topN,resultFile,'doc2vec-dbow',len_dim)
       os.system('rm -f '+inferFileTmp)
     else:
       demoDoc2Vec(  trainIds,modelDoc2Vecdbow_,tfidf_model, featuresDoc2VecDbow_, X_train_C_Dbow,testFile,testIds,'doc2vec-dbow',topN,resultFile)

   elif(algorithm=='doc2vec-dbpv'  and (isSearchDoc2vecDbpv)):
     inferFile=inferDir+'/'+'doc2vec-dbpv.txt'
     len_dim,lenInfer=getFeatureLineCnt(inferFile,delim=';')
     string1=inferFile.encode('utf-8')
     string2=';'.encode('utf-8')
     featuresDoc2VecDbpv_=libSim.loadVectFileC(string1,string2)  #idx,similarity

     print('Doc2Vec dbpv load feature completed...')
     modelDoc2Vecdbpv_=getLoadModel('doc2vec-dbpv', modelDoc2VecDbpv)
     if(isMultiProcess):
       inferFileTmp=inferDir+'/'+'doc2vec-dbpv.txt.tmp'
       getFeatureFromTextListMT(modelDoc2Vecdbpv_,tfidf_model,testFile,'doc2vec-dbpv',inferFileTmp,thProb=0.01)
       demoGenericForMT( trainIds, featuresDoc2VecDbpv_,inferFileTmp,testIds,topN,resultFile,'doc2vec-dbpv',len_dim)
       os.system('rm -f '+inferFileTmp)
     else:
       demoDoc2Vec(  trainIds,modelDoc2Vecdbpv_,tfidf_model, featuresDoc2VecDbpv_, X_train_C_Dbpv,testFile,testIds,'doc2vec-dbpv',topN,resultFile)

   elif(algorithm=='lda' and (isSearchLda)):
     modelLda_=getLoadModel('lda',modelLda)
     featuresLda_=getLoadVectorFloatArray(inferDir+'/'+'lda.txt')
     X_train_CLda= c_float_2darr(featuresLda_)
     if(isMultiProcess):
       inferFileTmp=inferDir+'/'+'lda.txt.tmp'
       getFeatureFromTextListMT(modelLda_,tfidf_model,testFile,'lda',inferFileTmp,thProb=0.01)
       len_dim=featuresLda
       demoGenericForMT( trainIds,  X_train_CLda,inferFileTmp,testIds,topN,resultFile,'lda',len_dim)
       os.system('rm -f '+inferFileTmp)
     else:
       demoGeneric(trainIds,modelLda_,tfidf_model,featuresLda_, X_train_CLda,testFile,testIds,'lda',topN,resultFile)

   elif(algorithm=='plsa' and (isSearchPlsa)):
     featuresPlsa_=getLoadVectorFloatArray(inferDir+'/'+'plsa.txt')
     X_train_CPlsa= c_float_2darr(featuresPlsa_)
     inferFileTmp=inferDir+'/'+'plsa.txt.tmp'
     os.system('python3 src/infer/createPlsaVec.py '+testFile+' '+modelPlsa+' '+inferFileTmp)
     len_dim=featuresPlsa
     demoGenericForMT( trainIds,  X_train_CPlsa,inferFileTmp,testIds,topN,resultFile,'plsa',len_dim)
     os.system('rm -f '+inferFileTmp)

   elif(algorithm=='rp' and (isSearchRp)):
     inferFile=inferDir+'/'+'rp.txt'
     len_dim,lenInfer=getFeatureLineCnt(inferFile,delim=';')
     string1=inferFile.encode('utf-8')
     string2=';'.encode('utf-8')
     featuresRp_=libSim.loadVectFileC(string1,string2)  #idx,similarity

     print('Random projection load feature completed...')
     modelRp_=getLoadModel('rp', modelRp)
     if(isMultiProcess):
       inferFileTmp=inferDir+'/'+'rp.txt.tmp'
       getFeatureFromTextListMT(modelRp_,tfidf_model,testFile,'rp',inferFileTmp,thProb=0.01)
       demoGenericForMT( trainIds, featuresRp_,inferFileTmp,testIds,topN,resultFile,'rp',len_dim)
       os.system('rm -f '+inferFileTmp)
     else:
       demoGeneric(trainIds,modelRp_,tfidf_model,featuresRp_, X_train_CRp,testFile,testIds,'rp',topN,resultFile)


   elif(algorithm=='minHash' and (isSearchMinHash)):
     featuresMinHash_=getLoadVectorFloatArray(inferDir+'/'+'minHash.txt')
     trainFeaturesMinHash= c_int_2darr(featuresMinHash_)
     if(isMultiProcess):
       inferFileTmp=inferDir+'/'+'minHash.txt.tmp'
       getFeatureFromTextListMT(None,tfidf_model,testFile,'minHash',inferFileTmp,thProb=0.01)
       len_dim=featuresMinHash
       demoGenericForMT( trainIds,trainFeaturesMinHash,inferFileTmp,testIds,topN,resultFile,'minHash',len_dim)
       os.system('rm -f '+inferFileTmp)
     else:
       demoGeneric(trainIds,'',tfidf_model,featuresMinHash_,  trainFeaturesMinHash,testFile,testIds,'minHash',topN,resultFile)

   
   elif(algorithm=='minHashLshForest' and (isSearchMinHashLshForest)):
      modelLshFile=inferDir+'/'+'minHashLshForest'
      lshForest = pickle.load( open( modelLshFile, "rb" ) )
      getDemoLshForest(lshForest,testFile,testIds,topN)

   elif(algorithm=='btm' and (isSearchBtm)):
     modelDir=modelDir+'/'
     dictVocab= loadDict(vocabFile)
     featuresBtm_=getLoadVectorFloatArray(inferDir+'/'+'btm.txt',';')
     inferFileTmp=inferDir+'/'+'btm.txt.infer'
    
     if(isMultiProcess):
       os.system('python3 src/infer/createBtmVec.py '+testFile+' '+str(btmTopic)+' '+inferDir+' '+modelDir+' '+inferFileTmp+' '+str(1))
     else:
       inferBtm(testFile,dictVocab,inferFileTmp,modelDir,inferType=inferType_) #test file should be preprocessed

     #demoBtm(featuresBtm_,inferFileTmp,testIds,topN,resultFile)
     X_train_CBtm= c_float_2darr( featuresBtm_)
     
     demoBtmC( trainIds, X_train_CBtm,inferFileTmp,testIds,topN,resultFile,len_dim=btmTopic)
     os.system('rm -f '+inferFileTmp)

   elif(algorithm=='ldaSparse' and (isSearchLdaSparse)):
     modelLda_=getLoadModel('lda',modelLda)
     indices,values,sizes=loadSparseVectFile(inferDir+'/'+'ldaSparse.txt')
     trainIndices= c_uint16_2darr(indices,len(indices),sizes)
     trainFeatures= c_float_2darr(values)
     trainLens=np.array(sizes,dtype=np.uint16)
    
     if(isMultiProcess):
       inferFileTmp=inferDir+'/'+'lda.txt.tmp'
       inferFileTmpSparse=inferDir+'/'+'ldaSparse.txt.tmp'
       getFeatureFromTextListMT(modelLda_,tfidf_model,testFile,'lda',inferFileTmp)
       convertDocVecToSparseVector(inferFileTmp,inferFileTmpSparse,indelim=';',outdelim=';',th=0.01)
       demoGenericSparseVectForMT(trainFeatures,inferFileTmpSparse,testIds,topN,resultFile,trainIndices,trainLens,len_dim=featuresLda)
       os.system('rm -f '+inferFileTmp)
       os.system('rm -f '+inferFileTmpSparse)
     else:
       demoGenericSparseVect(modelLda_,tfidf_model,trainFeatures,testFile,testIds,'ldaSparse',topN,resultFile,trainIndices,trainLens,len_dim=featuresLda)

   elif(algorithm=='plsaSparse' and (isSearchPlsaSparse)):
     indices,values,sizes=loadSparseVectFile(inferDir+'/'+'plsaSparse.txt')
     trainIndices= c_uint16_2darr(indices,len(indices),sizes)
     trainFeatures= c_float_2darr(values)
     trainLens=np.array(sizes,dtype=np.uint16)
     inferFileTmp=inferDir+'/'+'plsa.txt.tmp'
     inferFileTmpSparse=inferDir+'/'+'plsaSparse.txt.tmp'
     os.system('python3 src/infer/createPlsaVec.py '+testFile+' '+modelPlsa+' '+inferFileTmp)
     len_dim=featuresPlsa
     convertDocVecToSparseVector(inferFileTmp,inferFileTmpSparse,indelim=';',outdelim=';',th=0.01)
     demoGenericSparseVectForMT(trainFeatures,inferFileTmpSparse,testIds,topN,resultFile,trainIndices,trainLens,len_dim=featuresPlsa)
     os.system('rm -f '+inferFileTmp)
     os.system('rm -f '+inferFileTmpSparse)

   elif(algorithm=='ldaApproximateFast' and (isSearchLdaApproximateFast)):
     resultFileFilter=os.path.dirname(resultFile)+'/'+'ldaApproximateFast.txt'
     modelLda_=getLoadModel('lda',modelLda)
     vectFile=inferDir+'/'+'ldaSparse.txt'    
     inferFileTmp=inferDir+'/'+'lda.txt.tmp'
     inferFileTmpSparse=inferDir+'/'+'ldaSparse.txt.tmp'
     getFeatureFromTextListMT(modelLda_,tfidf_model,testFile,'lda',inferFileTmp)
     convertDocVecToSparseVector(inferFileTmp,inferFileTmpSparse,indelim=';',outdelim=';',th=0.01)
     os.system('rm -f '+inferFileTmp)
     demoTestSparseVectApproximateFast(vectFile,inferFileTmpSparse,featuresLda,testIds,testFile,'ldaApproximateFast',resultFile,resultFileFilter,topN)
     os.system('rm -f '+inferFileTmpSparse)

   elif(algorithm=='nmfApproximateFast' and (isSearchNmfApproximateFast)):
     tfidf_model=gensim.models.TfidfModel.load(tfidfFile)
     resultFileFilter=os.path.dirname(resultFile)+'/'+'nmfApproximateFast.txt'
     modelNmf_=getLoadModel('nmf',modelNmf)
     vectFile=inferDir+'/'+'nmfSparse.txt'    
     inferFileTmp=inferDir+'/'+'nmf.txt.tmp'
     inferFileTmpSparse=inferDir+'/'+'nmfSparse.txt.tmp'
     getFeatureFromTextListMT(modelNmf_,tfidf_model,testFile,'nmf',inferFileTmp)
     convertDocVecToSparseVector(inferFileTmp,inferFileTmpSparse,indelim=';',outdelim=';',th=0.01)
     os.system('rm -f '+inferFileTmp)
     demoTestSparseVectApproximateFast(vectFile,inferFileTmpSparse,featuresNmf,testIds,testFile,'nmfApproximateFast',resultFile,resultFileFilter,topN)
     os.system('rm -f '+inferFileTmpSparse)



   elif(algorithm=='nmfSparse' and (isSearchNmfSparse)):
     tfidf_model=gensim.models.TfidfModel.load(tfidfFile)
     modelNmf_=getLoadModel('nmf',modelNmf)
     indices,values,sizes=loadSparseVectFile(inferDir+'/'+'nmfSparse.txt')
     trainIndices= c_uint16_2darr(indices,len(indices),sizes)
     trainFeatures= c_float_2darr(values)
     trainLens=np.array(sizes,dtype=np.uint16)
     
     if(isMultiProcess):
       inferFileTmp=inferDir+'/'+'nmf.txt.tmp'
       inferFileTmpSparse=inferDir+'/'+'nmfSparse.txt.tmp'
       getFeatureFromTextListMT(modelNmf_,tfidf_model,testFile,'nmf',inferFileTmp)
       convertDocVecToSparseVector(inferFileTmp,inferFileTmpSparse,indelim=';',outdelim=';',th=0.01)
       demoGenericSparseVectForMT(trainFeatures,inferFileTmpSparse,testIds,topN,resultFile,trainIndices,trainLens,len_dim=featuresNmf)
       os.system('rm -f '+inferFileTmp)
       os.system('rm -f '+inferFileTmpSparse)

     else:
       demoGenericSparseVect(modelNmf_,tfidf_model,trainFeatures,testFile,testIds,'nmfSparse',topN,resultFile,trainIndices,trainLens,len_dim=featuresNmf)
     
   elif(algorithm=='btmSparse' and (isSearchBtmSparse)):
     if(modelDir[len(modelDir)-1]!='/'):
        modelDir=modelDir+'/'
     dictVocab= loadDict(vocabFile)
     inferFileTmp=inferDir+'/'+'btm.txt.infer'
     inferFileTmpSparse=inferDir+'/'+'btmSparse.txt.infer'
     if(isMultiProcess):
       os.system('python3 src/infer/createBtmVec.py '+testFile+' '+str(btmTopic)+' '+inferDir+' '+modelDir+' '+inferFileTmp+' '+str(1))

     else:
       inferBtm(testFile,dictVocab,inferFileTmp,modelDir,inferType=inferType_,tagId='')
      
     convertDocVecToSparseVector(inferFileTmp,inferFileTmpSparse,indelim=' ',outdelim=';',th=sparseBtmTh)
     indices,values,sizes=loadSparseVectFile(inferDir+'/'+'btmSparse.txt')
     trainIndices= c_uint16_2darr(indices,len(indices),sizes)
     trainFeatures= c_float_2darr(values)
     trainLens=np.array(sizes,dtype=np.uint16)
     if(isMultiProcess):
       demoGenericSparseVectForMT(trainFeatures,inferFileTmpSparse,testIds,topN,resultFile,trainIndices,trainLens,len_dim=btmTopic)
     else:
       demoBtmSparseVect(trainFeatures,inferFileTmp,testIds,topN,resultFile,trainIndices,trainLens,len_dim=btmTopic)
     os.system('rm -f '+inferFileTmp)
     os.system('rm -f '+inferFileTmpSparse)
