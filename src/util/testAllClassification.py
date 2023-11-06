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

def preprocess(text):
    text = re.sub(r"[^\w\s]"," ",text)
    textLower = text.lower()
    return ' '.join(textLower.strip().split())

def writeListToFile(outfile,list):
    outfp=open(outfile,'w',encoding='utf-8')
    for elem in list:
        outfp.write(str(elem)+'\n')
    outfp.close()    


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

def most_frequentElement(list):
    numeral=[[list.count(nb), nb] for nb in list]
    numeral.sort(key=lambda x:x[0], reverse=True)
    return(numeral[0][1])

def getTopSimDocByCosineSimWithLabel(trainFeatures,trainIds,testFeature,topN):

    sims= np.zeros(len(trainFeatures), float) 
    for i in range(0,len(trainFeatures)):
        cosineSim=cosine_similarity_numpy(trainFeatures[i],testFeature)  #cosine similarity
        sims[i]=-1.0*cosineSim # for reverse sort

    idx=np.argsort(sims)
    sims_=-1.0*np.sort(sims)
    sim_idx=idx[:topN]
    sim_ids=[]
    for index in sim_idx:
        sim_ids.append(trainIds[index])
    
    predictLabel=most_frequentElement(sim_ids)
    return predictLabel

def  getTopSimDocByCosineSimCWithLabel(trainFeatures,trainIds,testFeature,len_train,len_dim,topN):

    sim_obj=libSim.cosine_similarity_top(trainFeatures,testFeature,len_train,len_dim,topN)  #idx,similarity
    sim_ids=[]
    for t in range(topN):
        sim_ids.append(trainIds[int(sim_obj[t][0])]) #id,sim
    predictLabel=most_frequentElement(sim_ids)
    return predictLabel

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


def cosine_similarity_numpy(arr1,arr2):
    len1=math.sqrt(np.dot(arr1,arr1))
    len2=math.sqrt(np.dot(arr2,arr2))
    if(len1==0 or len2==0):
       return 0.0

    cosine = np.dot(arr1,arr2)/len1/len2
    return cosine

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

def infer(testFile,dictVocab,inferFile,inferType='sum_b'):

    rand=random.randint(0,1000000)
    tagTime = datetime.now().strftime("%Y%m%d-%H%M%S")+'-'+str(rand)
    outfileDwid=inferDir+'dwid-'+tagTime+'.txt'
    createDwidFile(testFile,dictVocab,outfileDwid)
    
    if(inferType=='sum_b'):
       os.system('src/bin/btm inf sum_b'+' '+str(btmTopic)+' '+outfileDwid+' '+modelDir+' > '+'src/log/btmLog.txt')
    elif(inferType=='sum_w'):
       os.system('src/bin/btm inf sum_w'+' '+str(btmTopic)+' '+outfileDwid+' '+modelDir+' > '+'src/log/btmLog.txt') 
    elif(inferType=='mix'):
       os.system('src/bin/btm inf mix'+' '+str(btmTopic)+' '+outfileDwid+' '+modelDir+' > '+'src/log/btmLog.txt')

    os.system('rm -f '+ outfileDwid)
    tmpInferFile=modelDir+'k'+str(btmTopic)+'.pz_d'
    os.system('mv '+tmpInferFile+' '+inferFile)

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

def getFeatureFromText(model,tfidf_model,text,algorithm):
   
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
         docVector= model.get_document_topics(doc_bow, minimum_probability=0.01)
         feature =docVector

       elif(algorithm=="lda"):
         doc_bow = model.id2word.doc2bow(content.split())
         docVector= model.get_document_topics(doc_bow, minimum_probability=0.01)
         docVectorTmp=[]
         for i in range(featuresLda):
           docVectorTmp.append(0)
         for x in  docVector:
           docVectorTmp[x[0]]=round(x[1],precision)
         
         feature=docVectorTmp

       elif(algorithm=="ldaTfidf"):
         doc_bow = model.id2word.doc2bow(content.split())
         corpus_tfidf = tfidf_model[doc_bow]
         docVector= model.get_document_topics(corpus_tfidf, minimum_probability=0.01)
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

       else: #doc2vec
         docVector=[x for x in model.infer_vector(content.split(), alpha=start_alpha,min_alpha=min_alpha_, steps=infer_epoch)]
         feature =docVector 
      
       return feature


def getLoadVector(vectFile,delim=';'):
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

def getLoadBtmVector(vectFile,delim=' '):
   X_train=[]
   infp=open(vectFile,'r',encoding='utf-8')
   for line in infp:
      text=line.replace('\n','')
      btmVect = [float(x) for x in text.split()]
      X_train.append(np.array(btmVect))
   return X_train


def demoGeneric(model,tfidf_model,features, X_train_C,testFile,trainIds,testIds,algorithm,topN,resultFile):

   outfp=open(resultFile,'a',encoding='utf-8')
   len_train=len(features)
   len_dim=len(features[0])
   infp=open(testFile,"r",encoding="utf-8")
   correct=0
   idx=0
  
   for line in infp:
      text=line.replace("\n","")
      feauture=getFeatureFromText(model,tfidf_model,text,algorithm)
      docVectorNp=np.array([ list(feauture) ],'f')
      label=getTopSimDocByCosineSimCWithLabel(X_train_C,trainIds,docVectorNp,len_train,len_dim,topN)
      testSimId=testIds[idx].split('\t')[0]
      isCorrect=False
      if(testSimId ==label):
          correct=correct+1
          isCorrect=True
      print("True: ",testSimId,'Estimated: ',label)
      idx=idx+1
   print('Algorithm: '+str(algorithm)+' Total: ',len(testIds),' Correct: ',correct, ' Acc: ', round(100*correct/len(testIds),2))
   outfp.write('TestFile:'+str(testFile)+' Algorithm: '+str(algorithm)+' TopN: '+str(topN)+' Total: '+str(len(testIds))+' Correct: '+str(correct)+ ' Acc: '+str( round(100*correct/len(testIds),2))+'\n')
   outfp.close()

def demoBtm(dictVocab,features,inferFile,trainIds,testIds,topN,resultFile,inferType_='sum_b'):
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
         label=getTopSimDocByCosineSimWithLabel(features,trainIds,np.array(btmVect),topN)
         testSimId=testIds[idx].split('\t')[0]
         if(testSimId ==label):
            correct=correct+1
            isCorrect=True
         print("True: ",testSimId,'Estimated: ',label)
         idx=idx+1
   print('Algorithm: '+str(algorithm)+' Total: ',len(testIds),' Correct: ',correct, ' Acc: ', round(100*correct/len(testIds),2))
   outfp.write('TestFile:'+str(testFile)+' Algorithm: '+str(algorithm)+' TopN: '+str(topN)+' Total: '+str(len(testIds))+' Correct: '+str(correct)+ ' Acc: '+str( round(100*correct/len(testIds),2))+'\n')
   outfp.close()

def demoDoc2Vec(model,tfidf_model,features, X_train_C,testFile,trainIds,testIds,algorithm,topN,resultFile):
  
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

      label=getTopSimDocByCosineSimCWithLabel(X_train_C,trainIds,docVectorNp,len_train,len_dim,topN)
      testSimId=testIds[idx].split('\t')[0]
      isCorrect=False
      if(testSimId ==label):
          correct=correct+1
          isCorrect=True
      print("True: ",testSimId,'Estimated: ',label)
      idx=idx+1
   print('Algorithm: '+str(algorithm)+' Total: ',len(testIds),' Correct: ',correct, ' Acc: ', round(100*correct/len(testIds),2))
   outfp.write('TestFile:'+str(testFile)+' Algorithm: '+str(algorithm)+' TopN: '+str(topN)+' Total: '+str(len(testIds))+' Correct: '+str(correct)+ ' Acc: '+str( round(100*correct/len(testIds),2))+'\n')
   outfp.close()

if __name__=="__main__":
    
  modelDir=sys.argv[1]
  inferDir=sys.argv[2]
  fileIdsFile=sys.argv[3]
  testFile=sys.argv[4]
  fileIdsTestFile=sys.argv[5]
  resultFile=sys.argv[6]
  topN=int(sys.argv[7])

  algorithms=['lsa','nmf','lda','doc2vec-dbow','doc2vec-dbpv','rp','btm']


  trainIds=getLines(fileIdsFile)
  testIds=getLines(fileIdsTestFile)

  with open('input/configTest.json', 'r') as f:
    config = json.load(f)

  # konfigürasyonlar
  featuresLda=config["featuresLda"]
  featuresNmf=config["featuresNmf"]
  featuresRp=config["featuresRp"]
  btmTopic=config["featuresBtm"]
  infer_epoch=config['infer_epoch'] 
  start_alpha=config['start_alpha'] # learning rate
  min_alpha_=config['min_alpha']
  isSearchLda=config['isSearchLda']
  isSearchLsa=config['isSearchLsa']
  isSearchNmf=config['isSearchNmf']
  isSearchDoc2vecDbow=config['isSearchDoc2vecDbow']
  isSearchDoc2vecDbpv=config['isSearchDoc2vecDbpv']
  isSearchRp=config['isSearchRp']
  isSearchBtm=config['isSearchBtm']

  modelLda=modelDir+'/'+'lda'
  modelLsa=modelDir+'/'+'lsa'
  modelNmf=modelDir+'/'+'nmf'
  modelDoc2VecDbow=modelDir+'/'+'doc2vec-dbow'
  modelDoc2VecDbpv=modelDir+'/'+'doc2vec-dbpv'
  modelRp=modelDir+'/'+'rp'
  vocabFile=modelDir+'/'+'btmVocab.txt'

  tfidfFile=modelDir+'/'+'tfidf'
  inferType_='sum_b'
  precision=6
  
  resultDir=os.path.dirname(resultFile)
  if(os.path.exists(resultDir)==False):
     os.system('mkdir '+resultDir)

 
  libSim= ctypes.cdll.LoadLibrary('src/bin/sim.so')

  libSim.cosine_similarity_top.restype =  POINTER(POINTER(ctypes.c_float))
  libSim.cosine_similarity_top.argtypes = [ POINTER(POINTER(ctypes.c_float))  , ndpointer(ctypes.c_float),ctypes.c_int,ctypes.c_int,ctypes.c_int]
   
  libSim.cosine_similarity_topSparseFilter.restype =  POINTER(POINTER(ctypes.c_float))
  #( uint16_t ** trainIndices,uint16_t * testIndice,float** trainValues,float* testValue,uint16_t * trainLens,uint16_t testLen,int dataSize,int dim,int topN) 
  libSim.cosine_similarity_topSparseFilter.argtypes = [  POINTER(POINTER(ctypes.c_uint16)),ndpointer(ctypes.c_uint16), POINTER(POINTER(ctypes.c_float))  , ndpointer(ctypes.c_float), ndpointer(ctypes.c_uint16), ctypes.c_uint16, ctypes.c_int,ctypes.c_int,ctypes.c_int,ndpointer(ctypes.c_uint32),ctypes.c_int]
    
   
  tfidf_model=gensim.models.TfidfModel.load(tfidfFile)
  for algorithm in algorithms:

   if(algorithm=='lsa' and (isSearchLsa)):
     modelLsa_=getLoadModel('lsa',modelLsa)
     featuresLsa_=getLoadVector(inferDir+'/'+'lsa.txt')
     print('Lsa load feature completed...')
     X_train_CLsa= c_float_2darr(featuresLsa_)
     demoGeneric(modelLsa_,tfidf_model,featuresLsa_, X_train_CLsa,testFile,trainIds,testIds,'lsa',topN,resultFile)

   elif(algorithm=='nmf'  and (isSearchNmf)):
     modelNmf_=getLoadModel('nmf',modelNmf)
     featuresNmf_=getLoadVector(inferDir+'/'+'nmf.txt')
     print('Nmf load feature completed...')
     X_train_CNmf= c_float_2darr(featuresNmf_)
     demoGeneric(modelNmf_,tfidf_model,featuresNmf_, X_train_CNmf,testFile,trainIds,testIds,'nmf',topN,resultFile)

   elif(algorithm=='doc2vec-dbow' and (isSearchDoc2vecDbow)):
     modelDoc2Vecdbow_=getLoadModel('doc2vec-dbow', modelDoc2VecDbow)
     featuresDoc2VecDbow_=getLoadVector(inferDir+'/'+'doc2vec-dbow.txt')
     print('Doc2Vec load feature completed...')
     X_train_C_Dbow= c_float_2darr( featuresDoc2VecDbow_)
     demoDoc2Vec(  modelDoc2Vecdbow_,tfidf_model, featuresDoc2VecDbow_, X_train_C_Dbow,testFile,trainIds,testIds,'doc2vec-dbow',topN,resultFile)
  
   elif(algorithm=='doc2vec-dbpv'  and (isSearchDoc2vecDbpv)):
     modelDoc2Vecdbpv_=getLoadModel('doc2vec-dbpv', modelDoc2VecDbpv)
     featuresDoc2VecDbpv_=getLoadVector(inferDir+'/'+'doc2vec-dbpv.txt')
     print('Doc2Vec load feature completed...')
     X_train_C_Dbpv= c_float_2darr( featuresDoc2VecDbpv_)
     demoDoc2Vec(  modelDoc2Vecdbpv_,tfidf_model, featuresDoc2VecDbpv_, X_train_C_Dbpv,testFile,trainIds,testIds,'doc2vec-dbpv',topN,resultFile)

   elif(algorithm=='lda' and (isSearchLda)):
     modelLda_=getLoadModel('lda',modelLda)
     featuresLda_=getLoadVector(inferDir+'/'+'lda.txt')
     X_train_CLda= c_float_2darr(featuresLda_)
     demoGeneric(modelLda_,tfidf_model,featuresLda_, X_train_CLda,testFile,trainIds,testIds,'lda',topN,resultFile)

   elif(algorithm=='rp' and (isSearchRp)):
     modelRp_=getLoadModel('rp',modelRp)
     featuresRp_=getLoadVector(inferDir+'/'+'rp.txt')
     X_train_CRp= c_float_2darr(featuresRp_)
     demoGeneric(modelRp_,tfidf_model,featuresRp_, X_train_CRp,testFile,trainIds,testIds,'rp',topN,resultFile)

   elif(algorithm=='btm' and (isSearchBtm)):
     modelDir=modelDir+'/'
     dictVocab= loadDict(vocabFile)
     featuresBtm_=getLoadBtmVector(inferDir+'/'+'btm.txt')
     inferFileTmp=inferDir+'/'+'btm.txt.infer'
     infer(testFile,dictVocab,inferFileTmp,inferType=inferType_) #test file should be preprocessed
     demoBtm(dictVocab,featuresBtm_,inferFileTmp,trainIds,testIds,topN,resultFile)
