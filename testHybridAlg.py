# python3 testHybridAlg.py src/conf/configHybrid.json model/doc2vec-dbow model/hdp infer/doc2vec-dbow.txt input/fileIds.txt input/importantSent-20/luhn.txt input/testFileIds.txt infer/labelsHdp.txt 1 100 0.80 0.60 0.40 0.20 results/resultsAll.txt
import os
import sys
import re
import time
import ctypes # for c dynamic library
import math
from numpy.ctypeslib import ndpointer
from ctypes import POINTER
import numpy as np
import gensim
from collections import defaultdict
import json
import random
import multiprocessing
from multiprocessing import Process

def preprocess(text):
    text = re.sub(r"[^\w\s]"," ",text)
    textLower = text.lower()
    return ' '.join(textLower.strip().split())


def split_list(lst, n):
    splitted = []
    for i in reversed(range(1, n + 1)):
        split_point = len(lst)//i
        splitted.append(lst[:split_point])
        lst = lst[split_point:]
    return splitted

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

#id\tdocUid
def getOnlyIdFromFile(fileIdsFile):
    fileIds=[]
    infp=open(fileIdsFile,'r',encoding='utf-8')
    for line in infp:
        text=line.replace('\n','')
        id=text.split('\t')[0] 
        fileIds.append(id)
    return fileIds

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
 

def getFeatureFromText(model,tfidf_model,text,algorithm,thProb=0.01):
   
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

       elif(algorithm=="nmf"):
         doc_bow = model.id2word.doc2bow(content.split())
         corpus_tfidf = tfidf_model[doc_bow]
         docVector= model[corpus_tfidf]
         docVectorTmp=[]
         for i in range(featuresNmf):
           docVectorTmp.append(0)
         for x in  docVector:
             if(len(x)>1):
               docVectorTmp[x[0]]=round(x[1],precision)
         
         feature=docVectorTmp

       elif(algorithm=="lda"):
         doc_bow = model.id2word.doc2bow(content.split())
         docVector= model.get_document_topics(doc_bow, minimum_probability=thProb)
         docVectorTmp=[]
         for i in range(featuresLda):
           docVectorTmp.append(0)
         for x in  docVector:
             if(len(x)>1):
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
             if(len(x)>1):
               docVectorTmp[x[0]]=round(x[1],precision)
         feature=docVectorTmp

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

def getSpecificArray(begin,end):
    array=[]
    for i in range(begin,end):
        array.append(i)
    return array

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


def getFeatureFromTextList(model,tfidf_model,textList,algorithm,fileIds,dictDocVectors,thProb=0.01):


    featuresArr=[]
    idx=0
    for elem in textList:
        feature=getFeatureFromText(model,tfidf_model,elem,algorithm,thProb=0.01)
        dictDocVectors[fileIds[idx]]=feature
        idx=idx+1

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



def demoHybridAlg(modelCluster,modelSemantic,trainFeatures,trainIds,testFile,testIds,resultFile):
   
   if(isFastSmartSearchUsingCluster):
     algorithmStr="fastSmartSearchUsingCluster"
   else:
     algorithmStr=algorithmSemantic

   outfp=open(resultFile,'a',encoding='utf-8')
   len_train=len(trainIds)
   idx=0
   correct=0
   totalTime=0.0

   if(isInferMultiThread):
     inferFileTmpSemantic=inferDir+'/'+algorithmSemantic+'.tmp'
     inferFileTmpCluster=inferDir+'/'+algorithmCluster+'.tmp'
     getFeatureFromTextListMT(modelSemantic,tfidf_model,testFile,algorithmSemantic,inferFileTmpSemantic,thProb=0.01)
     getFeatureFromTextListMT(modelCluster,tfidf_model,testFile,algorithmCluster,inferFileTmpCluster,thProb=0.01)
     inferVectorsSemanticArr=getLoadVectorFloatArray(inferFileTmpSemantic,vectFileDelim)
     inferVectorsClusterArr=getLoadVectorFloatArray(inferFileTmpCluster,vectFileDelim)
     os.system('rm -f '+inferFileTmpSemantic)
     os.system('rm -f '+inferFileTmpCluster)
     lenTest=len(inferVectorsSemanticArr)

   else:
      testDataArr = [line.strip() for line in open(testFile,"r",encoding="utf-8")]
      lenTest=len(testDataArr)

   for ij in range(lenTest):
     if(isInferMultiThread):
       inferVectorSemantic=inferVectorsSemanticArr[ij]
       inferVectorCluster=inferVectorsClusterArr[ij]
     else:
       inferVectorSemantic=getFeatureFromText(modelSemantic,tfidf_model,testDataArr[ij],algorithmSemantic,thProb=0.01)
       inferVectorCluster=getFeatureFromText(modelCluster,tfidf_model,testDataArr[ij],algorithmCluster,thProb=0.01)

     testFeature=np.array([list(inferVectorSemantic)],'f')
     if(isFastSmartSearchUsingCluster):
       start=time.time()
       npArrCluster=np.array(inferVectorCluster)
       thProb=minProb
       maxTmpProb=np.max(inferVectorCluster)
       if(thProb>maxTmpProb):
         thProb=maxTmpProb
       indices = np.nonzero(npArrCluster>=thProb)[0]
       maxLabels =indices[np.argsort(npArrCluster[indices])[::-1][:maxSearchTopNCluster]]
       #print(maxLabels)
       #maxProb= inferVectorCluster[maxLabels[0]]
       labelList= np.concatenate([ labelsDict[maxLabels[x]] for x in range(len(maxLabels)) if maxLabels[x] in labelsDict ])   
       sim_obj=libSim.cosine_similarity_topMTSingleCluster(trainFeatures,testFeature,len_dim,topN,labelList,len(labelList),10)  #idx,similarity
       end=time.time()
       totalTime+=(end-start)
     else:
       start=time.time()
       sim_obj=libSim.cosine_similarity_topMTSingle(trainFeatures,testFeature,len_train,len_dim,topN,10)  #idx,similarity
       end=time.time()
       totalTime+=(end-start)

     sim_ids=[]
     sim_s=[]
     for t in range(topN):
         sim_ids.append(int(sim_obj[t][0])) #id
         sim_s.append(np.round(sim_obj[t][1],4)) #id,sim
    
     #print("simIds: ",sim_ids)
     testSimId=testIds[idx].split('\t')[0]
     isCorrect=False
     if(int(testSimId) in sim_ids):
        correct=correct+1
        isCorrect=True
     print("True: ",int(testSimId)," Contain: ",str(isCorrect))
     idx=idx+1

   totalSecond=round(totalTime,1)
   if(isFastSmartSearchUsingCluster):
     thresholdStr=" minProb: "+str(minProb)+" maxSearchTopNCluster: "+str(maxSearchTopNCluster)
     print('TestFile: '+str(testFile)+' Algorithm: '+str(algorithmStr)+' TopN: '+str(topN)+thresholdStr,' Total: ',len(testIds),' Correct: ',correct, ' Acc: ', round(100*correct/len(testIds),2),' TotalSearchTimeSec: ',str(totalSecond))
     outfp.write('TestFile: '+str(testFile)+' Algorithm: '+str(algorithmStr)+' TopN: '+str(topN)+thresholdStr+' Total: '+str(len(testIds))+' Correct: '+str(correct)+ ' Acc: '+str( round(100*correct/len(testIds),2))+' TotalSearchTimeSec: '+str(totalSecond)+'\n')
   else:
     print('TestFile: '+str(testFile)+' Algorithm: '+str(algorithmStr)+' TopN: '+str(topN)+' Total: ',len(testIds),' Correct: ',correct, ' Acc: ', round(100*correct/len(testIds),2),' TotalSearchTimeSec: ',str(totalSecond))
     outfp.write('TestFile: '+str(testFile)+' Algorithm: '+str(algorithmStr)+' TopN: '+str(topN)+' Total: '+str(len(testIds))+' Correct: '+str(correct)+ ' Acc: '+str( round(100*correct/len(testIds),2))+' TotalSearchTimeSec: '+str(totalSecond)+'\n')
   outfp.close()


if __name__=="__main__":

  configFile=sys.argv[1]
  semanticAlgModelFile=sys.argv[2]
  clusterAlgModelFile=sys.argv[3]
  inferFile=sys.argv[4]
  trainIdsFile=sys.argv[5]
  testFile=sys.argv[6]
  testIdsFile=sys.argv[7]
  labelsFile=sys.argv[8]
  isFastSmartSearchUsingClusterArg=int(sys.argv[9])
  topN=int(sys.argv[10])
  minProb=float(sys.argv[11])
  maxSearchTopNCluster=int(sys.argv[12])
 

  resultFile=sys.argv[13]
  
  with open(configFile, 'r') as f:
    config = json.load(f)

  
  algorithmCluster=config["algorithmCluster"]
  algorithmSemantic=config["algorithmSemantic"]
  featuresHdp=config["featuresHdp"]
  isHdpDivide=config["isHdpDivide"]
  hdpDivideCnt=config["hdpDivideCnt"]
  minHdpDivideSent=config["minHdpDivideSent"]
  maxHdpDivideSent=config["maxHdpDivideSent"]
  start_alpha=config["start_alpha"]
  min_alpha_=config["min_alpha"]
  infer_epoch=config["infer_epoch"]
  isInferMultiThread=config["isInferMultiThread"]

  vectFileDelim=config["vectFileDelim"]

  # other configurations
  modelDir=os.path.dirname(semanticAlgModelFile)
  tfidfFile=modelDir+'/'+'tfidf'
  inferDir=os.path.dirname(inferFile)
 
  libSimPath='src/bin/sim.so'
  

  isFastSmartSearchUsingCluster=True
  if(isFastSmartSearchUsingClusterArg==0):
     isFastSmartSearchUsingCluster=False

 
  libSim= ctypes.cdll.LoadLibrary(libSimPath)
  charptr= ctypes.c_char_p
  libSim.loadVectFileC.restype =  POINTER(POINTER(ctypes.c_float))
  libSim.loadVectFileC.argtypes = [ charptr,charptr];

  libSim.cosine_similarity_top.restype =  POINTER(POINTER(ctypes.c_float))
  libSim.cosine_similarity_top.argtypes = [ POINTER(POINTER(ctypes.c_float))  , ndpointer(ctypes.c_float),ctypes.c_int,ctypes.c_int,ctypes.c_int]
  
  libSim.cosine_similarity_topMTSingle.restype =  POINTER(POINTER(ctypes.c_float))
  libSim.cosine_similarity_topMTSingle.argtypes = [ POINTER(POINTER(ctypes.c_float))  , ndpointer(ctypes.c_float),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]

  libSim.cosine_similarity_topMTSingleCluster.restype =  POINTER(POINTER(ctypes.c_float))
  libSim.cosine_similarity_topMTSingleCluster.argtypes = [ POINTER(POINTER(ctypes.c_float))  , ndpointer(ctypes.c_float),ctypes.c_int,ctypes.c_int,ndpointer(ctypes.c_int),ctypes.c_int,ctypes.c_int]


  tfidf_model=gensim.models.TfidfModel.load(tfidfFile)
  modelSemantic=getLoadModel(algorithmSemantic,semanticAlgModelFile)
  modelCluster=getLoadModel(algorithmCluster,clusterAlgModelFile)

  labelsDict=loadLabelsDictFromFile(labelsFile)
  with open(inferFile,"r",encoding="utf-8") as inferF:
    firstline = next(inferF)
  len_dim=len(firstline.split(vectFileDelim))

  string1=inferFile.encode('utf-8')
  string2=vectFileDelim.encode('utf-8')
  featuresDoc2VecDbow_=libSim.loadVectFileC(string1,string2)  #idx,similarity
  start=time.time()
  trainIds= getOnlyIdFromFile(trainIdsFile)
  testIds=getOnlyIdFromFile(testIdsFile)
  demoHybridAlg(modelCluster,modelSemantic,featuresDoc2VecDbow_,trainIds,testFile,testIds,resultFile)
  end=time.time() 
