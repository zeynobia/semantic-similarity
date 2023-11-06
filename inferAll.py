import sys
import os
from datasketch import MinHashLSHForest

import gensim
import re
from gensim.models.nmf import Nmf as GensimNmf
import numpy as np
import json
from datasketch import MinHash
import multiprocessing
from multiprocessing import Process
import uuid
import pickle
from memory_profiler import profile
import random
import math

def preprocess(text):
    text = re.sub(r"[^\w\s]"," ",text)
    textLower = text.lower()
    return ' '.join(textLower.strip().split())

def readSpecificLinesAll(inFile,beginLine,endLine):
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

def readSpecificLines(infile,beginLine,endLine):

  infpBytes=open(corpusByteFile,'r',encoding='utf-8')
  linesResult=[]
  lineCnt=0
  initByte=0
  for line in infpBytes:
     if(lineCnt==beginLine):
       text=line.replace('\n','')
       initByte=int(text)
       break
     lineCnt=lineCnt+1
  lineProcess=endLine-beginLine  
  infp=open(infile,'r',encoding='utf-8')
  infp.seek(initByte)
  lineCnt=1
  for line in infp:
    text=line.replace('\n','')
    linesResult.append(text)
    if(lineCnt>=lineProcess):
       break
    lineCnt=lineCnt+1
  return linesResult



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

def getLoadModel(algorithm,modelFile):

  if(algorithm=="lsa"):
      model =gensim.models.LsiModel.load(modelFile) 
  elif(algorithm=="rp"):
      model =gensim.models.RpModel.load(modelFile) 
  elif((algorithm=="nmf") or (algorithm=="nmfSparse") ):
      model =GensimNmf.load(modelFile)
  elif((algorithm=="lda") or (algorithm=="ldaSparse") or (algorithm=="ldaTfidf")):
      model =gensim.models.LdaMulticore.load(modelFile)
  elif((algorithm=="hdp") or (algorithm=="hdpSparse")):
      model=gensim.models.HdpModel.load(modelFile)
  elif(algorithm=="doc2vec-dbow" or algorithm=="doc2vec-dbpv"):
      #inference hyper-parameters
      model =gensim.models.Doc2Vec.load(modelFile)
  return model
 

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


def getFeatureFromFile(model,tfidf_model,inFile,tfidfFile,modelFile,algorithm,thProb=0.01):
   
  infp=open(inFile,'r',encoding='utf-8')
  features=[]
   
  for line in infp:
       text=line.replace('\n','')
       content=preprocess(text)
       if(algorithm=="lsa"):
         doc_bow = model.id2word.doc2bow(content.split())
         corpus_tfidf = tfidf_model[doc_bow]
         docVector= model[corpus_tfidf]
         featureArr =list(zip(*docVector ))
         if(len(featureArr)>1):
           features.append(featureArr[1])
         else:
           features.append([])

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
         features.append(feature)
       elif(algorithm=="lda"):
         doc_bow = model.id2word.doc2bow(content.split())
         docVector= model.get_document_topics(doc_bow, minimum_probability=thProb)
         docVectorTmp=[]
         for i in range(featuresLda):
           docVectorTmp.append(0)
         for x in  docVector:
             if(len(x)>1):
               docVectorTmp[x[0]]=round(x[1],precision)
         features.append(docVectorTmp)

       elif(algorithm=="ldaTfidf"):
         doc_bow = model.id2word.doc2bow(content.split())
         corpus_tfidf = tfidf_model[doc_bow]
         docVector= model.get_document_topics( corpus_tfidf, minimum_probability=thProb)
         docVectorTmp=[]
         for i in range(featuresLda):
           docVectorTmp.append(0)
         for x in  docVector:
             if(len(x)>1):
               docVectorTmp[x[0]]=round(x[1],precision)
         features.append(docVectorTmp)

       elif((algorithm=="hdp") and (isHdpDivide==False)):
         doc_bow = model.id2word.doc2bow(content.split())
         docVector=model[doc_bow]
         docVectorTmp=[]
         for i in range(featuresHdp):
           docVectorTmp.append(0)
         for x in docVector:
           if(len(x)>1):
             docVectorTmp[x[0]]=round(x[1],precision)
         features.append(docVectorTmp)

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
           #print(contentPre)
           #print(docVector)
           for x in docVector:
             if(len(x)>1):
               docVectorHdpTmp[x[0]]=docVectorHdpTmp[x[0]]+round(x[1]/hdpDivideCntTmp,6)
         features.append(docVectorHdpTmp)


       elif(algorithm=='rp'):
          doc_bow = model.id2word.doc2bow(content.split())
          docVector= model.__getitem__(doc_bow)
          featureArr =list(zip(*docVector ))
          if(len(featureArr)>1):
             feature=featureArr[1]
          else:
             feature=[]
          testFeature=np.pad(feature, (0, featuresRp-len(feature)))
          features.append(testFeature)

       else: #doc2vec
         docVector=[x for x in model.infer_vector(content.split(), alpha=start_alpha,min_alpha=min_alpha_, steps=infer_epoch)]
         feature =docVector 
         features.append(feature)
      
  return features

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

def getLines(inFile):
   lines=[]
   infp=open(inFile,'r',encoding='utf-8')
   for line in infp:
      text=line.replace("\n","")
      lines.append(text)

   return lines

def getMinHashForestIndex(inTextFile,fileIds,modelOutFile,num_permArg=128):

   infp=open(inTextFile,'r',encoding='utf-8')
   forest = MinHashLSHForest(num_perm=num_permArg)
   idx=0
   for line in infp:
        text=line.replace('\n','')
        textPre=preprocess(text)
        minHashObj=minHash(textPre,num_perm=num_permArg)
        strId=fileIds[idx].split('\t')[0]
        forest.add(strId,minHashObj)
        idx=idx+1
   forest.index()
   pickle.dump(forest, open(modelOutFile, "wb" ) )
   

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

#@profile
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
  
  
  

def getLines(inFile):
   lines=[]
   infp=open(inFile,'r',encoding='utf-8')
   for line in infp:
      text=line.replace("\n","")
      lines.append(text)

   return lines


def writeVector(vectors,outfile,writeMode="w"):
    outfp=open(outfile,writeMode)
    cnt=0
    for vector in vectors:
        #print(vector)
        for i in range(0,len(vector)-1):
           outfp.write(str(round(vector[i],6))+";")
        if(len(vector)>1):
           outfp.write(str(round(vector[len(vector)-1],6))+"\n")
        else:
          outfp.write("\n")
        cnt=cnt+1
    outfp.close()



def getMaxFileId(maxFileId):
   infp=open(maxFileId,'r',encoding='utf-8')
   for line in infp:
      text=line.replace('\n','') #tek satır
      break
   count=int(text)
   return count

         
def updateMaxFileId(maxFileId,oldMaxId,sizeDoc):

   newCount=oldMaxId+sizeDoc
   outfp=open(maxFileId,'w',encoding='utf-8')
   outfp.write(str(newCount))
   outfp.close()

def updateSparseVectFile(vectFile,listFeatures):

    outfp=open(vectFile,'a',encoding='utf-8') #append
    for elem in listFeatures:
        outfp.write(str(round_all(elem, precision)).replace(", ",";").replace("[","").replace("]","")+'\n')
    outfp.close()

def updateVectFile(vectFile,listFeatures):
    outfp=open(vectFile,'a',encoding='utf-8') #append
    for elem in listFeatures:
       np_array = np.array(elem)
       np_round= np.around(np_array, precision)
       stri=map(str, np_round)
       tmpStr=';'.join(stri)
       outfp.write(tmpStr+'\n')
    outfp.close()

def generateFileIds(beginId,size): #idx\tdocId
    listIds=[]
    for idx in range(beginId,beginId+size):
        myuuid = uuid.uuid4()
        docId= str(myuuid)
        listIds.append(str(idx)+"\t"+docId)
    return listIds

def updateFileIdsWithId(fileIdsFile,listFileIds):

    outfp=open(fileIdsFile,'a',encoding='utf-8') #append
    for elem in listFileIds:
        outfp.write(elem+'\n')
    outfp.close()

def getLineCntFromFile(inFile):
    lineCnt=0
    infp=open(inFile,'r',encoding='utf-8')
    for line in infp:
        lineCnt=lineCnt+1
    return lineCnt


def getLoadVector(vectFile,delim=';',vectorType=np.float32):
   X_train=[]
   infp=open(vectFile,'r',encoding='utf-8')
   for line in infp:
        #print(line)
        text=line.replace('\n','')
        if(text!='' and text!=';'  and text!=' ; '):
          vectParts=text.split(delim)
          string_array = np.array(vectParts)
          float_array = string_array.astype(vectorType) 
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


if __name__=="__main__":
    
  inFile=sys.argv[1]
  modelDir=sys.argv[2]
  inferDir=sys.argv[3]
  configFile=sys.argv[4]
  with open(configFile, 'r') as f:
    config = json.load(f)


  # konfigürasyonlar
 
  featuresLda=config["featuresLda"]
  featuresNmf=config["featuresNmf"]
  featuresRp=config["featuresRp"]
  featuresBtm=config["featuresBtm"]
  featuresPlsa=config["featuresPlsa"]
  featuresHdp=config["featuresHdp"]
  featuresMinHash=config["featuresMinHash"]
  num_perm_=featuresMinHash
  featuresMinHashLshForest=config["featuresMinHashLshForest"]
  infer_epoch=config['infer_epoch'] 
  start_alpha=config['start_alpha'] # learning rate
  min_alpha_=config['min_alpha']
  isHdpDivide=config['isHdpDivide']
  hdpDivideCnt=config['hdpDivideCnt']
  minHdpDivideSent=config['minHdpDivideSent']
  maxHdpDivideSent=config['maxHdpDivideSent']

  isMultiProcess=config['isMultiProcess']
  isInferLda=config['isInferLda']
  isInferLsa=config['isInferLsa']
  isInferNmf=config['isInferNmf']
  isInferDoc2vecDbow=config['isInferDoc2vecDbow']
  isInferDoc2vecDbpv=config['isInferDoc2vecDbpv']
  isInferRp=config['isInferRp']
  isInferBtm=config['isInferBtm']
  isInferPlsa=config['isInferPlsa']
  isInferHdp=config['isInferHdp']
  isInferMinHash=config['isInferMinHash']
  isInferMinHashLshForest=config['isInferMinHashLshForest']
  

  isSparseLda=config['isSparseLda']
  isSparseNmf=config['isSparseNmf']
  isSparseBtm=config['isSparseBtm']
  isSparsePlsa=config['isSparsePlsa']
  isSparseHdp=config['isInferHdp']
  sparseBtmTh=config['sparseBtmTh']

  corpusByteFile=config["corpusByteFile"]
  isCreateFileIds=config["isCreateFileIds"]
  fileIdsFile=config["fileIdsFile"]
  maxfileIdFile=config["maxfileIdFile"]

  modelLda=modelDir+'/'+'lda'
  modelHdp=modelDir+'/'+'hdp'
  modelLsa=modelDir+'/'+'lsa'
  modelNmf=modelDir+'/'+'nmf'
  modelDoc2VecDbow=modelDir+'/'+'doc2vec-dbow'
  modelDoc2VecDbpv=modelDir+'/'+'doc2vec-dbpv'
  modelRp=modelDir+'/'+'rp'

  tfidfFile=modelDir+'/'+'tfidf'

  precision=6

 
  if(os.path.exists(inferDir)==False):
    os.system('mkdir '+inferDir)

  os.system('python3 src/train/createCorpusByteFile.py '+inFile+' '+corpusByteFile)

  if(isCreateFileIds):
    if(os.path.exists(maxfileIdFile)==False):
       os.system('python3 src/train/createFileIds.py '+inFile+' '+fileIdsFile+' '+maxfileIdFile)
       print('FileIds files created...')
    else:
      maxFileId=getMaxFileId(maxfileIdFile)
      lenDoc=getLineCntFromFile(inFile)
      listFileIds=generateFileIds(maxFileId,lenDoc)
      updateFileIdsWithId(fileIdsFile,listFileIds)
      updateMaxFileId(maxfileIdFile,maxFileId,lenDoc)
      print('FileIds files updated...')

  if(isInferLsa):
 
    vectFile=inferDir+'/'+'lsa.txt'
    vectFileNew=vectFile+'.new'
    tfidf_model=gensim.models.TfidfModel.load(tfidfFile)
    modelLsa_=getLoadModel('lsa',modelLsa)
    if(isMultiProcess):
     
      getFeatureFromTextListMT(modelLsa_,tfidf_model,inFile,'lsa',vectFileNew)
      featuresTmp=getLoadVector(vectFileNew,delim=';')
      updateVectFile(vectFile,featuresTmp)
      os.system('rm -f '+vectFileNew)
    else:
      featuresLsa_=getFeatureFromFile(modelLsa_,tfidf_model,inFile,tfidfFile,modelLsa,'lsa')
      writeVector(featuresLsa_,vectFile,'a')
    print('Lsa vector created...')

  if(isInferNmf):
    vectFile=inferDir+'/'+'nmf.txt'
    vectFileNew=vectFile+'.new'
    tfidf_model=gensim.models.TfidfModel.load(tfidfFile)
    modelNmf_=getLoadModel('nmf',modelNmf)
   
    if(isMultiProcess):
   
      getFeatureFromTextListMT(modelNmf_,tfidf_model,inFile,'nmf',vectFileNew)
      featuresTmp=getLoadVector(vectFileNew,delim=';')
      updateVectFile(vectFile,featuresTmp)
      os.system('rm -f '+vectFileNew)
    else:
      featuresNmf_=getFeatureFromFile(modelNmf_,tfidf_model,inFile,tfidfFile,modelNmf,'nmf')
      writeVector(featuresNmf_,vectFile,'a')
    if(isSparseNmf):
        convertDocVecToSparseVector(vectFile,inferDir+'/'+'nmfSparse.txt',indelim=';',outdelim=';',th=0.01)
    print('Nmf vector created...')

  if(isInferLda):
    vectFile=inferDir+'/'+'lda.txt'
    vectFileNew=vectFile+'.new'
    tfidf_model=gensim.models.TfidfModel.load(tfidfFile)
    modelLda_=getLoadModel('lda',modelLda)
    if(isMultiProcess):
 
      getFeatureFromTextListMT(modelLda_,tfidf_model,inFile,'lda',vectFileNew)
      featuresTmp=getLoadVector(vectFileNew,delim=';')
      updateVectFile(vectFile,featuresTmp)
      os.system('rm -f '+vectFileNew)
    else:
      featuresLda_=getFeatureFromFile(modelLda_,tfidf_model,inFile,tfidfFile,modelLda,'lda')
      writeVector(featuresLda_,vectFile,'a')
    if(isSparseLda):
        vectFileSparse=inferDir+'/'+'ldaSparse.txt'
        convertDocVecToSparseVector(vectFile,vectFileSparse,indelim=';',outdelim=';',th=0.01)
        
    print('Lda vector created...')

  if(isInferHdp):
    vectFile=inferDir+'/'+'hdp.txt'
    vectFileNew=vectFile+'.new'
    tfidf_model=gensim.models.TfidfModel.load(tfidfFile)
    modelHdp_=getLoadModel('hdp',modelHdp)
    if(isMultiProcess):
 
      getFeatureFromTextListMT(modelHdp_,tfidf_model,inFile,'hdp',vectFileNew)
      featuresTmp=getLoadVector(vectFileNew,delim=';')
      updateVectFile(vectFile,featuresTmp)
      os.system('rm -f '+vectFileNew)
    else:
      featuresHdp_=getFeatureFromFile(modelHdp_,tfidf_model,inFile,tfidfFile,modelHdp,'hdp')
      writeVector(featuresHdp_,vectFile,'a')
    if(isSparseHdp):
        vectFileSparse=inferDir+'/'+'hdpSparse.txt'
        convertDocVecToSparseVector(vectFile,vectFileSparse,indelim=';',outdelim=';',th=0.01)
        
    print('Hdp vector created...')


  if(isInferDoc2vecDbow):
    vectFile=inferDir+'/'+'doc2vec-dbow.txt'
    vectFileNew=vectFile+'.new'
    tfidf_model=gensim.models.TfidfModel.load(tfidfFile)
    modelDoc2VecDbow_=getLoadModel('doc2vec-dbow',modelDoc2VecDbow)
    if(isMultiProcess):
      getFeatureFromTextListMT(modelDoc2VecDbow_,tfidf_model,inFile,'doc2vec-dbow',vectFileNew)
      featuresTmp=getLoadVector(vectFileNew,delim=';')
      updateVectFile(vectFile,featuresTmp)
      os.system('rm -f '+vectFileNew)
    else:
      featuresDoc2VecDbow_=getFeatureFromFile( modelDoc2VecDbow_,tfidf_model,inFile,tfidfFile,modelDoc2VecDbow,'doc2vec-dbow')
      writeVector( featuresDoc2VecDbow_,vectFile,'a')
    print('Doc2vec dbow vector created...')
 
  if(isInferDoc2vecDbpv):
    vectFile=inferDir+'/'+'doc2vec-dbpv.txt'
    vectFileNew=vectFile+'.new'
    tfidf_model=gensim.models.TfidfModel.load(tfidfFile)
    modelDoc2VecDbpv_=getLoadModel('doc2vec-dbpv',modelDoc2VecDbpv)
    if(isMultiProcess):
     
      getFeatureFromTextListMT(modelDoc2VecDbpv_,tfidf_model,inFile,'doc2vec-dbpv',vectFileNew)
      featuresTmp=getLoadVector(vectFileNew,delim=';')
      updateVectFile(vectFile,featuresTmp)
      os.system('rm -f '+vectFileNew)
    else:
      featuresDoc2VecDbpv_=getFeatureFromFile( modelDoc2VecDbpv_,tfidf_model,inFile,tfidfFile,modelDoc2VecDbpv,'doc2vec-dbpv')
      writeVector( featuresDoc2VecDbpv_,vectFile,'a')
    print('Doc2vec dbpv vector created...')

  if(isInferRp):
    vectFile=inferDir+'/'+'rp.txt'
    vectFileNew=vectFile+'.new'
    tfidf_model=gensim.models.TfidfModel.load(tfidfFile)
    modelRp_=getLoadModel('rp',modelRp)
    if(isMultiProcess):
     
      getFeatureFromTextListMT(modelRp_,tfidf_model,inFile,'rp',vectFileNew)
      featuresTmp=getLoadVector(vectFileNew,delim=';')
      updateVectFile(vectFile,featuresTmp)
      os.system('rm -f '+vectFileNew)

    else:
      featuresRp_=getFeatureFromFile(modelRp_,tfidf_model,inFile,tfidfFile,modelRp,'rp')
      writeVector(featuresRp_,vectFile,'a')
    print('Random  projection vector created...')

  if(isInferMinHash):
    vectFile=inferDir+'/'+'minHash.txt'
    vectFileNew=vectFile+'.new'
    #os.system('python3 src/infer/createMinHashVec.py '+inFile+' '+str(featuresMinHash)+' '+inferDir+'/'+'minHash.txt'+' '+str(1))
    #print('Minhash vector created...')
    tfidf_model=gensim.models.TfidfModel.load(tfidfFile)
    if(isMultiProcess):
      getFeatureFromTextListMT(None,tfidf_model,inFile,'minHash',vectFileNew)
      featuresTmp=getLoadVector(vectFileNew,delim=';',vectorType=np.int32)
      updateVectFile(vectFile,featuresTmp)
      os.system('rm -f '+vectFileNew)

    else:
      featuresMinHash_=getFeatureFromFile(None,tfidf_model,inFile,tfidfFile,None,'minHash')
      writeVector(featuresMinHash_,vectFile,'a')
   
    print('MinHash vector created...')

  if(isInferMinHashLshForest):
    
    modelOutFile=inferDir+'/'+'minHashLshForest'
    fileIds=getLines(fileIdsFile)
    getMinHashForestIndex(inFile,fileIds,modelOutFile,num_permArg=featuresMinHashLshForest)
    print('MinHashLshForest model created...')

  if(isInferBtm):
     vectFileNew=inferDir+'/'+'btmNewFiles.txt'
     vectFile=inferDir+'/'+'btm.txt'
     if(isMultiProcess):
       os.system('python3 src/infer/createBtmVec.py '+inFile+' '+str(featuresBtm)+' '+inferDir+' '+modelDir+' '+vectFileNew+' '+str(1))
     else:
       os.system('python3 src/infer/createBtmVec.py '+inFile+' '+str(featuresBtm)+' '+inferDir+' '+modelDir+' '+vectFileNew+' '+str(0))
     featuresTmp= getLoadBtmVector(vectFileNew,delim=' ')
     updateVectFile(vectFile,featuresTmp)
     os.system('rm -f '+vectFileNew)

     if(isSparseBtm):
        convertDocVecToSparseVector(vectFile,inferDir+'/'+'btmSparse.txt',indelim=';',outdelim=';',th=sparseBtmTh)

     print('Biterm vector created...')


  if(isInferPlsa):
    vectFile=inferDir+'/'+'plsa.txt'
    vectFileNew=vectFile+'.new'
    modelFile=modelDir+'/'+'plsa'
    os.system('python3 src/infer/createPlsaVec.py '+inFile+' '+modelFile+' '+vectFileNew)
    featuresTmp=getLoadVector(vectFileNew,delim=';')
    updateVectFile(vectFile,featuresTmp)
    os.system('rm -f '+vectFileNew)
    if(isSparsePlsa):
        vectFileSparse=inferDir+'/'+'plsaSparse.txt'
        convertDocVecToSparseVector(vectFile,vectFileSparse,indelim=';',outdelim=';',th=0.01)
        
    print('Plsa vector created...')

  os.system('rm -f '+corpusByteFile)

  





  
  

  


