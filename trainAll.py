import sys
import os
import json

if __name__=="__main__":


  corpusFile=sys.argv[1]
  modelDir=sys.argv[2]
  configTrainFile=sys.argv[3]
  with open(configTrainFile, 'r') as f:
    config = json.load(f)

  isCreateFileIds=config["isCreateFileIds"]
  fileIdsFile=config["fileIdsFile"]
  maxfileIdFile=config["maxfileIdFile"]

  inputDir=config["inputDir"]


  featuresLda=config["featuresLda"]
  epochLda=config["epochLda"]
  iterLda=config["iterLda"]
  featuresLsa=config["featuresLsa"]
  featuresNmf=config["featuresNmf"]
  featuresBtm=config["featuresBtm"]
  featuresPlsa=config["featuresPlsa"]
  featuresHdp=config["featuresHdp"]
  iterBtm=config["iterBtm"]
  epochNmf=config["epochNmf"]
  featuresDoc2vecDbow=config["featuresDoc2vecDbow"]
  featuresDoc2vecDbpv=config["featuresDoc2vecDbpv"]
  featuresRp=config["featuresRp"]
  isDictCreate=config["isDictCreate"]
  isTfidfVectorCreate=config["isTfidfVectorCreate"]

  isTrainLda=config["isTrainLda"]
  isTrainLsa=config["isTrainLsa"]
  isTrainNmf=config["isTrainNmf"]
  isTrainDoc2vecDbow=config["isTrainDoc2vecDbow"]
  isTrainDoc2vecDbpv=config["isTrainDoc2vecDbpv"]
  isTrainRp=config["isTrainRp"]
  isTrainBtm=config["isTrainBtm"]
  isTrainPlsa=config["isTrainPlsa"]
  isTrainHdp=config["isTrainHdp"]

  if(isCreateFileIds==True):
    isCreateFileIds=1
  else:
    isCreateFileIds=0

  if(isTfidfVectorCreate==True):
      isTfidfVectorCreate=1
  else:
      isTfidfVectorCreate=0


     
  if(os.path.exists(modelDir)==False):
     os.system('mkdir '+modelDir)
  
  if(os.path.exists(inputDir)==False):
     os.system('mkdir '+inputDir)

  if(isDictCreate):
    os.system('python3 src/train/prepareCorpus.py '+corpusFile+' '+modelDir+' '+str(isTfidfVectorCreate))

  if(isCreateFileIds):
    os.system('python3 src/train/createFileIds.py '+corpusFile+' '+fileIdsFile+' '+maxfileIdFile)

  if(isTrainDoc2vecDbow):
    os.system('python3 src/train/trainDoc2vec.py '+corpusFile+' '+str(featuresDoc2vecDbow)+' '+str(0)+' '+modelDir+'/'+'doc2vec-dbow')
  
  if(isTrainDoc2vecDbpv):
    os.system('python3 src/train/trainDoc2vec.py '+corpusFile+' '+str(featuresDoc2vecDbpv)+' '+str(1)+' '+modelDir+'/'+'doc2vec-dbpv')

  if(isTrainLsa):
    os.system('python3 src/train/trainLsa.py '+corpusFile+' '+str(featuresLsa)+' '+modelDir+'/'+'lsa'+' '+modelDir+'/'+'tfidf')

  if(isTrainNmf):
    os.system('python3 src/train/trainNmf.py '+corpusFile+' '+str(featuresNmf)+' '+modelDir+'/'+'nmf' +' '+str( epochNmf))
  
  if(isTrainLda):
     os.system('python3 src/train/trainLda.py '+corpusFile+' '+str(featuresLda)+' '+str(epochLda)+' '+str(iterLda)+' '+modelDir+'/'+'lda')

  if(isTrainRp):
    os.system('python3 src/train/trainRp.py '+corpusFile+' '+str(featuresRp)+' '+modelDir+'/'+'rp')

  if(isTrainBtm):
    os.system('python3 src/train/trainBtm.py '+corpusFile+' '+str(featuresBtm)+' '+' '+str(iterBtm)+' '+modelDir)

  if(isTrainPlsa):
    os.system('python3 src/train/trainPlsa.py '+corpusFile+' '+str(featuresPlsa)+' '+modelDir+'/'+'plsa')

  if(isTrainHdp):
    os.system('python3 src/train/trainHdp.py '+corpusFile+' '+str(featuresHdp)+' 20 256 1.0 2.0 1.0 0.01 '+' '+modelDir+'/'+'hdp')  #corpus,T,K,batchSize,kappa,alpha,gamma,eta,modelFile




