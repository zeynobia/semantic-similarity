# python3 testCluster.py input/rake.txt model input/testSample.txt src/conf/configInferExtFast.json hdp inferTest results/resultsAllCluster.txt infer/inferRef

import os
import sys
import numpy as np
import collections

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
  return indices,values

def convertSparseStrToTuple(strTuple):
   indices,values=getIndiceValue(strTuple,delim=";")
  
   tmpTuples=[]
   for idx in range(len(indices)):
      tmpTuples.append((indices[idx],values[idx]))
   return tmpTuples
# first ids...
def demo(labels,inferTestFile,th1=0.90):

  outfp=open(resultFile,'a',encoding='utf-8')
  infpTest=open(inferTestFile,'r',encoding='utf-8')
  idx=0
  correct=0
  correct2=0
  correct3=0
  correctTh=0
  correctTh08=0
  correctTh07=0
  totalTh=0
  totalTh08=0
  totalTh07=0

  for line in infpTest:
    text=line.replace('\n','')
    
    inferVector=convertSparseStrToTuple(text)
    sortedFeature=sorted(inferVector, key=lambda x: x[1],reverse=True)
    predictProb=0
    predictTag=0
    if(len(sortedFeature)>=1 ):
      predictProb=np.float32(sortedFeature[0][1])
      predictTag=sortedFeature[0][0]
      label2=[]
      if(len(sortedFeature)>=2):
        label2=[sortedFeature[0][0],sortedFeature[1][0]]
      else:
        label2=[sortedFeature[0][0]]
  
      if(len(sortedFeature)>=3):
        label3=[sortedFeature[0][0],sortedFeature[1][0],sortedFeature[2][0]]
      else:
        label3=label2

      if (int(predictTag)==int(labels[idx])):
        correct=correct+1
      if(int(labels[idx]) in label2):
        correct2=correct2+1
      if(int(labels[idx]) in label3):
        correct3=correct3+1

      if(predictProb>th1):
        totalTh=totalTh+1
        if (int(predictTag)==int(labels[idx])):
          correctTh=correctTh+1

      if(predictProb>0.80):
        totalTh08=totalTh08+1
        if (int(predictTag)==int(labels[idx])):
          correctTh08=correctTh08+1

      if(predictProb>0.70):
        totalTh07=totalTh07+1
        if (int(predictTag)==int(labels[idx])):
          correctTh07=correctTh07+1


      print('idx: ',idx,' predict: ',predictTag,' label: ',labels[idx],' prob: ',predictProb)
    idx=idx+1

  acc1=round(100*correct/idx,2)
  acc2=round(100*correct2/idx,2)
  acc3=round(100*correct3/idx,2)
  accTh=round(100*correctTh/totalTh,2)
  accTh08=round(100*correctTh08/totalTh08,2)
  accTh07=round(100*correctTh07/totalTh07,2)
  print('Correct 1 label: ',correct,' Total: ',idx,' Acc: ',acc1)
  print('Correct 2 label: ',correct2,' Total: ',idx,' Acc: ',acc2)
  print('Correct 3 label: ',correct3,' Total: ',idx,' Acc: ',acc3)

  print('Th:0.70 Correct: ',correctTh07,' Total: ',totalTh07,' Acc: ',accTh07)
  print('Th:0.80 Correct: ',correctTh08,' Total: ',totalTh08,' Acc: ',accTh08)
  print('Th:0.90 Correct: ',correctTh,' Total: ',totalTh,' Acc: ',accTh)

  outfp.write('TestFile: '+testFile+' Algorithm: '+str(algorithm)+' Total: '+str(idx)+' Correct1: '+str(correct)+' Acc1: '+str(acc1)+' Correct2: '+str(correct2)+' Acc2: '+str(acc2)+' Correct3: '+str(correct3)+' Acc3: '+str(acc3)+' Th0.90: '+str(correctTh)+' AccTh0.9: '+str(accTh)+' Th0.80: '+str(correctTh08)+' AccTh0.8: '+str(accTh08)+' Th0.70: '+str(correctTh07)+ ' AccTh0.7: '+str(accTh07)+'\n')
  outfp.close()
   
if __name__=="__main__":
  testFile=sys.argv[1]
  modelDir=sys.argv[2]
  inferRefFile=sys.argv[3]
  inferConfFile=sys.argv[4]
  algorithm=sys.argv[5]
  inferTestDir=sys.argv[6]
  resultFile=sys.argv[7]
  inferRefDir=sys.argv[8]

  

  
  inferRefResultFile=inferRefDir+'/'+str(algorithm)+'Sparse.txt'
  inferTestFile=inferTestDir+'/'+str(algorithm)+'Sparse.txt'
  if(os.path.exists(inferRefResultFile)==False):
    os.system('python3 src/train/createCorpusByteFile.py '+inferRefFile+' '+'input/corpusByte.txt')
    os.system('python3 inferAll.py '+inferRefFile+' '+modelDir+' '+inferRefDir+' '+inferConfFile)


  labelsFile=inferTestDir+'/'+'labels-'+str(algorithm)+'.txt'

  os.system('python3 src/util/createLabels.py '+inferRefResultFile+' '+labelsFile)
  labels=[]
  infpLabel=open(labelsFile,'r',encoding='utf-8')
  for line in infpLabel:
    text=line.replace('\n','')
    labels.append(text)
  counter=collections.Counter(labels)
  
  os.system('python3 src/train/createCorpusByteFile.py '+testFile+' '+'input/corpusByte.txt')
  os.system('python3 inferAll.py '+testFile+' '+modelDir+' '+inferTestDir+' '+inferConfFile)
 
  demo(labels,inferTestFile,th1=0.90)
  print(counter)
