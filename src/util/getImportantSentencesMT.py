#python example to infer document vectors from trained doc2vec model
import gensim
import codecs
import sys
import os
import multiprocessing
from multiprocessing import Process
import numpy as np
import re
from threading import Thread
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.reduction import ReductionSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


from memory_profiler import profile

def preprocess(text):
    text = re.sub(r"[^\w\s]"," ",text)
    textLower = text.lower()
    return ' '.join(textLower.strip().split())

def listToText(sentences,isPreProcess=True):

    strRes=''
    for elem in sentences:
        strRes+=str(elem)+' '
    if(isPreProcess):
        strRes=preprocess(strRes)
    return strRes

def split_list(lst, n):
    splitted = []
    for i in reversed(range(1, n + 1)):
        split_point = len(lst)//i
        splitted.append(lst[:split_point])
        lst = lst[split_point:]
    return splitted


def getSum(docs,fileIds,algorithm,sentCnt=10):
   idx=0
   for doc in docs:
      content=doc.replace('\n','')
      parser = PlaintextParser.from_string(content, Tokenizer(LANGUAGE))
      if(algorithm=='lsa'):
        sentences=lsaSummarizer(parser.document,sentCnt)
      elif(algorithm=='lexRank'):
        sentences=lexRankSummarizer(parser.document,sentCnt)
      elif(algorithm=='textRank'):
        sentences=textRankSummarizer(parser.document,sentCnt)
      elif(algorithm=='luhn'):
        sentences=luhnSummarizer(parser.document,sentCnt)
      elif(algorithm=='sumBasic'):
        sentences=sumBasicSummarizer(parser.document,sentCnt)
      elif(algorithm=='reduction'):
        sentences=reductionSummarizer(parser.document,sentCnt)

      strRes=listToText(sentences,afterPreProcess)
      dictDocVectors[fileIds[idx]]=strRes
      idx=idx+1


def writeVector(vectors,label,outfile):
    outfp=open(outfile,"w")
    cnt=0
    for vector in vectors:
        outfp.write(label[cnt]+"\t")
        for i in range(0,len(vector)-1):
           outfp.write(str(round(vector[i],6))+";")
        outfp.write(str(round(vector[len(vector)-1],6))+"\n")
        cnt=cnt+1
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



if __name__ == "__main__":

  infile=sys.argv[1]
  outDir=sys.argv[2]
  SENTENCES_COUNT = int(sys.argv[3])
  afterPreProcess=int(sys.argv[4])
  algorithm=sys.argv[5]
  idsFile=sys.argv[6]
  vec_file=outDir+'/'+str(algorithm)+'.txt'

  if(afterPreProcess==1):
     afterPreProcess=True
  else:
     afterPreProcess=False
       
  if(os.path.exists(outDir)==False):
       os.mkdir(outDir)

  sentCnt=SENTENCES_COUNT
  LANGUAGE = "english"
  stemmer = Stemmer(LANGUAGE)
  stopWords=get_stop_words(LANGUAGE)

  lsaSummarizer = LsaSummarizer(stemmer)
  lsaSummarizer.stop_words = stopWords

  lexRankSummarizer = LexRankSummarizer(stemmer)
  lexRankSummarizer.stop_words = stopWords
    
  textRankSummarizer = TextRankSummarizer(stemmer)
  textRankSummarizer.stop_words = stopWords

  luhnSummarizer = LuhnSummarizer(stemmer)
  luhnSummarizer.stop_words = stopWords

  sumBasicSummarizer = SumBasicSummarizer(stemmer)
  sumBasicSummarizer.stop_words = stopWords

  reductionSummarizer = ReductionSummarizer(stemmer)
  reductionSummarizer.stop_words = stopWords


  processBatch=200 #100-2500
  
  cntThread= multiprocessing.cpu_count()

  #load Random projection model


  infpIds=open(idsFile,"r",encoding="utf-8")
  lineCount=0
  for line in infpIds:
        lineCount= lineCount+1

  batchSize=cntThread* processBatch
  iterCnt=int(  lineCount/batchSize)
  remainCnt=  lineCount-(iterCnt*batchSize)
  outfp = open(vec_file+".tmp", "w",encoding="utf-8")
  manager = multiprocessing.Manager()
  for it in range(iterCnt):
    docs=readSpecificLines(infile,it*batchSize,(it+1)*batchSize)
    fileIds=readSpecificLines(idsFile,it*batchSize,(it+1)*batchSize)
    dictDocVectors= manager.dict()
    docSplit = split_list(docs, cntThread)
    fileIdsSplit=split_list(fileIds, cntThread)

    processes=[]
    for th in range(cntThread):
      processes.append( Process(target= getSum, args=(docSplit[th],fileIdsSplit[th],algorithm,sentCnt)) )
    for th in range(cntThread):
      processes[th].start()
    for th in range(cntThread):
      processes[th].join()
  
    for elem in dictDocVectors:
      tmpStr=dictDocVectors[elem]
      outfp.write(elem+'\t'+tmpStr+"\n")
          

  docs=readSpecificLines(infile,iterCnt*batchSize,lineCount)
  fileIds=readSpecificLines(idsFile,iterCnt*batchSize,lineCount)
  if(len(fileIds)>0):
     processes=[]
     dictDocVectors= manager.dict()
     processes.append( Process(target= getSum, args=(docs,fileIds,algorithm,sentCnt)) )
     processes[0].start()
     processes[0].join()
     for elem in dictDocVectors:
         tmpStr=dictDocVectors[elem]
         outfp.write(elem+'\t'+tmpStr+"\n")


  outfp.close()
  del dictDocVectors
  os.system("sort -n "+vec_file+".tmp"+" > "+vec_file+".sort")
  infp=open(vec_file+".sort","r",encoding="utf-8")
  outfp=open(vec_file,"w",encoding="utf-8")
  for line in infp:
       text=line.replace("\n","")
       parts=line.split("\t")
       if(len(parts)>=3):
          outfp.write(parts[2])

  outfp.close()   
  os.system("rm -f "+vec_file+".tmp")
  os.system("rm -f "+vec_file+".sort") #attention control after
  
