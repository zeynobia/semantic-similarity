# -*- coding: utf-8 -*-
# python3 src/util/getImportantSentencesMTFile.py input/importantSent/testPunct.txt input/importantSent sumBasic 10 10 1 1
# python3 src/util/getImportantSentencesMTFile.py input/importantSent/testPunct.txt input/importantSent luhn 10 10 1 1
# python3 src/util/getImportantSentencesMTFile.py input/importantSent/testPunct.txt input/importantSent reduction 10 10 1 1
# python3 src/util/getImportantSentencesMTFile.py input/importantSent/testPunct.txt input/importantSent lexRank 10 10 1 1
#  python3 src/util/getImportantSentencesMTFile.py input/importantSent/testPunct.txt input/importantSent textrank 10 10 1 1
#  python3 src/util/getImportantSentencesMTFile.py input/importantSent/testPunct.txt input/importantSent lsa 10 10 1 1

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
import re
import sys
import os
from multiprocessing import Process

def preprocess(text):
    text = re.sub(r"[^\w\s]"," ",text)
    textLower = text.lower()
    return ' '.join(textLower.strip().split())

def preprocessExceptDot(text):
    text = re.sub(r"[^\w\s.]"," ",text)
    textLower = text.lower()
    return ' '.join(textLower.strip().split())

def listToText(sentences,isPreProcess=True):

    strRes=''
    for elem in sentences:
        strRes+=str(elem)+' '
    if(isPreProcess):
      if(isAfterSentDots==1):
        strRes=preprocessExceptDot(strRes)
      else:
        strRes=preprocess(strRes)
    return strRes

def writeImportantSentToFile(infile,outFile,algorithm,sentCnt):

    outfp=open(outFile,'w',encoding='utf-8')

    infp=open(infile,'r',encoding='utf-8')
    for line in infp:
      content=line.replace('\n','')
      parser = PlaintextParser.from_string(content, Tokenizer(LANGUAGE))
      document=parser.document
     
      if(algorithm=='lsa'):
        sentences=lsaSummarizer(document,sentCnt)
      elif(algorithm=='lexRank'):
        sentences=lexRankSummarizer(document,sentCnt)
      elif(algorithm=='textRank'):
        sentences=textRankSummarizer(document,sentCnt)
      elif(algorithm=='luhn'):
        sentences=luhnSummarizer(document,sentCnt)
      elif(algorithm=='sumBasic'):
        sentences=sumBasicSummarizer(document,sentCnt)
      elif(algorithm=='reduction'):
        sentences=reductionSummarizer(document,sentCnt)
      strRes=listToText(sentences,afterPreProcess)
      outfp.write(strRes+'\n')
    outfp.close()



def writeImportantSentToFileMT(infile,outDir,algorithm,sentCnt,processCnt=10):
    
    tmpDirName='tmpSplitOut'
    os.system('mkdir -p '+tmpDirName)
    splitFiles=[]
    splitFilesOut=[]
    for i in range(processCnt):
        if(i<10):
          suffix='0'+str(i)
        else:
          suffix=str(i)
        splitFiles.append(tmpDirName+'/'+suffix)
        splitFilesOut.append(outDir+'/'+suffix+'.'+tmpDirName)

    infp=open(infile,'r',encoding='utf-8')
    lineCnt=0
    for line in infp:
        lineCnt=lineCnt+1

    listAll=range(lineCnt)
    splitCnt=int(lineCnt/processCnt)+1
    os.system('split '+infile+' -l '+str(splitCnt)+' -d '+tmpDirName+'/')  # split command preinstall system

    processList=[]
    outResFile=outDir+'/'+str(algorithm)+'.txt'
    for i in range(processCnt):
       process= Process(target=writeImportantSentToFile, args=(splitFiles[i],splitFilesOut[i],algorithm,sentCnt))
       processList.append(process)
    for i in range(processCnt):
       processList[i].start()
    for i in range(processCnt):
       processList[i].join()
    strCmd='cat '
    for i in range(processCnt):
        strCmd+=splitFilesOut[i]+' '
    strCmd+='> '+outResFile
    os.system(strCmd)
    os.system('rm -rf '+tmpDirName)
    for elem in splitFilesOut:
        os.system('rm -f '+elem)


if __name__ == "__main__":

    infile=sys.argv[1]
    outDir=sys.argv[2]
    algorithm=sys.argv[3]
    sentCnt = int(sys.argv[4])
    processCnt=int(sys.argv[5])
    afterPreProcess=int(sys.argv[6])
    isAfterSentDots=int(sys.argv[7])

    if(afterPreProcess==1):
       afterPreProcess=True
    else:
       afterPreProcess=False
       
    if(os.path.exists(outDir)==False):
       os.mkdir(outDir)

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

    writeImportantSentToFileMT(infile,outDir,algorithm,sentCnt,processCnt)
 
   