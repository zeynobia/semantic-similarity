# -*- coding: utf-8 -*-

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

def listToText(sentences,isPreProcess=True):

    strRes=''
    for elem in sentences:
        strRes+=str(elem)+' '
    if(isPreProcess):
        strRes=preprocess(strRes)
    return strRes

def writeImportantSentToFile(algorithm,document,sentCnt):

    outfp=open(outDir+'/'+algorithm+'ImportantSent.txt','a',encoding='utf-8')
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

if __name__ == "__main__":

    infile=sys.argv[1]
    outDir=sys.argv[2]
    SENTENCES_COUNT = int(sys.argv[3])
    afterPreProcess=int(sys.argv[4])
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

 
    infp=open(infile,'r',encoding='utf-8')
    for line in infp:
      content=line.replace('\n','')
      parser = PlaintextParser.from_string(content, Tokenizer(LANGUAGE))

      pro1=Process(target=writeImportantSentToFile,args=('lexRank',parser.document,SENTENCES_COUNT))
      pro2=Process(target=writeImportantSentToFile,args=('textRank',parser.document,SENTENCES_COUNT))
      pro3=Process(target=writeImportantSentToFile,args=('luhn',parser.document,SENTENCES_COUNT))
      pro4=Process(target=writeImportantSentToFile,args=('sumBasic',parser.document,SENTENCES_COUNT))
      pro5=Process(target=writeImportantSentToFile,args=('reduction',parser.document,SENTENCES_COUNT))

      pro1.start()
      pro2.start()
      pro3.start()
      pro4.start()
      pro5.start()

      pro1.join()
      pro2.join()
      pro3.join()
      pro4.join()
      pro5.join()

