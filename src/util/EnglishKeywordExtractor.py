from summa import keywords
import yake
from rake_nltk import Rake
from keybert import KeyBERT
from topicrank import TopicRank
import re
import sys
import os

def preprocess(text):
    text = re.sub(r"[^\w\s]"," ",text)
    textLower = text.lower()
    return ' '.join(textLower.strip().split())

def getPhrasesExtractAllAlg(text,stopWords,keyBertModel=None,numOfPhrases=10,max_ngram_size=2,isKeyBertExtract=False):
    
   rake = Rake()
   rake.extract_keywords_from_text(text)
   # keyword phrases ranked highest to lowest.
   rakePhrases=rake.get_ranked_phrases()[0:numOfPhrases]
   rakeStr = ' '.join([str(elem) for elem in rakePhrases])

   textRankPhrases=keywords.keywords(text,ratio=0.20).split('\n')[0:numOfPhrases]
   textRankStr = ' '.join([str(elem) for elem in textRankPhrases])

   language = "en"
   yake_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size,top=numOfPhrases, features=None,stopwords=stopWords)
   yakePhrasesWithScores= yake_kw_extractor.extract_keywords(text)
   if(len(yakePhrasesWithScores)==0):
      yakeStr=rakeStr
   else:
      yakePhrases=[elem for elem,score in  yakePhrasesWithScores ]
      yakeStr = ' '.join([str(elem) for elem in yakePhrases])

   try:
      topicRank = TopicRank(text)
      topicRankPhrases=topicRank.get_top_n(n=numOfPhrases)
      topicRankStr = ' '.join([str(elem) for elem in topicRankPhrases])
   except:
      topicRankStr=rakeStr


   if(isKeyBertExtract):
     keyBertPhrsesWithScores=keyBertModel.extract_keywords(text, keyphrase_ngram_range=(1, max_ngram_size), stop_words='english')
     keyBertPhrases=[elem for elem,score in keyBertPhrsesWithScores]
     keyBertStr=' '.join([str(elem) for elem in keyBertPhrases])
     return yakeStr,textRankStr,rakeStr,topicRankStr,keyBertStr
   
   
   return yakeStr,textRankStr,rakeStr,topicRankStr

if __name__ == "__main__":

  infile=sys.argv[1]
  outDir=sys.argv[2]
  numOfPhrases_=int(sys.argv[3])
  isKeyBertExtract=int(sys.argv[4])

  if(isKeyBertExtract==1):
    isKeyBertExtract=True
  else:
    isKeyBertExtract=False

  
  max_ngram_size_=2
  stopWordsFile='input/stopwords_en.txt'
  infpStopWords=open(stopWordsFile,'r',encoding='utf-8')
  stopWords=[]
  for line in infpStopWords:
      stopWords.append(line.replace('\n',''))
  
  if(isKeyBertExtract):
     keyBertModel= KeyBERT('distilbert-base-nli-mean-tokens')
  else:
     keyBertModel=None
  
  os.makedirs(outDir, exist_ok = True)
  
  infp=open(infile,'r',encoding='utf-8')
  outfpYake=open(outDir+'/'+'yake.txt','w',encoding='utf-8')
  outfpTextRank=open(outDir+'/'+'textRank.txt','w',encoding='utf-8')
  outfpRake=open(outDir+'/'+'rake.txt','w',encoding='utf-8')
  outfpTopicRank=open(outDir+'/'+'topicRank.txt','w',encoding='utf-8')
  if(isKeyBertExtract):
     outfpKeyBert=open(outDir+'/'+'keyBert.txt','w',encoding='utf-8')

  for line in infp:
     textLine=line.replace('\n','')
     content=preprocess(textLine)
     if(isKeyBertExtract):
        yakeStr,textRankStr,rakeStr,topicRankStr,keyBertStr=getPhrasesExtractAllAlg(content,stopWords,keyBertModel,numOfPhrases=numOfPhrases_,max_ngram_size=max_ngram_size_,isKeyBertExtract=True)
     else:
        yakeStr,textRankStr,rakeStr,topicRankStr=getPhrasesExtractAllAlg(content,stopWords,keyBertModel,numOfPhrases=numOfPhrases_,max_ngram_size=max_ngram_size_,isKeyBertExtract=False)

     outfpYake.write(yakeStr+'\n')
     outfpTextRank.write(textRankStr+'\n')
     outfpRake.write(rakeStr+'\n')
     outfpTopicRank.write(topicRankStr+'\n')
     if(isKeyBertExtract):
        outfpKeyBert.write(keyBertStr+'\n')
     
  outfpYake.close()
  outfpTextRank.close()
  outfpRake.close()
  outfpTopicRank.close()
  if(isKeyBertExtract):
    outfpKeyBert.close()
  
  