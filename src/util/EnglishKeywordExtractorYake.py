from rake_nltk import Rake
import yake
import re
import sys
import os

def preprocess(text):
    text = re.sub(r"[^\w\s]"," ",text)
    textLower = text.lower()
    return ' '.join(textLower.strip().split())

def getPhrasesExtractYake(text,stopWords,max_ngram_size=2,numOfPhrases=10):
   
   rake = Rake()
   language = "en"
   yake_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size,top=numOfPhrases, features=None,stopwords=stopWords)
   yakePhrasesWithScores= yake_kw_extractor.extract_keywords(text)

   if(len(yakePhrasesWithScores)==0):
      rake.extract_keywords_from_text(text)
      # keyword phrases ranked highest to lowest.
      rakePhrases=rake.get_ranked_phrases()[0:numOfPhrases]
      rakeStr = ' '.join([str(elem) for elem in rakePhrases])
      yakeStr=rakeStr
   else:
      yakePhrases=[elem for elem,score in  yakePhrasesWithScores ]
      yakeStr = ' '.join([str(elem) for elem in yakePhrases])
  
   return yakeStr

if __name__ == "__main__":

  infile=sys.argv[1]
  outDir=sys.argv[2]
  numOfPhrases_=int(sys.argv[3])
  
  if(os.path.exists(outDir)==False):
     os.system('mkdir '+outDir)
  max_ngram_size_=2
  stopWordsFile='input/stopwords_en.txt'
  infpStopWords=open(stopWordsFile,'r',encoding='utf-8')
  stopWords=[]
  for line in infpStopWords:
      stopWords.append(line.replace('\n',''))

  infp=open(infile,'r',encoding='utf-8')
  outfpYake=open(outDir+'/'+'yake.txt','w',encoding='utf-8')
  for line in infp:
     textLine=line.replace('\n','')
     content=preprocess(textLine)
     yakeStr=getPhrasesExtractYake(content,stopWords,max_ngram_size_,numOfPhrases=numOfPhrases_)
     outfpYake.write(yakeStr+'\n')

  outfpYake.close()  