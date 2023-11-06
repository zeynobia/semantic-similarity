from rake_nltk import Rake
import re
import sys
import os

def preprocess(text):
    text = re.sub(r"[^\w\s]"," ",text)
    textLower = text.lower()
    return ' '.join(textLower.strip().split())

def getPhrasesExtractRake(text,numOfPhrases=10):
    
   rake = Rake()
   rake.extract_keywords_from_text(text)
   # keyword phrases ranked highest to lowest.
   rakePhrases=rake.get_ranked_phrases()[0:numOfPhrases]
   rakeStr = ' '.join([str(elem) for elem in rakePhrases])
   return rakeStr

if __name__ == "__main__":

  infile=sys.argv[1]
  outDir=sys.argv[2]
  numOfPhrases_=int(sys.argv[3])
  
  if(os.path.exists(outDir)==False):
     os.system('mkdir '+outDir)

  infp=open(infile,'r',encoding='utf-8')
  outfpRake=open(outDir+'/'+'rake.txt','w',encoding='utf-8')
  for line in infp:
     textLine=line.replace('\n','')
     content=preprocess(textLine)
     rakeStr=getPhrasesExtractRake(content,numOfPhrases=numOfPhrases_)
     outfpRake.write(rakeStr+'\n')

  outfpRake.close()  