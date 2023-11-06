# python3 src/util/getRandomSent.py input/test-ori/testPunct.txt input/test-ori/randomSent.txt 50 1
import random
import sys
import re

def preprocess(text):
    text = re.sub(r"[^\w\s]"," ",text)
    textLower = text.lower()
    return ' '.join(textLower.strip().split())

def preprocessExceptDot(text):
    text = re.sub(r"[^\w\s.]"," ",text)
    textLower = text.lower()
    return ' '.join(textLower.strip().split())


infile=sys.argv[1]
outfile=sys.argv[2]
sentCnt=int(sys.argv[3])



infp=open(infile,'r',encoding='utf-8')
outfp=open(outfile,'w',encoding='utf-8')

for line in infp:
   text=line.replace('\n','')
   parts=text.split('.')
   if(len(parts)>=sentCnt):
     randTmp=random.sample(range(0, len(parts)), sentCnt)
   else:
     randTmp=random.sample(range(0, len(parts)), len(parts))
   strRes=''
   for elem in randTmp:
      strRes+=parts[elem]+' . '
   strRes=preprocessExceptDot(strRes).replace(' . . ',' . ')
   if(strRes[0:3]==' . '):
      srtRes=strRes[3:]
   outfp.write(strRes+'\n')
outfp.close()
   