import re
import sys

infile=sys.argv[1]
outfile=sys.argv[2]

infp=open(infile,'r',encoding='utf-8')
outfp=open(outfile,'w',encoding='utf-8')
idx=0
window=12
punc='. '
for line in infp:
    text=line.replace('\n','')
    
    #words=text.split()
    words=re.split('\s+',text)
    puncText=''
    wordCnt=0
    
    for word in words:
        puncText=puncText+word+' '
        if(wordCnt!=0 and wordCnt%window==0):
           puncText+=punc
        wordCnt+=1
    if(puncText[len(puncText)-2]!='.'):
       puncText+=punc
    outfp.write(puncText+'\n')
    
    idx=idx+1
outfp.close()
