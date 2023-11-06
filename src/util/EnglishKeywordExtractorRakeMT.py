# python3 src/util/EnglishKeywordExtractorRakeMT.py input/importantSent/testPunct.txt input/importantSent 10 10

from rake_nltk import Rake
import re
import sys
import os
from multiprocessing import Process
def preprocess(text):
    text = re.sub(r"[^\w\s]"," ",text)
    textLower = text.lower()
    return ' '.join(textLower.strip().split())

def getPhrasesExtractRake(infile,outfile,numOfPhrases=10):
    
   rake = Rake()
   infp=open(infile,'r',encoding='utf-8')
   outfpRake=open(outfile,'w',encoding='utf-8')
   for line in infp:
     text=line.replace('\n','')
     content=preprocess(text)
     rake.extract_keywords_from_text(content)
     # keyword phrases ranked highest to lowest.
     rakePhrases=rake.get_ranked_phrases()[0:numOfPhrases]
     rakeStr = ' '.join([str(elem) for elem in rakePhrases])
     outfpRake.write(rakeStr+'\n')
   outfpRake.close()




def writeRakePhrasesToFileMT(infile,outDir,numOfPhrases,processCnt=10):
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
    outResFile=outDir+'/'+'rake.txt'
    for i in range(processCnt):
       process= Process(target=getPhrasesExtractRake, args=(splitFiles[i],splitFilesOut[i],numOfPhrases))
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
  numOfPhrases=int(sys.argv[3])
  processCnt=int(sys.argv[4])
  if(os.path.exists(outDir)==False):
     os.system('mkdir '+outDir)
  writeRakePhrasesToFileMT(infile,outDir,numOfPhrases,processCnt)
