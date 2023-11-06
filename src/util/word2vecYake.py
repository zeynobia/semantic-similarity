import gensim
import sys


if __name__=="__main__":


  model_file=sys.argv[1]
  infile=sys.argv[2]
  outfile=sys.argv[3] # yakeWord2vec
  outfileSet=infile+'.set'
  topN=1
  infp=open(infile,'r',encoding='utf-8')
  outfp=open(outfile,'w',encoding='utf-8')
  outfpSet=open(outfileSet,'w',encoding='utf-8')
  model =gensim.models.Word2Vec.load(model_file)
  print('word2vec model load...')
  idx=0
  for elem in infp:
    text=elem.replace('\n','')
    #print(idx)
    idx=idx+1
    textSet=set(text.split())
    textSetStr=''
    textStr=''
    for setElem in textSet:
      try:
        textSetStr+=str(setElem)+' '
        arraySim=model.wv.most_similar(setElem, topn=topN)
        #print(setElem,' sim: ',arraySim)
        for simKey in arraySim:
          textStr+=str(simKey[0])+' '
      except:
        textStr+=str(setElem)+' '

    outfp.write(textStr+'\n')
    outfpSet.write(textSetStr+'\n')

  outfp.close()
  outfpSet.close()
 