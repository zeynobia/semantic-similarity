import sys

def getMaxIndice(str,delim=";"):
  indices=[]
  values=[]
  parts=str.split(")"+delim)
  for token in parts:
     elem=(token.replace(")","").replace("(",""))
     partsElem=elem.split(delim)
     if(len(partsElem)>=2):
       indices.append(int(partsElem[0]))  
       values.append(float(partsElem[1])) 
     else:
       indices.append(-1)
       values.append(0)

  maxIndice=values.index(max(values))
  return indices[maxIndice]


if __name__=="__main__":
   
  infile=sys.argv[1]
  outfile=sys.argv[2]
  infp=open(infile,'r',encoding='utf-8')
  outfp=open(outfile,'w',encoding='utf-8')
  for line in infp:
    text=line.replace('\n','')
    maxIndice=getMaxIndice(text,delim=";")
    outfp.write(str(maxIndice)+'\n')
  outfp.close()