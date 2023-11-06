topN=10
rm -r model/model-arxiv
rm -r infer/infer-arxiv
rm -f results/resultsAllArxiv.txt
mkdir -p model
mkdir -p infer
mkdir -p results
mkdir -p model/model-arxiv
mkdir -p infer/infer-arxiv
mkdir -p input/arxiv-corpus
#chmod 777 src/bin/btm

# *** preprocess ***
# java -jar src/bin/tika-server-1.26.jar & #Tika server must be run to extract the pdf text content
#python3 src/preprocess/pdfToTextTika.py input/arxiv-sample input/arxiv-sample-txt  #inDir,outDir
python3 src/preprocess/createCorpusFromDir.py input/arxiv-sample-txt input/arxiv-corpus/arxiv-sampleOri.txt input/arxiv-corpus/arxivFileNames.txt 0 #inputDir,outFile,fileNamesFile,isContainDot
#python3 src/preprocess/removeStop.py input/arxiv-corpus/arxiv-sampleOri.txt input/arxiv-corpus/arxiv-sample.txt #infile,outfile (optional,only remove stopwords)
python3  src/preprocess/stemStopTextFile.py input/arxiv-corpus/arxiv-sampleOri.txt input/arxiv-corpus/arxiv-sample.txt 1 10 #infile,outfile,isMultiThread,processCnt (remove stopwords and stemming)
rm -f input/arxiv-corpus/arxiv-sampleOri.txt #optional

# *** train ***
python3 trainAll.py input/arxiv-corpus/arxiv-sample.txt model/model-arxiv src/conf/arxiv/configTrainRec.json #corpusFile,modelDir,configDir

# *** infer ***
python3 inferAll.py input/arxiv-corpus/arxiv-sample.txt model/model-arxiv infer/infer-arxiv src/conf/arxiv/configInferRec.json #infile,modelDir,inferDir,conf
python3 src/util/addPunctionWindow.py input/arxiv-corpus/arxiv-sample.txt input/arxiv-corpus/arxiv-samplePunct.txt # infile,outfile (for sentence tokenize)
python3 src/util/getImportantSentencesMTFile.py input/arxiv-corpus/arxiv-samplePunct.txt input/arxiv-corpus sumBasic 60 10 1 1 #infile,outDir,algorithm,sentCnt,processCnt,afterPreProcess,isAfterSentDots
python3 inferAll.py input/arxiv-corpus/sumBasic.txt model/model-arxiv infer/infer-arxiv src/conf/arxiv/configInferHdp.json #infile,modelDir,inferDir,conf
python3 src/util/createLabels.py infer/infer-arxiv/hdpSparse.txt infer/infer-arxiv/labelsHdp.txt #inferSparsefile,labelsFile
rm -f input/arxiv-corpus/sumBasic.txt

# *** test files create ***
head -1000 input/arxiv-corpus/arxiv-samplePunct.txt > input/arxiv-corpus/testSampleArxiv.txt
head -1000 input/arxiv-corpus/arxivFileNames.txt > input/arxiv-corpus/testArxivFileNames.txt
rm -f input/arxiv-corpus/arxiv-samplePunct.txt 
mkdir -p input/importantSentArxiv-40

python3 src/util/getImportantSentencesMTFile.py input/arxiv-corpus/testSampleArxiv.txt input/importantSentArxiv-40 lexRank 40 10 1 1 #infile,outDir,algorithm,sentCnt,processCnt,afterPreProcess,isAfterSentDots
python3 src/util/getImportantSentencesMTFile.py input/arxiv-corpus/testSampleArxiv.txt input/importantSentArxiv-40 luhn 40 10 1 1 #infile,outDir,algorithm,sentCnt,processCnt,afterPreProcess,isAfterSentDots

# *** test and get accuracy of algorithms ***
# modelDir,inferDir,confFile,fileIdsFile,testFile,testFileIdsFile,resultsFile,topN
python3 testAll.py model/model-arxiv infer/infer-arxiv src/conf/arxiv/configTestRec.json input/arxiv-corpus/arxivFileNames.txt input/importantSentArxiv-40/lexRank.txt input/arxiv-corpus/arxivFileNames.txt results/resultsAllArxiv.txt ${topN}
python3 testAll.py model/model-arxiv infer/infer-arxiv src/conf/arxiv/configTestRec.json input/arxiv-corpus/arxivFileNames.txt input/importantSentArxiv-40/luhn.txt input/arxiv-corpus/arxivFileNames.txt results/resultsAllArxiv.txt ${topN} 
python3 testHybridAlg.py src/conf/hybrid/configTestHybrid.json model/model-arxiv/doc2vec-dbow model/model-arxiv/hdp infer/infer-arxiv/doc2vec-dbow.txt  input/arxiv-corpus/arxivFileNames.txt input/importantSentArxiv-40/lexRank.txt input/arxiv-corpus/arxivFileNames.txt infer/infer-arxiv/labelsHdp.txt 0 ${topN} 0.04 100 results/resultsAllArxiv.txt
python3 testHybridAlg.py src/conf/hybrid/configTestHybrid.json model/model-arxiv/doc2vec-dbow model/model-arxiv/hdp infer/infer-arxiv/doc2vec-dbow.txt  input/arxiv-corpus/arxivFileNames.txt input/importantSentArxiv-40/lexRank.txt input/arxiv-corpus/arxivFileNames.txt infer/infer-arxiv/labelsHdp.txt 1 ${topN} 0.01 100 results/resultsAllArxiv.txt
python3 testHybridAlg.py src/conf/hybrid/configTestHybrid.json model/model-arxiv/doc2vec-dbow model/model-arxiv/hdp infer/infer-arxiv/doc2vec-dbow.txt  input/arxiv-corpus/arxivFileNames.txt input/importantSentArxiv-40/lexRank.txt input/arxiv-corpus/arxivFileNames.txt infer/infer-arxiv/labelsHdp.txt 1 ${topN} 0.04 100 results/resultsAllArxiv.txt

# *** semantic web service started ***
python3 searchWebServiceAll.py src/conf/arxiv/configSearchWebServiceRec.json
