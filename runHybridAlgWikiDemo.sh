topN=250
: '
#rm -r model
#rm -r infer
rm -f results/resultsAllWiki.txt
mkdir -p model
mkdir -p infer
mkdir -p results

# *** preprocess ***
#python3 src/preprocess/preprocess.py input/corpus-noisy.txt input/corpusOri.txt 0 #infile,outfile,isAfterDot (simple preprocess)
# rm -f corpus-noisy.txt #optional
#python3  src/preprocess/stemStopTextFile.py input/corpusOri.txt input/corpus.txt 1 10 #infile,outfile,isMultiThread,processCnt
# rm -f input/corpusOri.txt #optional

# *** train ***
python3 trainAll.py input/corpus.txt model src/conf/hybrid/configTrainHdp.json #corpusFile,modelDir,configDir
python3 trainAll.py input/corpus.txt model src/conf/hybrid/configTrainDoc2vecDbow.json #corpusFile,modelDir,configDir
#python3 trainAll.py input/corpus.txt model src/conf/hybrid/configTrainHybrid.json #corpusFile,modelDir,configDir (for same corpus)

# *** infer ***
python3 inferAll.py input/corpus.txt model infer src/conf/hybrid/configInferDoc2vecDbow.json  #infile,modelDir,inferDir,conf
python3 src/util/addPunctionWindow.py input/corpus.txt input/corpusPunct.txt # infile,outfile (for sentence tokenize)
python3 src/util/getImportantSentencesMTFile.py input/corpusPunct.txt input sumBasic 60 10 1 1 #infile,outDir,algorithm,sentCnt,processCnt,afterPreProcess,isAfterSentDots
python3 inferAll.py input/sumBasic.txt model infer src/conf/hybrid/configInferHdp.json  #infile,modelDir,inferDir,conf
python3 src/util/createLabels.py infer/hdpSparse.txt infer/labelsHdp.txt #inferSparsefile,labelsFile
rm -f input/sumBasic.txt

# *** test files create ***
head -2500 input/corpusPunct.txt > input/testSample.txt
head -2500 input/fileIds.txt > input/testFileIds.txt
rm -f input/corpusPunct.txt

mkdir -p input/importantSent-24
python3 src/util/getImportantSentencesMTFile.py input/testSample.txt input/importantSent-24 lexRank 24 10 1 1  #infile,outDir,algorithm,sentCnt,processCnt,afterPreProcess,isAfterSentDots
python3 src/util/getImportantSentencesMTFile.py input/testSample.txt input/importantSent-24 textRank 24 10 1 1  #infile,outDir,algorithm,sentCnt,processCnt,afterPreProcess,isAfterSentDots
python3 src/util/getImportantSentencesMTFile.py input/testSample.txt input/importantSent-24 lsa 24 10 1 1  #infile,outDir,algorithm,sentCnt,processCnt,afterPreProcess,isAfterSentDots
python3 src/util/getImportantSentencesMTFile.py input/testSample.txt input/importantSent-24 luhn 24 10 1 1  #infile,outDir,algorithm,sentCnt,processCnt,afterPreProcess,isAfterSentDots
 '

# *** test and get accuracy of algorithms ***
#confFile,modelSemantic,modelCluster,inferSemantic,fileIds,testFile,testFileIds,labelsFile,isSmartSearch,topN,minProb,clusterCnt,resultsFile
python3 testHybridAlg.py src/conf/hybrid/configTestHybrid.json model/doc2vec-dbow model/hdp infer/doc2vec-dbow.txt input/fileIds.txt input/importantSent-24/lexRank.txt input/testFileIds.txt infer/labelsHdp.txt 0 ${topN} 0.04 100 results/resultsAllWiki.txt
python3 testHybridAlg.py src/conf/hybrid/configTestHybrid.json model/doc2vec-dbow model/hdp infer/doc2vec-dbow.txt input/fileIds.txt input/importantSent-24/lexRank.txt input/testFileIds.txt infer/labelsHdp.txt 1 ${topN} 0.01 100 results/resultsAllWiki.txt
python3 testHybridAlg.py src/conf/hybrid/configTestHybrid.json model/doc2vec-dbow model/hdp infer/doc2vec-dbow.txt input/fileIds.txt input/importantSent-24/lexRank.txt input/testFileIds.txt infer/labelsHdp.txt 1 ${topN} 0.04 100 results/resultsAllWiki.txt
python3 testHybridAlg.py src/conf/hybrid/configTestHybrid.json model/doc2vec-dbow model/hdp infer/doc2vec-dbow.txt input/fileIds.txt input/importantSent-24/lexRank.txt input/testFileIds.txt infer/labelsHdp.txt 1 ${topN} 0.1 100 results/resultsAllWiki.txt
python3 testHybridAlg.py src/conf/hybrid/configTestHybrid.json model/doc2vec-dbow model/hdp infer/doc2vec-dbow.txt input/fileIds.txt input/importantSent-24/lexRank.txt input/testFileIds.txt infer/labelsHdp.txt 1 ${topN} 0.01 1 results/resultsAllWiki.txt
python3 testHybridAlg.py src/conf/hybrid/configTestHybrid.json model/doc2vec-dbow model/hdp infer/doc2vec-dbow.txt input/fileIds.txt input/importantSent-24/lexRank.txt input/testFileIds.txt infer/labelsHdp.txt 1 ${topN} 0.01 2 results/resultsAllWiki.txt
python3 testHybridAlg.py src/conf/hybrid/configTestHybrid.json model/doc2vec-dbow model/hdp infer/doc2vec-dbow.txt input/fileIds.txt input/importantSent-24/lexRank.txt input/testFileIds.txt infer/labelsHdp.txt 1 ${topN} 0.01 3 results/resultsAllWiki.txt

python3 testHybridAlg.pysrc/conf/hybrid//configTestHybrid.json model/doc2vec-dbow model/hdp infer/doc2vec-dbow.txt input/fileIds.txt input/importantSent-24/textRank.txt input/testFileIds.txt infer/labelsHdp.txt 0 ${topN} 0.04 100 results/resultsAllWiki.txt
python3 testHybridAlg.py src/conf/hybrid/configTestHybrid.json model/doc2vec-dbow model/hdp infer/doc2vec-dbow.txt input/fileIds.txt input/importantSent-24/textRank.txt input/testFileIds.txt infer/labelsHdp.txt 1 ${topN} 0.01 100 results/resultsAllWiki.txt
python3 testHybridAlg.py src/conf/hybrid/configTestHybrid.json model/doc2vec-dbow model/hdp infer/doc2vec-dbow.txt input/fileIds.txt input/importantSent-24/textRank.txt input/testFileIds.txt infer/labelsHdp.txt 1 ${topN} 0.04 100 results/resultsAllWiki.txt

python3 testHybridAlg.py src/conf/hybrid/configTestHybrid.json model/doc2vec-dbow model/hdp infer/doc2vec-dbow.txt input/fileIds.txt input/importantSent-24/lsa.txt input/testFileIds.txt infer/labelsHdp.txt 0 ${topN} 0.04 100 results/resultsAllWiki.txt
python3 testHybridAlg.py src/conf/hybrid/configTestHybrid.json model/doc2vec-dbow model/hdp infer/doc2vec-dbow.txt input/fileIds.txt input/importantSent-24/lsa.txt input/testFileIds.txt infer/labelsHdp.txt 1 ${topN} 0.01 100 results/resultsAllWiki.txt
python3 testHybridAlg.py src/conf/hybrid/configTestHybrid.json model/doc2vec-dbow model/hdp infer/doc2vec-dbow.txt input/fileIds.txt input/importantSent-24/lsa.txt input/testFileIds.txt infer/labelsHdp.txt 1 ${topN} 0.04 100 results/resultsAllWiki.txt

python3 testHybridAlg.py src/conf/hybrid/configTestHybrid.json model/doc2vec-dbow model/hdp infer/doc2vec-dbow.txt input/fileIds.txt input/importantSent-24/luhn.txt input/testFileIds.txt infer/labelsHdp.txt 0 ${topN} 0.04 100 results/resultsAllWiki.txt
python3 testHybridAlg.py src/conf/hybrid/configTestHybrid.json model/doc2vec-dbow model/hdp infer/doc2vec-dbow.txt input/fileIds.txt input/importantSent-24/luhn.txt input/testFileIds.txt infer/labelsHdp.txt 1 ${topN} 0.01 100 results/resultsAllWiki.txt
python3 testHybridAlg.py src/conf/hybrid/configTestHybrid.json model/doc2vec-dbow model/hdp infer/doc2vec-dbow.txt input/fileIds.txt input/importantSent-24/luhn.txt input/testFileIds.txt infer/labelsHdp.txt 1 ${topN} 0.04 100 results/resultsAllWiki.txt

# *** semantic web service started ***
python3 searchWebServiceHybrid.py src/conf/hybrid/configSearchWebServiceHybrid.json
