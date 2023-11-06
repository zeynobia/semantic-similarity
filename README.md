# semantic-similarity

This document contains summary explanations related to the steps concerning demo projects.
Installation
Firstly, the application has been developed on the Linux operating system, in the Ubuntu distribution. The application is written in Python 3. For some important algorithms, the Gensim 3.8 Python library has been used. The search algorithm is written in C++ and is called from Python.

chmod 777 install.sh
./install.sh

Installation of the necessary Python libraries must be completed. If you want to extract text content from pdf texts and use the java library tika-server for this purpose, java dependencies must be installed. The scripts runArxivSampleDemo.sh with 1000 pdf samples and runHybridAlgWikiDemo.sh for the demo sample with 1,000,000 wikipedia samples can be examined and run, and minor edits can be made as needed.

Pre-processing Operations

The following piece of code can be used for simple preprocessing operations. The first parameter is selected as noisy corpus, the second parameter is selected as pre-processed corpus and 0 if the dot character will be removed, and the third parameter is selected as 1 if the dot character will not be removed.

python3 src/preprocess/preprocess.py input/corpus-noisy.txt input/corpus.txt 0

The code snippet can be edited for the example that extracts PDF text content. First of all, tika-server running on port 9998 should be started. PDF contents are written into separate files. Finally, txt files are combined to create a corpus with document content and preprocessing in each line. For both size and performance, stop words can be removed and stemming can be performed by running the stemStopTextFile.py script.

java -jar src/bin/tika-server-1.26.jar &
python3 src/preprocess/pdfToTextTika.py input/arxiv-sample input/arxiv-sample-txt  # inDir, outDir
python3 src/preprocess/createCorpusFromDir.py input/arxiv-sample-txt input/arxiv-corpus/arxivsampleOri.txt input/arxiv-corpus/arxivFileNames.txt 0 # inputDir, outFile, fileNamesFile, isContainDot

python3 src/preprocess/stemStopTextFile.py input/arxiv-corpus/arxiv-sampleOri.txt input/arxivcorpus/arxiv-sample.txt 1 10 #infile,outfile,isMultiThread,processCnt


EducationPhase

3 parameters are required for the training phase. Corpus txt file containing the content of the documents, the directory where the model files will be saved, and the training configuration file. These configuration files include which algorithms will be used, feature vector numbers of the algorithms, iteration numbers, and directories of some output files. Previously prepared configuration files can be used. For example, src/conf/hybrid configuration files can be used for the hybrid algorithm. First of all, a hybrid algorithm containing Doc2vec-dbow and HDP models can be recommended. Additionally, if different algorithms are wanted to be tested, Doc2vec-dbpv, Lsa, Rp models can be used in terms of success and running times.

python3 trainAll.py input/corpus.txt model src/conf/hybrid/configTrainHybrid.json #corpusFile,modelDir,configDir (for same corpus)


Inference Phase

This is the stage where the semantic document vector is created after the models are trained. 4 parameters are required. These parameters are text files containing the contents of the documents to be searched, model directory, inference directory and configuration files.

python3 inferAll.py input/corpus.txt model infer src/conf/hybrid/configInferDoc2vecDbow.json #infile,modelDir,inferDir,conf


Testing Phase

Preparation of Test Documents

These parameters are the test sample file, the directory to be saved, the algorithm that will extract important sentences, the number of sentences, the number of threads, the pre-processing process and whether there will be a period at the end of the sentence. When determining the number of sentences, it would be better to give the value according to the length of the texts. For example, while there may be around 20 for wikipedia, there may be around 50 for long article texts such as arxiv.

python3 src/util/getImportantSentencesMTFile.py input/testSample.txt input/importantSent-24 lexRank 24 10 1 1 #infile,outDir,algorithm,sentCnt,processCnt,afterPreProcess,isAfterSentDots

The testAll.py script is determined by the model Directory, the infer Directory, the test configuration indicating which algorithms will be tested, the infer document id or id file containing the names, the test txt file automatically created with the algorithms, the test id file, the file in which the results will be saved and the topN parameter. TopN can be determined between 10 and 2000, depending on the number of documents. Around 50 can be determined for 100,000 documents, around 200 for 1,000,000 documents, and around 1000 for 10,000,000 documents.

#modelDir,inferDir,confFile,fileIdsFile,testFile,testFileIdsFile,resultsFile, 
topN 
python3 testAll.py model/model-arxiv infer/infer-arxiv src/conf/arxiv/configTestRec.json input/arxiv-corpus/arxivFileNames.txt input/arxiv-corpus/testSampleArxivNoStop.txt input/arxivcorpus/arxivFileNames.txt results/resultsAllArxiv.txt ${topN}

testHybridAlg.py script includes the configuration file, semantic algorithm, cluster algorithm, infer index, infer document ids, the first txt file automatically created with algorithms, the id file of the test documents, the automatically created tag file specifying the cluster groups, whether to perform a quick search or not, the topN parameter. , minProb parameter that eliminates the minimum possibilities, clusterCnt parameter that specifies the number of clusters to be searched, and the result file where the results will be written.

#confFile,modelSemantic,modelCluster,inferSemantic,fileIds,testFile,testFileIds,labelsFile,isSmart 
Search,topN,minProb,clusterCnt,resultsFile 
python3 testHybridAlg.py src/conf/hybrid/configTestHybrid.json model/doc2vec-dbow model/hdp infer/doc2vec-dbow.txt input/fileIds.txt input/importantSent-24/lexRank.txt input/testFileIds.txt infer/labelsHdp.txt 0 ${topN} 0.01 100 results/resultsAllWiki.txt


Web Service

It only needs the configuration file. In the configuration file, make sure that the model, infer and fileIds are correct. Default parameters can also be used. The default parameters are model for the model directory, infer for the infer directory, and fileIds.txt for document ids. As a search algorithm, the default search algorithm is the hybrid algorithm, which has a much shorter search time, especially for large data.

python3 searchWebServiceAll.py

By default, it works on port 5555. By typing http://127.0.0.1:5555 in the browser, it is possible to send long text content or short text as in the example below and select the algorithm. It is also possible to make a request with the curl command. TopN parameter is set to 25. If necessary, the topN parameter can be adjusted in the web service configuration file. The default algorithm is fastSmartSearch, which is a hybrid algorithm that uses doc2vec-dbow and hdp semantic algorithms.

curl -F content='active passive learning' http://127.0.0.1:5555/getSemanticSimilarity
curl -F content='active passive learning' -F algorithm='fastSmartSearch' http://127.0.0.1:5555/getSemanticSimilarity
curl -F content='active passive learning' -F algorithm='doc2vec-dbow' http://127.0.0.1:5555/getSemanticSimilarity
curl -F content='active passive learning' -F algorithm='doc2vec-dbpv' http://127.0.0.1:5555/getSemanticSimilarity 
curl -F content='active passive learning' -F algorithm='lsa' http://127.0.0.1:5555/getSemanticSimilarity
curl -F content='active passive learning' -F algorithm='rp' http://127.0.0.1:5555/getSemanticSimilarity

Although there are algorithms that are not recommended in terms of processing power and performance, training can be done on lda, nmf, plsa, minHash, btm models. In this case, requests can be sent to the web service with the curl command as follows.

curl -F content='active passive learning' -F algorithm='lda' http://127.0.0.1:5555/getSemanticSimilarity
curl -F content='active passive learning' -F algorithm='nmf' http://127.0.0.1:5555/getSemanticSimilarity
curl -F content='active passive learning' -F algorithm='plsa' http://127.0.0.1:5555/getSemanticSimilarity
curl -F content='active passive learning' -F algorithm='minHash' http://127.0.0.1:5555/getSemanticSimilarity
curl -F content='active passive learning' -F algorithm='btm' http://127.0.0.1:5555/getSemanticSimilarity
