import os



#filesStem=['input/test-stem/rake.txt','input/test-stem/yake.txt','input/test-stem/reduction.txt','input/test-stem/textRank.txt']
filesStem=['input/test-1m/reduction.txt','input/test-1m/luhn.txt','input/test-1m/textRank.txt','input/test-1m/sumBasic.txt']
inferTestDir='infer/inferTest'




testRefFile='input/test-1m/testSample.txt'
inferRefDir='infer'+'/'+'inferRef'
 

for file in filesStem:
    os.system('mkdir -p infer/inferTest')
    os.system('mkdir -p '+inferRefDir)
    os.system('python3 src/test/testCluster.py '+file+' model/model-4m '+'infer/infer-1m/hdpSparse.txt '+'src/conf/configInferHdp64.json hdp infer/inferTest results/resultsAllCluster.txt')
    #os.system('python3 src/test/testClusterExt.py '+file+' model/model-4m '+testRefFile+' src/conf/configInferHdp64.json hdp infer/inferTest results/resultsAllCluster.txt '+str(inferRefDir))
    os.system('rm -rf infer/inferTest')

os.system('rm -rf '+inferRefDir)
for file in filesStem:
    os.system('mkdir -p infer/inferTest')
    os.system('mkdir -p '+inferRefDir)
    os.system('python3 src/test/testCluster.py '+file+' model/model-2m '+'infer/infer-1m-2/hdpSparse.txt '+'src/conf/configInferHdp64.json hdp infer/inferTest results/resultsAllCluster.txt')
    #os.system('python3 src/test/testClusterExt.py '+file+' model/model-2m '+testRefFile+' src/conf/configInferHdp64.json hdp infer/inferTest results/resultsAllCluster.txt '+str(inferRefDir))
    os.system('rm -rf infer/inferTest')


os.system('rm -rf '+inferRefDir)

