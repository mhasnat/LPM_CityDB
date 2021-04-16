import numpy as np
from difflib import SequenceMatcher
#import stringdist

def matchStringList(listLP):
    tScore = np.zeros((len(listLP), len(listLP)), dtype=np.float32)

    for ii in range(len(listLP)):
        for jj in range(ii, len(listLP)):
            if((len(listLP[ii])==0) or (len(listLP[jj])==0)):
                tScore[ii, jj] = 0.00001
                tScore[jj, ii] = 0.00001
            else:    
                #print((listLP[ii], listLP[jj]))
                tScore[ii, jj] = SequenceMatcher(None, str(listLP[ii]), str(listLP[jj])).ratio()
                #tScore[ii, jj] = stringdist.levenshtein_norm(str(listLP[ii][0]), str(listLP[jj][0]))
                tScore[jj, ii] = tScore[ii, jj]            
            
    return tScore

def get_similar_vehicles_fast(pdistArr, cIndx, thVal):
    tDists = pdistArr[cIndx, :]
    simIndx = np.where(tDists < thVal)[0]
    sortedIndx = simIndx[np.argsort(tDists[simIndx])]
    
    return sortedIndx    