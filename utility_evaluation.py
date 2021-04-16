import cv2
import numpy as np
from tqdm import tqdm
from utility_matching_functions import *

def get_accuracy_reject_characteristics(pdistArr, lp_text, thVal_list=None):
    ## Compute the Accuracy and Reject-Rates at different threshold values
    
    # Get the index, text and matching-score of the closest license plate in the Database
    tIndx   = np.argmin(pdistArr, axis=1)
    tPredLPText = lp_text[tIndx]    
    tScores = pdistArr[range(pdistArr.shape[0]), tIndx]

    # set the range of threshold to evaluate
    if thVal_list is None:
        thVal_list = np.arange(0.01, max(tScores), 0.02) #0.15

    # Construct the list of Accuracies and Reject Rates by searching in the list of threshold values
    acc = list()
    rejRate = list()    

    for ccnt in range(len(thVal_list)):
        thVal = thVal_list[ccnt]
        
        ###
        rejIndx = np.where(tScores > thVal)[0]
        accIndx = np.where(tScores <= thVal)[0]

        if len(rejIndx) == 0 or len(accIndx) == 0:
            continue

        
        rejLPText = lp_text[rejIndx]
        rejFracSamp = len(rejIndx) / float(len(tScores))
        rejLPMRate = len(np.where(tPredLPText[rejIndx] == lp_text[rejIndx])[0])/float(len(rejIndx))
        rejRate.append(rejFracSamp)


        accLPText = lp_text[accIndx]
        accFracSamp = len(accIndx) / float(len(tScores))
        accLPMRate = len(np.where(tPredLPText[accIndx] == lp_text[accIndx])[0])/float(len(accIndx))
        acc.append(accLPMRate)

    return acc, rejRate

def get_list_tp_fp_fn(pdistArr, lp_text, thVal, verbose=True):
    ## Returns the list of true positives, false positives and true negatives
    gtCnt = 0

    all_tp_list = []
    all_fp_list = []
    all_fn_list = []

    # Get the index, text and matching-score of the closest license plate in the Database
    tIndx   = np.argmin(pdistArr, axis=1)
    tPredLPText = lp_text[tIndx]    
    tScores = pdistArr[range(pdistArr.shape[0]), tIndx]
    
    if verbose:
        print('Compute the counts for a given threshold value ...')

    for ii in tqdm(range(pdistArr.shape[0])):
        tSimIndx = get_similar_vehicles_fast(pdistArr, ii, thVal)
        tSimIndx = np.delete(tSimIndx, np.where(tSimIndx==ii)[0])

        tGTIndx = np.where(lp_text[ii] == lp_text)[0]
        tGTIndx = tGTIndx[tGTIndx!=ii]
        
        # Construct the list of true positives
        t_tp_list = [tt for tt in tSimIndx if tt in tGTIndx]
        if len(t_tp_list)>0:
            for tt in range(len(t_tp_list)):
                all_tp_list.append(np.array([ii, t_tp_list[tt]]))     
                
        # Construct the list of false positives
        t_fp_list = [tt for tt in tSimIndx if tt not in tGTIndx]
        if len(t_fp_list)>0:
            for tt in range(len(t_fp_list)):
                all_fp_list.append(np.array([ii, t_fp_list[tt]]))   

        # Construct the list of false negatives
        t_fn_list = [tt for tt in tGTIndx if tt not in tSimIndx]
        if len(t_fn_list)>0:
            for tt in range(len(t_fn_list)):
                all_fn_list.append(np.array([ii, t_fn_list[tt]]))                  
        
    return all_tp_list, all_fp_list, all_fn_list
        
def get_counts_prf_measures_threshold(pdistArr, lp_text, thVal, verbose=False):
    ## Compute counts related to the Precision, Recall and F-measure for different threshold values            
    gt_l = []
    gt_r = []
    for ii in tqdm(range(pdistArr.shape[0])):
        tGTIndx = np.where(lp_text[ii] == lp_text)[0]
        tGTIndx = np.delete(tGTIndx, np.where(tGTIndx==ii)[0])
        gt_l.extend([ii]*len(tGTIndx))
        gt_r.extend(tGTIndx)
    gtCnt = len(gt_l)
    
    ## Compute the counts for different threshold values 
    tp_l = []
    tp_r = []
    fp_l = []
    fp_r = []
    #thVal = 0.1
    for ii in tqdm(range(pdistArr.shape[0])):
        tSimIndx = get_similar_vehicles_fast(pdistArr, ii, thVal)
        tSimIndx = np.delete(tSimIndx, np.where(tSimIndx==ii)[0])

        if(len(tSimIndx) > 0):
            for jj in range(len(tSimIndx)):
                if(lp_text[ii] == lp_text[tSimIndx[jj]]):
                    tp_l.append(ii)
                    tp_r.append(tSimIndx[jj])
                else:
                    fp_l.append(ii)
                    fp_r.append(tSimIndx[jj])
                     
    ## Different metrices, add them to the list
    trPos = len(tp_l)
    flPos = len(fp_l)    
    flNeg = gtCnt - trPos
    
    return trPos, flPos, flNeg, gtCnt, list(zip(tp_l, tp_r)), list(zip(fp_l, fp_r))

def get_counts_prf_measures(pdistArr, lp_text, thVal_list=None, verbose=False):
    ## Compute counts related to the Precision, Recall and F-measure for different threshold values
    
    # Get the index, text and matching-score of the closest license plate in the Database
    tIndx   = np.argmin(pdistArr, axis=1)
    tPredLPText = lp_text[tIndx]    
    tScores = pdistArr[range(pdistArr.shape[0]), tIndx]
    
    # set the range of threshold to evaluate
    if thVal_list is None:
        thVal_list = np.arange(0.01, max(tScores), 0.01) #0.15
    
    tPcList = list()
    fPcList = list()
    fNcList = list()
    gtcList = list()
    smcList = list()

    if verbose:        
        print('Computing the ground truth counts')
    gtCnt = 0    
    for ii in range(pdistArr.shape[0]):
        tGTIndx = np.where(lp_text[ii] == lp_text)[0]
        gtCnt += (len(tGTIndx) - 1)
        
    ## Compute the counts for different threshold values 
    if verbose:
        print('Compute the counts for different threshold values')
    for kk in range(len(thVal_list)):
        thVal = thVal_list[kk]

        if verbose:
            print('[ ' + str(kk) + ' / ' + str(len(thVal_list)) + ' ] ' + str(thVal))
            
        simCnt = 0
        actCnt = 0
        singlCnt = 0

        #for ii in tqdm(range(pdistArr.shape[0])):
        for ii in range(pdistArr.shape[0]):
            tSimIndx = get_similar_vehicles_fast(pdistArr, ii, thVal)
            tSimIndx = np.delete(tSimIndx, np.where(tSimIndx==ii)[0])
            simCnt += len(tSimIndx)

            #tGTIndx = np.where(lp_text[ii] == lp_text)[0]
            #gtCnt += (len(tGTIndx) - 1)

            # list the pairs which our method do not find
            '''
            for kk in range(len(tGTIndx)):
                if (tGTIndx[kk] not in tSimIndx):
                    if(ii != tGTIndx[kk]):
                        abc=0
            '''
            
            if(len(tSimIndx) > 0):
                for jj in range(len(tSimIndx)):
                    if(lp_text[ii] == lp_text[tSimIndx[jj]]):
                        actCnt +=1
                    #else:
                    #    abc=0

        ## Different metrices, add them to the list
        trPos = actCnt
        flNeg = gtCnt - trPos
        flPos = simCnt - trPos

        tPcList.append(trPos)
        fPcList.append(flPos)
        gtcList.append(gtCnt)
        fNcList.append(flNeg)
        
    return tPcList, fPcList, fNcList, gtcList, thVal_list


def analyze_precision_recall_list(tPcList, fPcList, fNcList, gtcList, thVal_list, verbose=True):
    tPcList = np.array(tPcList)
    fPcList = np.array(fPcList)
    fNcList = np.array(fNcList)
    gtcList = np.array(gtcList)

    ## Empty list of precision, recall and f-measure values
    prec = np.zeros((len(tPcList)), dtype=np.float32)
    recl = np.zeros((len(tPcList)), dtype=np.float32)
    f_measure = np.zeros((len(tPcList)), dtype=np.float32)

    for kk in range(len(tPcList)):
        cpPr = tPcList[kk]/float(gtcList[kk])
        fpPr = fPcList[kk]/ float(tPcList[kk]+fPcList[kk])       

        prec[kk] = tPcList[kk]/float(tPcList[kk]+fPcList[kk])
        recl[kk] = tPcList[kk]/float(tPcList[kk]+fNcList[kk])
        f_measure[kk] = 2 * ((prec[kk] * recl[kk]) / (prec[kk] + recl[kk]))

        if verbose:
            print('Threshold: ' + str(thVal_list[kk]))
            print('N.FP: ' + str(fPcList[kk]) + ' N.FN: ' + str(fNcList[kk]) + ' N.TP: ' + str(tPcList[kk]))    
            print('Precision: ' + str(prec[kk]) + ' Recall: ' + str(recl[kk]) + ' F-measure: ' + str(f_measure[kk]) )    
            print('===')
    
    # Get the best index according to f-measure and return
    f_measure[np.isnan(f_measure)] = -50 # to handle the nan values
    best_indx = np.argmax(f_measure)
    return thVal_list[best_indx], prec[best_indx], recl[best_indx], f_measure[best_indx]