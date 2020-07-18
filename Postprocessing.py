from utils import *
import copy
import random 


#######################################################################################################################
# YOU MUST FILL OUT YOUR SECONDARY OPTIMIZATION METRIC (either accuracy or cost)!
# The metric chosen must be the same for all 5 methods.
#
# Chosen Secondary Optimization Metric: #
#######################################################################################################################
""" Determines the thresholds such that each group has equal predictive positive rates within 
    a tolerance value epsilon. For the Naive Bayes Classifier and SVM you should be able to find
    a nontrivial solution with epsilon=0.02. 
    Chooses the best solution of those that satisfy this constraint based on chosen 
    secondary optimization criteria.
"""
def enforce_demographic_parity(categorical_results, epsilon):
    demographic_parity_data = {}
    thresholds = {}
    dp={}
    for k,v in categorical_results.items():
        pft={}
        for i in range(1,101):
            threshold=float(i)/100
            eval_copy = apply_threshold(v, threshold)
            predt = get_num_predicted_positives(eval_copy)
            predt2 = len(v)
            xy=predt/predt2
            pft[threshold]=xy
        dp[k]=pft
    res = list(dp.keys())
    revalues=dp.values()
    dpval=[]
    for r in revalues:
        rx= r.values()
        for rx1 in rx:
            dpval.append(rx1)
    listoffinals=[]
    valuesoffinals=[]
    final=[]
    finalth=[]
    for a in range(int(len(dpval)/2),300):
        finalth=[]
        final=[]
        for b in range(0,int(len(dpval)/4)):
            if dpval[a]-epsilon<=dpval[b]<=dpval[a]+epsilon:
                finalth.append(a+1)
                finalth.append(b+1)
                final.append(dpval[a])
                final.append(dpval[b])
                break
        if len(finalth)<2:
            continue
            
        for c in range(int(len(dpval)/4),int(len(dpval)/2)):
            if dpval[a]-epsilon<=dpval[c]<=dpval[a]+epsilon:
                finalth.append(c+1)
                final.append(dpval[c])
                break
        if len(finalth)<3:
            continue
        for d in range(300,int(len(dpval))):
            if dpval[a]-epsilon<=dpval[d]<=dpval[a]+epsilon:
                finalth.append(d+1)
                final.append(dpval[d])
                break
        if len(finalth)==4:
            listoffinals.append(finalth)
            valuesoffinals.append(final)
    #print(listoffinals)      
    tl1=[]
    l1d={}
    for i in listoffinals:
        l1=[]
        for j in i:
            if 0<=j<=100:
                l1.append(float(j)/100.0)
                continue
            if 100<=j<=200:
                l1.append(float(j-100)/100.0)
                continue
            if 200<=j<=300:
                l1.append(float(j-200)/100.0)
                continue
            if 300<=j<=400:
                l1.append(float(j-300)/100.0)
#            l1d={res[2]:l1[0],res[0]:l1[1],res[1]:l1[2],res[3]:l1[3]}
        l1d={res[0]:l1[1],res[1]:l1[2],res[2]:l1[0],res[3]:l1[3]}
        tl1.append(l1d)
       
    print(valuesoffinals)
#    #print(dpval[10])
#    tl1=[]
#    l1d={}
#    for i in listoffinals:
#        l1=[]
#        c=1
#        for j in i:
#            if 0<=j<=100:
#                l1.append(float(j)/100.0)
#                continue
#            elif c*100<=j<=(c+1)*100:
#                l1.append(float(j-c*100)/100.0)
#                c=c+1
#        l1d={res[xj]: l1[xj] for xj in range(len(res))}
#        tl1.append(l1d) 
#     
    mind={}
    mind1={}
    least=[]
    count=0
    for iz in tl1:
        mind1={}
        for k,v in categorical_results.items():
            eval_copy = apply_threshold(v, iz[k])
            mind1[k]=eval_copy
            
        least.append(apply_financials(mind1))
    index=least.index(max(least))
    finalth=tl1[index]
            
    thresholds = tl1[index] 
    for k,v in categorical_results.items():
            x1 = apply_threshold(v, thresholds[k])
            demographic_parity_data[k]=x1
    


    # Must complete this function!
    #return demographic_parity_data, thresholds

    return demographic_parity_data, thresholds

#######################################################################################################################
""" Determine thresholds such that all groups have equal TPR within some tolerance value epsilon, 
    and chooses best solution according to chosen secondary optimization criteria. For the Naive 
    Bayes Classifier and SVM you should be able to find a non-trivial solution with epsilon=0.01
"""
def enforce_equal_opportunity(categorical_results, epsilon):

    thresholds = {}
    equal_opportunity_data = {}
    check={}
    remain={}
    for k,v in categorical_results.items():
        tprx1={}
        for i in range(1,101):
            threshold=float(i)/100.0
            eval_copy = apply_threshold(v, threshold)
            tpr=get_true_positive_rate(eval_copy)
            tprx1[threshold]=tpr
        check[k]=tprx1
    remain=copy.deepcopy(check)
    res = list(remain.keys())
    revalues=remain.values()
    tprval=[]
    for r in revalues:
        rx= r.values()
        for rx1 in rx:
            tprval.append(rx1)
    listoffinals=[]
    finalth=[]
    for a in range(int(len(tprval)/4),int(len(tprval)/2)):
        finalth=[]
        final=[]
        for b in range(0,int(len(tprval)/4)):
            if tprval[a]-epsilon<=tprval[b]<=tprval[a]+epsilon:
                finalth.append(a+1)
                finalth.append(b+1)
                final.append(tprval[a])
                final.append(tprval[b])
                break
        if len(finalth)<2:
            continue
            
        for c in range(int(len(tprval)/2),300):
            if tprval[a]-epsilon<=tprval[c]<=tprval[a]+epsilon:
                finalth.append(c+1)
                final.append(tprval[c])
                break
        if len(finalth)<3:
            continue
        for d in range(300,int(len(tprval))):
            if tprval[a]-epsilon<=tprval[d]<=tprval[a]+epsilon:
                finalth.append(d+1)
                final.append(tprval[d])
                break
        if len(finalth)==4:
            listoffinals.append(finalth)
            
    tl1=[]
    l1d={}
    for i in listoffinals:
        l1=[]
        for j in i:
            if 0<=j<=100:
                l1.append(float(j)/100.0)
                continue
            if 100<=j<=200:
                l1.append(float(j-100)/100.0)
                continue
            if 200<=j<=300:
                l1.append(float(j-200)/100.0)
                continue
            if 300<=j<=400:
                l1.append(float(j-300)/100.0)
        l1d={res[0]:l1[1],res[1]:l1[0],res[2]:l1[2],res[3]:l1[3]}
        tl1.append(l1d)
       
 
    mind={}
    mind1={}
    least=[]
    for iz in tl1:
        
        mind1={}
        for k,v in categorical_results.items():
            eval_copy = apply_threshold(v, iz[k])
            mind1[k]=eval_copy
        least.append(apply_financials(mind1))
    index=least.index(max(least))
    finalth=tl1[index]
            
    thresholds = tl1[index] 
    for k,v in categorical_results.items():
            x1 = apply_threshold(v, thresholds[k])
            equal_opportunity_data[k]=x1
        
            
    #print(equal_opportunity_data)
    # Must complete this function!
    #return equal_opportunity_data, thresholds

#    return equal_opportunity_data, thresholds
    return equal_opportunity_data,thresholds

#######################################################################################################################

"""Determines which thresholds to use to achieve the maximum profit or maximum accuracy with the given data
"""

def enforce_maximum_profit(categorical_results):
    mp_data = {}
    thresholds = {}
    maxp={}
    for k,v in categorical_results.items():
        accx1={}
        for i in range(1,101):
            threshold = float(i) / 100.0
            eval_copy = apply_threshold(v, threshold)
            fp=get_num_false_positives(eval_copy)
            tn=get_num_true_negatives(eval_copy)
            fn=get_num_false_negatives(eval_copy)
            tp=get_num_true_positives(eval_copy)
            acc=(tp+tn)/(tp+tn+(fp)+(fn))
            accx1[threshold]=acc
        thr1 = max(accx1,key=accx1.get)
        maxp[k]=thr1
        x1 = apply_threshold(v, thr1)
        mp_data[k]=x1
        thresholds[k]=thr1

    # Must complete this function!
    #return mp_data, thresholds

    return mp_data, thresholds

#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""
def get_pred_parity(prediction_label_pairs):
    ppv = 0
    px = 0

    for pair in prediction_label_pairs:
        prediction = int(pair[0])
        label = int(pair[1])
        if prediction == 1:
            px += 1
            if label == 1:
                ppv += 1

    if px != 0:
        return ppv / px
    else:
        return 0
def get_final_list(a,b,v,finalth,ppvval,epsilon):
    for z in range(a,b):
        if abs(ppvval[z]-ppvval[v])<=epsilon:
            finalth.append(z+1)
            break
    return finalth

def enforce_predictive_parity(categorical_results, epsilon):
    predictive_parity_data = {}
    thresholds = {}
    equalppv={}
    for k,v in categorical_results.items():
        tres={}
        for i in range(1,101):
            threshold=float(i)/100.0
            eval_copy = apply_threshold(v, threshold)
            xy = get_pred_parity(eval_copy)
            
            tres[threshold]=xy
        equalppv[k]=tres
    res = list(equalppv.keys())
    revalues=equalppv.values()
    ppvval=[]
    for r in revalues:
        rx= r.values()
        for rx1 in rx:
            ppvval.append(rx1)    
            
    finalth=[]
    listoffinals=[]
    
    for a in range(0,int(len(ppvval)/len(res))):
        finalth=[]
        finalth.append(a+1)
        for cv in range(1,len(res)):
            y=get_final_list(cv*100,(cv+1)*100,a,finalth,ppvval,epsilon)
            
            if len(y)<cv+1:
                continue
        if len(y)==len(res):
            listoffinals.append(y)
    tl1=[]
    l1d={}
    for i in listoffinals:
        l1=[]
        c=1
        for j in i:
            #c=1
            if 0<=j<=100:
                l1.append(float(j)/100.0)
                continue
            elif c*100<=j<=(c+1)*100:
                l1.append(float(j-c*100)/100.0)
                c=c+1
        l1d={res[xj]: l1[xj] for xj in range(len(res))}
        tl1.append(l1d)

    
    #print(len(listoffinals)) 
    mind={}
    mind1={}
    least=[]
    for iz in tl1:
        mind1={}
        for k,v in categorical_results.items():
            eval_copy = apply_threshold(v, iz[k])
            mind1[k]=eval_copy
        least.append(apply_financials(mind1))
    index=least.index(max(least))
    finalth=tl1[index]
            
    thresholds = tl1[index] 
        
    for k,v in categorical_results.items():
            x1 = apply_threshold(v, thresholds[k])
            predictive_parity_data[k]=x1

                

    # Must complete this function!
    #return predictive_parity_data, thresholds

    return predictive_parity_data, thresholds

    ###################################################################################################################
""" Apply a single threshold to all groups, and return the best solution according to 
    chosen secondary optimization criteria
"""

def enforce_single_threshold(categorical_results):
    single_threshold_data = {}
    thresholds = {}    
    classifications={}
    least=[]
    for i in range(1, 101):
            
            threshold = float(i) / 100.0
            app={}
            for k,v in categorical_results.items():
                eval_copy2 = list.copy(v)
                eval_copy = apply_threshold(eval_copy2, threshold)
                app[k]=eval_copy
            least.append(apply_financials(app))
    index=least.index(max(least))
    iz=float(index+1)/100.0
    

    for k,v in categorical_results.items():
        x1 = apply_threshold(v,iz)
        single_threshold_data[k]= x1
        thresholds[k]=iz    
       

    # Must complete this function!
    #return single_threshold_data, thresholds

    return single_threshold_data, thresholds