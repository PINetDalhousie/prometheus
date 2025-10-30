import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE 

def simple_undersampling(rl_kpis_with_labels, majority_ratio):
    np.random.seed(42)
    cond_rlf = rl_kpis_with_labels["1-day-predict"].astype('bool')
    rlf_count = cond_rlf.sum()
    
    # Get sample index from non rlf columns with 1:3 ratio
    sampled_non_rlf_indicies = np.random.choice(rl_kpis_with_labels[~cond_rlf].index, size=rlf_count * majority_ratio)
    rlf_indicies = np.array(rl_kpis_with_labels[cond_rlf].index)

    sampled_data_indicies = list(sampled_non_rlf_indicies) + list(rlf_indicies)
    sampled_data = rl_kpis_with_labels.loc[sampled_data_indicies]
    
    return sampled_data

def SMOTE_undersampling(X, y, minority_ratio):
    #y = y.to_numpy().astype(int)
    rus = RandomUnderSampler(sampling_strategy=minority_ratio, random_state=42)
    X_res, y_res = rus.fit_resample(X, y)
    #y_res = y_res.astype('bool')
    #X_res["1-day-predict"] = y_res
    return X_res, y_res

def SMOTE_oversampling(X, y, minority_ratio):
    smote = SMOTE(sampling_strategy=minority_ratio, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res
    


if __name__ == '__main__':
    pass
