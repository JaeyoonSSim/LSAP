import numpy as np
import pandas as pd
import os
from scipy.stats import ttest_ind

### Bonferroni T-test between two raw data
def Bonferroni_t_test_raw_data():
    data_path = 'data/adni_2022'
    file_name = 'Tau_SUVR.xlsx'
    file_path = '/' + data_path + '/' + file_name

    table = pd.read_excel(os.getcwd() + file_path, engine='openpyxl')
    
    Bonferroni_setting = {'FP': 0.05,
                          'No': 160}

    if file_name == 'DT_thickness.xlsx':
        table_CN = table[table['DX']=='CN'].copy() 
        table_SMC = table[table['DX']=='SMC'].copy() 
        table_EMCI = table[table['DX']=='EMCI'].copy() 
        table_LMCI = table[table['DX']=='LMCI'].copy() 
        table_AD = table[table['DX']=='AD'].copy() 

        table_CN.drop(['Subject', 'PTID', 'DX'], axis=1, inplace=True)
        table_SMC.drop(['Subject', 'PTID', 'DX'], axis=1, inplace=True)
        table_EMCI.drop(['Subject', 'PTID', 'DX'], axis=1, inplace=True)
        table_LMCI.drop(['Subject', 'PTID', 'DX'], axis=1, inplace=True)
        table_AD.drop(['Subject', 'PTID', 'DX'], axis=1, inplace=True)

        CN = table_CN.to_numpy() # (844, 160)
        SMC = table_SMC.to_numpy() # (197, 160)
        EMCI = table_EMCI.to_numpy() # (490, 160)
        LMCI = table_LMCI.to_numpy() # (250, 160)
        AD = table_AD.to_numpy() # (240, 160)
        CN_SMC = np.concatenate((CN, SMC), axis=0) # (1041, 160)
    elif file_name == 'Amyloid_SUVR.xlsx' or file_name == 'FDG_SUVR.xlsx' or file_name == 'Tau_SUVR.xlsx':
        table_CN = table[table['DX']=='CN'].copy() 
        table_SMC = table[table['DX']=='SMC'].copy() 
        table_EMCI = table[table['DX']=='EMCI'].copy() 
        table_LMCI = table[table['DX']=='LMCI'].copy() 
        table_AD = table[table['DX']=='AD'].copy() 

        table_CN.drop(['PTID', 'SCAN', 'EXAMDATE', 'DX', 'AGE', 'PTGENDER', 'PTEDUCAT', 'PTETHCAT', 'PTRACCAT', 'PTMARRY', 'APOE4'], axis=1, inplace=True)
        table_SMC.drop(['PTID', 'SCAN', 'EXAMDATE', 'DX', 'AGE', 'PTGENDER', 'PTEDUCAT', 'PTETHCAT', 'PTRACCAT', 'PTMARRY', 'APOE4'], axis=1, inplace=True)
        table_EMCI.drop(['PTID', 'SCAN', 'EXAMDATE', 'DX', 'AGE', 'PTGENDER', 'PTEDUCAT', 'PTETHCAT', 'PTRACCAT', 'PTMARRY', 'APOE4'], axis=1, inplace=True)
        table_LMCI.drop(['PTID', 'SCAN', 'EXAMDATE', 'DX', 'AGE', 'PTGENDER', 'PTEDUCAT', 'PTETHCAT', 'PTRACCAT', 'PTMARRY', 'APOE4'], axis=1, inplace=True)
        table_AD.drop(['PTID', 'SCAN', 'EXAMDATE', 'DX', 'AGE', 'PTGENDER', 'PTEDUCAT', 'PTETHCAT', 'PTRACCAT', 'PTMARRY', 'APOE4'], axis=1, inplace=True)

        CN = table_CN.to_numpy() # A:(735, 160) / F:(861, 160) / T:(237, 160)
        SMC = table_SMC.to_numpy() # A:(300, 160) / F:(110, 160) / T:(210, 160)
        EMCI = table_EMCI.to_numpy() # A:(833, 160) / F:(597, 160) / T:(186, 160)
        LMCI = table_LMCI.to_numpy() # A:(447, 160) / F:(1138, 160) / T:(105, 160)
        AD = table_AD.to_numpy() # A:(442, 160) / F:(755, 160) / T:(85, 160) 
        CN_SMC = np.concatenate((CN, SMC), axis=0) # A:(1035, 160) / F:(971, 160) / T:(447, 160)

    ### Bonferroni T-test
    t_statistics_list = []
    p_values_list = []
    for i in range(Bonferroni_setting['No']):
        t_statistic, p_value = ttest_ind(CN[:,i], EMCI[:,i]) # T-test
        t_statistics_list.append(t_statistic)
        p_values_list.append(p_value)

    p_values_index_list = []
    for i, p in enumerate(p_values_list):
        p_values_index_list.append([i+1, p])
    
    print('\033[106m' + "Original p-values> "+ '\033[0m')
    print(*p_values_list) # Original p-value
    print('\033[106m' + "Original p-values with index> "+ '\033[0m')
    print(p_values_index_list) # Original p-value with RoI index
    print("Total number of p-values : ", '\033[43m' + ' ' + str(len(p_values_index_list)) + ' ' + '\033[0m')
    
    alpha_level = Bonferroni_setting['FP'] / Bonferroni_setting['No']
    
    output_list = []
    output_index_list = []
    for i, p in enumerate(p_values_index_list):
        if(p[1] < alpha_level):
            output_list.append(p[1])
            output_index_list.append(p)
        else:
            output_list.append(1)
    
    print(output_index_list) # Meaningful p-value with RoI index
    print("The number of meaningful p-values : ", '\033[43m' + ' ' + str(len(output_index_list)) + ' ' + '\033[0m')
    print('\033[106m' + "Scale-changed p-values(Before)> "+ '\033[0m')
    print(*output_list) # Meaningful p-value with Roi index and 1 for rest RoI
    
    ### Change the scale with -log10 (0~1 -> 0~infinity)
    scale_tuning_list = -np.log10(output_list)
    scale_tuning_list[scale_tuning_list == -0.] = 0
    scale_tuning_list[scale_tuning_list > 10] = 10
    print('\033[106m' + "Scale-changed p-values(After)> " + '\033[0m')
    print(*scale_tuning_list) # Scale-changed p-value


### Bonferroni T-test between two data
def Bonferroni_t_test(data1, data2):
    Bonferroni_setting = {'FP': 0.05,
                          'No': 160}
    
    t_statistics_list = []
    p_values_list = []
    for i in range(Bonferroni_setting['No']):
        t_statistic, p_value = ttest_ind(data1[:,i], data2[:,i]) # T-test
        t_statistics_list.append(t_statistic)
        p_values_list.append(p_value)

    p_values_index_list = []
    for i, p in enumerate(p_values_list):
        p_values_index_list.append([i+1, p])
    
    print('\033[106m' + "Original p-values> "+ '\033[0m')
    print(*p_values_list) # Original p-value
    print('\033[106m' + "Original p-values with index> "+ '\033[0m')
    print(p_values_index_list) # Original p-value with RoI index
    print("Total number of p-values : ", '\033[43m' + ' ' + str(len(p_values_index_list)) + ' ' + '\033[0m')
    
    alpha_level = Bonferroni_setting['FP'] / Bonferroni_setting['No']
    
    output_list = []
    output_index_list = []
    for i, p in enumerate(p_values_index_list):
        if(p[1] < alpha_level):
            output_list.append(p[1])
            output_index_list.append(p)
        else:
            output_list.append(1)
    
    print(output_index_list) # Meaningful p-value with RoI index
    print("The number of meaningful p-values : ", '\033[43m' + ' ' + str(len(output_index_list)) + ' ' + '\033[0m')
    print('\033[106m' + "Scale-changed p-values(Before)> "+ '\033[0m')
    print(*output_list) # Meaningful p-value with Roi index and 1 for rest RoI
    
    ### Change the scale with -log10 (0~1 -> 0~infinity)
    scale_tuning_list = -np.log10(output_list)
    scale_tuning_list[scale_tuning_list == -0.] = 0
    scale_tuning_list[scale_tuning_list > 10] = 10
    print('\033[106m' + "Scale-changed p-values(After)> " + '\033[0m')
    print(*scale_tuning_list) # Scale-changed p-value


### T-test
def t_test(data1, data2):
    t_statistic, p_value = ttest_ind(data1, data2) # T-test
    
    print("Original p-value : ", '\033[42m' + ' ' + str(p_value) + ' ' + '\033[0m')

    scale_tuning_p_value = -np.log10(p_value)
    print("Scale-changed p-value : ", '\033[42m' + ' ' + str(scale_tuning_p_value) + ' ' + '\033[0m')
    
    
if __name__ == "__main__":
    Bonferroni_t_test_raw_data()