# -*- coding:utf-8 -*-

'''
data sets for calib_rt

.. calib_rt:
    url

'''

import pandas as pd
import numpy as np
import os


class RTdatasets(object):
    """
    RTdatasets include some commonly used datasets
    
    Properties:
        info: a dict store of all information
        sample_types: a list of available sample type

    Attributes:
        get_datasets_list(): get basis information of all datasets
        get_sample_type_info(): get detail information of all datasets in one sample group
        get_dict(): get data of datasets in a dict
        get_pandas(): get data of datasets in a DataFrame type
    """

    info = {
        "distort_left":('fmeierab_T190525_CLL_diaPASEF_04_1979',
                        'fmeierab_T190525_CLL_diaPASEF_40_1965'),
        "distort_right":('20200505_Evosep_100SPD_SG06-16_MLHeLa_100ng_py8_S2-C1_1_2731',
                         '20211103_PRO2_LS_01_MA_HeLa_200_SDC_NS_RE2_1_1418'),
        "exp":('SC_HeLa_10min1_Slot1-9_1_807',
               'SC_HeLa_15min1_Slot1-7_1_805'),
        "linear":('20211101_PRO2_LS_04_MA_HeLaSCS_0.2_ngHS_GE2_1_1408',
                  '20220714_10ngK562_ZI_500ul60C3cm5min_P1-C1_1_9675'),
        "S":('20200505_Evosep_200SPD_SG06-16_MLHeLa_200ng_py8_S3-A1_1_2737',
             'CMs_Subject3_Lvmid_G10_BG11_1_7560')
    }
    sample_types = ["distort_left","distort_right","exp","linear","S"]

    __library_root = os.path.dirname(__file__)
    __file_path = os.path.join(__library_root, "data")

    def get_datasets_list() -> pd.DataFrame:
        """
        get basis information of all datasets
        
        Results:
            a DataFrame of all datasets
        
        Examples:
            >>> from calib_rt import RTdatasets
            >>> RTdatasets.get_datasets_list()
                 sample_type  datasets_num
            0   distort_left             2
            1  distort_right             2
            2            exp             2
            3         linear             2
            4              S             2
        
        References:
            url
        """
        result = [len(RTdatasets.info[sample_type])
                  for sample_type in RTdatasets.sample_types]
        df = pd.DataFrame({"sample_type":RTdatasets.sample_types,
                           "datasets_num":result})
        return df
    
    def get_sample_type_info(sample_type:str) -> dict:
        """
        get detail information of all datasets in one sample group

        Args:
            sample_type(str): a str which in RTdatasets.sample_types is OK

        Results:
            a dict of detail of datasets in is sample group
        
        Examples:
            >>> from calib_rt import RTdatasets
            >>> RTdatasets.get_sample_type_info("S")
        
        References:
            url
        """
        result = RTdatasets.info[sample_type]
        return {sample_type:result}

    
    def get_dict(sample_type:str,index_in_group:int) -> dict[str:np.ndarray]:
        """
        get data of datasets in a dict

        Args:
            sample_type(str): a str which in RTdatasets.sample_types is OK

            index_in_group(int): the index of datasets in it's group

        Results:
            dict of RT data

            like: {"Spectral library RT":ndarray,"Measured RT":ndarray}
        
        Examples:
            >>> from calib_rt import RTdatasets
            >>> RTdatasets.get_dict("linear",0)
        
        References:
            url
        """
        info = RTdatasets.info[sample_type][index_in_group]
        df_path = os.path.join(RTdatasets.__file_path,f"{info}.d.npz")
        data = np.load(df_path)
        df_dict = {"Spectral library RT":data["x"],
                   "Measured RT":data["y"]}
        return df_dict
    
    def get_pandas(sample_type:str,index_in_group:int) -> pd.DataFrame:
        """
        get data of datasets in a DataFrame type

        Args:
            sample_type(str): a str which in RTdatasets.sample_types is OK

            index_in_group(int): the index of datasets in it's group

        Results:
            DataFrame of RT data 
            
            The detail information can get from Name 
        
        Examples:
            >>> from calib_rt import RTdatasets
            >>> RTdatasets.get_pandas("linear",0)
        
        References:
            url
        """
        df_dict = RTdatasets.get_dict(sample_type,index_in_group)
        df = pd.DataFrame(df_dict)
        df.Name = RTdatasets.info[sample_type][index_in_group]
        return df