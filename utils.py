import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class CorrectRace(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.col = 'race'
        # self.map = { 'white' : 'caucasian'
        #            , 'africanamerican' : 'african-american'
        #            , 'african american' : 'african-american'
        #            , 'afro american' : 'african-american'
        #            , 'euro' : 'european'
        #            , '?' : 'other'
        #            , 'asian' : 'other'
        #            , 'latino' : 'other'
        #            }
        
        self.map = { 'white' : 'caucasian'
             , 'africanamerican' : 'black'
             , 'african american' : 'black'
             , 'afro american' : 'black'
             , 'euro' : 'caucasian'
             , 'european' : 'caucasian'
             , 'asian' : 'asian'
             , 'latino' : 'latino'
             , 'hispanic' : 'latino'
             }

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        X[self.col] = X[self.col].str.lower().replace(self.map)
        
        return X

class ConvertAge(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.col = 'age'
        self.map = { '[0-10)' : 10 # 5
                    ,'[10-20)' : 10 # 15
                    ,'[20-30)' : 25
                    ,'[30-40)' : 35
                    ,'[40-50)' : 45
                    ,'[50-60)' : 55
                    ,'[60-70)' : 65
                    ,'[70-80)' : 75
                    ,'[80-90)' : 85
                    ,'[90-100)' : 95
                   }
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        X[self.col] = X[self.col].map(self.map)
        
        return X

class CastString(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols = [ 'blood_transfusion',
                      # 'has_prosthesis'
                    ]
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        X_ = X.copy(deep = True)
        
        for col in self.cols:
            X_[col] = X_[col].astype(str)
            X_[col].replace('nan', np.nan, inplace = True)
        
        return X_
    
class CorrectAdmissionTypeCode(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.col = 'admission_type_code'
        self.map = { '1.0' : '1.0'
                    ,'2.0' : '1.0'
                    ,'3.0' : '3.0'
                    ,'4.0' : '4.0'
                    ,'5.0' : '5.0'
                    ,'6.0' : '5.0''2'
                    ,'7.0' : '7.0'
                    ,'8.0' : '5.0'}
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        X_ = X.copy(deep = True)
        
        X_[self.col] = X_[self.col].astype(float).astype(str)
        X_[self.col] = X_[self.col].replace(self.map)
        X_[self.col] = X_[self.col].replace('nan', np.nan)
        X_[self.col] = X_[self.col].fillna('5.0')
        
        return X_
    
class CorrectDischargeDispositionCode(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.col = 'discharge_disposition_code'
        self.map = { '1.0' : '1.0',
                     '2.0' : '3.0',
                     '3.0' : '3.0',
                     '4.0' : '3.0',
                     '5.0' : '3.0',
                     '6.0' : '1.0',
                     '7.0' : '7.0',
                     '8.0' : '1.0',
                     '9.0' : '9.0',
                     '10.0' : '3.0',
                     '11.0' : '11.0',
                     '12.0' : '12.0',
                     '13.0' : '1.0',
                     '14.0' : '3.0',
                     '15.0' : '15.0',
                     '16.0' : '16.0',
                     '17.0' : '17.0',
                     '18.0' : '18.0',
                     '19.0' : '19.0',
                     '20.0' : '20.0',
                     '21.0' : '21.0',
                     '22.0' : '22.0',
                     '23.0' : '3.0',
                     '24.0' : '3.0',
                     '25.0' : '18.0',
                     '26.0' : '18.0',
                     '27.0' : '3.0',
                     '28.0' : '3.0',
                     '29.0' : '3.0',
                     '30.0' : '3.0' }
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        X_ = X.copy(deep = True)
        
        X_[self.col] = X_[self.col].apply(float).apply(str)
        X_[self.col] = X_[self.col].replace(self.map)
        X_[self.col] = X_[self.col].replace('nan', np.nan)
        X_[self.col] = X_[self.col].fillna('18.0')
        
        return X_
    
class CorrectAdmissionSourceCode(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.col = 'admission_source_code'
        self.map = { '1.0' : '1.0',
                     '2.0' : '1.0',
                     '3.0' : '1.0',
                     '4.0' : '4.0',
                     '5.0' : '4.0',
                     '6.0' : '4.0',
                     '7.0' : '7.0',
                     '8.0' : '8.0',
                     '9.0' : '9.0',
                     '10.0' : '4.0',
                     '11.0' : '11.0',
                     '12.0' : '12.0',
                     '13.0' : '13.0',
                     '14.0' : '14.0',
                     '15.0' : '9.0',
                     '16.0' : '9.0',
                     '17.0' : '4.0',
                     '18.0' : '18.0',
                     '19.0' : '9.0',
                     '20.0' : '9.0',
                     '21.0' : '4.0',
                     '22.0' : '22.0',
                     '23.0' : '14.0',
                     '24.0' : '4.0',
                     '25.0' : '4.0' }
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        X_ = X.copy(deep = True) 
        
        X_[self.col] = X_[self.col].apply(float).apply(str)
        X_[self.col] = X_[self.col].replace(self.map)
        
        return X_

class CorrectMaxGluSerum(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.col = 'max_glu_serum'
        self.map = { 'norm' : 'normal',
                     '>200' : 'abnormal',
                     '>300' : 'abnormal'}
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        X_ = X.copy(deep = True)
        
        X_[self.col] = X_[self.col].astype(str)
        X_[self.col] = X_[self.col].str.lower()
        X_[self.col] = X_[self.col].replace(self.map)
        X_[self.col] = X_[self.col].replace('nan', np.nan)
        
        return X_

class CorrectA1CResult(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.col = 'A1Cresult'
        self.map = { 'norm' : 'normal',
                     '>8' : 'abnormal',
                     '>7' : 'abnormal'}
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        X_ = X.copy(deep = True)
        
        X_[self.col] = X_[self.col].str.lower()
        X_[self.col] = X_[self.col].replace(self.map)
        
        return X_

class ConvertDiag(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols = ['diag_1', 
                     'diag_2', 
                     'diag_3']

    def fit(self, X, y = None):
        return self
    
    def convert(self, x):
        try:
            x_num = float(x)
            
            if 0 <= x_num <= 139:
                interval = '0-139'
            elif 140 <= x_num <= 239:
                interval = '140-239'
            elif 240 <= x_num <= 279:
                interval = '240-279'
            elif 280 <= x_num <= 289:
                interval = '280-289'
            elif 290 <= x_num <= 319:
                interval = '290-319'
            elif 320 <= x_num <= 389:
                interval = '320-389'
            elif 390 <= x_num <= 459:
                interval = '390-459'
            elif 460 <= x_num <= 519:
                interval = '460-519'
            elif 520 <= x_num <= 579:
                interval = '520-579'
            elif 580 <= x_num <= 629:
                interval = '580-629'
            elif 630 <= x_num <= 679:
                interval = '630-679'
            elif 680 <= x_num <= 709:
                interval = '680-709'
            elif 710 <= x_num <= 739:
                interval = '710-739'
            elif 740 <= x_num <= 759:
                interval = '740-759'
            elif 760 <= x_num <= 779:
                interval = '760-779'
            elif 780 <= x_num <= 799:
                interval = '780-799'
            elif 800 <= x_num <= 999:
                interval = '800-999'
                
        except (ValueError, UnboundLocalError):
            if x == 'Unknown':
                interval = 'Unknown'
            else:
                interval = 'E-V'
            
        return interval

    def transform(self, X, y = None):
        X_ = X.copy(deep = True)
        
        for col in self.cols:
            X_[col] = X_[col].fillna('Unknown')
            X_[col] = X_[col].apply(self.convert)
        
        return X_