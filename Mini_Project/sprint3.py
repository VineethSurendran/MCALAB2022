# -*- coding: utf-8 -*-
"""Sprint3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fLTCIMXmLBCxWvfpfvu-h5lsw-kBTjh6
"""

import joblib
modelname="enoder1.pkl"
joblib.dump(label_Encoder,modelname)
from google.colab import files
files.download(modelname)

modelname="enoder2.pkl"
joblib.dump(encoder1,modelname)
from google.colab import files
files.download(modelname)

modelname="enoder3.pkl"
joblib.dump(encoder2,modelname)
from google.colab import files
files.download(modelname)

modelname="randomforest.pkl"
joblib.dump(rf_model,modelname)
from google.colab import files
files.download(modelname)

modelname="svm.pkl"
joblib.dump(classifier,modelname)
from google.colab import files
files.download(modelname)