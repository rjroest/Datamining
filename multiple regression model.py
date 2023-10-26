# Importeer de benodigde bibliotheken
import pandas as pd
import statsmodels.formula.api as sm

# Lees de dataset in
data = pd.read_excel('heart.xlsx')

# Selecteer alle variabelen in het model om de multiple regression model uit te kunnen voeren
Model1 = sm.ols('target ~ age + sex + cp + trestbps + chol + restecg + thalach + exang + oldpeak + slope + ca + thal', data = data).fit()

# Bekijk de samenvatting van het regressiemodel
print(Model1.summary())
