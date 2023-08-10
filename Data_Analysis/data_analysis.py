#numpy-Numerical Python
import numpy as np
#hız sağlar aşağıda gözüküyor - yüksek seviyeden işlem yapmak 
a = [1, 2, 3, 4]
b = [2, 3, 4, 5]

ab = []

for i in range(0, len(a)):
    ab.append(a[i] * b[i])
#numpy ile yapımı
a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b

#numpy array oluşturmak
import numpy as np

np.array([1,3,5,7,9])
type(np.array([1, 2, 3, 4, 5]))
np.zeros(10, dtype=int) #0 lardan oluşan bir liste
np.ones(10,dtype=int) #1lerden oluşan bir liste
np.random.randint(0, 10, size=10)
np.random.normal(10, 4, (3, 4)) #ortalama 10 standart sapma 4 olan 


#numpy array özellikleri
import numpy as np

# ndim- boyut sayısı
# shape- boyut bilgisi
# size- toplam eleman sayısı
# dtype- array veri tipi

a = np.random.randint(10, size=5)
a.ndim
a.shape
a.size
a.dtype

#boyut değişme reshape
import numpy as np

np.random.randint(1, 10, size=9)
np.random.randint(1, 10, size=9).reshape(3, 3)
array = np.random.randint(1, 10, size=9)
array.reshape(3, 3)


#index secim
import numpy as np
a = np.random.randint(10, size=10)
a[0]
a[0:5] #slicing dilimleme
a[0] = 8

m = np.random.randint(10, size=(3, 5))

m[0, 0]#virgülden önce satırlar sonra sütunlar

m[2, 3] = 9

m[2, 3] = 2.1#numpy sabit tipli array yani floatı int yapar

m[:, 0]
m[1, :]
m[0:2, 0:3]

#fancy index
import numpy as np

v = np.arange(0, 30, 3)#3er 3er artacak array
v[2]

index = [1, 2, 3]#index bilgisi

v[index]

#koşulu işlemler
import numpy as np
v = np.array([1, 2, 3, 4, 5])

#döngü ile
ab = []
for i in v:
    if i < 3:
        ab.append(i)

#numpy ile
v < 3#true false arrayi döner

v[v < 3]#truelar seçildi
v[v > 3]
v[v != 3]
v[v == 3]
v[v >= 3]

#matematiksel işlemler
import numpy as np
v = np.array([1, 2, 3, 4, 5])

v / 5
v * 5 / 10
v ** 2
v - 1

np.subtract(v, 1)
np.add(v, 1)
np.mean(v)
np.sum(v)
np.min(v)
np.max(v)
np.var(v)
v = np.subtract(v, 1)

#2bilinmeyenli denklem çözümü

# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5, 1], [1, 3]])#katsayılar matrisi
b = np.array([12, 10])#cevaplar

np.linalg.solve(a, b)

#pandas - pandas series- veri manuplasyonu - vari analitiği -veri analizi
import pandas as pd

s = pd.Series([10, 77, 12, 4, 5])
type(s)
s.index
s.dtype
s.size
s.ndim
s.values#numpy array döner
type(s.values)
s.head(3)
s.tail(3)


#read data
import pandas as pd

df = pd.read_csv("datasets/advertising.csv")
df.head()
# pandas cheatsheet

#veriye hızlı bakis
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T #transpoz daha okunabilir yapar
df.isnull().values.any() #eksik değer var mı
df.isnull().sum()
df["sex"].head()
df["sex"].value_counts()

#pandas secim islemleri
import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df.index
df[0:10]#0 dahil 10 hariç
df.drop(0, axis=0).head()

delete_indexes = [1, 2, 3, 4]
df.drop(delete_indexes, axis=0).head(10)

# df = df.drop(delete_indexes, axis=0) #kalıcı hale gelir
# df.drop(delete_indexes, axis=0, inplace=True) #kalıcı hale gelir inplace yaygın kullanılır

#degiskeni index yapmak

df["age"].head()
df.age.head()

df.index = df["age"]

df.drop("age", axis=1).head() #axis=0 satır axis=1 sütun

df.drop("age", axis=1, inplace=True)
df.head()

#indexi değişken yapmak 

df.index

df["age"] = df.index

df.head()
df.drop("age", axis=1, inplace=True)

df.reset_index().head()
df = df.reset_index()
df.head()

#degisken üzerinde islemler
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

"age" in df

df["age"].head()
df.age.head()

df["age"].head()
type(df["age"].head())#pandas serisi


df[["age"]].head()
type(df[["age"]].head())#dataframe olarak kalır

df[["age", "alive"]]

col_names = ["age", "adult_male", "alive"]
df[col_names]

df["age2"] = df["age"]**2 #yenidegisken eklenmis olur
df["age3"] = df["age"] / df["age2"]

df.drop("age3", axis=1).head()

df.drop(col_names, axis=1).head()

df.loc[:, ~df.columns.str.contains("age")].head() # ~ bunun dışındaki hepsini seç


#loc ve iloc
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

# iloc: integer based selection index bilgisi vererek seçim yapma
df.iloc[0:3]
df.iloc[0, 0]

# loc: label based selection label vererek seçim yapma
df.loc[0:3]

df.iloc[0:3, 0:3]
df.loc[0:3, "age"]

col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names]

#kosullu secim
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df[df["age"] > 50].head()
df[df["age"] > 50]["age"].count() #saydırma

df.loc[df["age"] > 50, ["age", "class"]].head()

df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age", "class"]].head()

df["embark_town"].value_counts()

df_new = df.loc[(df["age"] > 50) & (df["sex"] == "male")
       & ((df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton")),
       ["age", "class", "embark_town"]]

df_new["embark_town"].value_counts()

#toplulaştırma ve gruplama

#fonksiyonlar group by işlemi ile kullanılır
# - count()
# - first()
# - last()
# - mean()
# - median()
# - min()
# - max()
# - std()
# - var()
# - sum()
# - pivot table

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df["age"].mean()

df.groupby("sex")["age"].mean() #önce gruplanır sonra toplulaştırılır

df.groupby("sex").agg({"age": "mean"})
df.groupby("sex").agg({"age": ["mean", "sum"]})

df.groupby("sex").agg({"age": ["mean", "sum"],
                       "survived": "mean"})


df.groupby(["sex", "embark_town"]).agg({"age": ["mean"],"survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean"],"survived": "mean"})


df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean"],"survived": "mean","sex": "count"})


#pivot table
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df.pivot_table("survived", "sex", "embarked") #ön tanımlı değer ortalamadır

df.pivot_table("survived", "sex", "embarked",aggfunc="std")

df.pivot_table("survived", "sex", ["embarked", "class"])

df.head()

df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90]) #sayısal değişkeni kategorik yapma değişkeni tanıyorsak cut tanımıyorsak qcut

df.pivot_table("survived", "sex", ["new_age", "class"])

pd.set_option('display.width', 500)


#lambda &apply 
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["age2"] = df["age"]*2
df["age3"] = df["age"]*5

(df["age"]/10).head()
(df["age2"]/10).head()
(df["age3"]/10).head()
#fonksiyon  ile yapımı
for col in df.columns:
    if "age" in col:
        print(col)

for col in df.columns:
    if "age" in col:
        print((df[col]/10).head())

for col in df.columns:
    if "age" in col:
        df[col] = df[col]/10

df.head()
#apply ile uygulama
df[["age", "age2", "age3"]].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std()).head()

def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head() #fonksiyonlarda kullanılır

# df.loc[:, ["age","age2","age3"]] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)

df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)

df.head()

#birleştirme işlemleri  

import numpy as np
import pandas as pd
m = np.random.randint(1, 30, size=(5, 3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99

pd.concat([df1, df2])

pd.concat([df1, df2], ignore_index=True)

#merge ile birleştirme

df1 = pd.DataFrame({'employees': ['john', 'dennis', 'mark', 'maria'],
                    'group': ['accounting', 'engineering', 'engineering', 'hr']})

df2 = pd.DataFrame({'employees': ['mark', 'john', 'dennis', 'maria'],
                    'start_date': [2010, 2009, 2014, 2019]})

pd.merge(df1, df2)
pd.merge(df1, df2, on="employees") #çalşanlara göre birleştir

# her calısanın müdür bilgisi
df3 = pd.merge(df1, df2)

df4 = pd.DataFrame({'group': ['accounting', 'engineering', 'hr'],
                    'manager': ['Caner', 'Mustafa', 'Berkcan']})

pd.merge(df3, df4)
















































