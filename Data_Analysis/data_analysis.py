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
s.values
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
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df["sex"].head()
df["sex"].value_counts()

#pandas secim islemleri
import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df.index
df[0:10]
df.drop(0, axis=0).head()

delete_indexes = [1, 2, 3, 4]
df.drop(delete_indexes, axis=0).head(10)
















































