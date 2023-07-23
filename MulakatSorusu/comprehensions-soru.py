##################################soru-1

# Amaç: çift sayıların karesi alınarak bir sözlüğe eklenmek istemektedir
# Key'ler orjinal değerler value'lar ise değiştirilmiş değerler olacak.


numbers = range(10)
new_dict = {}

for n in numbers:
    if n % 2 == 0:
        new_dict[n] = n ** 2 #sol taraftaki key sağ taraftaki value

{n: n ** 2 for n in numbers if n % 2 == 0}

#veri setindeki değişken isimlerini değiştirmek

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

for col in df.columns:
    print(col.upper())


A = []

for col in df.columns:
    A.append(col.upper())

df.columns = A

df = sns.load_dataset("car_crashes")

df.columns = [col.upper() for col in df.columns]

#isminde ins olanların başına flag olmayanların başına no_flag

[col for col in df.columns if "INS" in col]

["FLAG_" + col for col in df.columns if "INS" in col]

["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns]

df.columns = ["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns]


#amaç keyi string valuesu aşağıdaki gibi bir liste olan sözlük oluşturmak
#sadece sayısal kolonlara yapmamız isteniyor

import seaborn as sns
df = sns.load_dataset("car_crashes")
agg_list = ["mean", "min", "max", "sum"]

num_cols = [col for col in df.columns if df[col].dtype != "O"]
dic = {}

for col in num_cols:
    dic[col] = agg_list

#kısa
new_dict = {col: agg_list for col in num_cols}

df[num_cols].head()

df[num_cols].agg(new_dict)