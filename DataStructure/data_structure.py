# VERİ YAPILARI (DATA STRUCTURES)

#integer
x = 11
type(x)

#float
x = 14.3
type(x)

#complex !!! complex sayılarda j nin katsayısı olmalıdır yani j+10 complex kabul etmez
x = 2j + 1
type(x)

#string
x = "hello world i'm muyux"
type(x)

#boolean
True
False
type(True)
2 == 4
4>2
2<4
5 == 5
type(3>=2)

#liste
x = ["btc", "eth", "xrp"]
type(x)

#dictionary
x = {"name": "Peter", "Age": 36}
type(x)

# Tuple
x = ("python", "ml", "ds")
type(x)

# Set
x = {"python", "ml", "ds"}
type(x)

# list, tuple,dict,set python da arrays olarak geçer.

#tip degistirmet

a = 3
b = 3.5

int(b)
float(a)

int(a * b / 10)

c = a * b / 10

int(c)


#string

print("ck")
print('ck')

"ck"
name = "ck"
name = 'ck'

#cok satırlı

"""Veri Yapıları: Hızlı Özet, 
Sayılar (Numbers): int, float, complex, 
Karakter Dizileri (Strings): str, 
List, Dictionary, Tuple, Set, 
Boolean (TRUE-FALSE): bool"""

long_str = """Veri Yapıları: Hızlı Özet, 
Sayılar (Numbers): int, float, complex, 
Karakter Dizileri (Strings): str, 
List, Dictionary, Tuple, Set, 
Boolean (TRUE-FALSE): bool"""

#string eleman erisimi

name
name[0]
name[3]
name[2]

#slice

name[0:2]
long_str[0:10]

#string icinde kelime sorgu !!ilk denk gelenin adresini verir

long_str

"veri" in long_str

"Veri" in long_str

"bool" in long_str

#string fonksiyonları

dir(str)#fonksiyonları gösterir

#uzunluk-len

name = "ck"
type(name)
type(len)

len(name)
len("muyux")
len("miuul")

#büyük-küçük harf dönüşüm

"mx".upper()
"MUYUX".lower()

#karakter değişme replace

hi = "celal karahan"
hi.replace("l", "k")

#bölme-strip

"celal karahan".split() #boşluklardan böler

# strip: başından sonundan kırpar

" ayayay ".strip()
"lcelal".strip("l")

#ilk harfi capitalize büyütür

"vaapv".capitalize()

dir("vaoov")

"vaoov".startswith("v") #bununla mı başlıyor


# liste özellikleri
# değiştirilebilir
# sıralıdır index işlemi yapılır
# kapsayıcı

notes = [1, 2, 3, 4]
type(notes)
names = ["c", "e", "l", "a","l"]
not_nam = [1, 2, 3, "m", "x", False, [1, 2, 3]]#liste içinde liste

not_nam[0]
not_nam[5]
not_nam[6]
not_nam[6][1]

type(not_nam[6])

type(not_nam[6][1])

notes[0] = 99#değiştirilebilir

not_nam[0:4]


#liste fonksiyonları

dir(notes)

#dizi büyüklüğü len

len(notes)
len(not_nam)

#sona eleman ekleme append

notes
notes.append(100)

#pop indexe göre siler

notes.pop(0)

#araya eleman ekleme insert

notes.insert(2, 99)


#sözlük-dict

# Değiştirilebilir
# Sırasız yeni pythonlarda sıralı
# Kapsayıcı

# key-value

dictionary = {"REG": "Regression",
              "LOG": "Logistic Regression",
              "CART": "Classification and Reg"}

dictionary["REG"]


dictionary = {"REG": ["RMSE", 10],
              "LOG": ["MSE", 20],
              "CART": ["SSE", 30]}

dictionary = {"REG": 10,
              "LOG": 20,
              "CART": 30}

dictionary["CART"][1]

#key sorgu

"YSA" in dictionary

#keye göre value erişimi

dictionary["REG"]
dictionary.get("REG")

#değer değiştirme

dictionary["REG"] = ["YSA", 10]

#tüm keylere erişme

dictionary.keys()

#tüm değerlere erişme

dictionary.values()

#tüm key-value çiftlerini tuple şeklinde listeye dönüştürme

dictionary.items()

#key value değeri güncelleme

dictionary.update({"REG": 11})

#yeni key-value ekleme

dictionary.update({"RF": 10})

#tuple

# değiştirilemez
# sıralıdır
# kapsayıcıdır
#yuvarlak parantez

t = ("john", "mark", 1, 2)
type(t)

t[0]
t[0:3]

t[0] = 99 #hata verir

t = list(t) #listeye çevirdik
t[0] = 99 #liste olduğu için hata vermez
t = tuple(t) #tuple çevirdik



#set

# değiştirilebilir
# sırasız + eşsizdir her veriden 1 tane olur
# kapsayıcıdır

#iki kümenin farkı difference

set1 = set([1, 3, 5])
set2 = set([1, 2, 3])

# set1'de olup set2'de olmayanlar.
set1.difference(set2)
set1 - set2

# set2'de olup set1'de olmayanlar.
set2.difference(set1)
set2 - set1

#iki kümede birbirine göre olmayanlar symmetric_difference

set1.symmetric_difference(set2)
set2.symmetric_difference(set1)

#iki kümenin kesişimi intersection

set1 = set([1, 3, 5])
set2 = set([1, 2, 3])

set1.intersection(set2)
set2.intersection(set1)

set1 & set2 #intersection benzer hali

#iki küme birleşimi union

set1.union(set2)
set2.union(set1)


#iki kümenin kesişimi boş mu isdisjoint()

set1 = set([7, 8, 9])
set2 = set([5, 6, 7, 8, 9, 10])

set1.isdisjoint(set2)#bool dönderir
set2.isdisjoint(set1)


#bir küme diğerinin alt kümesi mi issubset()

set1.issubset(set2)
set2.issubset(set1)

#biri diğerini kapsıyor mu issuperset()

set2.issuperset(set1)
set1.issuperset(set2)

