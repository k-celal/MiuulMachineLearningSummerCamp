#Fonksiyonlar

print("a", "b")

print("a", "b", sep="__")


#tanımlama

def hesapla(x):
    print(x * 2)


hesapla(5)


# İki argümanlı/parametreli bir fonksiyon tanımlayalım.

def toplayici(arg1, arg2):
    print(arg1 + arg2)


toplayici(7, 8)

toplayici(8, 7)

toplayici(arg2=8, arg1=7)


#fonksiyon bilgileri

def toplayici(arg1, arg2):
    print(arg1 + arg2)


def toplayici(arg1, arg2):
    """
    İki sayının toplamı

    Gerekenler:
        arg1: int, float
        arg2: int, float

    return:
        int, float

    """

    print(arg1 + arg2)


toplayici(1, 3)


# girilen değerleri bir liste içinde saklayacak fonksiyon

list_store = []


def listeye_ekle(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)


listeye_ekle(1, 8)
listeye_ekle(18, 8)
listeye_ekle(180, 10)

#ön tanımlı parametreler

def say_hi(string="Merhaba"):
    print(string)
    print("Hi")
    print("Hello")


say_hi("mrb")

# ne zaman fonksiyon yazarız

(10 + 15) / 20
(16 + 95) / 30
(47 + 15) / 40


#func

def calculate(varm, moisture, charge):
    print((varm + moisture) / charge)

# calculate(28,12,70) * 10 ifadesi çalışmaz çünkü print dönüyor
calculate(28,12,70)
#return ifadesi

def calculate(varm, moisture, charge):
    return (varm + moisture) / charge

calculate(28,12,70) * 10

a = calculate(28,12,70)


def calculate(varm, moisture, charge):
    varm = varm * 2
    moisture = moisture * 2
    charge = charge * 2
    output = (varm + moisture) / charge
    return varm, moisture, charge, output


type(calculate(28,12,70))

varma, moisturea, chargea, output = calculate(28,12,70)

#fonksiyon içinden fonksiyon

def calculate(varm, moisture, charge):
    return int((varm + moisture) / charge)


calculate(28,12,70) * 10


def standardization(a, p):
    return a * 10 / 100 * p * p

standardization(15, 1)


def all_calculation(varm, moisture, charge, p):
    a = calculate(varm, moisture, charge)
    b = standardization(a, p)
    print(b * 10)


all_calculation(1, 2, 3, 4)


def all_calculation(varm, moisture, charge, a, p):
    print(calculate(varm, moisture, charge))
    b = standardization(a, p)
    print(b * 10)


all_calculation(1, 2, 3, 4, 5)

#lokal ve global variable

list_store = [1, 2]

def listeye_ekle(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)

listeye_ekle(1, 9)

#if-else

sayi = 11

if sayi == 10:
    print("sayi 10")

sayi = 10
sayi = 20


def sayi_kontrol(number):
    if number == 10:
        print("sayi 10")

sayi_kontrol(12)

#else

def sayi_kontrol(number):
    if number == 10:
        print("sayi 10")

sayi_kontrol(12)


def number_check(number):
    if number == 10:
        print("sayi 10")
    else:
        print("sayi 10 degil")

number_check(12)

#elif

def number_check(number):
    if number > 10:
        print("10 dan büyük")
    elif number < 10:
        print("10 dan küçük")
    else:
        print("10")

number_check(6)

#döngüler
#for

students = ["celal","ahmet","mehmet","deniz"]

#aşağıdaki gibi yapılmaz kolay yolu döngülerdir
students[0]
students[1]
students[2]

for student in students:
    print(student)

salaries = [1, 2, 3, 4, 5]

for salary in salaries:
    print(salary)

def yeni_maas(salary, rate):
    return int(salary*rate/100 + salary)

yeni_maas(15, 1)
yeni_maas(20, 2)

for salary in salaries:
    print(yeni_maas(salary, 20))

#break-continue

maaslar = [1000, 2000, 3000, 4000, 5000]

for maas in maaslar:
    if maas == 3000:
        break
    print(maas)


for maas in maaslar:
    if maas == 3000:
        continue
    print(maas)


# while

number = 6
while number < 1000:
    print(number)
    number += 1

#enumerate

students = ["celal","ahmet","mehmet","deniz"]

for index, student in enumerate(students):
    print(index, student)

A = []
B = []

for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)



#zip

students = ["celal","ahmet","mehmet","deniz"]

departments = ["mat", "istatistik", "fizik", "astronomi"]

ages = [23, 30, 26, 22]

list(zip(students, departments, ages))

#lambda,map,filter,reduce

def topla(a, b):
    return a + b

topla(1, 3) * 9

toplam = lambda a, b: a + b

toplam(4, 5)

# map
maaslar = [1000, 2000, 3000, 4000, 5000]

def yeni_maas(x):
    return x * 20 / 100 + x

yeni_maas(5000)

for maas in maaslar:
    print(yeni_maas(maas))

list(map(yeni_maas, maaslar))

list(map(lambda x: x * 20 / 100 + x, maaslar))
list(map(lambda x: x ** 2 , maaslar))

# filter
lists = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list(filter(lambda x: x % 2 == 0, lists))

# reduce
from functools import reduce
lists = [1, 2, 3, 4]
reduce(lambda a, b: a + b, lists)

#comprehensions

maaslar = [1000, 2000, 3000, 4000, 5000]

def yeni_maas(x):
    return x * 20 / 100 + x

for maas in maaslar:
    print(yeni_maas(maas))

bos_liste = []

for maas in maaslar:
    bos_liste.append(yeni_maas(maas))

bos_liste = []

for salary in maaslar:
    if salary > 3000:
        bos_liste.append(yeni_maas(salary))
    else:
        bos_liste.append(yeni_maas(salary * 2))

[yeni_maas(maas * 2) if maas < 3000 else yeni_maas(maas) for maas in maaslar]
[maas * 2 for maas in maaslar]
[maas * 2 for maas in maaslar if salary < 3000] #eğer sadece if varsa forun sağına
[maas * 2 if maas < 3000 else maas * 0 for maas in maaslar] #eğer else ile birlikteyse forun soluna
[yeni_maas(maas * 2) if maas < 3000 else yeni_maas(maas * 0.2) for maas in maaslar] #fonksiyon kullanma


ogrenciler = ["John", "Mark", "Venessa", "Mariam"]
ogrenci_kontrol = ["John", "Venessa"]


[ogrenci.lower() if ogrenci in ogrenci_kontrol else ogrenci.upper() for ogrenci in ogrenciler]
[ogrenci.upper() if ogrenci not in ogrenci_kontrol else ogrenci.lower() for ogrenci in ogrenciler]#üsttekinin tam tersi
 
#dict comprehensions

dictionary = {'a': 1,
              'b': 2,
              'c': 3,
              'd': 4}

dictionary.keys()
dictionary.values()
dictionary.items()

{k: v ** 2 for (k, v) in dictionary.items()}#valuelerin karesi alınır

{k.upper(): v for (k, v) in dictionary.items()}#keylerin harfleri büyütülür

{k.upper(): v*2 for (k, v) in dictionary.items()}#ikisinede müdahale





