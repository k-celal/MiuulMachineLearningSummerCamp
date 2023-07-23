##################################soru-1

#before : hi my name
#after : Hi mY NaMe

before=input("Uygulamak istediginiz ifade giriniz.")

def alternating(string):
    new_str=""
    for i in range(len(before)):
        if i%2==0:
            new_str+=string[i].upper()
        else:
            new_str+=string[i].lower()
    return new_str

print(alternating(before))

##################################soru-2

students = ["john","mark","venessa","mariam"]

def divide_Student(list):
    liste=[[],[]]
    for index,student in enumerate(students):
        if index%2==0:
            liste[0].append(student)
        else:
            liste[1].append(student)
    return liste

print(divide_Student(students))