import datetime
import math

name = ""
yearOfBirth = 0
monthOfBirth = 0
dayOfBirth = 0
dateOfBirth = None
yourDaysOfLiving = 0


def getData():
    global name, yearOfBirth, monthOfBirth, dayOfBirth, dateOfBirth, yourDaysOfLiving
    name = input("Whats your name?\n")

    while True:
        try:
            yearOfBirth = int(input("Whats your year of birth?\n"))
            if yearOfBirth < 1 and yearOfBirth < datetime.date.today().year:
                raise Exception("Wrong number.")
            break
        except:
            print("Only positive integer.")

    while True:
        try:
            monthOfBirth = int(input("Whats your month of birth?\n"))
            if monthOfBirth < 1 or monthOfBirth > 12:
                raise Exception("Wrong number.")
            break
        except:
            print("Only positive integer.")

    while True:
        try:
            dayOfBirth = int(input("Whats your day of birth?\n"))
            if dayOfBirth < 1 or dayOfBirth > 31:
                raise Exception("Wrong number.")
            break
        except:
            print("Only positive integer.")

    dateOfBirth = datetime.date(yearOfBirth, monthOfBirth, dayOfBirth)
    todayDate = datetime.date.today()

    yourDaysOfLiving = (todayDate - dateOfBirth).days


def zadA():
    global name, dateOfBirth, yourDaysOfLiving
    print("\nZad A")

    print(f'Hello, {name} who has been born in {dateOfBirth}.')

    print(f'It is your {yourDaysOfLiving} day of living on this planet.')

    physicalWave = math.sin((2 * math.pi / 23) * yourDaysOfLiving)
    print(f'Your physical wave is {physicalWave}')

    emotionalWave = math.sin((2 * math.pi / 28) * yourDaysOfLiving)
    print(f'Your emotional wave is {emotionalWave}')


def zadB():
    global yourDaysOfLiving
    print("\nZad B")

    physicalWave = math.sin((2 * math.pi / 23) * yourDaysOfLiving)
    print(f'Your physical wave is {physicalWave}')

    emotionalWave = math.sin((2 * math.pi / 28) * yourDaysOfLiving)
    print(f'Your emotional wave is {emotionalWave}')

    if physicalWave > 0.5:
        print(f'Good physical day {physicalWave}')
    else:
        print(f'Bad physical day {physicalWave}, it will be better... maybe.')
        physicalWaveTommorow = math.sin((2 * math.pi / 23) * (yourDaysOfLiving + 1))
        print(f'Your physical wave for tommorow is {physicalWaveTommorow}.')

    if emotionalWave > 0.5:
        print(f'Good emotional day {emotionalWave}')
    else:
        print(f'Bad emotional day {emotionalWave}, it will be better... maybe.')
        emotionalWaveTommorow = math.sin((2 * math.pi / 28) * (yourDaysOfLiving + 1))
        print(f'Your emotional wave for tommorow is {emotionalWaveTommorow}.')


def zadC():
    print("\nZad C")
    print(f"Nie liczyłem ile zajęło, wiem że najwięcej mi zajęło ogarnięcie żeby zmienne były globalne.")


def zadD():
    print("\nZad D")
    print("Done")


def zadE():
    print("\nZad E")
    print("Szybciej było i zrobił w inny sposób.")


getData()
zadA()
zadB()
zadC()
zadD()
zadE()
