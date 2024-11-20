import math
import random
import matplotlib.pyplot as plt
import numpy as np

distanceToTarget = 0
h = 100
v0 = 50
g = 9.81
alpha = 0


def zad1():
    print("zad1")
    global distanceToTarget
    distanceToTarget = random.randint(50, 340)
    print(f"Distance to target: {distanceToTarget}")


def shotTarget():
    global distanceToTarget, h, v0, g, alpha
    # Thanks for help chatGBT
    alpha_rad = math.radians(alpha)
    term1 = v0 * math.sin(alpha_rad)
    term2 = math.sqrt(v0 ** 2 * math.sin(alpha_rad) ** 2 + 2 * g * h)
    term3 = (v0 * math.cos(alpha_rad)) / g
    distance = (term1 + term2) * term3
    print(f"Your distance is: {distance}")
    for x in range(-5, 5):
        if int(distance) + x == distanceToTarget:
            return True
    return False


def zad2():
    print("zad2")
    global alpha
    numberOfTries = 1

    while True:
        while True:
            try:
                alpha = int(input("Give me angle:\n"))
                if alpha < 0 or alpha > 360:
                    raise Exception("Wrong number.")
                break
            except:
                print("Wrong integer(range is 0-360).")
        if shotTarget():
            print(f"You hit the target after {numberOfTries} tries!")
            break
        else:
            print("You missed, try again.")
            numberOfTries += 1


def zad2a():
    print("\nzad2a")
    print("Nope")


def zad2b():
    print("\nzad2b")
    print("ChatGbt gave me formula from screen.")


def zad2c():
    print("\nzad2c")
    print("Nope")


def calculate_trajectory(v0, angle, h):
    t_points = 1000
    alpha_rad = np.radians(angle)
    t_flight = (v0 * np.sin(alpha_rad) + np.sqrt((v0 * np.sin(alpha_rad))**2 + 2 * g * h)) / g
    t = np.linspace(0, t_flight, t_points)
    x = v0 * np.cos(alpha_rad) * t
    y = h + v0 * np.sin(alpha_rad) * t - 0.5 * g * t**2
    return x, y

def zad3():
    print("\nzad3")
    global h, v0, g, alpha

    x, y = calculate_trajectory(v0, alpha, h)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, color='blue', label='Trajektoria pocisku')

    plt.text(30, 100, f'α = {alpha}°', fontsize=12, color='black', rotation=30)

    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.title('Projectile Motion for the Trebuchet')

    plt.grid(True)

    plt.ylim(bottom=0)

    plt.savefig('trajektoria.png')

    plt.show()


def zad4():
    print("\nzad4")
    print("I've done that when i was doing those tasks.")


zad1()
zad2()
zad2a()
zad2b()
zad2c()
zad3()
zad4()
