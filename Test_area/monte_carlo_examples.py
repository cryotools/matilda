"""Example Monte Carlo simulation
Monty Hall Problem"""

import random
import matplotlib.pyplot as plt

# There are three doors
#1 - Car
#2 - Goats

doors = ["goat", "goat", "car"]

# Empty lists to store probability values:
switch_win_prob = []
stick_win_prob = []

plt.axhline(y=0.66666, color="r", linestyle="-")
plt.axhline(y=0.33333, color="g", linestyle="-")

# Monte Carlo Simulation
def monte_carlo(n):

    # Calculation switch and stick wins:
    switch_wins = 0
    stick_wins = 0

    for i in range(n):
        # randomly placing the car and goats behind the three doors
        random.shuffle(doors)
        # Contestan's choice
        k = random.randrange(2)
        # if the contestant doesn't get the car
        if doors[k] != "car":
            switch_wins += 1
        # if the contestant got car
        else:
            stick_wins += 1

        # Updating the list values:
        switch_win_prob.append(switch_wins/(i+1))
        stick_win_prob.append(stick_wins/(i+1))

    print("Winning probability if you always switch:" ,switch_win_prob[-1])
    print("Winning probability if you always stick:" ,stick_win_prob[-1])

monte_carlo(1000)
plt.plot(switch_win_prob)
plt.plot(stick_win_prob)
plt.show()