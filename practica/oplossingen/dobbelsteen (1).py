#!/usr/bin/env python

import numpy as np
import random

class Dobbelsteen:
    
    # Constructor
    def __init__(self):
        self.values = set(range(1, 7))  
        self.history = list()
        self.roll()
        self.history = list()
        self.faces = {

            1: (

                "┌─────────┐\n"
                "│         │\n"
                "│    ●    │\n"
                "│         │\n"
                "└─────────┘\n"
            ),

            2: (

                "┌─────────┐\n"
                "│  ●      │\n"
                "│         │\n"
                "│      ●  │\n"
                "└─────────┘"
            ),

            3: (

                "┌─────────┐\n"
                "│  ●      │\n"
                "│    ●    │\n"
                "│      ●  │\n"
                "└─────────┘"
            ),

            4: (

                "┌─────────┐\n"
                "│  ●   ●  │\n"
                "│         │\n"
                "│  ●   ●  │\n"
                "└─────────┘"

            ),

            5: (

                "┌─────────┐\n"
                "│  ●   ●  │\n"
                "│    ●    │\n"
                "│  ●   ●  │\n"
                "└─────────┘"
            ),

            6: (

                "┌─────────┐\n"
                "│  ●   ●  │\n"
                "│  ●   ●  │\n"
                "│  ●   ●  │\n"
                "└─────────┘"

            )

        }

    def getList(self):
        return list(self.values)

    def roll(self):
        newValue = random.choice(self.getList())
        self.history.append(newValue)
        self.number = newValue

    def getNumber(self):
        return self.number

    def getHistory(self):
        return np.asarray(self.history)

    def show(self):        
        return str(self.faces.get(self.number))

def main():
    d = Dobbelsteen()
    d.roll()
    # print(d.show())
    print(d.getHistory())

if __name__ == main():
    main()