import pyautogui as pag
import random
import time

while True:
    x, y = random.randint(600, 700), random.randint(200, 400)
    pag.moveTo(x, y, duration=0.25)
    time.sleep(2)
