from PIL import Image
from selenium import webdriver
from pynput.mouse import Button, Listener, Controller
from pynput import keyboard
import pdb
import time
from random import randint
import pandas as pd
import numpy as np

class MyException(Exception):
    pass

def rgb2hex(rgb):
    return '#%02x%02x%02x' % rgb

def getNumber(image, pos):
    rgb = image.getpixel(pos)
    hexColor = rgb2hex(rgb[:3])
    if hexColor in COLOR2NUMBER.keys():
        return COLOR2NUMBER[hexColor]
    else:
        return 0

def getField(image):
    newField = [None]*20
    for ind, field in enumerate(FIELDPOS.values()):
        newField[ind] = getNumber(image, field)
    return newField


def newNumbers(img):
    num1 = getNumber(img, POS1)
    num2 = getNumber(img, POS2)
    return (num1, num2)

def generatePosition(numbers):
    pos1 = randint(0,19)
    pos2 = (pos1 + 10) % 20
    return (pos1, pos2)

def getScore(driver):
    scores = driver.find_elements_by_xpath("//div[@class='score-value']")
    score = scores[0].text
    score = score.replace(' ','')
    if score=='':
        score = 0
    return int(score)

def getGameState():
    pass


def on_keyf8(key):
    if (key==keyboard.Key.f8):
        pdb.set_trace()

def playTutorial():
    actionSequence = [15, 13, 11, 'SWITCH', 13, 1]
    for step in actionSequence:
        mouse.position = BROWSERFIELDS[step]
        mouse.click(Button.left, 2)
        time.sleep(1)

def shotBall(newPos):
    mouse.position = BROWSERFIELDS[newPos[0]]
    mouse.click(Button.left, 2)

def gameOver(field):
    return 0 not in field



key_listener = keyboard.Listener(on_release=on_keyf8)
key_listener.start()


COLOR2NUMBER = {'#303440':-1, '#c876d6':1, '#7cc2f7':2, '#b1e07b':3, '#ffd59':4, '#ffd659':4, '#987ee6':5, '#de8a64':6, '#64debf':7, '#64debf':8}

POS1= (617, 450)
POS2 = (617, 550)
FIELDPOS = {0:(635,285), 1:(699,295), 2:(755,321), 3:(799,365), 4:(826,421), 5:(837,485), 6:(828,548), 7:(799,601), 8:(754,647), 9:(698, 674), 10:(639,683), 11:(580,674), 12:(523, 648), 13:(474,600), 14:(444,546), 15:(435,483), 16:(446,422), 17:(474,366), 18:(517,323), 19:(573, 294)}
BROWSERFIELDS = {0:(710,440), 1:(770,450), 2:(830,480), 3:(875,525), 4:(900,585), 5:(910,640), 6:(900,705), 7:(870,760), 8:(830,800), 9:(770,830), 10:(710,840), 11:(650,830), 12:(595,800), 13:(546,753), 14:(520,705), 15:(510,640), 16:(520,580), 17:(550,525), 18:(595,480), 19:(650,455), 'SWITCH':(490, 860)}


html_file = 'http://www.dotowheel.com'
save_path = "screenshots/image001.png"


driver = webdriver.Firefox()
driver.get(html_file)
driver.set_window_position(0,0)

center = (710,640)

mouse = Controller()
time.sleep(2)

playTutorial()

GAME_OVER = False
NumberOfMouseClicks = 1

#pdb.set_trace()

#columns =
#Q = pd.DataFrame(np.zeros(), columns=range(0,20).append('state'), index=range(0,10))
REWARD = -1
LAST_SCORE = 0
LAST_FIELD = [None] * 20

def on_click(x, y, button, pressed):
    global GAME_OVER
    global LAST_SCORE
    global LAST_FIELD
    global NumberOfMouseClicks
    print 'NumberOfMouseClicks: ' + str(NumberOfMouseClicks)

    if NumberOfMouseClicks%2==0:

        time.sleep(2)
        driver.save_screenshot(save_path)
        img = Image.open(save_path)
        DOTS = newNumbers(img)
        FIELD = getField(img)

        GAME_OVER = gameOver(FIELD)
        if GAME_OVER:
            raise MyException()

        SCORE = getScore(driver)
        REWARD = SCORE - LAST_SCORE
        print 'SCORE: ' + str(SCORE)
        print 'REWARD: ' + str(REWARD)
        print 'FIELD: ' + str(FIELD)
        print 'DOTS: ' + str(DOTS)

        #pdb.set_trace()
        newPos = generatePosition(DOTS)
        print 'PLACE DOTS ON: ' + str(newPos)
        shotBall(newPos)
        time.sleep(2)

        LAST_FIELD = FIELD
        LAST_SCORE = SCORE

    NumberOfMouseClicks += 1


with Listener(on_click=on_click) as listener:
    try:
        print 'TRY'
        listener.join()
    except MyException as e:
        print 'GAME OVER!!! :((('
        listener.stop()

    pdb.set_trace()



driver.quit()



