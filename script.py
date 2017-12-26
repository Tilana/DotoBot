#!/usr/bin/env python
from PIL import Image
from selenium import webdriver
from pynput.mouse import Button, Listener, Controller
from pynput import keyboard, mouse
import pdb
import time
from random import randint
import numpy as np
import pandas as pd
from Q_Net import Q_Net
import tensorflow as tf
import os

COLOR2NUMBER = {'#303440':-1, '#c876d6':1, '#7cc2f7':2, '#b1e07b':3, '#ffd59':4, '#ffd659':4, '#987ee6':5, '#de8a64':6, '#64debf':7, '#64debf':8}

POS1= (617, 450)
POS2 = (617, 550)
FIELDPOS = {0:(635,285), 1:(699,295), 2:(755,321), 3:(799,365), 4:(826,421), 5:(837,485), 6:(828,548), 7:(799,601), 8:(754,647), 9:(698, 674), 10:(639,683), 11:(580,674), 12:(523, 648), 13:(474,600), 14:(444,546), 15:(435,483), 16:(446,422), 17:(474,366), 18:(517,323), 19:(573, 294)}
BROWSERFIELDS = {0:(710,440), 1:(770,450), 2:(830,480), 3:(875,525), 4:(900,585), 5:(910,640), 6:(900,705), 7:(870,760), 8:(830,800), 9:(770,830), 10:(710,840), 11:(650,830), 12:(595,800), 13:(546,753), 14:(520,705), 15:(510,640), 16:(520,580), 17:(550,525), 18:(595,480), 19:(650,455), 'SWITCH':(490, 860)}


mouse = Controller()
Q_Net = Q_Net(22,20)

GAME_OVER = False
NumberOfMouseClicks = 1

LAST_SCORE = 0
LAST_FIELD = [None] * 20


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

def generateComplementaryPosition(pos1):
    pos2 = (pos1 + 10) % 20
    return (pos1, pos2)

def getScores(driver):
    scores = driver.find_elements_by_xpath("//div[@class='score-value']")
    score = scores[0].text
    score = score.replace(' ','')
    high_number = scores[2].text
    high_number = high_number.replace(' ','')
    if score=='':
        score = 0
    return (int(score), int(high_number))

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

html_file = 'http://www.dotowheel.com'
save_path = "screenshots/image001.png"

infoPath = 'model/info.csv'

driver = webdriver.Firefox()
driver.get(html_file)
driver.set_window_position(0,0)

center = (710,640)

time.sleep(2)

playTutorial()

e = 0.25
info = pd.DataFrame(columns=['score', 'highest number', 'clicks'])
memory = pd.DataFrame(columns=['state', 'action', 'reward', 'state+1', 'action+1'])

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if os.path.exists('model/doto_model-1.meta'):
        pretrained_model = tf.train.import_meta_graph('model/doto_model-1.meta')
        pretrained_model.restore(sess, tf.train.latest_checkpoint('./model/'))
        info = pd.read_csv(infoPath)

    def on_click(x, y, button, pressed):
        global GAME_OVER
        global LAST_SCORE
        global LAST_FIELD
        global LAST_POS
        global NumberOfMouseClicks
        global Q_values
        global LAST_STATE
        global memory
        global info

        if NumberOfMouseClicks%2==0:

            time.sleep(2)
            driver.save_screenshot(save_path)
            img = Image.open(save_path)
            DOTS = newNumbers(img)
            FIELD = getField(img)

            GAME_OVER = gameOver(FIELD)

            SCORE, HIGH_NUMBER = getScores(driver)
            REWARD = SCORE - LAST_SCORE
            if REWARD == 0:
                REWARD = 0
            if LAST_FIELD == FIELD:
                REWARD = -1
            if REWARD > 0:
                REWARD = 2

            #pdb.set_trace()

            print 'SCORE: ' + str(SCORE)
            print 'REWARD: ' + str(REWARD)
            print '\n - - - \n'
            print 'FIELD: ' + str(FIELD)
            print 'DOTS: ' + str(DOTS)

            state = np.array(FIELD + list(DOTS))

            if NumberOfMouseClicks>2:

                Q1 = sess.run(Q_Net.logits, feed_dict={Q_Net.inputs:state.reshape((1,-1,1))})

                maxInd= Q1.argmax()
                maxQ1 = Q1[0,maxInd]
                target_Q = Q_values.reshape(20)
                target_Q[LAST_POS] = REWARD + 0.99 * maxQ1

                loss, _, W1 = sess.run([Q_Net.loss, Q_Net.update, Q_Net.logits], feed_dict={Q_Net.inputs:LAST_STATE.reshape((1,-1,1)), Q_Net.target_Q:target_Q.reshape(1,-1)})
                print 'W1: ' + str(W1)
                print 'LOSS ' + str(loss)

                experience = {'state': LAST_STATE, 'action':LAST_POS, 'reward': REWARD, 'state+1': state, 'action+1':maxInd}
                memory.loc[len(memory)] = experience

            if GAME_OVER:
                print '\n GAME OVER :((( \n'
                info.loc[len(info)] = {'score': SCORE, 'highest number': HIGH_NUMBER, 'clicks': NumberOfMouseClicks}
                info.to_csv(infoPath, index=False)
                REWARD = -5
                loss, _, W1 = sess.run([Q_Net.loss, Q_Net.update, Q_Net.logits], feed_dict={Q_Net.inputs:LAST_STATE.reshape((1,-1,1)), Q_Net.target_Q:target_Q.reshape(1,-1)})
                saver.save(sess, 'model/doto_model', global_step=1)
                sess.close()
                driver.quit()
                os.execv('script.py', ['python'])

            action, Q_values = sess.run([Q_Net.predict, Q_Net.logits], feed_dict={Q_Net.inputs:state.reshape((1,-1,1))})

            if np.random.rand(1) < e:
                print 'Random Action'
                action[0] = randint(0,19)

            POS = generateComplementaryPosition(action[0])
            print 'PLACE DOTS ON: ' + str(POS)
            shotBall(POS)
            time.sleep(2)

            LAST_FIELD = FIELD
            LAST_SCORE = SCORE
            LAST_POS = POS[1]
            LAST_STATE = state
            Q_values = Q_values

        NumberOfMouseClicks += 1


    time.sleep(2)

    mouse_listener = Listener(on_click=on_click)
    mouse_listener.start()

    # Start Game
    mouse.click(Button.left, 1)
    time.sleep(2)
    mouse.position = BROWSERFIELDS[2]
    mouse.click(Button.left, 1)
    time.sleep(6000)

