#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 10:47:31 2021

@author: sven
"""

import csv
import os

# import random, sys
# filenames = os.listdir('./500x500')
# random.shuffle(filenames)
# with open('img_list_sample.csv', 'w') as f:
#     for filename in filenames[:200]:
#         f.write(filename+'\n')
# sys.exit(1)

SIZE = 500
CHAR_ROOT = './Character charts/CSV/'
FONT_ROOT = './Fonts/'
CHAR_OUT  = './500x500/'
FONT = {'alphabetum':'ALPHABETUM_v_14.25.otf',
        'bamum':'NotoSansBamum-Regular.ttf',
        'duployan':'DuployanProp.ttf',
        'han':'BabelStoneHan.ttf',
        'kikakui':'KikakuiSans.ot.ttf',
        'legacycomp':'LegacyComputing.otf',
        'miao':'MiaoUnicode-Regular.ttf',
        'quivira':'Quivira.otf',
        'suttonline':'SuttonSignWritingLine.ttf',
        'symbola':'Symbola.otf',
        'yi':'NotoSansYi-Regular.ttf'}

char_lists = {}

for (k,v) in FONT.items():
    with open(CHAR_ROOT+k+'.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        char_list = []
        for row in reader:
            for string in row:
                char = string.strip()
                if len(char)==1:
                    char_list.append(char)
        char_lists[k] = char_list

total = 0
for (k,v) in char_lists.items():
    print('List: {0}. Length: {1} characters.'.format(k, len(v)))
    total+=len(v)
print('\nTotal character count: {}'.format(total))
        
for (k,v) in char_lists.items():
    if k == 'symbola':
        for char in v:
            if k == 'suttonline':
                outcome = os.system(f'convert -size {SIZE}x{SIZE} -gravity North -pointsize 300 -font "{FONT_ROOT+FONT[k]}" label:{char} {CHAR_OUT}{hex(ord(char))}_{k}.png')
            else:
                outcome = os.system(f'convert -size {SIZE}x{SIZE} -gravity center -font "{FONT_ROOT+FONT[k]}" label:{char} {CHAR_OUT}{hex(ord(char))}_{k}.png')
            if outcome!=0:
                outcome = os.system(f'convert -size {SIZE}x{SIZE} -gravity center -font "{FONT_ROOT+FONT[k]}" label:"\{char}" {CHAR_OUT}{hex(ord(char))}_{k}.png')