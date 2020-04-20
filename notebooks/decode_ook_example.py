# Script from http://morethanuser.blogspot.com/2016/08/decode-ook-with-rtlfm-and-python-script.html
from numpy import memmap, where, trim_zeros, char, mean, max
from scipy.signal import medfilt
from re import split
from sys import argv
 
import rf_analyzer
 
# load raw audio file with uint16 encoding and
# 38400 sample rate
data = memmap(argv[1],  dtype='uint16', mode='r')
 
# "calculate" noise level from begining of a file
noise = max(data[0:1000])
 
# Sample rate was set to 38400, which is 4 * 9600
# now downsample it 4 times by reading every 4-th integer
data = data[0::4]
 
# set baudrate for OOK analyzer, it will calculate original
# signal baudrate
baud = 9600
 
# create median vector
median = medfilt(data)
 
# every signal above will be set to 1, every below to 0
# use additional noise floor
normalized = where(median > mean(median) + noise, 1, 0)
 
# trim trailing and leading zeros
normalized_no_zeros = trim_zeros(normalized)
 
# convert above [1,1, ... , 0, 0, ... , 1,1] to string 11...00...11
bin_str =  char.mod('%d', normalized_no_zeros).tostring()
 
# split string into packets separted by high count of '0', 20 is enought for
# 2400 to 9600 data rate
for splitted in split('0{20,}', bin_str):
   if (len(splitted) > 0):
      # make sure that we've got some zeroes at the end for regexp
      splitted+="0" * 20
      # analyze and print
      rf_analyzer.analyze_ook(splitted, baud, bin_input=True)