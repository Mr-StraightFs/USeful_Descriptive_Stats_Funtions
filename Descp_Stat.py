import numpy as np
import math

def mean (numbers):
    return (sum(numbers))/(len(numbers))

def std (numbers) :
    num_trans = map(lambda x: math.pow((x- mean(numbers)),2) , numbers)
    return  math.sqrt(sum (num_trans)/len(numbers))

def kurt (listX):
    num_trans = map(lambda x: math.pow((x - mean(listX)), 4), listX)
    return sum (num_trans) / (len(listX) * math.pow(std(listX),4))

def skew (listX ):
    num_trans = map(lambda x: math.pow((x - mean(listX)), 3), listX)
    return sum(num_trans) / (len(listX) * math.pow(std(listX),3))

def Cov (listX , listY) :
    num_trans = map(lambda x,y : (x-mean(listX)) * (y-mean(listY)) , listX,listY)
    return sum (num_trans)/len(listX)

def Cov_test (listX , listY) :
    num_trans = map(lambda x,y : (x-1) * (y-1) , listX,listY)
    return num_trans

def Corr (listX , listY) :
    return Cov (listX,listY) * (1/ (std(listX) * std(listY)))

list_X = [1,2,3,4,5,6,7,8,9,10]
list_Y= [7,6,5,4,5,6,7,8,9,10]

# Univariate for X
print('Mean of X : ' + str(mean(list_X)))
print('The Standard Deviation of X : ' + str(std(list_X)))
print('The Skewness of X : ' + str(skew(list_X)))
print('The Kurtosis of X : ' + str(kurt(list_X)))

# Univariate for Y
print('Mean of Y : ' + str(mean(list_Y)))
print('The Standard Deviation of Y : ' + str(std(list_Y)))
print('The Skewness of Y : ' + str(skew(list_Y)))
print('The Kurtosis of Y : ' + str(kurt(list_Y)))

# Bivariate (X,Y)
print ('Cov (X,Y) is : ' +str(Cov(list_X,list_Y)))
print ('Corr (X,Y) is : ' +str(Corr(list_X,list_Y)))
