import collections
import re
import math
import csv
class Tokenizer:
    def __init__(self):
        super().__init__()
        return

    def lowerCaseAndRemovePunctuation(self,review):
        tokens = re.findall("[-\w']+", review)
        tokens = [tokens[0]] + [token.lower() for token in tokens[1:]]
        return tokens

class FeatureExtractor:



    def __init__(self):
        super().__init__()
        return

    def f1_countPositive(self,doc):
        posCount = 0
        for val in doc:
            if val in setOfPositiveWords:
                # print(val)
                posCount += 1
        return posCount

    def f2_countNegative(self,doc):
        negCount = 0
        for val in doc:
            if val in setOfNegativeWords:
                # print(val)
                negCount += 1
        return negCount

    def f3_existsNo(self,doc):
        if 'no' in doc:
            return 1
        return 0

    def f4_count1stAnd2ndPronouns(self,doc):
        pronounSet = {"i", "me", "mine", "my", "you", "your", "yours", "we", "us", "ours"}
        pronounCount = 0
        for val in doc:
            if val in pronounSet:
                pronounCount += 1
        return pronounCount

    def f5_existsExclamation(self,review):
        tokens = re.findall("[!]+", review)
        if len(tokens) > 0:
            return 1 
        return 0

    def f6_wordCount(self,doc):

        return round(math.log(len(doc)),2)

positiveWords = open('positive-words.txt')
negativeWords = open('negative-words.txt')

def addToSet(givenSet,fileName):
    with fileName as reader:
        line = reader.readline()
        while line != '':  
            givenSet.add(line.rstrip("\n"))
            line = reader.readline()
    return

setOfPositiveWords = set()
setOfNegativeWords = set()

addToSet(setOfPositiveWords,positiveWords)
addToSet(setOfNegativeWords,negativeWords)

myTokenizer = Tokenizer()
myFeatureExtractor = FeatureExtractor()

file = open(r'c:\Users\hunar\Downloads\CU Boulder Semester 1\CSCI 5832 NLP\Assignment_1\nlp-env-1\hotelPosT-train.txt', 'r')
posReviews =  list(map(str.strip,file.readlines()))
file.close()
file = open(r'c:\Users\hunar\Downloads\CU Boulder Semester 1\CSCI 5832 NLP\Assignment_1\nlp-env-1\hotelNegT-train.txt', 'r',encoding="utf8")
negReviews = file.readlines()
file.close()

with open('jain-hunar-assgn2-part1.csv', 'w',newline='') as file:     
    write = csv.writer(file)
    
    def addReview(totalReviews,val=None):
        for review in totalReviews:
            # print(type(review),review)
            tokens = myTokenizer.lowerCaseAndRemovePunctuation(review)
            posCount = myFeatureExtractor.f1_countPositive(tokens[1:])
            negCount = myFeatureExtractor.f2_countNegative(tokens[1:])
            existsNo = myFeatureExtractor.f3_existsNo(tokens[1:])
            pronounCount = myFeatureExtractor.f4_count1stAnd2ndPronouns(tokens[1:])
            existsExclamation = myFeatureExtractor.f5_existsExclamation(review)
            wordCount = myFeatureExtractor.f6_wordCount(tokens[1:])
            if val == None:
                array = list(map(str,[tokens[0],posCount,negCount,existsNo,pronounCount,existsExclamation,wordCount]))
            else:
                array = list(map(str,[tokens[0],posCount,negCount,existsNo,pronounCount,existsExclamation,wordCount,val]))
            write.writerow(array) 
    addReview(posReviews,1)
    addReview(negReviews,0)   

