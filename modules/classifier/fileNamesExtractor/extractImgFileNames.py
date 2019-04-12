import os
import random

# Function to gather the filenames and diseases, and divide them in
# trainingsset, validationset and testset
def extractImgFileNames(dir = "[PATH TO DIRECTORY WITH IMAGES]",
                        overviewPath = "[PATH TO CSV WITH DISEASES]"):

  filenames = [[],[],[]]
  indexDict = {
    "atelectasis" : 0,
    "infiltration" : 1,
    "nodule" : 2
  }
  diseaseDict = getDDict(overviewPath)
  for filename in os.listdir(dir):
      if filename.endswith(".png"):
        disease = diseaseDict[filename]
        name = dir+os.sep+filename
        disNo = indexDict[disease]
        filenames[disNo] += [name]


  x,y = sortData(filenames)
  return x[0], y[0], x[1], y[1], x[2], y[2]

# Create a dictionary that contains filenames and diseases
def getDDict(overviewPath):
  file = open(overviewPath)
  dict = {}
  for rule in file:
    part = rule.split(",")
    dict[part[0]] = part[1].split("|")[0].lower()

  return dict

# Sort the data into multiple datasets
def sortData(filenames):
  sortArray = [[],[],[]]
  diseases = [[],[],[]]

  disNo = 0
  disDict = {
    0 : "atelectasis",
    1 : "infiltration",
    2 : "nodule"
  }
  for array in filenames:
    lenArray = len(array)
    lenTest = int(lenArray * 0.2)
    lenVal = int((lenArray - lenTest) * 0.2)

    array, testArray = getRandomPop(lenTest, array)
    array, valArray = getRandomPop(lenVal, array)

    train = sortArray[0] + array
    val = sortArray[1] + valArray
    test = sortArray[2] + testArray

    trainDis = diseases[0] + ([disDict[disNo]] * len(array))
    valDis = diseases[1] + ([disDict[disNo]] * lenVal)
    testDis = diseases[2] + ([disDict[disNo]] * lenTest)

    if disNo > 0:
      random.seed(disNo)

      random.shuffle(train)
      random.shuffle(val)
      random.shuffle(test)

      random.seed(disNo)

      random.shuffle(trainDis)
      random.shuffle(valDis)
      random.shuffle(testDis)

    sortArray = [train,val,test]
    diseases = [trainDis,valDis,testDis]

    disNo += 1

  return sortArray, diseases


# Extract items randomly from array
def getRandomPop(count,array):
  newArray = []
  for i in range(0,count):
    newArray += [array.pop(random.randrange(len(array)))]
    random.shuffle(newArray)
    random.shuffle(array)

  return array, newArray