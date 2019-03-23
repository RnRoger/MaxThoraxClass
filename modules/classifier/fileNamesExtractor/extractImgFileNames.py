import os
import re

def extractImgFileNames(dir = "[PATH TO MAP WITH IMAGES]",
                        overviewPath = "[PATH TO CSV WITH DISEASES]"):
  regexp = re.compile("(atelectasis|infiltration|nodule)")

  first = True
  array = []
  diseases = []
  diseaseDict = getDDict(overviewPath)
  for filename in os.listdir(dir):
      if filename.endswith(".png"):
        disease = diseaseDict[filename]
        name = dir+os.sep+filename
        if first:
          first = False
          array = [name]
        else:
          array += [name]

        if len(disease) > 1:
          for i in range(1,len(disease)+1):
            if regexp.search(disease[i-1]):
              array += [name]
              diseases = diseases + [disease[i]]
        else:
          diseases = diseases + disease

  return array, diseases

def getDDict(overviewPath):
  file = open(overviewPath)
  dict = {}
  for rule in file:
    part = rule.split(",")
    dict[part[0]] = part[1].split("|")

  return dict


