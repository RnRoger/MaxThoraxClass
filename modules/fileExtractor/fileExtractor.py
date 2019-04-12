import shutil
import os
import re

#' Function to extract photos with the correct diseases from the xRay directory
def fileExtractor(inputPath = os.getcwd(),
                  outputPath = "dataset",
                  diseasesRe = "(atelectasis|infiltration|nodule)",
                  indexPath = "Data_Entry_2017.csv",
                  overviewPath = "overview.csv"):

    os.makedirs(outputPath)

    regexp = re.compile(diseasesRe)

    indexFile = open(indexPath,"r")                         # Contains a file with the diseases and patient information
    overview = open(overviewPath,"w")                       # File that wil contain the filename and the corresponding disease
    newText = ""
    patDict = {}
    for rule in indexFile:
        ruleParts = rule.split(",")

        if ruleParts[4].endswith("Y"):
            age = int(ruleParts[4][0:len(ruleParts[4])-1])
        else:
            age = 0

        if regexp.search(ruleParts[1].lower()) and \
                age > 20 and len(ruleParts[1].split("|")) == 1:
            if ruleParts[3] in patDict.keys():
                dis = patDict[ruleParts[3]]
                if not ruleParts[1] in dis:
                    temp = patDict[ruleParts[3]]
                    temp.append(dis)
                    patDict[ruleParts[3]] = temp
                    doIt = True
                else:
                    doIt = False
            else:
                patDict[ruleParts[3]] = [ruleParts[1]]
                doIt  = True

            if doIt:
                shutil.copyfile(inputPath+os.sep+ruleParts[0],
                                outputPath+os.path+ruleParts[0])
                newText = newText + rule

    indexFile.close()
    overview.write(newText)
    overview.close()