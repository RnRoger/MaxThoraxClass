import shutil
import os
import re
import shutil

os.makedirs("dataset")
regexp = re.compile("(atelectasis|infiltration|nodule)")

indexFile = open("Data_Entry_2017.csv","r")
overview = open("overview.csv","w")
newText = ""
patDict = {}
for rule in indexFile:
    ruleParts = rule.split(",")
    if ruleParts[4].endswith("Y"):
        age = int(ruleParts[4][0:len(ruleParts[4])-1])
    else:
        age = 0

    if regexp.search(ruleParts[1].lower()) and age > 20 and len(ruleParts[1].split("|")) == 1:
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
            shutil.copyfile(ruleParts[0],"dataset/"+ruleParts[0])
            newText = newText + rule

indexFile.close()
overview.write(newText)
overview.close()