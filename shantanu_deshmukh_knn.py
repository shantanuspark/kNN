'''
Created on Sep 7, 2017

@author: Shantanu Deshmukh

'''
from matplotlib import pyplot as plt 
import sys,os

class Agent(object):
    '''
    Agent class senses input, perform computations on input(function()) and takes action(actuator())
    '''

    def __init__(self, k, traRows, attributeInfo, optimizationParams):
        '''
        initializes k, training percept, information of each attribute and optimization parameters like data distribution, distance algorithm
        '''
        self.k=k
        self.traRows=traRows
        self.attributeInfo = attributeInfo
        self.optimizationParams = optimizationParams
        
    def sensor(self, tstRow):
        '''
        senses the testing row and passes to the function
        '''
        return self.function(tstRow)
        
    def function(self, tstRow):
        '''
        Finds k nearest neighbours of the test row, according to the optimization parameters, and passes to the actuator to find the label
        '''
        neighbourDistance={}
        neighbours=[]
        if self.optimizationParams["dataDistribution"] == 1:
            tstRow = generalize(tstRow,self.attributeInfo)
        for i in range(len(self.traRows)):
            if self.optimizationParams["dataDistribution"] == 1:
                '''
                Enters here if user has selected to distribute the data and as per the distance algorithm fill in the neighbourDistance list
                '''
                if self.optimizationParams["distanceAlgo"] == 1:
                    attrType = []
                    for info in self.attributeInfo:
                        if info["maxRange"]-info["minRange"]<4:
                            attrType.append("cat")
                        else:
                            attrType.append("conti")
                    neighbourDistance[i]=intelligentDistance(tstRow, generalize(self.traRows[i], self.attributeInfo), attrType)
                elif self.optimizationParams["distanceAlgo"] == 2:
                    neighbourDistance[i]=manhattanDistance(tstRow, generalize(self.traRows[i], self.attributeInfo))
                else:
                    neighbourDistance[i]=euclidianDistance(tstRow, generalize(self.traRows[i], self.attributeInfo))            
            else:
                '''
                if user does not want to distribute data evenly, calculate distance without generalize() and fill the neighboutDistance list
                '''
                if self.optimizationParams["distanceAlgo"] == 1:
                    attrType = []
                    for info in self.attributeInfo:
                        if info["maxRange"]-info["minRange"]<4:
                            attrType.append("cat")
                        else:
                            attrType.append("conti")
                    neighbourDistance[i]=intelligentDistance(tstRow, self.traRows[i],attrType)
                elif self.optimizationParams["distanceAlgo"] == 2:
                    neighbourDistance[i]=manhattanDistance(tstRow, self.traRows[i])
                else:
                    neighbourDistance[i]=euclidianDistance(tstRow, self.traRows[i])
        for i in sorted(neighbourDistance, key=neighbourDistance.get):
            neighbours.append(self.traRows[i])
        return self.actuator(neighbours[0:int(self.k)])
        
    def actuator(self, neighbours):
        '''
        From the nearest neighbours counts same type of neighours and returns label of neighbour with maximumm count 
        '''
        labels={}
        for neighbour in neighbours:
            label = neighbour[len(neighbour)-1]
            if labels.has_key(label):
                labels[label]+=1
            else:
                labels[label]=1
        maxCount = 0
        predictedLabel = ""
        for label in labels.items():
            if label[1]>maxCount:
                maxCount=label[1]
                predictedLabel=label[0]
        return predictedLabel

def generalize(row, attributeInfo):
    '''
    Generalizes attribute data in a row.
    for e.g. if value of one attribute is 250 and the range of attribute is (150,350) then this function will reduce its range to (0,100) so its new value will be 50
    This will be done on each attribute so that each attribute will have same weight while calculating distance
    '''
    generalizedRow = []
    i=0
    for info in attributeInfo:
        generalizedRow.append(round(100/(info["maxRange"]-info["minRange"])*(float(row[i])-info["minRange"]),2))
        i+=1
    generalizedRow.append(row[i])
    return generalizedRow
    
def euclidianDistance(row1,row2):
    '''
    Calculates euclidian distance between 2 rows
    '''
    distance = 0
    for i in range(len(row1)-1):
        distance += pow((float(row1[i]) - float(row2[i])), 2)
    return distance

def manhattanDistance(row1,row2):
    '''
    Calculates manhattan distance between 2 rows
    '''
    distance = 0
    for i in range(len(row1)-1):
        distance += abs(float(row1[i]) - float(row2[i]))
    return distance

def intelligentDistance(row1,row2,attrType):
    '''
    If an attribute is Categorical with less then 4 categories, it will use manhattan distance else it will use euclidian distance
    '''
    distance = 0
    for i in range(len(row1)-1):
        if(attrType[i] == "cat"):
            distance += abs(float(row1[i]) - float(row2[i]))
        else:
            distance += pow((float(row1[i]) - float(row2[i])), 2)
    return distance


class Environment(object):
    '''
    Environment for the agent. It feeds percepts to the agent's sensor function, calculates accuracy and plots graphs accordingly. 
    '''

    def __init__(self, k, folderName, optimizationParams):
        '''
        Sets value of k, folderName and optimizationParams as set by the user during bigbang()
        initializes accuracy and attributeInfo lists
        '''
        self.folderName = folderName
        self.k = k 
        self.attributeInfo = []
        self.accuracy = []
        self.optimizationParams = optimizationParams
        
    def createAgent(self, trainingRows):
        '''
        Creates agent with the essential parameters
        '''
        return Agent(self.k, trainingRows, self.attributeInfo, self.optimizationParams)
    
    def classifyAndComputeAccuracy(self):
        '''
        Calls agent to get the test data classified and then computes accuracy
        '''
        isAttrbuiteInfoExtracted = 0;
        for j in range(1,11):
            with open(self.folderName+"/"+self.folderName+"-10dobscv-"+repr(j)+"tra.dat",'r') as dataFile:
                traRows = []
                i = 0
                for line in dataFile:
                    line = line.strip()
                    if line[0] != '@':
                        traRows.insert(i, line.split(",")) 
                        i+=1
                    elif not isAttrbuiteInfoExtracted:
                        self.extractAttributeInfo(line.split())
                if len(self.attributeInfo) > 0:
                    isAttrbuiteInfoExtracted = 1        
                agent = self.createAgent(traRows)
                tstCount,correctPrediction = 0,0
                with open(self.folderName+"/"+self.folderName+"-10dobscv-"+repr(j)+"tst.dat",'r') as tstRows:
                    for tstRow in tstRows:
                        tstRow = tstRow.strip()
                        if tstRow[0] != '@':
                            tstRow = tstRow.split(",")
                            pLabel = agent.sensor(tstRow)
                            if tstRow[len(tstRow)-1] == pLabel:
                                correctPrediction+=1
                            tstCount+=1
                    self.accuracy.insert(j, round(correctPrediction/float(tstCount),2))
        
    def process(self):
        '''
        If k is 0, evaluates accuracy for different values of k. Otherwise, computes accuracy for each iteration of percept for given value of k.
        Draws accuracy graphs accordingly
        '''
        if int(self.k) == 0:
            averageAccuracy = {}
            minAccuracy = 0
            optimalK = 0
            for kNew in range(1,self.optimizationParams["maxK"]):
                self.attributeInfo = []
                self.k = kNew
                self.classifyAndComputeAccuracy()
                averageAccuracy[kNew]=sum(self.accuracy)/len(self.accuracy)
            for kNew in averageAccuracy.keys():
                if averageAccuracy[kNew] > minAccuracy:
                    minAccuracy =  averageAccuracy[kNew]
                    optimalK = kNew
            x = range(1,self.optimizationParams["maxK"])
            y = averageAccuracy.values()
            title = "For k = "+repr(optimalK)+" accuracy is maximum ("+repr(round(averageAccuracy[optimalK],2))+")"
            supTitle = "Accuracy vs k plot"
            self.drawGraph(x, y, title, supTitle, "Value of k", "Accuracy")
        else:
            self.classifyAndComputeAccuracy()        
            overallAcc = sum(self.accuracy)/len(self.accuracy)
            x = range(0,len(self.accuracy))
            y = self.accuracy
            title = "For k = "+repr(self.k)+", average accuracy obtained is "+repr(round(float(overallAcc),2))
            supTitle = "Classification Accuracy Plot"
            self.drawGraph(x, y, title, supTitle, "Iteration", "Accuracy")
              
    def extractAttributeInfo(self,row):
        '''
        Reads attribute info at the start of the training data and intelligently extracts information 
        '''
        if row[0] == "@attribute" and row[1].find("{")<0:
            attribute={}
            attribute["name"] = row[1]
            attribute["minRange"] = float(row[2][row[2].find("[")+1:row[2].find(",")])
            attribute["maxRange"] = float(row[2][row[2].find(",")+1:row[2].find("]")])
            self.attributeInfo.append(attribute)
     
    def drawGraph(self,x,y,title,supTitle,xlabel,ylabel):
        '''
        Plots graph as per the information provided in parameters
        '''
        plt.suptitle(supTitle)
        plt.title(title)
        plt.xlabel(xlabel) 
        plt.ylabel(ylabel) 
        plt.plot(x,y,'b', markersize=15)
        plt.grid(True) 
        plt.show()

                
print "..####  Hey there!! Welcome to the class predictor bot  ####.."

error = ""

def bigBang(k, folderName, optimizationParams):
    '''
    Creates Environment as per the user's preferences
    '''
    env = Environment(k, folderName, optimizationParams)
    env.process()
    
if len(sys.argv)>1:
    '''
    If command line arguments are provided, menu is not displayed and processing tarts right away
    '''
    bigBang(sys.argv[1], sys.argv[2], {"distanceAlgo":"1","dataDistribution":"1"})
else:    
    '''
    If command line arguments are not provided, menu is displayed to accept input from the user
    '''
    while 1:
        '''
        Display Menu
        '''
        optimizationParams = {}
        if error == "":
            k = raw_input("\nEnter the value of k (Enter 0 if you want me to evaluate how accuracy depends on k) \n")
            if int(k) == 0:
                maxK = raw_input("\nEnter maximum value of k (Note - Computing time will be higher for high values of k) \n") 
                optimizationParams["maxK"]=int(maxK)
            distanceAlgo = raw_input("\nSelect one of the distance algorithms below:\n1.Intelligent(I will decide distance algorithm for each attribute individually)\n2.Euclidian\n3.Manhattan\n")
            optimizationParams["distanceAlgo"]=int(distanceAlgo)
            dataDistribution = raw_input("\nDo you want attribute data to be in the same range interval?\n1.Yes\n2.No\n")
            optimizationParams["dataDistribution"]=int(dataDistribution)
        
        dirs = next(os.walk('.'))[1]

        getFolderName = "\nSelect the folder with data\n"
        i=0
        folderNames=[]
        for fileName in dirs:
            i+=1
            getFolderName+=repr(i)+"."+fileName+"\n" 
            folderNames.append(fileName)
        folderName = raw_input(getFolderName)  
        folderName = folderNames[i-1]
    
        try:
            '''
            Start Processing after accepting user input
            '''
            bigBang(k, folderName, optimizationParams)
        except IOError:
            print "#### Data files not correctly formatted or not found!!"
            error = "IOError"
            
        
        exitInput = raw_input("\nEnter e to exit.. Enter any other key to rerun..\n")
        if exitInput == 'e':
            print "THANK YOU !!"
            exit()

