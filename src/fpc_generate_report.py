#!/usr/bin/env python

import argparse
import os
import json
import datetime

locTable = {}
INT_MIN = -2147483648
INT_MAX = 2147483647
filesAnalyzed = 0

htmlHeader = """
<!DOCTYPE html>
<html>

<head>
<style>
table, th, td, tr {
    border: 0px solid black;
    border-spacing: 3px;
    font-family: courier;
    font-size: 90%;
    text-align: left;
}

style1 {
    font-family: courier;
    font-size: 100%;
    color: #703A10;
}

h3 {
    font-family: Arial;
}

h2 {
    font-family: Arial;
}

h1 {
    font-family: Arial;
}

p {
    font-family: Arial;
    color: #667CAD;
}

</style>
</head>

<body>
"""

htmlEnd = """
    
    </body>
    </html>
"""

def generateHTML():
    fd = open('fpc_report_integers.html', 'w')
    fd.write(htmlHeader)
    
    fd.write("<h1>\n")
    fd.write("FPChecker Report")
    fd.write("</h1>\n")
    
    now = datetime.datetime.now()
    d = str(now.year) + "-" + str(now.month) + "-" + str(now.day)
    fd.write("<p>\n")
    fd.write("Date: " + d + " <br>")
    lines = str(len(locTable.keys()))
    fd.write("Lines of code: " + lines)
    fd.write("</p>\n")
    
    fd.write("<h3>\n")
    fd.write("Signed 32-bit Integer Overflows")
    fd.write("</h3>\n")
    
    fd.write("<table>\n")
    overflows = []
    for k in locTable.keys():
        if locTable[k][2] > 0:
            overflows.append("(" + str(locTable[k][3]) + ")" + k)
    
    if (len(overflows) == 0):
        fd.write("<tr> <tr><td>NONE</td></tr> </tr>\n")
    else:
        for e in overflows:
            fd.write("<tr> <tr><td><style1>" + e + "</style1></td></tr> </tr>\n")
    fd.write("</table>\n")
        
    
    for i in [0, 1]:
        fileData = genTable(i)

        fd.write("<h3>\n")
        if i==0:
            fd.write("Int Range Analysis (Sorted by Max Value)")
        else:
            fd.write("Int Range Analysis (Sorted by Min Value)")
        fd.write("</h3>\n")

        fd.write("<table>\n")
        fd.write("<tr> <th></th> <th>Min</th> <th>Max</th> <th>Location</th> </tr>\n")

        for l in fileData.split("\n"):
            if len(l)==0:
                continue
        
            fd.write("<tr>\n")
        
            t = l.split(",")
        
            bar = t[0]
            minVal = t[1]
            maxVal = t[2]
            fileName = t[3]
        
            fd.write("<td>\n")
            fd.write(bar)
            fd.write("</td>\n")
        
            fd.write("<td>\n")
            fd.write(minVal)
            fd.write("</td>\n")
        
            fd.write("<td>\n")
            fd.write(maxVal)
            fd.write("</td>\n")
        
            fd.write("<td>\n")
            fd.write(fileName)
            fd.write("</td>\n")
        
            fd.write("</tr>\n")
        fd.write("</table>\n")
    
    fd.write(htmlEnd)
    fd.close()

def printBar(val):
    #emptyChar = '-'
    #filledChar = 'X'
    emptyChar = '&#9634;'
    filledChar = '&#9609;'
    numChars = 20
    
    ret = ''
    total = INT_MAX - INT_MIN
    chunkSize = float(total) / numChars
    val = float(val)
    filledCharPrinted = False
    for i in range(numChars):
        if (i*chunkSize + INT_MIN) <= val <= ((i+1)*chunkSize + INT_MIN) and filledCharPrinted==False:
            ret = ret + filledChar
            filledCharPrinted = True
        else:
            ret = ret + emptyChar

    return "&#9475;" + ret + "&#9475;"

def isFPCFile(input):
    v = input.split(".")[-1:]
    if v[0] == 'json' and 'fpc' in input.split('_')[0]:
        return True
    return False

def analyzeFile(input):
    global locTable, filesAnalyzed
    
    if isFPCFile(input):
        print "File found:", input
        filesAnalyzed = filesAnalyzed + 1
        
        with open(input) as f:
            data = json.load(f)

        for i in range(len(data)):
            fileName = data[i]['file'] + ":" + str(data[i]['line'])
            minVal = data[i]['min']
            maxVal = data[i]['max']
            over = data[i]['over']
            overRes = data[i]['over_res']
            
            if fileName in locTable.keys():
                if minVal < locTable[fileName][0]:
                    locTable[fileName][0] = minVal
                if maxVal > locTable[fileName][1]:
                    locTable[fileName][1] = maxVal
                if over > locTable[fileName][2]:
                    locTable[fileName][2] = over
                    locTable[fileName][3] = overRes
            else:
                locTable[fileName] = [minVal, maxVal, over, overRes]

def genTable(sortedBy):
    global locTable
    
    fileData = ""
    
    sortedMax = sorted(locTable.items(), key=lambda x: x[1][1], reverse=True)
    sortedMin = sorted(locTable.items(), key=lambda x: x[1][0])

    if (sortedBy == 0):
        for k in sortedMax:
            maxVal = k[1][1]
            fileData = fileData + printBar(maxVal) + "," + str(k[1][0]) + "," + str(k[1][1]) + "," + k[0] + "\n"
    else:
        for k in sortedMin:
            minVal = k[1][0]
            fileData = fileData + printBar(minVal) + "," + str(k[1][0]) + "," + str(k[1][1]) + "," + k[0] + "\n"

    return fileData

def main():
    parseArgs()
    #input = sys.argv[1]
    input = inputPath
    print "Analyzing dir:", input
    
    # Check if input exist
    if os.path.exists(input):
        
        # Traver tree if input is a directory
        if os.path.isdir(input):
            
            # This loop traverses the entire directory tree
            rootDir = input
            for dirName, subdirList, fileList in os.walk(rootDir):
                for fname in fileList:
                    # Full path of the file
                    filePath = dirName + "/" + fname
                    #printOut([filePath])
                    analyzeFile(filePath)
        
        else: ## this is a file
            #print input
            analyzeFile(input)
    else:
        print "Error:", input, "does not exist"
        exit()

    if (filesAnalyzed == 0):
        print "No fpc_*.json files found"
        exit()

    generateHTML()

def parseArgs():
    global inputPath, outputFileName, verbose
    
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str,
                        help="file or directory to analyze")
    parser.add_argument("-o", "--output", help="name of output file", type=str)
    parser.add_argument("-v", "--verbose", help="print what the script does", action="store_true")
    args = parser.parse_args()

    inputPath = args.path

    if args.output:
        outputFileName = args.output
    if args.verbose:
        verbose = True


main()
