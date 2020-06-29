import sys
from API.ApiMain import CustomConsole
# ApiMain [begin|options|help|create]
# ApiMain --begin/-b -d/--d <directory> or -f <file> <json>
# ApiMain --options/-o id
# ApiMain --help/-h//? return this tooltip



if __name__ == '__main__':
    CC = CustomConsole()
    while CC.running==True:
        line = input("Virus-API>> ")
        CC.run(CC.delim.split(line))