import sys
# ApiMain [begin|options|help|create]
# ApiMain --begin/-b -d/--d <directory> or -f <file> <json>
# ApiMain --options/-o id
# ApiMain --help/-h//? return this tooltip


def update_readme():
    HF = open("CLine_help.txt", "r")
    help_string = HF.open()
    
    rm = open("../README.md", "r")
    s = rm.read()
    rm.close()
    
    s = s.format(__docstring__=help_string)
    
    rm = open("../README.md", "w")
    rm.write(s)
    rm.close()

if __name__ == "__main__":
    uarg = sys.argv[1:]
    if len(uarg) > 0:
        if "-ur" in uarg:
            update_readme()
