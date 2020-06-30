import sys
from API.ApiMain import CustomConsole

# ApiMain [begin|options|help|create]
# ApiMain --begin/-b -d/--d <directory> or -f <file> <json>
# ApiMain --options/-o id
# ApiMain --help/-h//? return this tooltip


def update_readme():
    rm = open("../README.md", "+")
    s = rm.read()
    s = s.format(__docstring__=CustomConsole.help_string)
    rm.write(s)

if __name__ == "__main__":
    uarg = sys.argv[1:]
    if len(uarg) > 0:
        if "-ur" in uarg:
            update_readme()
