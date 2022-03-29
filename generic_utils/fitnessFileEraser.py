
# import required module
import os
import sys

def fitnessFilesEraser(directory, substring):    
    # iterate over subdirectories in
    # that directory
    for filename in os.scandir(directory):
        if substring in filename.name:
            os.remove(filename)

def main(argv):
    fitnessFilesEraser(argv[0], argv[1])

if __name__ == "__main__":
    main(sys.argv[1:])