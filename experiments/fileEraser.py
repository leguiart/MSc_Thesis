
# import required module
import os
import sys

def fileEraser(directory, substring):    
    # iterate over subdirectories in
    # that directory
    for subdir in os.scandir(directory):
        if subdir.is_dir() and "Gen" in subdir.name:
            for filename in os.scandir(subdir):
                if substring in filename.name:
                    os.remove(filename)

def main(argv):
    fileEraser(argv[0], argv[1])

if __name__ == "__main__":
    main(sys.argv[1:])