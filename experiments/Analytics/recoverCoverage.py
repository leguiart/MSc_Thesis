import sys
import json

from Utils import save_json
# sys.path.append(os.getcwd() + "/..")

def computeCoverage(qd_history):
    for i in range(len(qd_history["coverage"])):
        bins = len(qd_history["f_archive_progression"][i])
        coverage = 0
        for vect in qd_history["f_archive_progression"][i]:
            if vect != 0:
                coverage += 1
        coverage /= bins
        qd_history["coverage"][i] = coverage


def coverageFromJsons(filename):
    newFilename = filename.replace(".json", "")
    newFilename += "coverage"
    with open(filename, 'r') as fh:
        for line in fh:
            analytics = json.loads(line)
            keys = list(analytics.keys())
            run = keys[0]
            qd_history = analytics[run]
            computeCoverage(qd_history)
            analytics[run] = qd_history
            save_json(newFilename + ".json", analytics)
            

def main(argv):
    coverageFromJsons(argv[0])

if __name__ == "__main__":
    main(sys.argv[1:])