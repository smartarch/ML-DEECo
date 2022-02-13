import csv
from typing import List, Dict, Union


class Log:
    def __init__(self, header: Union[List[str], Dict[str, str]]):
        if type(header) == dict:
            self.format = header
            self.header = list(header.keys())
        else:
            self.format = None
            self.header = header
        self.records = []
        self.columns = len(header)

    def register(self, newData):
        self.records.append(newData)

    def getHeader(self) -> List[str]:
        return self.header

    def getColumn(self, column):
        index = self.getHeader().index(column)
        return [record[index] for record in self.records]

    def formatRow(self, row):
        if self.format:
            return [
                format(col, fmt) if fmt else col
                for col, fmt in zip(row, self.format.values())
            ]
        else:
            return row

    def export(self, filename):
        with open(filename, 'w', newline='') as myFile:
            wr = csv.writer(myFile)
            wr.writerow(self.header)
            for rec in self.records:
                wr.writerow(self.formatRow(rec))
