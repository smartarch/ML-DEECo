import csv
from typing import List, Dict, Union
from statistics import mean


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

    def getColumn(self, column, data=None):
        if data is None:
            data = self.records
        index = self.getHeader().index(column)
        return [record[index] for record in data]

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


class AverageLog(Log):

    def __init__(self, header: Union[List[str], Dict[str, str]]):
        super().__init__(header)
        self.avgRecords = []
        self.lastAvgIndex = 0

    def registerAvg(self):
        newRecords = self.records[self.lastAvgIndex:]
        self.lastAvgIndex = len(self.records)
        self.avgRecords.append(self.computeAverage(newRecords))

    def computeAverage(self, records):
        return [
            self.computeAverageForColumn(col, records)
            for col in self.header
        ]

    def computeAverageForColumn(self, column, records):
        data = self.getColumn(column, records)

        if len(data) == 0:
            return None

        if type(data[0]) == int or type(data[0]) == float:
            return mean(data)
        else:
            return data[0]  # TODO: mode

    def exportAvg(self, filename):
        with open(filename, 'w', newline='') as myFile:
            wr = csv.writer(myFile)
            wr.writerow(self.header)
            for rec in self.avgRecords:
                wr.writerow(self.formatRow(rec))

    def getColumnAvg(self, column):
        return self.getColumn(column, self.avgRecords)
