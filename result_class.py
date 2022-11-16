"""
This is a part of the autonomous car project.
This simulates how to apply Reinforcement Learning in dynamic obstacles avoidance for a self-driving car.
author: Binh Tran Thanh / email:thanhbinh@hcmut.edu.vn or thanhbinh.hcmut@gmail.com
"""
import csv
import pandas as pd

class Result_Log:
    def __init__(self, header_csv = ["Reached goal","Car status"]):
        self.results_data = []
        self.header_csv = header_csv
    
    ''' save result in to list '''
    def add_result(self, result):
        self.results_data.append (result)
    
    ''' set header '''
    def set_header (self, header_csv):
        self.header_csv = header_csv

    ''' write result to csv file '''
    def write_csv(self, file_name):
        f = open(file_name, 'w', newline='', encoding="utf-8")
        writer = csv.writer(f, delimiter=",")
        writer.writerow(self.header_csv)
        for result_items in self.results_data:
            writer.writerow(result_items)
        f.close()

    ''' read csv as dataframe '''
    def read_csv_as_dataframe(self, result_file):
        # read result as frame
        return pd.read_csv(result_file)