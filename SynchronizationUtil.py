import pandas as pd
import numpy as np

class SynchronizationUtil(object):
    def trim_start_end_times(data_frames):
        # Initialize earlist/latest time variables, we want the latest start_time and earliest end_time
        # so we can clip all the files to the same time span, so initialize earlist to 0 and latest to inf
        earliest_time = 0.0
        latest_time = float("inf")
        
        #Collect earliest start time and latest end time for the data frames
        for df in data_frames:
            # If the start time is later than current, update earliest_time
            if df.time_milisec.iloc[0] > earliest_time:
                earliest_time = df.time_milisec.iloc[0]

            # If the start time is later than current, update earliest_time
            if df.time_milisec.iloc[0] > earliest_time:
                earliest_time = df.time_milisec.iloc[-1]
        
       # Trim starts and ends of dataframes with obtained start/end times
        for df in data_frames:
            df = df.drop(df[(df.time_milisec < earliest_time) & (df.time_milisec > latest_time)].index)
            print(df.time_milisec.iloc[0])
            print(df.time_milisec.iloc[-1])
