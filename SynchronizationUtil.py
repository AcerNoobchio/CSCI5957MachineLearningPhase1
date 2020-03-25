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
            if df.iloc[0][0] > earliest_time:
                earliest_time = df.iloc[0][0]

            # If the start time is later than current, update earliest_time
            if df.iloc[-1][0] < latest_time:
                latest_time = df.iloc[-1][0]
        
       # Trim starts and ends of dataframes with obtained start/end times
        for df in data_frames:
            df = df.drop(df[(df.iloc[:,0] < earliest_time) & (df.iloc[:,0] > latest_time)].index)

        return data_frames

    #Takes in a list of dataframes for a single event, combines them and sorts them by timestamp
    def join_event_data(data_frames):
        result = pd.concat(data_frames)
        result = result.sort_values(result.columns[0])

        return result
