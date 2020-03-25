import pandas as pd
import numpy as np
import math

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

    #Takes a sorted dataFrame and a second interval and returns a list of smaller dataFrames where each contains data only for the given interval size
    #Example, a dataFrame representing 10 seconds and an interval of 2 returns a list of 5 dataFrames each representing 2 seconds
    def chunkify_data_frame(dataFrame, miliSecondInterval=2000):
        result = []
        #Get start and end times
        start_time = dataFrame.iloc[0][0]
        end_time = dataFrame.iloc[-1][0]
        # in seconds, rounding up to accompany all data
        time_span = math.ceil((end_time - start_time)/1000 / (miliSecondInterval/1000))
        # Increase end_time by one percent to ensure we get the end of the data
        end_time = end_time + end_time*.01
        # Initialize our last_time and next_time loop variables to create the chunks
        last_time = start_time
        next_time = start_time + miliSecondInterval
        # Steph through each interval, creating the corresponding chunk and appending it to result
        for chunk in range(1,time_span):
            df = dataFrame[(dataFrame.iloc[:,0] >= last_time) & (dataFrame.iloc[:,0] < next_time)]
            result.append(df)
            next_time += miliSecondInterval
            last_time += miliSecondInterval

        return result

