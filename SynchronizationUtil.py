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

            # If the end time is earlier than current, update latest_time
            if df.iloc[-1][0] < latest_time:
                latest_time = df.iloc[-1][0]
        
        # Shift start and end times by 3 seconds each to trim noise
        earliest_time += 3000
        latest_time -= 3000

        time_span = (latest_time - earliest_time)/1000

        # Trim starts and ends of dataframes with obtained start/end times
        for df in range(0, len(data_frames)):
            curr_latest = data_frames[df]['Time (Milliseconds)'].iloc[-1]
            curr_earliest = data_frames[df]['Time (Milliseconds)'].iloc[0]
            curr_time_span = (curr_latest - curr_earliest)/1000
            too_early = data_frames[df][(data_frames[df]['Time (Milliseconds)'] < earliest_time)].index
            too_late = data_frames[df][(data_frames[df]['Time (Milliseconds)'] > latest_time)].index
            temp_df = data_frames[df].drop(too_early)
            temp_df = temp_df.drop(too_late)
            data_frames[df] = temp_df
            curr_latest = temp_df['Time (Milliseconds)'].iloc[-1]
            curr_earliest = temp_df['Time (Milliseconds)'].iloc[0]
            curr_time_span = (curr_latest - curr_earliest)/1000

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
            #if len(df) > 1:
            result.append(df)
            next_time += miliSecondInterval
            last_time += miliSecondInterval

        return result

