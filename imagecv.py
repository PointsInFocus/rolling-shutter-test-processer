#!/usr/bin/env python3

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import itertools
import os
from exiftool import ExifToolHelper
import argparse
from dataclasses import dataclass,field

WIDTH = 1
HEIGHT = 0
BLACK = 0
WHITE = 255
RS_TEST_PULSE_TIMES = [ (16/1023)*(1/239) * 1e6, 
                        (31/1023)*(1/239) * 1e6, 
                        (62/1023)*(1/239) * 1e6, 
                        (123/1023)*(1/239) * 1e6,
                        (246/1023)*(1/239) * 1e6,
                        (492/1023)*(1/239) * 1e6,
                    ]

@dataclass
class Metadata:
    shutter_speed: float = field(init=False)  
    shutter_speed_value: float = field(init=False)  
    total_frames: int = field(init=False)  
    image_width: int = field(init=False)  
    image_height: int = field(init=False)  
    fps: float = field(init=False)  


def main():
    parser = argparse.ArgumentParser( description="Rolling shutter test calculator." )

    parser.add_argument( 
        "-p", 
        "--pulse-time", 
        type=float, 
        default=100.0, 
        nargs="+", 
        help="Pulsetime in microseconds, can be included upto 3 times for automated testing." 
        )

    parser.add_argument(
        "--shutter-speed",
        type=int,
        default=2000,
        help="Denominator of shutter speed"
    )

    parser.add_argument(
        '--std-test', 
        action="store_true", 
        help="Uses standard pulse times." 
        )
    parser.add_argument( 
        '-d', 
        '--debug', 
        action="store_true", 
        help="Enable printing of debug images." 
        )

    parser.add_argument(
        '-s', 
        '--sort-files', 
        action="store_true", 
        help="Sort files in numerical order."
        ) 
    parser.add_argument(
        '--stat-file', 
        action="store_true", 
        help="Reads image stats from stat file insted of reprocessing image." 
        ) 

    parser.add_argument( 
        "source_files", 
        nargs="+" 
        )

    args = parser.parse_args()

    if args.std_test:
        args.pulse_time = RS_TEST_PULSE_TIMES

    if args.debug:
        if not os.path.exists( "blur" ):
            os.mkdir( "blur" )
        if not os.path.exists( "edges" ):
            os.mkdir( "edges" )
        if not os.path.exists( "gray" ):
            os.mkdir( "gray" )
        if not os.path.exists( "threshold" ):
            os.mkdir( "threshold" )

    if args.sort_files:
        args.source_files = sorted( args.source_files ) 
        
    print( args.source_files ) 
    print( args.pulse_time )

    if len(args.source_files ) % len(args.pulse_time) != 0:
        print( "Automated testing requires the number of source files to be integer multiple of pulse times." )
        exit() 

    averages = []

    for frame_index, source_file in enumerate( args.source_files ):
        print( f"Processing {source_file} - pulse time {args.pulse_time[frame_index % len(args.pulse_time)]} us" )
        processed_frame_count = 0
        frame_data = []


        cap = cv.VideoCapture( source_file )
        file_metadata = get_file_metadata( source_file, cap, args )

        if args.debug:
            print( file_metadata )

        while cap.isOpened():
            _, frame = cap.read()

            if frame is None:
                break

            print( f"Processing frame: {processed_frame_count}", end='\r')

            # edge_iamge, threshold_image = process_frame_image( frame )

            gray_image = cv.cvtColor( frame, cv.COLOR_BGR2GRAY )

            gray_image = gray_image[
                            int(gray_image.shape[HEIGHT]*0.05):int(gray_image.shape[HEIGHT]*0.95), 
                            int(gray_image.shape[WIDTH]/2)-20:int(gray_image.shape[WIDTH]/2)+20
                        ]

            blurred_image = cv.bilateralFilter( gray_image, 3, 75, 11 )

            # Calcualte threshold value
            gray_min = np.min( gray_image )
            _, threshold_image = cv.threshold( blurred_image, 
                                                np.min(gray_image) + 16,
                                                WHITE,
                                                cv.THRESH_BINARY )
            
            # threshold_image = remove_isolated_pixels( threshold_image )
            
            threshold_image = remove_isolated_blobs( threshold_image )

            edges = cv.Canny( threshold_image, 33, 100 )

            if args.debug:
                cv.imwrite( f"gray/gray-{source_file}-{processed_frame_count}.png", gray_image )
                cv.imwrite( f"blur/blur-{source_file}-{processed_frame_count}.png", blurred_image )
                cv.imwrite( f"threshold/thres-{source_file}-{processed_frame_count}.png", threshold_image )
                cv.imwrite( f"edges/edges-{source_file}-{processed_frame_count}.png", edges )

            lines = np.where( edges[0:edges.shape[HEIGHT], int(edges.shape[WIDTH]/2)] == WHITE )[0].tolist()

            filter_lines( lines, threshold_image )
            
            for line_1,line_2 in pairwise(lines):
                if threshold_image[int( (line_1 + line_2) / 2), 20] != WHITE:
                    continue

                band_height = line_2 - line_1
                if band_height < 10:
                    continue       
    
                frame_data.append( band_height )
            
            if len(lines) > 1: 
                processed_frame_count += 1


        print()
        cap.release()

        average_lines = round( np.mean( frame_data ), 2)
        lines_sd = round( np.std( frame_data ), 2)

        if args.debug:
            print( average_lines, lines_sd ) 
            print( frame_data )
        
        frame_data = [i for i in frame_data if abs( i - average_lines ) <= (1.0 * lines_sd) ]

        histogram = np.hstack( frame_data )
        _ = plt.hist( histogram, bins="auto" )
        plt.savefig( f"{source_file}-histogram.png" )
        plt.close()
        
        if args.debug:
            print( frame_data )
        
        average_lines = np.mean( frame_data )
        lines_sd = np.std( frame_data )

        strobe_time = args.pulse_time[frame_index % len(args.pulse_time)] / 1e6 
        print( strobe_time )
        
        # average_lines = sum(data)/len(data)
        line_time = (file_metadata.shutter_speed_value + strobe_time) / average_lines
        frame_time = file_metadata.image_height * line_time

        averages.append( (line_time, frame_time, average_lines, lines_sd ) )

        print( "File | Avg. Len | Min Len | Max Len | Std. dev | # of Samples | # of Frames | Line Time | Frame Time | Shutter Speed | Shutter Spd. Value | Image Res | FPS ")
        print( f"{source_file}, {round(average_lines,1)}, {min(frame_data)}, {max(frame_data)}, {lines_sd}, {len(frame_data)}, {processed_frame_count}, {line_time}, {frame_time}, {file_metadata.shutter_speed}, {file_metadata.shutter_speed_value}, {file_metadata.image_width}x{file_metadata.image_height}, {file_metadata.fps}"  )
        
        if os.path.exists( "stats.txt" ):
            with open( "stats.txt", "a") as f:
                f.write( f"{source_file}, {average_lines}, {min(frame_data)}, {max(frame_data)}, {lines_sd}, {len(frame_data)}, {processed_frame_count}, {line_time}, {frame_time}, {file_metadata.shutter_speed}, {file_metadata.shutter_speed_value}, {strobe_time}, {file_metadata.image_width}x{file_metadata.image_height}, {file_metadata.fps}\n" )
        else:
            with open( "stats.txt", "a" ) as f:
                f.write("File, Avg. Len, Min Len, Max Len, Std. dev, # of Samples, # of Frames, Line Time, Frame Time, Shutter Speed, Shutter Spd. Value, Storbe Time, Image Dims, FPS\n" )
                f.write( f"{source_file}, {average_lines}, {min(frame_data)}, {max(frame_data)}, {lines_sd}, {len(frame_data)}, {processed_frame_count}, {line_time}, {frame_time}, {file_metadata.shutter_speed}, {file_metadata.shutter_speed_value}, {strobe_time}, {file_metadata.image_width}x{file_metadata.image_height}, {file_metadata.fps}\n" )
        frame_index += 1


    '''
    My reasoning behing the averaging.
    Due to noise, and other image processing deficiencies I wanted to weight the final values based on the impact their known errors would have on the final average. For example, if the total number of lines counted is small, say around 100-150, then an error of even as little as 1 line can have an apprciable impact on the resulting timing (as much as several 0.1s of a millisecond). Conversely when the number of coutned lines is high, the impact of a 1 line error is much smaller.

    As a reuslt I settled on the idea of using the standard deviation in counted lines (so basically how confident in the line counts) to figure out the error that would be created by that devication based on the average number of lines counted. The resulting averages from the sub tests are then weighted based on how their std deviation driven error compares to the minmium of all of the tests.

    In the event that the std deviation is 0, then it's simply replaced with a small non-zero value of 0.01. This prevent a divide by zero condition while still weighting the results heavily.

    This process does favor tests wiht more lines, which currently I consider more relaible as a whole.
    '''
    line_averages = np.array([ i[-2] for i in averages ])
    line_error = np.array([i[-1] if i[-1] > 0 else 0.01 for i in averages ] )
    line_errors = line_error / line_averages
    line_weights = np.min( line_errors ) / line_errors

    # print( line_averages )
    # print( line_error )
    # print( line_errors )
    # print( line_weights )

    myAverages = list( zip( *averages ) )

    with open( "stats.txt", "a") as f:
        f.write( "\n\n" )
        f.write( f"{np.mean( myAverages[0] )}\n{np.average(myAverages[1], weights=line_weights )}\n{np.std(myAverages[1])}\n\n" )



def pairwise( iterable ):
    a, b, = itertools.tee(iterable)
    next(b,None)
    return zip(a,b)



def get_file_metadata( source_file : str, cap_file : cv.VideoCapture, args ) -> Metadata:
    metadata = Metadata()

    with ExifToolHelper() as et:
        exif_data = et.get_tags( source_file, tags=['ShutterSpeed', 'ShutterSpeedValue' ] )[0]
        metadata.shutter_speed = float( exif_data.get('Composite:ShutterSpeed') or 1/args.shutter_speed )
        metadata.shutter_speed_value = float( exif_data.get('EXIF:ShutterSpeedValue') or metadata.shutter_speed )
    
    metadata.total_frames = int( cap_file.get( cv.CAP_PROP_FRAME_COUNT ) )
    metadata.image_width = int( cap_file.get( cv.CAP_PROP_FRAME_WIDTH ) )
    metadata.image_height = int( cap_file.get( cv.CAP_PROP_FRAME_HEIGHT ) ) 
    metadata.fps = float( cap_file.get( cv.CAP_PROP_FPS ) )

    return metadata



def remove_isolated_pixels(image):
    connectivity = 8

    output = cv.connectedComponentsWithStats(image, connectivity, cv.CV_32S)

    num_stats = output[0]
    labels = output[1]
    stats = output[2]

    new_image = image.copy()

    for label in range(num_stats):
        if stats[label,cv.CC_STAT_AREA] == 1:
            new_image[labels == label] = 0

    return new_image



def remove_isolated_blobs( image ):
    blobs, im_with_searated_blobs, stats, centroids = cv.connectedComponentsWithStats(image, 8)
 
    sizes = stats[:, -1]
    sizes = sizes[1:]

    blobs -= 1

    min_size = 79

    im_result = np.zeros_like(image)

    for blob in range(blobs):
        if sizes[blob] >= min_size and abs(centroids[blob+1,0] - 20) < 1:
            im_result[im_with_searated_blobs == blob + 1] = 255

    return im_result



def filter_lines( lines, threshold_image ):
    threshold_image_line = threshold_image[0:,20]

    if not lines: 
        return

    if threshold_image_line[0] == WHITE:
        lines = lines[1:]

    # Find and remove small line artifacts
    if len(lines) <= 2:
        return

    for index, value in enumerate( lines ):
        if index == len(lines) - 1:
            break
        
        start_of_band = True if threshold_image_line[value - 1] < threshold_image_line[value+1] else False

        recursive_filter_lines( index, start_of_band, lines )
       


def recursive_filter_lines( index, black_to_white, lines ):
    # Look at the next index to see if transition is less than 8 pixels away, recurse to it and repeat
    # If next transition is > 8 pixels away:
    #   Determine if the the current transition is from black to white or white to black
    #   If direction is black to white:
    #       return min of the two indicies
    #   Else:
    #       return max of two indicies
    # print( index, black_to_white, lines )
    try:
        if lines[index+1] - lines[index] <= 8:
            recursive_filter_lines(index + 1, black_to_white, lines )
            if black_to_white:
                lines.remove( lines[index+1] )
            else:
                lines.remove( lines[index])
    except IndexError:
        pass




if __name__ == "__main__":
    main()