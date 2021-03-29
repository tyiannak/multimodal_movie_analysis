"""
This script takes a video file and generates video files based on shots. 

Usage example:

for single file:
python3 shot_generator.py -f dataset/data/trump.mp4

for directory:
python3 shot_generator.py -d dataset/data
"""

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
import os.path
import sys
import glob
sys.path.insert(0, '..')
from analyze_visual.analyze_visual import process_video

def crop_shots(video_path, shot_change_times):
    """
    Crop video based on given timestamps
    :param video_path: path of video
    :shot_change_times: timestamps
    """
    for index in range(0, len(shot_change_times)):
        shot_change_times = [int(i) for i in shot_change_times]
        print (shot_change_times[index])
        i = shot_change_times[index]
        extension = os.path.splitext(video_path)[1]
        if i < shot_change_times[-1]:
            shot_file_name = f"{video_path}_shot_{shot_change_times[index]}_" \
                             f"{shot_change_times[index+1]}.{extension}"
#            ffmpeg_extract_subclip(video_path, i, shot_change_times[index+1],
#                                   targetname=shot_file_name)

            ffmpeg_command = f"ffmpeg -i \"{video_path}\" " \
                             f"-ss {i} -to {shot_change_times[index+1]} " \
                             f"-c copy " \
                             f"\"{shot_file_name}\""
            print(ffmpeg_command)
            os.system(ffmpeg_command)


def crop_dir(dir_name):
    """
    Crop directory videos based on given timestamps
    :param dir_name: path of difectory
    """
    shot_change_t = []
    types = ('*.avi', '*.mpeg', '*.mpg', '*.mp4', '*.mkv')
    video_files_list = []
    for files in types:
        video_files_list.extend(glob.glob(os.path.join(dir_name, files)))
    video_files_list = sorted(video_files_list)
    for movieFile in video_files_list:
        print(movieFile)
        _, _, _, _, shot_change_times = process_video(movieFile, 2, True, False)
        shot_change_t.append(shot_change_times)
        print(shot_change_t)
        crop_shots(movieFile,shot_change_times)


def main(argv):
    if len(argv) == 3:
        if argv[1] == "-f":
            _, _, _, _, shot_change_t = process_video(argv[2], 2, True, True)
            crop_shots(argv[2],shot_change_t)
            print(shot_change_t)
        elif argv[1] == "-d":  # directory
            dir_name = argv[2]
            crop_dir(dir_name)
        else:
            print('Error: Unsupported flag.')
            print('For supported flags please read the modules\' docstring.')
    elif len(argv) < 3:
        print('Error: There are 3 required arguments, but less were given.')
        print('For help please read the modules\' docstring.')
    else:
        print('Error: There are 3 required arguments, but more were given.')
        print('For help please read the modules\' docstring.')


if __name__ == '__main__':
    main(sys.argv)
