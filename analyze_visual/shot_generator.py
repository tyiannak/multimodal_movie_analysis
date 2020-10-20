"""
TODO
"""

from analyze_visual import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def crop_shots(video_path, shot_change_times):
    """
    TODO
    """
    for index in range(0, len(shot_change_times)):
        shot_change_times = [int(i) for i in shot_change_times]
        print (shot_change_times[index])
        i = shot_change_times[index]
        if i < shot_change_times[-1]:
            shot_file_name = f"{video_path}_shot_{shot_change_times[index]}_" \
                             f"{shot_change_times[index+1]}.mp4"
            ffmpeg_extract_subclip(video_path, i, shot_change_times[index+1],
                                   targetname=shot_file_name)


def crop_dir(dir_name):
    """
    TODO
    """
    shot_change_t = []
    types = ('*.avi', '*.mpeg', '*.mpg', '*.mp4', '*.mkv')
    video_files_list = []
    for files in types:
        video_files_list.extend(glob.glob(os.path.join(dir_name, files)))
    video_files_list = sorted(video_files_list)
    for movieFile in video_files_list:
        print(movieFile)
        _, _, shot_change_times = process_video(movieFile, 2, True, False)
        shot_change_t.append(shot_change_times)
        print(shot_change_t)
        crop_shots(movieFile,shot_change_times)


def main(argv):
    if len(argv) == 3:
        if argv[1] == "-f":
            _, _, shot_change_t = process_video(argv[2], 2, True, True)
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