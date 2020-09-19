'''
Visual Analysis.
The script extracts shots from single video file or from a directory with videos

Flags:

    - f: extract features from specific file
        E.g. python3 analyze_visual.py -f <filename>
    - d: extract features from all files of a directory
        E.g. python3 analyze_visual.py -d <directory_name>

Basic functions:

    - process_video :  extracts features from specific file
    - dir_process_video : extracts features from all files of a directory

'''
from analyze_visual import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def crop_shots(video_path,shot_change_times):
 
    # ---Initializations-------------------------------------------------------    
    t_start = time.time()
    t_0 = t_start
    cap = cv2.VideoCapture(video_path)
    frames_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_secs = frames_number / fps
    hrs, mins, secs, tenths = seconds_to_time(duration_secs)
    duration_str = '{0:02d}:{1:02d}:{2:02d}.{3:02d}'.format(hrs, mins,
       
                                                            secs, tenths)


    for index in range(0,len(shot_change_times)):

        shot_change_times = [int(i) for i in shot_change_times]

        try:

            print (shot_change_times[index])
        
            i = shot_change_times[index]

            if i < shot_change_times[-1]:
                
                ffmpeg_extract_subclip(video_path, i, shot_change_times[index+1],
                 targetname=str(shot_change_times[index])+"__"+str(shot_change_times[index+1])+".mp4")
                
        except:
            
                pass

def crop_dir(dir_name):

    dir_name_no_path = os.path.basename(os.path.normpath(dir_name))

    features_all = np.array([])

    shot_change_t = []
    types = ('*.avi', '*.mpeg', '*.mpg', '*.mp4', '*.mkv')
    video_files_list = []
    for files in types:
        video_files_list.extend(glob.glob(os.path.join(dir_name, files)))
    video_files_list = sorted(video_files_list)

    for movieFile in video_files_list:

        print(movieFile)

        _, _, shot_change_times = process_video(movieFile, 2,True, True)
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