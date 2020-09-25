from analyze_visual import *
import statistics

def read_file(process_mode,file,moviefile,shots):
	'''
	Read the file and return informations about lists of shots
	'''
	lines = []


	filename1 = os.path.basename(moviefile)  
	filename2 = os.path.basename(file) 
	name1 = filename1.rsplit('.',1)[0]
	name2 = filename2.rsplit('.',1)[0]

	if name1 == name2:
		with open(file) as w:
			lines = w.read().splitlines() 

		y= len(lines)
		x= len(shots)

		print("------ \nThe number of shots founded: ",x-1)
		print("The actual number of shots: ",y-1)


		for i in range(0, len(lines)): 
			lines[i] = float(lines[i]) 

	return lines


def calc(annotated_shots,shot_change_times):
	tolerance, correct_shot,correct_shot_h = 0.5, 0,0

	correct_recall, correct_precision = 0, 0
	for t in annotated_shots:
		if np.abs(t - np.array(shot_change_times)).min() < tolerance:
			correct_recall += 1

	for t in shot_change_times:
		if np.abs(t - np.array(annotated_shots)).min() < tolerance:
			correct_precision += 1



	precision= correct_precision/(len(shot_change_times))
	recall = correct_recall/(len(annotated_shots))

	
	print ("Precision: {0:.0f}%".format(precision*100))
	print ("Recall: {0:.0f}%".format(recall*100))	

	return precision,recall
def single_acc(process_mode,video_path,txt_path,shot_change_times):


	annotated_shots =read_file(process_mode,txt_path,video_path,shot_change_times)
	
	for i in range(0, len(shot_change_times)): 
		shot_change_times[i] = float(shot_change_times[i])
	
	print("Timestamps for predicted shots: \n",shot_change_times)
	print("Timestamps for actual shots: \n",annotated_shots)

	calc(annotated_shots,shot_change_times)

def dir_acc(video_path,process_mode,txt_path):

	shot_change_t = []
	types = ('*.avi', '*.mpeg', '*.mpg', '*.mp4', '*.mkv')
	shot_types = ('*.txt')
	video_files_list = []
	shot_list = []
	prec_list,rec_list=[],[]

	for files in types:
		video_files_list.extend(glob.glob(os.path.join(video_path, files)))

	for files in shot_types:
		shot_list.extend(glob.glob(os.path.join(txt_path, files)))

	video_files_list = sorted(video_files_list)
	shot_list = sorted(shot_list)
	shot_list.pop(0)
	print(video_files_list,shot_list)

	for movieFile,shotFile in zip(video_files_list,shot_list):

		
		_, _, shot_change_times = process_video(movieFile, 2, True, False)
		#shot_change_t.append(shot_change_times)
		#print(shot_change_t)
		annotated_shots = read_file(process_mode,shotFile,movieFile,shot_change_times)

		print("Timestamps for predicted shots: \n",shot_change_times)
		print("Timestamps for actual shots: \n",annotated_shots)

		precision,recall=calc(annotated_shots,shot_change_times)

		prec_list.append(precision)
		rec_list.append(recall)

		shot_change_times.clear()

	#print(statistics.mean(prec_list),statistics.mean(rec_list))

	print ("AVG of Precision: {0:.0f}%".format(statistics.mean(prec_list)*100))
	print ("AVG of Recall: {0:.0f}%".format(statistics.mean(rec_list)*100))		
        #acc(movieFile,txt_path,process_mode,shot_change_times)

def main(argv):
    if len(argv) == 4:
        if argv[1] == "-f":
            _, _, shot_change_t = process_video(argv[2], 2, True, True)
            single_acc(argv[1],argv[2],argv[3],shot_change_t)
        elif argv[1] == "-d":  # directory
            dir_name = argv[2]
            dir_acc(dir_name,argv[1],argv[3])
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
