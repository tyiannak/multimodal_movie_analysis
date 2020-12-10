import sys
sys.path.append('../')
from annotation.aggregate_annotations import aggregate_annotations,save_to_csv
import os.path
import shutil


def create_file(file):

    df = aggregate_annotations(file)

    ann_gr_2 = df[(df['Number_annotations'] >= 2)
                    & (df['Confidence'] == 100)]

    save_to_csv(ann_gr_2,'annot_with_conf.csv')

    return ann_gr_2


def create_dataset(df, path_of_shots, final_path):

    classes = ['Static','Vertical_movement','Titl','Panoramic',
                'Panoramic_lateral','Panoramic_360','Travelling_in',
                'Travelling_out','Zoom_in','Zoom_out','Vertigo',
                'Aerial','Handheld','Car_front_windshield','Car_side_mirror'
                'None']
    
    #Create folders of classes
    if os.path.exists('dataset'):
        shutil.rmtree('dataset')
    os.mkdir('dataset')

    owd = os.getcwd()

    print('Creating folders..')
    for _class_ in classes:
        os.mkdir(os.path.join('dataset', _class_))

    os.chdir(path_of_shots)

    print('Moving shots to folders..')
    #Move shots to class folder
    for _class_ in classes:

        df2 = df[(df['Winner_annotation'] == _class_)]    

        names = df2['Sample_Name']
        names = names.values.tolist()
        names = set(names) 
        print(final_path, _class_)
        for filename in os.listdir('.'):
            if filename in names:
                try: 
                    shutil.copy2(filename,os.path.join(final_path,_class_))
                except:
                    print('File not found')

    os.chdir(owd)
    print('End of process!')


if __name__ == "__main__":   
    
    file = '../annotation/annotations_database.txt'
    aggr_file = create_file(file)
    create_dataset(aggr_file,'/Users/tyiannak/Downloads/shots_final_selected_fixed','dataset')