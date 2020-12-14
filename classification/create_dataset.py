import sys
import argparse
import os.path
import shutil
sys.path.append('../')
from annotation.aggregate_annotations import aggregate_annotations, save_to_csv


def parse_arguments():
    """Parse arguments for real time demo.
    """
    parser = argparse.ArgumentParser(description="Create Shot "
                                                 "Classification Dataset")
    parser.add_argument("-a", "--annotation_file", required=True, nargs=None,
                        help="Annotation database file")
    parser.add_argument("-v", "--videos_path", required=True, nargs=None,
                        help="Videos folder path")

    return parser.parse_args()


def read_annotation_data_and_aggregate(file_path):
    """
    Reads aggregated annotation data (using annotation.aggregate_annotations()
    function)
    :param file_path: initial annotation database
    :return: pandas dataframe of aggregated annotations
    """
    df = aggregate_annotations(file_path)
    ann_gr_2 = df[(df['Number_annotations'] >= 3)
                  & (df['Confidence'] > 50)]
    save_to_csv(ann_gr_2, 'temp.csv')

    return ann_gr_2


def create_dataset(df, path_of_shots, output_path):
    """
    Takes a pandas data frame of aggregated annotations, and it copies each 
    file to a class-folder. 
    :param df: pandas dataframe of aggregated annotations. To compute it, run
    read_annotation_data_and_aggregate()
    :param path_of_shots: path to the actual video files of the dataset
    :param output_path: output path
    :return: 
    """

    # get list of all possible classes:
    classes = (df['Winner_annotation'].unique())

    # Create folders of classes
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    print('Creating folders..')
    for _class_ in classes:
        os.mkdir(os.path.join('dataset', _class_))

    print('Copying shots to folders..')
    count = 0
    count_moved = 0
    list_of_videos_in_folder = os.listdir(path_of_shots)
    for _class_ in classes:  # for each shot class:
        print(' - - - - - - - - - - - - - - - - -')
        # get the filenames of the videos belonging to the class:
        df2 = df[(df['Winner_annotation'] == _class_)]
        names = df2['Sample_Name']
        names = names.values.tolist()
        names = set(names)
        print("Class: " + _class_)
        print(f"Files annotated to this class: {len(names)}")
        count_moved_class = 0
        count += len(names)
        for f in names:
            if f in list_of_videos_in_folder:
                try: 
                    shutil.copy2(os.path.join(path_of_shots, f),
                                 os.path.join(output_path, _class_))
                    count_moved_class += 1
                    count_moved += 1
                except:
                    print('File not found')
            else:
                print(f + " not found!")
        print(f"Files found and copied: {count_moved_class}")

    print(f"Total files annotated {count}. "
          f"Total files found and moved {count_moved}")
    print('End of process!')


if __name__ == "__main__":
    args = parse_arguments()
    data_base_file_name = args.annotation_file
    videos_path = args.videos_path

    aggr_file = read_annotation_data_and_aggregate(data_base_file_name)

    assert os.path.exists(videos_path), "Video Path doesn't exist, " + \
                                        str(videos_path)
    
    create_dataset(aggr_file, videos_path, 'dataset')

