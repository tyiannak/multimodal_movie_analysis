from aggregate_annotations import aggregate_annotations,save_to_csv

df = aggregate_annotations('annotations_database.txt')

ann_gr_2 = df[(df['Number_annotations'] >= 2)
                 & (df['Confidence'] == 100)
                 & (df['Winner_annotation'] == 'Static') ]

save_to_csv(ann_gr_2,'find_statics.csv')

