import os
import csv

def generate_traning_list():
    list_filename='trainingfile_list.csv'
    dir1='./mfcc'
    dir3=['AE','AJ','AL','AW','BD','CB','CF','CR','DL','DN','EH','EL','FC','FD','FF','FI','FJ','FK','FL','GG']

    # Clean
    if(os.path.exists(list_filename)):
        os.remove(list_filename)

    # Read and write trainingfile
    for dir2 in dir3:
        dir=dir1+'/'+dir2
        for path,name,files in os.walk(dir):
            for file in files:
                label=file[0]
                if label == 'O':
                    label=10
                elif label == 'Z':
                    label=11
                else:
                    label=int(label)
                row=[label,path+'/'+file]
                with open(list_filename,'a') as of:
                    writer=csv.writer(of,lineterminator='\n')
                    writer.writerow(row)

    print("Generate trainingfile Done.\n")

    