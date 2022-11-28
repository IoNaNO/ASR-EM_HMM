import os
import csv

def generate_tesing_list():
    list_filename='testingfile_list.csv'
    dir1='./mfcc'
    dir3=['AH','AR','AT','BC','BE','BM','BN','CC','CE','CP','DF','DJ','ED','EF','ET','FA','FG','FH','FM','FP','FR','FS','FT','GA','GP','GS','GW','HC','HJ','HM','HR','IA','IB','IM','IP','JA','JH','KA','KE','KG','LE','LG','MI','NL','NP','NT','PC','PG','PH','PR','RK','SA','SL','SR','SW','TC']

    # Clean
    if(os.path.exists(list_filename)):
        os.remove(list_filename)

    # Read and write testingfile
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

    print("Generate testingfile Done.\n")

