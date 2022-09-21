from os import listdir
from os.path import isfile, join
import zipfile

def unzipFiles(org_folder,dest_folder):
 

    filesPath = [f for f in listdir(org_folder) if isfile(join(org_folder, f))]


    
    for filePath in filesPath:
        with zipfile.ZipFile(join(org_folder,filePath),'r') as zip_ref:
            zip_ref.extractall(dest_folder)


