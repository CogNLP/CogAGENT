import os

path = 'raw_data/connectivity'
name_list = []
Note = open('raw_data/connectivity/scans.txt', mode='w')
for file_name in os.listdir(path):
    if file_name.split("_")[0] != "README.md" and file_name.split("_")[0] != "scans.txt":
        Note.write(file_name.split("_")[0] + "\n")

print("end")
