import csv
import sys 

# dont write / read text as binary    
with open('validation.csv', "r") as f:
    data = list(csv.reader(f))
l = 0
print(len(data))
with open('validation.csv', "w") as f:
    writer = csv.writer(f)
    for row in data:
        if row:
            if row[0] != 'None' and row[1] != 'None':
                writer.writerow(row)
                l+=1
print(l)
