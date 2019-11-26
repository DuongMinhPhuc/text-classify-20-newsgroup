import csv

with open('train.csv') as csv_file:
    # csv_reader = csv.reader(csv_file, delimiter=',')
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 1:
            print(row)
            print('BBBBBBBBBBBBBBBBBB')
            print(row["topic"])
            break
        line_count += 1

