import os, sys, random
import numpy as np

input_file_structure_labels = {0:"NO.", 1:"TIME", 2:"SOURCE", 3:"DESTINATION", 4:"PROTOCOL", 5:"LENGTH",6:"INFO"}
MASTER = ""
SLAVE = ""

def read_input_file(input_file=os.path.join(os.getcwd(), "dataset","input_data.csv"), delim=','):
    with open(input_file,'r') as in_f:
        _ = in_f.readline()
        content = in_f.readlines()
        content = [line.strip('\n').split(delim) for line in content]

    return content

def choose_columns(input_content, columns_indices=[1,2,3,5]):
    chosen_columns = [[item.replace('"',"") for item in sublist if sublist.index(item) in columns_indices] for sublist in input_content]
    chosen_labels = [input_file_structure_labels[idx] for idx in columns_indices]
    return chosen_columns, chosen_labels

def split_dataset(content, train_ratio=0.7, test_ratio=0.3, shuffle=False):
    if shuffle:
        random.shuffle(content)

    # ratio must sum up to 1
    if train_ratio + test_ratio != 1:
        sys.stderr.write("Invalid dataset split ratio")
        sys.exit(1)

    train_relative_freq = int(len(content) * train_ratio)

    # slice the input data
    train_data = content[0:train_relative_freq]
    test_data = content[train_relative_freq:]
    return train_data, test_data

def parse_time(time):
    return float(time.replace('"',''))

def analyze_time(data, interval_len=3):
    max_time = parse_time(data[-1][0])
    print(max_time)
    resulting_counts = []
    resulting_counts_MS = []
    resulting_counts_SM = []
    t = 0

    while t <= max_time:
        current_cnt = 0 
        ms_cnt = 0 
        sm_cnt = 0
        for point in data:
            current_time = parse_time(point[0])

            if current_time >= t and current_time <= t + interval_len:
                current_cnt += 1
                if point[1] == MASTER:
                    ms_cnt += 1
                else:
                    sm_cnt += 1

        resulting_counts_MS.append(ms_cnt)
        resulting_counts_SM.append(sm_cnt)
        resulting_counts.append(current_cnt)
        t += interval_len

    print(resulting_counts)
    print(resulting_counts_MS)
    print(resulting_counts_SM)
    print(len(data))
    print(np.sum(resulting_counts))

def analyze_delta_time(data, interval_len=3):
    max_time = parse_time(data[-1][0])
    print(max_time)
    delta_time = []
    t = 0
    cnt = 0
    while t <= max_time:

        for i in range(0,len(data)):
            current_time = parse_time(data[i][0])
            cnt += 1
            if cnt >= len(data): 
                break
            delta_time.append(parse_time(data[i+1][0]) - parse_time(data[i][0]))

        t += interval_len
    delta_time.append('0.0')
    
    #print(delta_time)
    #print(len(data))
    #print(len(delta_time))


def analyze_data(train_data):
    analyze_time(train_data)
    analyze_delta_time(train_data)




content = read_input_file()
if content[0][6].replace('"', '')[0:2] == "<-":
    SLAVE = content[0][2].replace('"','')
    MASTER = content[0][3].replace('"','')
else:
    MASTER = content[0][2].replace('"','')
    SLAVE = content[0][3].replace('"','')

chosen_columns, columns_labels = choose_columns(content)
#print(chosen_columns)

train_data, test_data = split_dataset(chosen_columns)
analyze_data(train_data)
