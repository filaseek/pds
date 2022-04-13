import os, sys, random
import numpy as np
import matplotlib.pyplot as plt

input_file_structure_labels = {0:"NO.", 1:"TIME", 2:"SOURCE", 3:"DESTINATION", 4:"PROTOCOL", 5:"LENGTH",6:"INFO"}
MASTER = ""
SLAVE = ""


def read_input_file(input_file=os.path.join(os.getcwd(), "dataset","input_data.csv"), delim=','):
    with open(input_file,'r') as in_f:
        _ = in_f.readline()
        content = in_f.readlines()
        content = [line.strip('\n').split(delim,6) for line in content]
        
    for i in range(0, len(content)):
        content[i][0] = int(content[i][0].replace('"',''))
    
    return content

def choose_columns(input_content, columns_indices=[1,2,3,5]):
    chosen_columns = [[item for item in sublist if sublist.index(item) in columns_indices] for sublist in input_content]
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
    MS_communication = []
    SM_communication = []
    resulting_counts = []
    resulting_counts_MS = []
    resulting_counts_SM = []
    MS_avg_size = []
    SM_avg_size = []
    t = 0

    while t <= max_time:
        current_cnt = 0 
        ms_cnt = 0
        ms_sum = 0 
        sm_cnt = 0
        sm_sum = 0
        for point in data:
            current_time = parse_time(point[0])
            #print(point)
            if current_time >= t and current_time <= t + interval_len:
                current_cnt += 1
                if point[1] == MASTER:
                    MS_communication.append(point)
                    ms_cnt += 1
                    # try:
                    #     ms_sum += float(point[3])
                    # except:
                    #     pass 
                else:
                    SM_communication.append(point)
                    sm_cnt += 1
                    # try:
                    #     sm_sum += float(point[3])
                    # except:
                    #     pass
        # try:            
        #     MS_avg_size.append(ms_sum/ms_cnt)
        # except:
        #     MS_avg_size.append('0')
        # try:    
        #     SM_avg_size.append(sm_sum/sm_cnt)
        # except:
        #     SM_avg_size.append('0')
        resulting_counts_MS.append(ms_cnt)
        resulting_counts_SM.append(sm_cnt)
        resulting_counts.append(current_cnt)
        t += interval_len
        #print(t)

    #print(MS_communication)
    #print(SM_avg_size)
    #print(resulting_counts)
    #print(resulting_counts_MS)
    #print(resulting_counts_SM)
    #print(len(data))
    #print(np.sum(resulting_counts))
    return MS_communication, SM_communication


def analyze_delta_time(data, interval_len=3):
    max_time = parse_time(data[-1][0])
    #print(max_time)
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
    master, slave = analyze_time(train_data)
    analyze_delta_time(train_data)
    return master, slave

def box_plot(content_master, content_slave, idx=3):
    correct_column_master = [int(line[idx]) for line in content_master]
    #print(content_slave)

    correct_column_slave = [int(line[idx]) for line in content_slave]
    #print(correct_column_master)

    plt.figure(figsize=(6,4)) 

    # pro vsechna ruzna nastaveni mutace vykreslete boxplot
    plt.boxplot([correct_column_master, correct_column_slave], labels=["Master","Slave"], notch=True)

    plt.xlabel('Varianty')
    plt.ylabel('Hodnoty')
    plt.title('Boxplot')

    plt.show()                                                           
    plt.close()

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
master, slave = analyze_data(train_data)
box_plot(master, slave)
