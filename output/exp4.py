from cgi import test
import os, sys, random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



input_file_structure_labels = {0:"NO.", 1:"TIME", 2:"SOURCE", 3:"DESTINATION", 4:"PROTOCOL", 5:"LENGTH",6:"INFO"}
MASTER = ""
SLAVE = ""
interval_len=20
def read_input_file(input_file=os.path.join(os.getcwd(), "../dataset","input_data.csv"), delim=','):
    with open(input_file,'r') as in_f:
        _ = in_f.readline()
        content = in_f.readlines()
        content = [line.strip('\n').split(delim,6) for line in content]
        
    for i in range(0, len(content)):
        content[i][0] = int(content[i][0].replace('"',''))
    
    return content

def choose_columns(input_content, columns_indices=[1,2,3,5]):
    chosen_columns = [[item for item in sublist if sublist.index(item) in columns_indices] for sublist in input_content]
    chosen_columns = [[item.replace('"','') for item in sublist] for sublist in chosen_columns]
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

def analyze_time(data):
    max_time = parse_time(data[-1][0])
    #print(max_time)
    resulting_counts = []
    avg_size = []
    total_size = []
    t = 0

    while t <= max_time:
        current_cnt = 0 
        sum = 0
        for point in data:
            current_time = parse_time(point[0])
            if current_time >= t and current_time <= t + interval_len:
                current_cnt += 1
                sum += float(point[3])
        if current_cnt != 0:            
            avg_size.append(sum/current_cnt)
        else:
            avg_size.append('0')
        total_size.append(sum)
        resulting_counts.append(current_cnt)
        t += interval_len

    #print(resulting_counts)

    return avg_size, total_size, resulting_counts


def analyze_delta_time(data):

    delta_time = []

    for i in range(0,len(data)):
        if i+1 < len(data):
            delta_time.append(float(data[i+1][0]) - float(data[i][0]))

    return delta_time

def separate_communication(data):

    communication = []
    pck_size = []
    for i in range(0,len(data)):
            communication.append(data[i])
            pck_size.append(data[i][3])


    return communication, pck_size


def box_plot(content_master, content_slave, idx=3):
    correct_column_master = [int(line[idx]) for line in content_master]

    correct_column_slave = [int(line[idx]) for line in content_slave]

    plt.figure(figsize=(6,4)) 

    plt.boxplot([correct_column_master, correct_column_slave], labels=["Master","Slave"], notch=True)

    plt.xlabel('Varianty')
    plt.ylabel('Hodnoty')
    plt.title('Boxplot')

    plt.show()                                                           
    plt.close()

def sigma_rule(content, idx=3):
    correct_column = [int(line[idx]) for line in content]

    plt.figure(figsize=(6,4))

    mean = np.mean(correct_column)
    sigma = np.std(correct_column)

    x = np.linspace(mean - 3*sigma, mean + 3*sigma)
    y = stats.norm.pdf(x, mean, sigma)

    #plt.stem(correct_column)
    y_values = np.full(len(correct_column),0)
    plt.scatter(correct_column, y_values)
    plt.plot(x, y)
    plt.show()

def graph_packets(master_packets, slave_packets):
    x = np.linspace(0,len(master_packets))

    plt.figure(figsize=(6,4))
    plt.plot(master_packets)

    plt.show()


def showplot(interarrival_time, size_of_pck):
    plt.figure(figsize=(6,4))
    plt.scatter(size_of_pck, interarrival_time)
    plt.show()
    plt.close()

def calculate_statistics(content):
    min = np.min(content)
    max = np.max(content)
    q = np.quantile(content, [0,0.25,0.5,0.75,1])
    print("-----------------QUANTILE STATISTICS--------------------")
    print("<min> <Q1> <Q2> <Q3> <Q4> <max>")
    print(q)
    print(min)
    print(max)
    print("--------------------------------------------------------")

def train_svm(data, gamma = 0.1, nu = 0.015):
    model = OneClassSVM(kernel = 'rbf', gamma = gamma, nu = nu)
    model.fit(data)
    return model

def evaluate(model, test_data, ground_truth):
    predicted_y = model.predict(test_data)
    predicted_y[predicted_y == 1] = 1
    predicted_y[predicted_y == -1] = 0

    conf_mat = confusion_matrix(ground_truth, predicted_y)
    tn, fp, fn, tp = confusion_matrix(ground_truth, predicted_y).ravel()
    print(predicted_y)
    print(conf_mat)
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    print("ACCURACY:\t" + str(accuracy))



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

# ----------------------------------------------------------REGULAR COMMUNICATION -------------------------
MS_communication, pck_size_ms = separate_communication(train_data)

pck_size_ms = pck_size_ms[:-1]

avg_size_ms, total_size_ms, count_ms = analyze_time(MS_communication)
delta_time_ms = analyze_delta_time(MS_communication)


# box_plot(MS_communication, SM_communication)
# sigma_rule(MS_communication)
# sigma_rule(SM_communication)
# calculate_statistics(delta_time_ms)
# calculate_statistics(delta_time_sm)
# showplot(delta_time_ms, pck_size_ms)
# showplot(delta_time_sm, pck_size_sm)

#------------------------------------------------------------ TEST COMMUNICATION ---------------------------
test_MS_communication, test_pck_size_ms = separate_communication(test_data)

test_pck_size_ms = test_pck_size_ms[:-1]

test_delta_time_ms = analyze_delta_time(test_MS_communication)
test_avg_size_ms, test_total_size_ms, test_count_ms = analyze_time(test_MS_communication)

#---------------------------------------------------------- ANOMALIOUS COMMUNICATION ---------------------------
fake_data = read_input_file(os.path.join(os.getcwd(),"../dataset","fake_data.csv"))
fake_columns, fake_labels = choose_columns(fake_data)

fake_MS_communication, fake_pck_size_ms = separate_communication(fake_columns)
fake_pck_size_ms = fake_pck_size_ms[:-1]

fake_avg_size_ms, fake_total_size_ms, fake_count_ms = analyze_time(fake_MS_communication)
fake_delta_time_ms = analyze_delta_time(fake_MS_communication)
#---------------------------------------------------------- MODEL PREPARATION --------------------------------

while test_total_size_ms[0] == 0 and test_count_ms[0] == 0 : 
    test_total_size_ms.remove(0) 
    test_count_ms.remove(0)

while fake_total_size_ms[0] == 0 and fake_count_ms[0] == 0 : 
    fake_total_size_ms.remove(0) 
    fake_count_ms.remove(0)

df = [pair for pair in zip(delta_time_ms, pck_size_ms)]
df1= [pair for pair in zip(total_size_ms, count_ms)]

# svm_model = train_svm(df)
svm_model1= train_svm(df1, gamma=0.001, nu=0.07) #0.001, 0.07, interval 20s = 92.95% precision


positive_len = len([pair for pair in zip(test_delta_time_ms, test_pck_size_ms)] )
negative_len = len([pair for pair in zip(fake_delta_time_ms, fake_pck_size_ms)])
test_points_expanded = [pair for pair in zip(test_delta_time_ms, test_pck_size_ms)] + [pair for pair in zip(fake_delta_time_ms, fake_pck_size_ms)]
ground_truth_expanded = np.concatenate((np.full(positive_len,1),np.full(negative_len,0)))


positive_len = len([pair for pair in zip(test_total_size_ms, test_count_ms)])
negative_len = len([pair for pair in zip(fake_total_size_ms, fake_count_ms)])
test_points_expanded1 = [pair for pair in zip(test_total_size_ms, test_count_ms)] + [pair for pair in zip(fake_total_size_ms, fake_total_size_ms)]
ground_truth_expanded1= np.concatenate((np.full(positive_len,1),np.full(negative_len,0)))

#evaluate(svm_model, test_points_expanded, ground_truth_expanded)
#evaluate(svm_model1,test_points_expanded1, ground_truth_expanded1)

test_points = [pair for pair in zip(test_delta_time_ms, test_pck_size_ms)]
ground_truth = np.full(len(test_points),1)
svm_model = train_svm(df)
evaluate(svm_model, test_points, ground_truth)

test_points = [pair for pair in zip(test_total_size_ms, test_count_ms)]
ground_truth = np.full(len(test_points),1)
#print(ground_truth)
#print(len(ground_truth))
#svm_model = train_svm(df)
evaluate(svm_model1, test_points, ground_truth)

#master, slave, master_packets, slave_packets = analyze_time(train_data)
#graph_packets(master_packets, slave_packets)