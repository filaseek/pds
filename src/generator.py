import random
start_number = 102970
i = start_number
start_time = 17601.00000
time = float(start_time)
master = "192.168.11.248"
slave = "192.168.11.111"
number_of_generated_pcks = 10000

f = open("fake_data.csv","w")
f.write('"No.","Time","Source","Destination","Protocol","Length","Info"\n')

while i < start_number+number_of_generated_pcks:
    time += random.triangular(0.00001, 0.5, 0.001)
    pck_size = int(random.triangular(50, 1000, 125))
    f.write('"'+str(i)+'","'+str(time)+'","'+master+'","'+slave+'","IEC 60870-5-104","'+str(pck_size)+'","-> S (15250)"\n')
    i += 1
    time += random.triangular(0.00001, 0.5, 0.001)
    pck_size = int(random.triangular(200, 1000, 260))
    f.write('"'+str(i)+'","'+str(time)+'","'+slave+'","'+master+'","TCP","'+str(pck_size)+'","-> S (15250)"\n')
    i += 1