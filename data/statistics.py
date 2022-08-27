
import numpy as np
import matplotlib.pyplot as plt

cutoff_read_file = 100000000


def read_data( file_name ):

    #Read file
    can_log_file = open(file_name, 'r')
    lines = can_log_file.readlines()

    times = []

    ids = []
    count = 0
    longest = 0.0
    shortest = 100000.0
    total = 0.0
    lessthan = 0

    prev_time = float(lines[0][1:][:16])
    #Convert hex identifier data to numbers
    for line in lines[1:len(lines)]:

        time = float(line[1:][:16])

        diff = time - prev_time
        times.append(diff * 1000) #Fromseconds to mili seconds
        if diff > longest:
            longest = diff
        if diff < shortest:
            shortest = diff
        total += diff

        if diff < 0.0001:
            lessthan += 1

        prev_time = time

        count +=1
        ids.append(  int("0x"+line[27:][:3], 16))
        #print(int("0x"+line[27:][:3], 16))
        if count >= cutoff_read_file:
            break

    print("shortest: " + str(shortest))
    print("longest: " + str(longest))
    print("total: " + str(total))
    print("count: " + str(count))
    print("average: " + str(total / count))
    print("amount of times less than 0.0001: " + str(lessthan))

    #n, bins, patches = plt.hist(times, [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9], density=True, facecolor='g', alpha=0.75)

    #plt.xlabel('Time between messages in ms')
    #plt.ylabel('Probability in %')
    #plt.title('Distribution of times between messages')
    #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    #plt.xlim(40, 160)
    #plt.ylim(0, 0.03)
    #plt.grid(True)
    #plt.show()

    #Normalize ids

    ids_set = set(ids)
    id_dict = {}
    count = 0
    for id in ids_set:
        id_dict[id] = count
        count +=1

    normalized_ids = []
    for id in ids:
        normalized_ids.append(id_dict[id])

    n, bins, patches = plt.hist(normalized_ids, range(0,85),
                                density=True, facecolor='g', alpha=0.75)

    plt.xlabel('ID')
    plt.ylabel('% of occurence')
    plt.title('Distribution of IDS training data')
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    # plt.xlim(40, 160)
    # plt.ylim(0, 0.03)
    plt.grid(True)
    plt.show()


    #return normalized_ids

    can_log_file = open('../data/testing.log', 'r')
    lines = can_log_file.readlines()

    times = []

    ids = []
    count = 0
    longest = 0.0
    shortest = 100000.0
    total = 0.0
    lessthan = 0

    prev_time = float(lines[0][1:][:16])
    # Convert hex identifier data to numbers
    for line in lines[1:len(lines)]:

        time = float(line[1:][:16])

        diff = time - prev_time
        times.append(diff * 1000)  # Fromseconds to mili seconds
        if diff > longest:
            longest = diff
        if diff < shortest:
            shortest = diff
        total += diff

        if diff < 0.0001:
            lessthan += 1

        prev_time = time

        count += 1
        ids.append(int("0x" + line[27:][:3], 16))
        # print(int("0x"+line[27:][:3], 16))
        if count >= cutoff_read_file:
            break

    normalized_ids = []
    for id in ids:
        normalized_ids.append(id_dict[id])

    n, bins, patches = plt.hist(normalized_ids, range(0, 85),
                                density=True, facecolor='g', alpha=0.75)

    plt.xlabel('ID')
    plt.ylabel('% of occurence')
    plt.title('Distribution of IDS test data')
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    # plt.xlim(40, 160)
    # plt.ylim(0, 0.03)
    plt.grid(True)
    plt.show()


read_data('../data/training.log')
#read_data('../data/testing.log')


