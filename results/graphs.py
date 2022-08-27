import numpy as np

import matplotlib
import matplotlib.pyplot as plt

final_data = []


def plot_time():

    for name in ["LSTM64", "GRU64", "TCN64"]:
        data = []
        for cat in ["base","float","float0","4c","85c"]:
            result_file = open("../models/results/timing_info/" + name + cat + "_output.txt", 'r')
            lines = result_file.readlines()

            data.append(float(lines[3].split(':')[1]))


        final_data.append((data[0],data[1],data[2],data[3],data[4]))

    # set width of bar
    barWidth = 0.25
    fig, ax = plt.subplots(figsize=(9, 7))

    # Set position of bar on X axis
    br1 = np.arange(5)
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, final_data[0], color='tab:orange', width=barWidth,
            edgecolor='grey', label='LSTM')
    plt.bar(br2, final_data[1], color='tab:blue', width=barWidth,
            edgecolor='grey', label='GRU')
    plt.bar(br3, final_data[2], color='tab:olive', width=barWidth,
            edgecolor='grey', label='TCN')

    # Adding Xticks
    plt.xlabel('Method', fontweight='bold', fontsize=20)
    plt.ylabel('Accuracy [%]', fontweight='bold', fontsize=20)
    plt.xticks([r + barWidth for r in range(0, 5)],
               ['Base', 'Float', 'Float0', '4C', '85C'], fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.rcParams.update({'font.size': 20})
    plt.legend()

    #plt.show()
    plt.savefig('timing_info.svg')

def plot_varying_tw():
    data = []
    datax = [4, 8, 16, 32, 64, 128]
    c = 0
    for name in ["LSTM", "GRU", "TCN"]:
        data.append([])
        for tw in datax:
            result_file = open("../models/results/syn_var_tw/" + name + str(tw) +"_output.txt", 'r')
            lines = result_file.readlines()
            data[c].append(100*float(lines[3].split(':')[1]))
        c+=1
    fig , ax = plt.subplots(figsize=(12, 9))

    plt.rcParams.update({'font.size': 20})
    # Adding Xticks



    plt.xlabel('Train window', fontweight='bold', fontsize=20)
    plt.ylabel('Accuracy [%]', fontweight='bold', fontsize=20)

    ax.plot(datax, data[0], label = "LSTM", marker="o")
    ax.plot(datax, data[1], label = "GRU", marker="^")
    ax.plot(datax, data[2], label = "TCN", marker="s")



    ax.set_xscale('log')
    ax.set_xticks(datax )
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend()

    #plt.show()
    plt.savefig('syn_var_tw.svg')

def plot_varying_hidden_time():
    data = []
    datax = [4,8,16,32,64,128]
    c = 0
    for name in ["LSTM", "GRU","TCN"]:
        data.append([])
        for tw in datax:
            result_file = open("../models/results/syn_var_tw/" + name + str(tw) +"_output.txt", 'r')
            lines = result_file.readlines()
            data[c].append( float(lines[1].split(':')[1].split(' ')[1]))
        c+=1
    fig , ax = plt.subplots(figsize=(12, 9))

    plt.rcParams.update({'font.size': 20})
    # Adding Xticks
    plt.xlabel('Train window', fontweight='bold', fontsize=20)
    plt.ylabel('Execution time[μs]', fontweight='bold', fontsize=20)

    ax.plot(datax, data[0], label = "LSTM", marker="o")
    ax.plot(datax, data[1] , label = "GRU", marker="^")
    ax.plot(datax, data[2], label = "TCN", marker="s")

    ax.set_xscale('log')
    ax.set_xticks(datax )
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend()

    #plt.show()
    plt.savefig('syn_var_tw_ex.svg')

def plot_varying_hidden():
    data = []
    datax = [1,10,25,40,55, 70, 85, 100, 115,130,145]
    c = 0
    for name in ["LSTM", "GRU"]:
        data.append([])
        for tw in datax:
            result_file = open("../models/results/dat_var_hidden/" + name + str(tw) +"_output.txt", 'r')
            lines = result_file.readlines()
            data[c].append(100* float(lines[3].split(':')[1]))
        c+=1
    fig , ax = plt.subplots(figsize=(12, 9))

    plt.rcParams.update({'font.size': 20})
    # Adding Xticks
    plt.xlabel('Hidden size', fontweight='bold', fontsize=20)
    plt.ylabel('Accuracy [%]', fontweight='bold', fontsize=20)

    ax.plot(datax, data[0], label = "LSTM", marker="o")
    ax.plot(datax, data[1] , label = "GRU", marker="^")

    #ax.set_xscale('log')
    ax.set_xticks(datax )
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend()

    #plt.show()
    plt.savefig('dat_var_hidden.svg')

def plot_varying_jitter():
    data = []
    datax = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    c = 0
    for name in ["LSTM64", "GRU64", "TCN64"]:
        data.append([])
        for tw in datax:
            result_file = open("../models/results/jitter/" + name + str(tw) +"_output.txt", 'r')
            lines = result_file.readlines()
            data[c].append(float(lines[3].split(':')[1]))
        c+=1
    fig , ax = plt.subplots(figsize=(12, 9))

    plt.rcParams.update({'font.size': 20})
    # Adding Xticks



    plt.xlabel('Jitter', fontweight='bold', fontsize=20)
    plt.ylabel('Accuracy [%]', fontweight='bold', fontsize=20)

    ax.plot(datax, data[0], label = "LSTM", marker="o")
    ax.plot(datax, data[1], label = "GRU", marker="^")
    ax.plot(datax, data[2], label = "TCN", marker="s")



    #ax.set_xscale('log')
    ax.set_xticks(datax )
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend()

    #plt.show()
    plt.savefig('jitter.svg')

def plot_all_metrics_all_models():

    for name in ["LSTM", "GRU", "TCN"]:

        result_file = open("../models/results/dat_main/" + name + "_output.txt", 'r')
        lines = result_file.readlines()

        data = []
        amount_data = []
        count = 0
        for num in lines[0][17:].split(','):
            if num != ' \n':
                nums = num.split('/')
                data.append((float(nums[0]), float(nums[1]), str(count) ))
                #amount_data.append(float(nums[1]))
                count +=1

        poaam = 0.0
        paam = 0.0
        oaam = 0.0
        am = 0.0

        max_occurence = 0
        for _, ao, _ in data:
            max_occurence = max(ao, max_occurence)

        total_occurence = 0
        for _, ao, _ in data:
            total_occurence += ao

        for dc, ao, idnum in data:
            # * (ao/max_occurence)
            poaam += dc * ((85 - (float(idnum)))/85) * (ao/max_occurence)
            paam += dc * ((85 - (float(idnum)))/85)
            oaam += dc * (ao/max_occurence)
            am += dc * (ao)

        poaam /= 85
        paam /= 85
        oaam /= 85
        am /= total_occurence


        #print(f'accuracy metric: {am}')
        #print(f'Priority adjusted accuracy metric: {paam}')
        #print(f'occurence adjusted accuracy metric: {oaam}')
        #print(f'Priority occurence adjusted accuracy metric: {poaam}')

        final_data.append((am*100,paam*100,oaam*100,poaam*100))


    # set width of bar
    barWidth = 0.25
    fig ,ax = plt.subplots(figsize =(9, 7))



    # Set position of bar on X axis
    br1 = np.arange(4)
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, final_data[0], color='tab:orange', width=barWidth,
            edgecolor='grey', label='LSTM')
    plt.bar(br2, final_data[1], color='tab:blue', width=barWidth,
            edgecolor='grey', label='GRU')
    plt.bar(br3, final_data[2], color='tab:olive', width=barWidth,
            edgecolor='grey', label='TCN')

    # Adding Xticks
    plt.xlabel('Metrics', fontweight='bold', fontsize=20)
    plt.ylabel('Accuracy [%]', fontweight='bold', fontsize=20)
    plt.xticks([r + barWidth for r in range(0,4)],
               ['Accuracy', 'Priority', 'Occurence', 'Both'], fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.rcParams.update({'font.size': 20})
    plt.legend()

    #plt.show()
    plt.savefig('main_metrics_accuracy.svg')


def plot_varying_ck():
    data = []
    datax = [2,3,4]
    c = 0
    for channel in [2,3,4]:
        data.append([])
        for kernel in datax:
            result_file = open("../models/results/dat_var_ck/TCN(" + str(channel) +", " + str(kernel) +")_output.txt", 'r')
            lines = result_file.readlines()
            data[c].append(100*float(lines[3].split(':')[1]))
            #data[c].append(float(lines[1].split(':')[1].split(' ')[1]))
        c+=1
    fig , ax = plt.subplots(figsize=(12, 9))

    plt.rcParams.update({'font.size': 20})
    # Adding Xticks

    plt.xlabel('Kernel size', fontweight='bold', fontsize=20)
    plt.ylabel('Accuracy[%]', fontweight='bold', fontsize=20)

    ax.plot(datax, data[0], label = "Channel size=2", marker="o")
    ax.plot(datax, data[1], label = "Channel size=3", marker="^")
    ax.plot(datax, data[2], label = "Channel size=4", marker="s")

    #ax.set_xscale('log')
    ax.set_xticks(datax )
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend()

    #plt.show()
    plt.savefig('dat_var_ck.svg')

def plot_per_id(data):

    data.sort(key=lambda y: y[1], reverse=True)
    data_correct, amount_occurence, idnum = zip(*data)


    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(idnum, data_correct, color='maroon',
            width=0.4)

    plt.show()

def plot_varying_u():
    data = []
    datax = [10, 30, 50, 70, 90]
    c = 0
    for name in ["LSTM", "GRU", "TCN"]:
        data.append([])
        for tw in datax:
            result_file = open("../models/results/syn_var_u/" + name + str(tw/100) +"_output.txt", 'r')
            lines = result_file.readlines()
            data[c].append(100*float(lines[3].split(':')[1]))
        c+=1
    fig , ax = plt.subplots(figsize=(12, 9))

    plt.rcParams.update({'font.size': 20})
    # Adding Xticks



    plt.xlabel('Utilization [%]', fontweight='bold', fontsize=20)
    plt.ylabel('Accuracy [%]', fontweight='bold', fontsize=20)

    ax.plot(datax, data[0], label = "LSTM", marker="o")
    ax.plot(datax, data[1], label = "GRU", marker="^")
    ax.plot(datax, data[2], label = "TCN", marker="s")



    #ax.set_xscale('log')
    ax.set_xticks(datax)
    ax.set_yticks([30,50,70,90])
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend()

    plt.show()
    #plt.savefig('syn_var_u.svg')

def plot_varying_ids():
    data = []
    datax = [5, 15, 25, 35, 45, 55]
    c = 0
    for name in ["LSTM", "GRU", "TCN"]:
        data.append([])
        for tw in datax:
            result_file = open("../models/results/syn_var_ids/" + name + str(tw) +"_output.txt", 'r')
            lines = result_file.readlines()
            data[c].append(100*float(lines[3].split(':')[1]))
        c+=1
    fig , ax = plt.subplots(figsize=(12, 9))

    plt.rcParams.update({'font.size': 20})
    # Adding Xticks



    plt.xlabel('Number of IDs', fontweight='bold', fontsize=20)
    plt.ylabel('Accuracy [%]', fontweight='bold', fontsize=20)

    ax.plot(datax, data[0], label = "LSTM", marker="o")
    ax.plot(datax, data[1], label = "GRU", marker="^")
    ax.plot(datax, data[2], label = "TCN", marker="s")



    #ax.set_xscale('log')
    ax.set_xticks(datax)
    ax.set_yticks([30,50,70,90])
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend()

    #plt.show()
    plt.savefig('syn_var_ids.svg')

def plot_varying_u_time():
    data = []
    datax = [10,30,50,70,90]
    c = 0
    for name in ["LSTM", "GRU", "TCN"]:
        data.append([])
        for tw in datax:
            result_file = open("../models/results/syn_var_u/" + name + str(tw/100) +"_output.txt", 'r')
            lines = result_file.readlines()
            data[c].append(float(lines[1].split(':')[1].split(' ')[1]))
        c+=1
    fig , ax = plt.subplots(figsize=(12, 9))

    plt.rcParams.update({'font.size': 20})
    # Adding Xticks
    plt.xlabel('Utilization[%]', fontweight='bold', fontsize=20)
    plt.ylabel('Execution time[μs]', fontweight='bold', fontsize=20)

    ax.plot(datax, data[0], label = "LSTM", marker="o")
    ax.plot(datax, data[1] , label = "GRU", marker="^")
    ax.plot(datax, data[2], label = "TCN", marker="s")

    ax.set_xticks(datax)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend()

    #plt.show()
    plt.savefig('syn_var_u_ex.svg')

def plot_varying_ids_time():
    data = []
    datax = [5, 15, 25, 35, 45, 55]
    c = 0
    for name in ["LSTM", "GRU", "TCN"]:
        data.append([])
        for tw in datax:
            result_file = open("../models/results/syn_var_ids/" + name + str(tw) +"_output.txt", 'r')
            lines = result_file.readlines()
            data[c].append(float(lines[1].split(':')[1].split(' ')[1]))
        c+=1
    fig , ax = plt.subplots(figsize=(12, 9))

    plt.rcParams.update({'font.size': 20})
    # Adding Xticks
    plt.xlabel('Number of IDs', fontweight='bold', fontsize=20)
    plt.ylabel('Execution time[μs]', fontweight='bold', fontsize=20)

    ax.plot(datax, data[0], label = "LSTM", marker="o")
    ax.plot(datax, data[1] , label = "GRU", marker="^")
    ax.plot(datax, data[2], label = "TCN", marker="s")

    ax.set_xticks(datax)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend()

    #plt.show()
    plt.savefig('syn_var_ids_ex.svg')


def plot_varying_ids_par():
    data = []
    datax = [5, 15, 25, 35, 45, 55]
    c = 0
    for name in ["LSTM", "GRU", "TCN"]:
        data.append([])
        for tw in datax:
            result_file = open("../models/results/syn_var_ids/" + name + str(tw) +"_output.txt", 'r')
            lines = result_file.readlines()
            if name == "TCN":
                data[c].append(int(lines[17].split(': ')[1].replace(',', '')))
            else:
                data[c].append(int(lines[13].split(': ')[1].replace(',','')))
        c+=1

    fig , ax = plt.subplots(figsize=(12, 9))

    plt.rcParams.update({'font.size': 20})
    # Adding Xticks
    plt.xlabel('Number of IDs', fontweight='bold', fontsize=20)
    plt.ylabel('Number of parameters', fontweight='bold', fontsize=20)

    ax.plot(datax, data[0], label = "LSTM and TCN", marker="o")
    ax.plot(datax, data[1] , label = "GRU", marker="^")
    #ax.plot(datax, data[2], label = "TCN", marker="s")

    ax.set_xticks(datax)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend()

    #plt.show()
    plt.savefig('syn_var_ids_par.svg')

plot_varying_ids_par()