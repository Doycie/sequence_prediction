
import torch

def read_generated_data(file_name, max_len, test):
    # Read file
    log_file = open(file_name, 'r')
    lines = log_file.readlines()
    ids = []

    count = 0
    prev_num = -1
    myrange = range(1, len(lines))
    if test:
        myrange = range(len(lines) -1, 1,-1)

    for linenum in myrange:
        for num in lines[linenum].split(']","[')[0][2:-3].split(','):

            if num != prev_num:
                count += 1
                if num == '':
                    break
                ids.append(int(num))
                prev_num = num
            if count >= max_len:
                break
    ids_set = set(ids)
    return ids, len(ids_set)


def read_data(file_name, max_len):
    # Read file
    can_log_file = open(file_name, 'r')
    lines = can_log_file.readlines()
    ids = []
    count = 0
    # Convert hex identifier data to numbers
    for line in lines:
        count += 1
        ids.append(int("0x" + line[27:][:3], 16))
        # print(int("0x"+line[27:][:3], 16))
        if count >= max_len:
            break

    # Normalize ids
    ids_set = set(ids)

    sorted_set = sorted(ids_set)


    id_dict = {}
    count = 0
    for id in sorted_set:
        id_dict[id] = count
        count += 1

    normalized_ids = []
    for id in ids:
        normalized_ids.append(id_dict[id])

    return normalized_ids, count

# Create some random data for testing purposes
def create_random_data(n_ids,max_len):
    normalized_ids = []
    count = 0
    while count < max_len:
        for i in range(n_ids):
            normalized_ids.append(i)

        count += n_ids
    return normalized_ids


def create_inout_sequence(input_data, tw, device, n_ids, num):
    train_seq = []
    for w in range(tw):
        ten = [0.0] * n_ids
        # Add for timing info + 3)
        # ten[int(input_data[num + w][0])] = 1.0 for time
        ten[int(input_data[num + w])] = 1.0
        # For timing info
        #ten[ min(85, int(input_data[num+w][1]/2.0))] = 1.0
        train_seq.append(ten)
        # print("seq: " + str(int(input_data[i+w])))
    train_seq = torch.FloatTensor(train_seq).to(device)
    train_label = [0.0] * n_ids

    # print("label" + str(int(input_data[i+tw])))
    # train_label[int(input_data[num + tw][0])] = 1.0
    train_label[int(input_data[num + tw])] = 1.0

    train_label = torch.FloatTensor(train_label).to(device)
    return train_seq, train_label


# Convert to 85 length tensor and labels format [0,0....,1,0....0] * window length
def create_inout_sequences(input_data, tw, device, n_ids):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw - 1):
        inout_seq.append(create_inout_sequence(input_data, tw, device, n_ids, i))
    return inout_seq


# Convert to window length tensor and labels, format: [2,3,4,...,2] (len = window length)
def create_inout_sequences_tcn(input_data, tw, device):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw - 1):
        train_seq = []
        for w in range(tw):
            train_seq.append(int(input_data[i + w]))
        train_seq = torch.IntTensor(train_seq).to(device)

        train_label = []
        for w in range(tw):
            train_label.append(int(input_data[i + w + 1]))
        train_label = torch.IntTensor(train_label).to(device)

        inout_seq.append((train_seq, train_label))
    return inout_seq


def batchify_tcn(input_data, batch_size,device):
    input_data_ten = torch.LongTensor(input_data).to(device)
    batch_amount = input_data_ten.size(0) // batch_size
    data = input_data_ten.narrow(0, 0, batch_amount * batch_size)
    data = data.view(batch_size, -1)
    return data


def get_batch_tcn(input_data, index, sequence_len):
    seq_len = min(sequence_len, input_data.size(1) - 1 - index)
    end_index = index + seq_len
    inp = input_data[:, index:end_index].contiguous()
    target = input_data[:, index + 1:end_index + 1].contiguous()  # The successors of the inp.
    return inp, target