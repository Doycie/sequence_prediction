# TCN model by:
# https://github.com/locuslab/TCN [1]
# @article{BaiTCN2018,
# 	author    = {Shaojie Bai and J. Zico Kolter and Vladlen Koltun},
# 	title     = {An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling},
# 	journal   = {arXiv:1803.01271},
# 	year      = {2018},
# }


import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from torchinfo import summary

import lstm
import gru
import tcn

import time
import data_util

torch.manual_seed(1)

# File parameters
training_data_len = 100000
validation_data_len = 25000
test_data_len = 50000


# Model parameters
num_layer = 1


# Training parameters
learning_rate = 0.01
epochs = 1

# Read the input data from file and divide into train and test
full_train_data, n_ids = data_util.read_data('../data/training.log', training_data_len + validation_data_len)
full_test_data, tn_ids = data_util.read_data('../data/testing.log', test_data_len)
#full_train_data, n_ids = data_util.read_generated_data(
#    '../data/IDEAL_automotive_84_tasks_0.5_utilization_1661381775.csv', training_data_len + validation_data_len, False)
#full_test_data, tn_ids = data_util.read_generated_data(
 #  '../data/IDEAL_automotive_84_tasks_0.5_utilization_1661381775.csv', test_data_len, True)

n_ids = max(n_ids, tn_ids)
test_data = full_test_data[:test_data_len]

print("number of ids:" + str(n_ids))

# Use cuda if available
print(torch.version.cuda)
print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Device: {device}')


# Training procedure with batch size of 1
def train(model, model_name, train_window, reset_hidden_layer, squeeze_input):

    print("Training model")

    # Bring model to correct device
    model.to(device)

    # For timeseries cross validation
    k = 10
    split_k = int(training_data_len / k)

    # Set up a loss function and optimizer nn.MSELoss used for lstm
    # loss_function = nn.L1Loss()
    loss_function = nn.CrossEntropyLoss()
    lr = learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Start measuring time
    prev_time = time.time()

    best_validation_loss = 1e8

    # For epochs
    for i in range(epochs):
        # For timeseries validation
        for ki in range(1, k+1):
            counter = 0

            train_data = full_train_data[:ki * split_k]
            validation_data = full_train_data[len(full_train_data) - validation_data_len:]

            model.train()
            try:
                for tr in range(len(train_data) - train_window - 1):

                    seq, labels = data_util.create_inout_sequence(train_data, train_window, device, n_ids, tr)
                    # Reset all gradients
                    optimizer.zero_grad()

                    # Some models need their hidden state reset
                    if reset_hidden_layer:
                        model.hidden_cell = (torch.zeros(num_layer, 1, model.hidden_layer_size).to(device),
                                             torch.zeros(num_layer, 1, model.hidden_layer_size).to(device))

                    # Evaluate the model
                    if squeeze_input:
                        output = model(seq.unsqueeze(0)).squeeze(0)[train_window-1]
                    else:
                        output = model(seq)

                    # Bookkeeping time
                    counter += 1
                    dt = time.time() - prev_time
                    if dt > 5:
                        prev_time = time.time()
                        print("%.2f %% " % (counter / (split_k*ki) * 100))

                    # Calculate loss
                    output.to(device)
                    labels.to(device)
                    # loss function for cross entropy
                    single_loss = loss_function(output, labels.argmax())
                    # loss function for L1:
                    # single_loss = loss_function(output, labels)
                    # Backpropagate the loss
                    single_loss.backward()
                    # Update the weights
                    optimizer.step()

                validation_loss = evaluate_the_model(model, model_name,train_window , validation_data, reset_hidden_layer, squeeze_input, False)
                if validation_loss < best_validation_loss:
                    torch.save(model.state_dict(), '../models/' + model_name)
                    print("Found better model with %.2f validation loss, saving it..." % validation_loss)
                    best_validation_loss = validation_loss
            except KeyboardInterrupt:
                validation_loss = evaluate_the_model(model, model_name, train_window ,validation_data, reset_hidden_layer, squeeze_input, False)
                torch.save(model.state_dict(), '../models/' + model_name)
                print("Interruption! saving with %.2f validation loss, saving it..." % validation_loss)
                break

        # Modify learning rate for slower decent
        lr = lr / 2.
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        print(f'epoch: {i:3} loss: {best_validation_loss:10.8f}')


def print_parameters(file, cur_model, cur_training_data_len, cur_validation_data_len,
                     cur_test_data_len, cur_model_name, cur_n_ids, cur_train_window, cur_learning_rate, cur_epochs):
    file.write(str(summary(cur_model)))
    file.write(str(cur_model))

    #file.write(f'training data len {cur_training_data_len}\n')
    #file.write(f'validation data len {cur_validation_data_len}\n')
    #file.write(f'test data len {cur_test_data_len}\n')
    #file.write(f'model name {cur_model_name}\n')

    #file.write(f'number ids: {cur_n_ids}\n')
    #file.write(f'train window length: {cur_train_window}\n')

    #file.write(f'learning rate: {cur_learning_rate}\n')
    #file.write(f'epochs trained: {cur_epochs}\n')


def evaluate_the_model(model,model_name, train_window ,data, reset_hidden, squeeze_input, should_print):

    data_len = len(data)
    # print("Testing for " + str(len(data) - train_window) + " cases. Random chance would be " + str(
    #    (len(data) - train_window) / n_ids))

    # Bookkeeping
    total_duration = 0
    count = 0
    total_count = 0

    # Put the model in evaluation mode
    model.eval()

    # Get the test data
    # tensor_data = torch.FloatTensor(data).to(device)
    # test_inout_seq = data_util.create_inout_sequences(tensor_data, train_window, device, n_ids)

    # The chosen loss function
    loss_function = nn.L1Loss()
    total_loss = 0

    per_id_result_correct = [0] * n_ids
    per_id_result_total = [0] * n_ids

    for tr in range(len(data) - train_window - 1):
        seq, label = data_util.create_inout_sequence(data, train_window, device, n_ids, tr)
        with torch.no_grad():
            if reset_hidden:
                model.hidden = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                                torch.zeros(1, 1, model.hidden_layer_size).to(device))
            prev_time = time.time_ns()
            if squeeze_input:
                res = model(seq.unsqueeze(0)).squeeze(0)[train_window-1]
            else:
                res = model(seq)

            # Calculate loss
            single_loss = loss_function(res, label)
            total_loss += single_loss.item()

            duration = time.time_ns() - prev_time
            total_duration += duration
            total_count += 1

        # print(f'label: {label.argmax()} predicted: {torch.argmax(res)}')
        per_id_result_total[label.argmax()] += 1
        if torch.argmax(res) == label.argmax():
            count += 1
            per_id_result_correct[label.argmax()] += 1

    if should_print:
        console_output_file = "results/" + model_name + "_output.txt"
        with open(console_output_file, 'w') as file:

            output_per_id = ""
            for current_id in range(0, n_ids):
                output_per_id += str(round(per_id_result_correct[current_id] / per_id_result_total[current_id], 2)) + \
                                 str("/") + str(per_id_result_total[current_id]) + str(", ")
            output_per_id = "Per id accuracy: " + output_per_id

            file.write(output_per_id)
            file.write("\n Average duration: %.2f us\n" % ((total_duration/1000) / total_count))
            file.write("Amount of times correct: \n" + str(count) + " out of " + str(data_len - train_window))
            file.write("Percentage correct: %.2f\n" % (count / (data_len - train_window)))
            file.write("MSE loss: \n" + str(total_loss))

            print_parameters(file, model, training_data_len, validation_data_len, test_data_len, model_name,
                             n_ids, train_window, learning_rate, epochs)

    return total_loss


def run_all_models():

    train_window = 64



    if int(model_choice) == 1:
        model = lstm.LSTM(device, n_ids+0, n_ids, n_ids, num_layer, 0.1)
        model_name = "LSTM"
    elif int(model_choice) == 2:
        model = gru.GRU(device, n_ids+0, n_ids, n_ids, num_layer, 0.1)
        model_name = "GRU"
    elif int(model_choice) == 3:
        # model = tcn.TCN_batch(200, n_ids, [900] + [900] + [200], 3, 0.1, 0.1)
        # model = tcn.TCN(n_ids, n_ids, [150] * 4, 5, 0.1)
        model = tcn.TCN(n_ids+0, n_ids, [n_ids] * 2, 2, 0.2)
        model_name = "TCN"

    # Train or load
    if int(user_input) == 1:
        if torch.cuda.is_available():
            model.cuda()
        if int(model_choice) == 3:
            train(model, model_name, train_window, False, True)
        else:
            train(model, model_name, train_window, True, False)


    # Reload the model to get the best saved result
    model.to(device)
    model.load_state_dict(torch.load('../models/' + model_name))

    # Evaluate
    if int(model_choice) == 3:
        # evaluate_tcn_batch()
        evaluate_the_model(model, model_name, train_window, test_data, False, True, True)
    else:
        evaluate_the_model(model, model_name, train_window, test_data, True, False, True)




user_input = input("Train(1) or load(2)?")
model_choice = input("LSTM(1) or GRU(2) or TCN(3)?")

run_all_models()

