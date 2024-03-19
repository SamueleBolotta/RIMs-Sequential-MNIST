import torch
from data import MnistData
from networks import MnistModel, LSTM
from tqdm import tqdm
import pickle
import argparse
import numpy as np
import os

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--cuda', type = str2bool, default = True, help = 'use gpu or not')
parser.add_argument('--epochs', type = int, default = 200)

parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--hidden_size', type = int, default = 100)
parser.add_argument('--input_size', type = int, default = 1)
parser.add_argument('--model', type = str, default = 'LSTM')
parser.add_argument('--train', type = str2bool, default = True)
parser.add_argument('--num_units', type = int, default = 6)
parser.add_argument('--rnn_cell', type = str, default = 'LSTM')
parser.add_argument('--key_size_input', type = int, default = 64)
parser.add_argument('--value_size_input', type = int, default =  400)
parser.add_argument('--query_size_input', type = int, default = 64)
parser.add_argument('--num_input_heads', type = int, default = 1)
parser.add_argument('--num_comm_heads', type = int, default = 4)
parser.add_argument('--input_dropout', type = float, default = 0.1)
parser.add_argument('--comm_dropout', type = float, default = 0.1)

parser.add_argument('--key_size_comm', type = int, default = 32)
parser.add_argument('--value_size_comm', type = int, default = 100)
parser.add_argument('--query_size_comm', type = int, default = 32)
parser.add_argument('--k', type = int, default = 4)

parser.add_argument('--size', type = int, default = 14)
parser.add_argument('--loadsaved', type = int, default = 0)
parser.add_argument('--log_dir_lstm', type=str, default='lstm_model_dir', help='Directory for saving/loading LSTM model checkpoints')
parser.add_argument('--log_dir_rim', type=str, default='rim_model_dir', help='Directory for saving/loading RIM model checkpoints')


args = vars(parser.parse_args())

log_dir_lstm = args['log_dir_lstm']
log_dir_rim = args['log_dir_rim']
torch.manual_seed(10)
np.random.seed(10)
torch.cuda.manual_seed(10)

if args['model'] == 'LSTM':
	model = LSTM
else:
	model = MnistModel

def test_model(model, loader, func):
	
	accuracy = 0
	loss = 0
	model.eval()
	with torch.no_grad():
		for i in tqdm(range(loader.val_len())):
			test_x, test_y = func(i)
			test_x = model.to_device(test_x)
			test_y = model.to_device(test_y).long()
			
			probs  = model( test_x)

			preds = torch.argmax(probs, dim=1)
			correct = preds == test_y
			accuracy += correct.sum().item()

	accuracy /= 100.0
	return accuracy

def train_model(model, epochs, data, log_dir):
    acc = []
    lossstats = []
    best_acc_sum = 0.0
    no_improvement_epochs = 0
    start_epoch = 0
    ctr = 0  # Ensure ctr is defined at the beginning
    
    if args['loadsaved'] == 1:
        if args['model'] == 'LSTM':
            acc_stats_file = os.path.join(log_dir['lstm'], 'accstats.pickle')
            loss_stats_file = os.path.join(log_dir['lstm'], 'lossstats.pickle')
            model_checkpoint_file = os.path.join(log_dir['lstm'], 'lstm_best_model.pt')
        elif args['model'] == 'RIM':
            acc_stats_file = os.path.join(log_dir['rim'], 'accstats.pickle')
            loss_stats_file = os.path.join(log_dir['rim'], 'lossstats.pickle')
            model_checkpoint_file = os.path.join(log_dir['rim'], 'rim_best_model.pt')
        else:
            raise ValueError('Unsupported model type')
        
        with open(acc_stats_file, 'rb') as f:
            acc = pickle.load(f)
        with open(loss_stats_file, 'rb') as f:
            losslist = pickle.load(f)
        start_epoch = len(acc) - 1
        best_acc = 0
        for i in acc:
            if i[0] > best_acc:
                best_acc = i[0]
        ctr = len(losslist) - 1
        saved = torch.load(model_checkpoint_file)
        model.load_state_dict(saved['net'])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(start_epoch, epochs):
        print(f'Epoch {epoch + 1}')
        epoch_loss = 0.
        iter_ctr = 0.
        t_accuracy = 0
        norm = 0
        model.train()

        for i in tqdm(range(data.train_len())):
            iter_ctr += 1.
            inp_x, inp_y = data.train_get(i)
            inp_x = model.to_device(inp_x)
            inp_y = model.to_device(inp_y)

            output, l = model(inp_x, inp_y)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            norm += model.grad_norm()
            epoch_loss += l.item()
            preds = torch.argmax(output, dim=1)
            correct = preds == inp_y.long()
            t_accuracy += correct.sum().item()

        # Use args['batch_size'] instead of data.batch_size
        train_accuracy = t_accuracy / (iter_ctr * args['batch_size'])
        print(f'Training Accuracy: {train_accuracy*100:.2f}%')

        # Validation accuracies for each set
        v_accuracy1 = test_model(model, data, data.val_get1)
        v_accuracy2 = test_model(model, data, data.val_get2)
        v_accuracy3 = test_model(model, data, data.val_get3)

        print(f'Validation Set 1 Accuracy: {v_accuracy1}%')
        print(f'Validation Set 2 Accuracy: {v_accuracy2}%')
        print(f'Validation Set 3 Accuracy: {v_accuracy3}%')

        # After calculating validation accuracies
        current_acc_sum = v_accuracy1 + v_accuracy2 + v_accuracy3
        print(f'Current Validation Accuracy Sum: {current_acc_sum}')

        # Determine the directory and filename based on the model type
        if args['model'] == 'LSTM':
            model_dir = 'lstm_model_dir'
            model_filename = 'lstm_best_model.pt'
        elif args['model'] == 'RIM':
            model_dir = 'rim_model_dir'
            model_filename = 'rim_best_model.pt'
        else:
            raise ValueError('Unsupported model type')

        # Ensure the directory exists
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir, model_filename)

        # Now, when saving the model, use the model_path which includes the correct directory and filename
        if current_acc_sum > best_acc_sum:
            best_acc_sum = current_acc_sum
            no_improvement_epochs = 0
            print('New best validation accuracy sum, saving model..')
            state = {'net': model.state_dict(), 'epoch': epoch, 'best_acc_sum': best_acc_sum}
            with open(model_path, 'wb') as f:
                torch.save(state, f)
        else:
            no_improvement_epochs += 1
        if no_improvement_epochs >= 15:
            print(f'No improvement in validation accuracies sum for {no_improvement_epochs} epochs, stopping training.')
            break

data = MnistData(args['batch_size'], (args['size'], args['size']), args['k'])
device = torch.device("cuda" if torch.cuda.is_available() and args['cuda'] else "cpu")

if args['model'] == 'LSTM':
    model = LSTM(args).to(device)  # Assuming LSTM constructor takes args as an argument
else:
    model = MnistModel(args).to(device)  # Assuming MnistModel constructor takes args as an argument


if args['train']:
    train_model(model, args['epochs'], data, {'lstm': args['log_dir_lstm'], 'rim': args['log_dir_rim']})
else:
    if args['model'] == 'LSTM':
        model_checkpoint_file = os.path.join(args['log_dir_lstm'], 'lstm_best_model.pt')
    elif args['model'] == 'RIM':
        model_checkpoint_file = os.path.join(args['log_dir_rim'], 'rim_best_model.pt')
    else:
        raise ValueError('Unsupported model type')

    saved = torch.load(model_checkpoint_file)
    model.load_state_dict(saved['net'])
    # Evaluate on all three validation sets
    validation_functions = [data.val_get1, data.val_get2, data.val_get3]
    validation_accuracies = []


    for func in validation_functions:
        accuracy = test_model(model, data, func)
        validation_accuracies.append(accuracy)

    # Print accuracies for all validation sets
    for i, accuracy in enumerate(validation_accuracies, 1):
        print(f'Validation Set {i} Accuracy: {accuracy:.2f}%')

