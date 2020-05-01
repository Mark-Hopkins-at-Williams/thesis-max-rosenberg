import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
from puzzle import WordnetPuzzleGenerator, make_puzzle_vector
from puzzle import make_puzzle_matrix, make_puzzle_targets
from networks import DropoutClassifier
from cuda import cudaify


parameters = {
        'batch_size': 1000,
        'num_epochs': 200000,
        'hidden_size': 300,
        'optimizer': 'adam',
        'training_size': 2000
        }

class TrainingParameters:
    
    def __init__(self, params):
        self._params = params
        
    def get_batch_size(self):
        return self._params['batch_size']
    
    def get_num_epochs(self):
        return self._params['num_epochs']
    
    def get_hidden_layer_size(self):
        return self._params['hidden_size']
    
    def get_training_data_size(self):
        return self._params['training_size']
    
    def get_optimizer_factory(self):
        optimizer = self._params['optimizer']
        if optimizer == 'adam':
            return optim.Adam
        elif optimizer == 'sgd':
            return lambda p: optim.SGD(p, lr=0.1)
        else:
            raise Exception('Optimizer "{}" is not supported.'.format(optimizer))

class PuzzleDataset(Dataset):

    def __init__(self, puzzles, vocab):
        #self.puzzles = puzzles
        self.evidence_matrix = make_puzzle_matrix(puzzles, vocab)
        self.response_vector = make_puzzle_targets([label for (_, label) in puzzles])

    def __getitem__(self, index):
        return self.evidence_matrix[index], self.response_vector[index]

    def __len__(self):
        return len(self.evidence_matrix)
    
class Trainer:
    
    def __init__(self, generator, params):
        self.generator = generator
        self.num_training_epochs = params.get_num_epochs()
        self.training_data_size = params.get_training_data_size()
        self.test_data_size = 100
        self.hidden_layer_size = params.get_hidden_layer_size()
        self.num_choices = 5
        self.optimizerFactory = params.get_optimizer_factory()
        self.batch_size = params.get_batch_size()


    def generate_data(self):
        self.data = set(self.generator.batch_generate(self.training_data_size))
        self.test_data = set()
        while len(self.test_data) < self.test_data_size:
            puzzle = self.generator.generate()
            if puzzle not in self.data:
                self.test_data.add(puzzle)
        self.data = list(self.data)
        self.dataset = PuzzleDataset(self.data, self.generator.get_vocab())
        self.dataloader = DataLoader(dataset = self.dataset, 
                                     batch_size = self.batch_size, 
                                     shuffle=True)
        self.test_data = list(self.test_data)

            

    def run(self):
        model = DropoutClassifier(self.generator.get_vocab(),
                                  self.num_choices, 
                                  self.hidden_layer_size)
        model = nn.DataParallel(model)
        cudaify(model)
        return self.batch_train(model)
        

    def batch_train(self, model):
        loss_function = nn.NLLLoss()
        optimizer = self.optimizerFactory(model.parameters())
        best_model = None
        best_test_acc = -1.0
        for epoch in range(self.num_training_epochs):
            if epoch % 100 == 0:
                sys.stdout.write('regenerating data...')
                sys.stdout.flush()
                self.generate_data()
                print('done!')
            model.train()
            model.zero_grad()
            for batch in self.dataloader:
                input_matrix = cudaify(batch[0])
                target = batch[1]
                log_probs = model(input_matrix)
                print("Outside: input size", input_matrix.size(), "output_size", log_probs.size())
                loss = loss_function(log_probs, target)
                loss.backward()
                optimizer.step()
            if epoch % 100 == 0:
                print('epoch {}'.format(epoch))
                test_acc = self.evaluate(model, self.test_data)
                print('test: {:.2f}'.format(test_acc))
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_model = model    
                    print('saving new model')
                    torch.save(best_model, 'best.model')
            if epoch % 100 == 99:
                print('epoch {}'.format(epoch))
                train_acc = self.evaluate(model, self.data[:200])
                print('train: {:.2f}'.format(train_acc))
        return best_model

    def evaluate(self, model, test_d):
        """Evaluates the trained network on test data."""
        model.eval()
        word_to_ix = model.vocab
        with torch.no_grad():
            correct = 0
            for instance, label in test_d:
                input_vec = make_puzzle_vector((instance, label), word_to_ix)
                log_probs = model(input_vec)
                probs = [math.exp(log_prob) for log_prob in log_probs.tolist()[0]]
                ranked_probs = list(zip(probs, range(len(probs))))
                response = max(ranked_probs)[1]
                if response == label:
                    correct += 1            
        return correct/len(test_d)


def run(params, puzzle_gen):
    trainer = Trainer(puzzle_gen, params)
    model = trainer.run() 
    print('training accuracy = {}'.format(trainer.evaluate(model, trainer.data[:1000])))
    print('test accuracy = {}'.format(trainer.evaluate(model, trainer.test_data)))        
    return model
    
model = run(TrainingParameters(parameters), 
            WordnetPuzzleGenerator('dog.n.1'))


