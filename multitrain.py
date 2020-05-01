import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from puzzle import make_puzzle_matrix, make_puzzle_targets, WordnetPuzzleGenerator
import time
import csv
from wordnet import hypernym_chain, Specificity


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PuzzleDataset(Dataset):

    def __init__(self, puzzles, vocab):
        self.vocab = vocab
        self.evidence_matrix = make_puzzle_matrix(puzzles, vocab)
        self.response_vector = make_puzzle_targets([label for (_, label) in puzzles])
        self.num_choices = 5


    def input_size(self):
        return len(self.vocab) * self.num_choices

    def __getitem__(self, index):
        return self.evidence_matrix[index], self.response_vector[index]

    def __len__(self):
        return len(self.evidence_matrix)   

    @staticmethod
    def generate(generator, num_train):
        data = list(set(generator.batch_generate(num_train)))
        return PuzzleDataset(data, generator.get_vocab())

    @staticmethod
    def compile_puzzle(generator, puzzle):
        return make_puzzle_matrix([(puzzle, -1)], generator.get_vocab())
   
    @staticmethod
    def create_data_loader(dataset, batch_size):
        dataloader = DataLoader(dataset = dataset, 
                                     batch_size = batch_size, 
                                     shuffle=True)
        return dataloader


class PhraseEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(PhraseEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(vocab_size, hidden_size)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
 
    def forward(self, input_vec):
        output = self.linear1(input_vec).clamp(min=0)
        output = self.dropout(output)
        output = self.linear2(output).clamp(min=0)
        output = self.dropout(output)
        output = self.linear3(output).clamp(min=0)
        output = self.linear4(output)
        return output
        


class TiedClassifier(nn.Module): 

    def __init__(self, input_size, num_labels, hidden_size):
        super(TiedClassifier, self).__init__()
        self.input_size = input_size
        self.vocab_size = input_size // 5
        self.hidden_size = hidden_size
        self.word_encoder = PhraseEncoder(self.vocab_size, hidden_size)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.linear3 = nn.Linear(5*hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.final_layer = nn.Linear(hidden_size, num_labels)

    def forward(self, input_vec):
        t = input_vec
        output1 = self.word_encoder(t[:,0*self.vocab_size:1*self.vocab_size])
        output2 = self.word_encoder(t[:,1*self.vocab_size:2*self.vocab_size])
        output3 = self.word_encoder(t[:,2*self.vocab_size:3*self.vocab_size])
        output4 = self.word_encoder(t[:,3*self.vocab_size:4*self.vocab_size])
        output5 = self.word_encoder(t[:,4*self.vocab_size:5*self.vocab_size])
        nextout = torch.cat([output1, output2, output3, output4, output5], dim=1) 
        nextout = self.linear3(nextout).clamp(min=0)
        nextout = self.dropout(nextout)
        nextout = self.linear4(nextout).clamp(min=0)
        nextout = self.dropout(nextout)
        nextout = self.linear5(nextout).clamp(min=0)
        nextout = self.dropout(nextout)
        nextout = self.final_layer(nextout)
        return F.log_softmax(nextout, dim=1)
     

def evaluate(model, loader):
    """Evaluates the trained network on test data."""
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, response in loader:
            input_matrix = data.to(device)
            log_probs = model(input_matrix)
            predictions = log_probs.argmax(dim=1)
            total += predictions.shape[0]
            for i in range(predictions.shape[0]):
                if response[i].item() == predictions[i].item():
                    correct += 1
    return correct / total

def predict(model, puzzle, generator):
    compiled = PuzzleDataset.compile_puzzle(generator, puzzle)
    model.eval()
    input_matrix = compiled.to(device)
    model = model.to(device)
    log_probs = model(input_matrix)
    predictions = log_probs.argmax(dim=1)
    return predictions    

def train(root_synset_lists, num_epochs, hidden_size, 
          num_puzzles_to_generate, batch_size, multigpu = False):
    def maybe_regenerate(puzzle_generator, epoch, prev_loader, prev_test_loader):
        if epoch % 100 == 0:
            dataset = PuzzleDataset.generate(puzzle_generator, num_puzzles_to_generate)
            loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle=True)
            test_dataset = PuzzleDataset.generate(puzzle_generator, 100)
            test_loader = DataLoader(dataset = test_dataset, batch_size = 100, shuffle=False)
            return loader, test_loader
        else:
            return prev_loader, prev_test_loader
    
    def maybe_evaluate(model, epoch, current_root, prev_best, prev_best_acc):
        best_model = prev_best
        best_test_acc = prev_best_acc
        if epoch % 100 == 99:
            test_acc = evaluate(model, test_loader)
            print('epoch {} test ({}): {:.2f}'.format(epoch, current_root, test_acc))
            if test_acc > prev_best_acc:
                best_test_acc = test_acc
                best_model = model
                print('saving new model')
                torch.save(best_model.state_dict, 'best.model')
        return best_model, best_test_acc
    
    def maybe_report_time():
        if False and epoch % 100 == 0 and epoch > 0:
            finish_time = time.clock()
            time_per_epoch = (finish_time - start_time) / epoch
            print('Average time per epoch: {:.2} sec'.format(time_per_epoch))
            
    def get_final_root_synset(initial_synset):
        chain = hypernym_chain(initial_root_synset)
        spec = Specificity()
        for w in chain[::-1]:
            if spec.evaluate(w) < 700:
                return w.name()
            
    def write_to_csv(word, num_epochs, best_accuracy, success):
        with open('multrain_data.csv', mode='a') as csv_file:
                    multitrain_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                
                    multitrain_writer.writerow([word, num_epochs, best_accuracy, success])

    # modified this part to loop through root synset lists
    for root_synset_list in root_synset_lists:
        initial_root_synset = root_synset_list[0]
        
        if len(root_synset_list) != 2:
            final_root_synset = get_final_root_synset(initial_root_synset)
        else:
            final_root_synset = root_synset_list[1]
            
        start_time = time.clock()
        puzzle_generator = WordnetPuzzleGenerator(final_root_synset)
        input_size = 5 * len(puzzle_generator.get_vocab())
        output_size = 5
        model = TiedClassifier(input_size, output_size, hidden_size)
        if multigpu and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            #dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)
        
        model.to(device)
        loader = None
        test_loader = None
        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters())
        best_model = None
        best_test_acc = -1.0
        puzzle_generator.reset_root(initial_root_synset)
        for epoch in range(num_epochs):
            model.train()
            model.zero_grad()
            loader, test_loader = maybe_regenerate(puzzle_generator, epoch, 
                                                   loader, test_loader)
            for data, response in loader:
                input_matrix = data.to(device)
                log_probs = model(input_matrix)
                loss = loss_function(log_probs, response)
                loss.backward()
                optimizer.step()
            best_model, best_test_acc = maybe_evaluate(model, epoch, initial_root_synset,
                                                       best_model, best_test_acc)
            
            if best_test_acc > .8:
                current_root = initial_root_synset
                initial_root_synset = hypernym_chain(initial_root_synset)[1].name()
                puzzle_generator.reset_root(initial_root_synset)
                print("Successful training of {}! Moving on to {}.".format(current_root, initial_root_synset))

                write_to_csv(current_root, epoch, best_test_acc, True)
                                    
                print('saving new model')
                torch.save(model.state_dict(), 'best.model')
                best_test_acc = -1.0
                loader, test_loader = maybe_regenerate(puzzle_generator, 100, 
                                                       loader, test_loader)
            
            maybe_report_time()
            
            if epoch == num_epochs - 1:
                write_to_csv(current_root, epoch, best_test_acc, False)
                print("took more than ", num_epochs, " epochs on word ", current_root, "moving on to the next set")
                break
            
    return best_model

if __name__ == '__main__':
    roots_to_test = [
                     ['alcohol.n.01'],
                     ['crime.n.01'],
                     ['dog.n.01'],
                     ['fluid.n.01'],
                     ['vehicle.n.01'],
                     ['creation.n.02']
                     ]
    train(roots_to_test,
          num_epochs= 10000, 
          hidden_size=500,
          num_puzzles_to_generate=2000,
          batch_size=256,
          multigpu=False)

