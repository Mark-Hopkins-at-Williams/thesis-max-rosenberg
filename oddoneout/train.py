import torch
import torch.nn as nn
import time
from ozone.util import cudaify
from ozone.oddone import OddOneOutDataset, OddOneOutDataloader


def evaluate(model, loader):
    """Evaluates the trained network on test data."""
    model.eval()
    correct = 0
    total = 0
    for data, response in loader:
        predictions = predict(model, data)
        total += predictions.shape[0]
        for i in range(predictions.shape[0]):
            if response[i].item() == predictions[i].item():                
                correct += 1
    return correct / total


def predict(model, input_tensor):
    with torch.no_grad():
        model.eval()
        input_matrix = cudaify(input_tensor)
        log_probs = model(input_matrix)
        predictions = log_probs.argmax(dim=1)
        return predictions


def train(num_epochs, config, data_loader, multigpu=False):
    
    def maybe_evaluate(prev_best, prev_best_acc):
        best = prev_best
        best_accuracy = prev_best_acc
        test_accuracy = None
        if epoch % 100 == 99:
            test_accuracy = evaluate(model, test_loader)
            print('epoch {} test: {:.2f}'.format(epoch, test_accuracy))
            if test_accuracy > prev_best_acc:
                best_accuracy = test_accuracy
                best = model
                print('saving new model')
                torch.save(best, 'best.model.testing')
        return best, best_accuracy, test_accuracy
    
    def maybe_report_time():
        if False and epoch % 100 == 0 and epoch > 0:
            finish_time = time.process_time()
            time_per_epoch = (finish_time - start_time) / epoch
            print('Average time per epoch: {:.2} sec'.format(time_per_epoch))

    start_time = time.process_time()
    print('epoch: {}'.format(epoch))
    net_factory = config.create_network_factory()
    model = net_factory(data_loader.input_size(), data_loader.output_size())
    if multigpu and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)    
    model = cudaify(model)
    loss_function = nn.NLLLoss()
    optimizer = config.create_optimizer_factory()(model.parameters())
    best_model = None
    best_test_acc = -1.0
    scores = []    
    for epoch in range(num_epochs):
        model.train()
        model.zero_grad()
        loader, test_loader = data_loader.get_loaders(epoch)
        for data, response in loader:
            input_matrix = cudaify(data)
            torch.set_printoptions(profile="full")
            log_probs = model(input_matrix)
            loss = loss_function(log_probs, cudaify(response))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        best_model, best_test_acc, test_acc = maybe_evaluate(best_model, best_test_acc)
        if test_acc is not None:
            scores.append((epoch, test_acc))
        if best_test_acc >= .95:
            break
        maybe_report_time()
    return best_model, scores
