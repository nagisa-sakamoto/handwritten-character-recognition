import torch
from torch.nn import CTCLoss
import torch.nn.functional as F
import os
from tqdm import tqdm

from data.image_dataset import ImageDataset
from data.image_data_loader import ImageDataLoader


def calculate_loss(input_data,
                    reference,
                    input_lengths,
                    reference_lengths,
                    model,
                    criterion,
                    cuda):
    if cuda:
        input_data, reference = input_data.cuda(), reference.cuda()
    output, output_lengths = model(input_data, input_lengths)
    output = F.log_softmax(output, dim=-1)
    output = output.transpose(0, 1) #frameN x batchN x referenceDim
    loss = criterion(output.float(), reference, output_lengths, reference_lengths)
    return loss


def check_loss(loss, loss_value):
    loss_valid = True
    if loss_value == float("inf") or loss_value == float("-inf"):
        loss_valid = False
        print("WARNING: received an inf loss")
    elif torch.isnan(loss).sum() > 0:
        loss_valid = False
        print('WARNING: received a nan loss, setting loss value to 0')
    elif loss_value < 0:
        loss_valid = False
        print("WARNING: received a negative loss")
    return loss_valid


def train(model, 
            train_data,
            targets,
            output_dir,
            dev_data=None,
            batch_size=16,
            n_epochs=30,
            learning_rate=0.0001,
            learning_rate_factor=0.5,
            cuda=False):
    criterion = CTCLoss(blank=targets.index('_'), zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=learning_rate_factor, patience=1)
    train_loader = ImageDataLoader(ImageDataset(train_data, targets), batch_size=batch_size)
    if dev_data:
        dev_loader = ImageDataLoader(ImageDataset(dev_data, targets), batch_size=batch_size)
    else:
        dev_loader = None
    model.train()
    best_loss = 100
    for epoch in range(n_epochs):
        train_loss = 0
        dev_loss = 0
        print(f'Train Epoch: {epoch+1} \tLearning Rate: {optimizer.param_groups[0]["lr"]}')
        for batch_idx, (input_data, reference, input_lengths, reference_lengths) in tqdm(enumerate(train_loader)): 
            loss = calculate_loss(input_data, reference, input_lengths, reference_lengths, model, criterion, cuda)
            if cuda:
                loss = loss.cuda()
            loss_value = loss.item()
            if check_loss(loss, loss_value):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss_value
            #print(loss_value)
        train_loss /= len(train_loader)
        print('Train Loss: {:.6f}'.format(train_loss))
        output_file = os.path.join(output_dir, f'{model.__class__.__name__}_checkpoint_epoch{epoch+1}.pth')
        print(f'Seve checkpoint: {output_file}')
        torch.save(model.state_dict(), output_file)

        if dev_loader:
            print(f'Validation Epoch: {epoch+1}')
            for batch_idx, (input_data, reference, input_lengths, reference_lengths) in tqdm(enumerate(dev_loader)): 
                loss = calculate_loss(input_data, reference, input_lengths, reference_lengths, model, criterion, cuda)
                dev_loss += loss.item()
            dev_loss /= len(dev_loader)
            scheduler.step(dev_loss)
            print('Validation Loss: {:.6f}'.format(dev_loss))
            if dev_loss < best_loss:
                output_best_file = os.path.join(output_dir, f'{model.__class__.__name__}_checkpoint_best.pth')
                print(f'Seve best checkpoint: {output_best_file}')
                torch.save(model.state_dict(), output_best_file)
                best_loss = dev_loss



