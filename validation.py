import torch
from sklearn.metrics import accuracy_score

def val_epoch(model, criterion, dataloader, device, epoch, logger, writer):
    model.eval()
    losses = []
    all_label = []
    all_pred = []

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # get the inputs and labels
            inputs, labels = data['data'].to(device), data['label'].to(device)
            # forward
            outputs = model(inputs)
            if isinstance(outputs, list):
                outputs = outputs[0]
            # compute the loss
            loss = criterion(outputs, labels.squeeze())
            losses.append(loss.item())
            # collect labels & prediction
            prediction = torch.max(outputs, 1)[1]
            all_label.extend(labels.squeeze())
            all_pred.extend(prediction)
    # Compute the average loss & accuracy
    validation_loss = sum(losses)/len(losses)
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    validation_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    # Log
    writer.add_scalars('Loss', {'validation': validation_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'validation': validation_acc}, epoch+1)
    logger.info("Average Validation Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch+1, validation_loss, validation_acc*100))


def val_seq2seq(model, criterion, dataloader, device, epoch, logger, writer):
    model.eval()
    losses = []
    all_trg = []
    all_pred = []

    with torch.no_grad():
        for batch_idx, (imgs, target) in enumerate(dataloader):
            imgs = imgs.to(device)
            target = target.to(device)

            # forward(no teacher forcing)
            outputs = model(imgs, target, 0)

            # target: (batch_size, trg len)
            # outputs: (trg_len, batch_size, output_dim)
            # skip sos
            output_dim = outputs.shape[-1]
            outputs = outputs[1:].view(-1, output_dim)
            target = target.permute(1,0)[1:].reshape(-1)

            # compute the loss
            loss = criterion(outputs, target)
            losses.append(loss.item())

            # compute the accuracy
            prediction = torch.max(outputs, 1)[1]
            score = accuracy_score(target.cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())
            all_trg.extend(target)
            all_pred.extend(prediction)

    # Compute the average loss & accuracy
    validation_loss = sum(losses)/len(losses)
    all_trg = torch.stack(all_trg, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    validation_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    # Log
    writer.add_scalars('Loss', {'validation': validation_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'validation': validation_acc}, epoch+1)
    logger.info("Average Validation Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch+1, validation_loss, validation_acc*100))
