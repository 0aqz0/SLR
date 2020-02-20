import torch
from sklearn.metrics import accuracy_score

def test(model, criterion, testloader, device, epoch, logger, writer):
    # Set testing mode
    model.eval()
    losses = []
    all_label = []
    all_pred = []

    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            # get the inputs and labels
            inputs, labels = data['images'].to(device), data['label'].to(device)
            # forward
            outputs = model(inputs)
            # compute the loss
            loss = criterion(outputs, labels.squeeze())
            losses.append(loss.item())
            # collect labels & prediction
            prediction = torch.max(outputs, 1)[1]
            all_label.extend(labels.squeeze())
            all_pred.extend(prediction)
    # Compute the average loss & accuracy
    testing_loss = sum(losses)/len(losses)
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    testing_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    # Log
    writer.add_scalars('Loss', {'test': testing_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'test': testing_acc}, epoch+1)
    logger.info("Average Testing Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch+1, testing_loss, testing_acc*100))
