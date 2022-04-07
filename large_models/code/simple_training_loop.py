# loop through batches
for inputs, targets in data_loader:

    inputs = inputs.to(device)
    targets = targets.to(device)

    # forward pass
    preds = model(inputs)
    loss  = criterion(preds, targets)

    # backward pass
    loss.backward()

    # weights update
    optimizer.step()
    optimizer.zero_grad()
