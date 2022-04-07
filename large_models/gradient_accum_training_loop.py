# batch accumulation parameter
accum_iter = 8 # we want to do model's update only after 64 images being processed

# loop through enumaretad batches
for batch_idx, (inputs, targets) in enumerate(data_loader):

    # extract inputs and labels
    inputs = inputs.to(device)
    targets = targets.to(device)

    # forward pass
    preds = model(inputs)
    loss  = criterion(preds, targets)

    # normalize loss to account for batch accumulation
    loss = loss / accum_iter

    # backward pass
    loss.backward()

    # weights update
    if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(data_loader)):
        optimizer.step()
        optimizer.zero_grad()
