import torch
from IPython.display import display, clear_output
import time
import src.run_utils as ru 
from  src.dataset import get_dataloader_training, get_dataloader_validation, get_dataloader_test, clean_cache

def training(training_params, model, cost, optimizer, device, hparam, paths):
    """
    Main training loop.

    Args:
        training_params (dict):
            contains training parameters set in params.json
        model (nn.Module):
            neural network model
        cost (function):
            cost function for computing the loss
        optimizer:
            optimizer for training
        device (str):
            cpu or gpu device
        hparam (dict):
            ordered dictionary with model hyperparameters
        paths (dict):
            dictionary of directory paths
    """

    start_time = time.time()

    logger = ru.Logger()
    logger.begin_run(training_params.model_name, hparam, paths.tensorboard_path)

    if bool(training_params.caching) == True:
        train_loader = get_dataloader_training(vars(paths), vars(training_params), hparam.batch_size,
                                               device=device, uuid=logger.get_uuid(), caching_mode='write')
        validation_loader = get_dataloader_validation(vars(paths), vars(training_params), hparam.batch_size,
                                                      device=device, uuid=logger.get_uuid(), caching_mode='write')
    else:
        train_loader = get_dataloader_training(vars(paths), vars(training_params), hparam.batch_size,
                                               caching_mode=None)
        validation_loader = get_dataloader_validation(vars(paths), vars(training_params), hparam.batch_size,
                                                      caching_mode=None)

    logger.read_loader(train_loader, validation_loader)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    stop_training = False
    print('Start training')
    for epoch in range(hparam.n_epochs):

        logger.begin_epoch()
        if epoch > 0 and bool(training_params.caching) == True:
            train_loader = get_dataloader_training(vars(paths), vars(training_params), hparam.batch_size,
                                                   device=device, uuid=logger.get_uuid(), caching_mode='load')
            validation_loader = get_dataloader_validation(vars(paths), vars(training_params), hparam.batch_size,
                                                          device=device, uuid=logger.get_uuid(), caching_mode='load')

        if stop_training: 
            break

        model.train()
        for idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()     
            yhat = model(x) 
            loss = cost(yhat, y) 
            loss.backward()
            optimizer.step()

            if training_params.feature_transform == 'linear':
                logger.track_train_loss(loss.mean().detach().cpu())
            else:
                logger.track_train_loss(loss.detach().cpu())

        model.eval() 
        with torch.no_grad():
            for idx, (x, y) in enumerate(validation_loader):
                x, y = x.to(device), y.to(device)

                yhat = model(x)
                loss = cost(yhat, y) 
                logger.track_validation_loss(loss.detach().cpu())

            stop_training = logger.early_stopping(patience=training_params.patience)
            logger.save_model(model, paths.scratch_path)
        logger.end_epoch()

        scheduler.step(logger.get_validation_loss())
        print_progress(logger, start_time)
    logger.end_run(vars(paths), vars(training_params))

    if bool(training_params.save_output):
        logger.save(vars(paths))

    if bool(training_params.caching) == True:
        clean_cache(f'{paths.cache_path}/{logger.get_uuid()}')


def print_progress(logger, start_time):
        
        print(f"    Epoch: {logger.get_progress()}   ",
              f"    Time: {time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))}   ",
              f"    Train loss: {logger.get_training_loss():3.3f}",
              f"    Validation loss: {logger.get_validation_loss():3.3f}",
              f"    ID: {logger.get_uuid()}")

        clear_output(wait=True)

