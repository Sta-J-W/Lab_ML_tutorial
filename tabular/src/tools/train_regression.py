import os

# import ML libs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# import internal libs
import model
from tools.plot import plot_curves



def train_regression_model(args, X_train, y_train, X_test, y_test, train_loader):
    """prepare the model the later usage. If the model haven't been trained, we will firstly
    train the model. If the model has been trained, we will certainly load the model.

    Args:
        args (Namespace): set containing all program arguments.
        X_train, y_train, X_test, y_test (tensor.to(deivice)): splitted dataset.
        train_loader (nn.dataloader) used for training model

    Returns:
        return the fitted model or trained model.
    """
    # get the net
    in_dims = X_train.size(-1)
    out_dims = 1
    if args.dataset in ["bike", "commercial_rule1_regression"]:
        net = model.__dict__[args.arch](in_dim=in_dims, hidd_dim=100, out_dim=1)
    else:
        raise Exception(f"Unknown dataset: {args.dataset} for regression.")
    print(net)
    net = net.float().to(args.device)

    # define loss function
    criterion = nn.MSELoss().to(args.device)

    if "model.pt" in os.listdir(args.model_path):
        print("The model has existed in model path '{}'. Load pretrained model.".format(args.model_path))
        net.load_state_dict(torch.load(os.path.join(args.model_path, "model.pt")))

        # evaluate the performance of the model
        net = eval_regression_model(net, X_test, y_test, criterion)

        return net
    else:
        print("The model doen't exist in model path '{}'. Train a model with new settings.".format(args.model_path))
        if args.model_seed != args.seed:
            print("argument model_seed should be as same as argument seed to train a new model with new settings. Terminate program with code -3.")
            exit(-3)

        # define the optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=args.train_lr)
        # set the decay of learning rate
        if args.logspace != 0:
            logspace_lr = np.logspace(np.log10(args.train_lr), np.log10(args.train_lr) - args.logspace, args.epoch)

        # define the train_csv
        learning_csv = os.path.join(args.model_path, "learning.csv")
        # define the res dict
        res_dict = {
            'train-loss': [], 'test-loss': []
        }

        # starts training
        for epoch in range(args.epoch + 1):
            # eval on test set
            net.eval()
            with torch.no_grad():
                # eval on train
                train_outputs = net(X_train).squeeze()
                train_loss = criterion(train_outputs, y_train)

                # eval on test
                test_outputs = net(X_test).squeeze()
                test_loss = criterion(test_outputs, y_test)

            # save the res in dict
            res_dict['train-loss'].append(train_loss.item())
            res_dict['test-loss'].append(test_loss.item())
            # store the res in csv
            pd.DataFrame.from_dict(res_dict).to_csv(learning_csv, index=False)

            # show loss
            if epoch % 10 == 0 or epoch == args.epoch:
                print('On train set - Epoch: {} \t Loss: {:.6f}'
                      .format(epoch, train_loss.item()))
                print('On test set - Epoch: {} \t Loss: {:.6f}'
                      .format(epoch, test_loss.item()))

                if epoch % 100 == 0 or epoch == args.epoch - 1:
                    # draw the curves
                    plot_curves(args.model_path, res_dict)

            if epoch >= args.epoch:
                break

            # set the lr
            if args.logspace != 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = logspace_lr[epoch]

            for step, (batch_X, batch_y) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(args.device), batch_y.to(args.device)
                net.train()
                net.zero_grad()
                outputs = net(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
                loss.backward()
                optimizer.step()

        # save the model
        torch.save(net.cpu().state_dict(), os.path.join(args.model_path, "model.pt"))
        print("The model has been trained and saved in model path '{}'.".format(args.model_path))

        return net


def eval_regression_model(net, X_test, y_test, criterion):
    # evaluate the performance of the model
    net.eval()
    with torch.no_grad():
        outputs = net(X_test).squeeze()
        loss = criterion(outputs, y_test)

    print('On test set - \t MSE Loss: {:.6f}'.format(loss.item()))

    return net

