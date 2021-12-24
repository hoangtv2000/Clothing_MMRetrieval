import torch
import time
import os
import json
import numpy as np
from utils.early_stopping import EarlyStopping
from tqdm import tqdm as tqdm
from logger.logger import Timer
from tester import test_retrieval




def train_loop(config, loss_weights, log, trainset, testset, model, optimizer, scheduler, cpkt_fol_name, start_epoch):
    """Function for train loop.
    """
    print('Number of test_queries of testset: ', len(testset.test_queries))
    torch.backends.cudnn.benchmark = True
    early_stopping = EarlyStopping(config, log)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and config.cuda) else "cpu")

    losses_tracking = {}

    # history dict contains loss diary
    history = {'Soft_triplet_loss': [], 'L2_loss_img': [], 'L2_loss_text': [], 'Rot_sym_loss':[], 'Total_loss':[]}

    timer = Timer()

    model = model.to(device)
    l2_loss = torch.nn.MSELoss().to(device)

    for epoch in range(start_epoch, config.num_epochs+1):

        # show/log stats
        log.info(f'Epoch: {epoch} | Elapsed time: {timer.get_time_since_start()}')
        timer.reset()

        # test
        if epoch % 3 == 1:
            print('!!!!!!!! Validating progress !!!!!!!!')
            log.info(f'Validating progress of epoch: {epoch}')

            precision_result, recall_result = [], []
            precision, recall = test_retrieval.test(config, model, testset)

            precision_result += [('test_precision ' + metric_name, metric_value)
                          for metric_name, metric_value in precision]
            recall_result += [('test_recall ' + metric_name, metric_value)
                          for metric_name, metric_value in recall]

            for metric_name, metric_value in precision_result:
                log.info(f'{metric_name} : {round(metric_value, 4)}')

            for metric_name, metric_value in recall_result:
                log.info(f'{metric_name} : {round(metric_value, 4)}')


        # run training for 1 epoch
        model.train()
        trainloader = iter(trainset.get_loader())

        total_loss = 0
        for b_idx, data in enumerate(tqdm(trainloader, desc='Training for epoch ' + str(epoch))):
            # print(total_loss)
            try:
                total_loss += _train_epoch(data, loss_weights, l2_loss, losses_tracking, model, optimizer, device)

                if b_idx % config.log_interval == 0:
                    log.info(f'Training statistics of Batch It: {b_idx}  | in epoch {epoch}')
                    for loss_name in losses_tracking:
                        this_batch_loss = np.mean(losses_tracking[loss_name][-1])
                        log.info(f'Loss "{loss_name}" : {round(this_batch_loss, 4)}')

            except KeyboardInterrupt:
                total_loss = torch.div(total_loss, b_idx)
                scheduler.step(total_loss)

                early_stopping(cpkt_dir=cpkt_fol_name, model=model, optimizer=optimizer,\
                        scheduler=scheduler, epoch=epoch, total_loss=total_loss.item())


        total_loss = torch.div(total_loss, len(trainloader))
        scheduler.step(total_loss)

        early_stopping(cpkt_dir=cpkt_fol_name, model=model, optimizer=optimizer,\
                        scheduler=scheduler, epoch=epoch, total_loss=total_loss.item())

        for loss_name in losses_tracking:
            avg_loss = np.mean(losses_tracking[loss_name][-len(trainloader):])

            if loss_name == 'soft_triplet_loss':
                history['Soft_triplet_loss'].append(avg_loss)
            elif loss_name == 'L2_loss':
                history['L2_loss_img'].append(avg_loss)
            elif loss_name == 'L2_loss_text':
                history['L2_loss_text'].append(avg_loss)
            elif loss_name == 'rot_sym_loss':
                history['Rot_sym_loss'].append(avg_loss)
            elif loss_name == 'total_training_loss':
                history['Total_loss'].append(avg_loss)

            log.info(f'Loss "{loss_name}" : {round(avg_loss, 4)}')

        writer = os.path.join(cpkt_fol_name, 'writer.txt')
        f = open(writer, "w")
        f.write(json.dumps(history))

        if early_stopping.early_stop == True:
            log.info(f'Training session ends at epoch: {epoch}')
            break

    print('Finished training!')



def _train_epoch(data, loss_weights, l2_loss, losses_tracking, model, optimizer, device):
    """Train for 1 epochs.
    """
    img1 = data['source_img_data']
    img1 = torch.autograd.Variable(img1).to(device)

    img2 = data['target_img_data']
    img2 = torch.autograd.Variable(img2).to(device)

    text_query = data['target_caption']

    losses = []
    loss_value, dct_with_representations = model.compute_loss(img1, text_query, img2)

    losses += [('soft_triplet_loss', loss_weights[0], loss_value.to(device))]


    dec_img_loss = l2_loss(dct_with_representations["repr_to_compare_with_source"],
                                   dct_with_representations["img_features"])
    dec_text_loss = l2_loss(dct_with_representations["repr_to_compare_with_mods"],
                                        dct_with_representations["text_features"])

    losses += [("L2_loss", loss_weights[1], dec_img_loss.to(device))]
    losses += [("L2_loss_text", loss_weights[2], dec_text_loss.to(device))]
    losses += [("rot_sym_loss", loss_weights[3], dct_with_representations["rot_sym_loss"].to(device))]

    total_loss = sum([
        loss_weight * loss_value
        for loss_name, loss_weight, loss_value in losses
    ])
    assert not torch.isnan(total_loss)
    losses += [('total_training_loss', None, total_loss.item())]

    # track losses
    for loss_name, loss_weight, loss_value in losses:
        if loss_name not in losses_tracking:
            losses_tracking[loss_name] = []
        losses_tracking[loss_name].append(float(loss_value))

    torch.autograd.set_detect_anomaly(True)

    # apply gradient descent
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss
