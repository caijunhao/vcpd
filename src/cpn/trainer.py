from collections import defaultdict
import torch
import time
import os


def cpn_trainer(model,
                train_loader,
                test_loader,
                device,
                criterion,
                optimizer,
                scheduler,
                writer,
                path,
                curr_epoch,
                curr_step,
                num_epochs,
                scope='cpn'):
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    loss_list = []
    global_step = curr_step
    # save summary around 20 times for one epoch
    num_save_summary = round(len(train_loader.dataset) / (20 * train_loader.batch_size)) + 1
    num_save_model = round(len(train_loader.dataset) / (3 * train_loader.batch_size)) + 1
    for epoch in range(curr_epoch, num_epochs):
        torch.cuda.empty_cache()
        print('epoch: {}/{}'.format(epoch, num_epochs - 1))
        print('**' * 20)
        model.train()
        for samples in train_loader:
            begin = time.time()
            for k, v in samples.items():
                samples[k] = v.to(device)
            optimizer.zero_grad()
            outputs = model.forward(samples)
            collections = criterion(outputs, samples)
            collections['total_loss'].backward()
            optimizer.step()
            end = time.time()
            loss_list.append(collections['total_loss'].item())
            printfunc(epoch, global_step, end-begin, collections)
            if global_step % num_save_summary == 0:
                [writer.add_scalar(k, v, global_step) for k, v in collections.items() if 'loss' in k]
            if global_step % num_save_model == 0:
                torch.save(model.module.state_dict(), os.path.join(path, scope + '_{}_{}.pth'.format(epoch, global_step)))
                print('successfully saves model at step {}'.format(global_step))
            global_step += 1
        scheduler.step()
        if test_loader is not None:
            with torch.no_grad():
                statistic = defaultdict(float)
                num_validate = 0
                for samples in test_loader:
                    for k, v in samples.items():
                        samples[k] = v.to(device)
                    outputs = model.forward(samples)
                    collections = criterion(outputs, samples)
                    for k, v in collections.items():
                        statistic[k] += v
                    num_validate += 1
                [writer.add_scalar('test_'+k, v/num_validate, global_step) for k, v in statistic.items() if 'loss' in k]
        torch.save(model.module.state_dict(), os.path.join(path, scope + '_{}_{}.pth'.format(epoch, global_step)))
        print('successfully saves model at step {}'.format(global_step))
    writer.close()


def printfunc(epoch, global_step, elapse_time, collections):
    print('='*40)
    print('epoch: {} | global_step: {} | elapse time: {}s'.format(epoch, global_step, elapse_time))
    print(' | '.join([k+': {:.4f}'.format(v) for k, v in collections.items() if 'loss' in k in k]))
