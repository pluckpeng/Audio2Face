#-*- coding : utf-8-*-
# coding:unicode_escape
from model import A2FNet
from torchsummary import summary
import pandas as pd
import torch.nn.functional as F
import torch
from dataset import BlendshapeDataset, Blendshape37Dataset
import os
import time
from datetime import datetime

keys = ['JawForward', 'JawLeft', 'JawRight', 'JawOpen', 'MouthClose', 'MouthFunnel', 'MouthPucker', 'MouthLeft', 'MouthRight', 'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft', 'MouthFrownRight', 'MouthDimpleLeft',
        'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight', 'MouthRollLower', 'MouthRollUpper', 'MouthShrugLower', 'MouthShrugUpper', 'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft', 'MouthLowerDownRight', 'MouthUpperUpLeft',
        'MouthUpperUpRight', 'BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight', 'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight', 'NoseSneerLeft', 'NoseSneerRight']

data_root = '/home/pzr/code/aiwin/data/audio2face_data_for_train/train'
result_target=os.path.join(data_root,'result_comb')
model=A2FNet(num_blendshapes=len(keys)).cuda()
# summary(model,(1, 64, 32))
# hyper parameters 
batch_size=32
epoch=5000
learning_rate=0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def main():

    
    train_loader = torch.utils.data.DataLoader(
                BlendshapeDataset(feature_file=os.path.join(data_root, 'train_data.npy'),
                                target_file=os.path.join(data_root, 'train_label.npy')),
                batch_size=batch_size, shuffle=False, num_workers=2
                )
    val_loader = torch.utils.data.DataLoader(
                    Blendshape37Dataset(feature_file=os.path.join(data_root, 'eval/x_val.npy'),
                                    target_file=os.path.join(data_root, 'eval/y_val_37.npy')),
                    batch_size=batch_size, shuffle=False, num_workers=2
                    )
    best_epoch=0
    count=0
    best_loss = 10000000
    for j in range(epoch):
        start_time = time.time()
        model.train()
        current_loss = 0.
        train_loss = 0.
        for i, (input, target) in enumerate(train_loader):
            target = target.cuda()
            input_var = torch.autograd.Variable(input.float()).cuda()
            target_var = torch.autograd.Variable(target.float())

            # compute model output
            # audio_z, bs_z, output = model(input_var, target_var)
            # loss = criterion(output, target_var)
            output = model(input_var) # method2: loss change
            loss = F.l1_loss(output, target_var)
            # print(loss)
            current_loss = loss.item()
            train_loss+=current_loss

            # compute gradient and do the backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)

        model.eval()
        eval_loss = 0.
        for input, target in val_loader:
            target = target.cuda()
            input_var = torch.autograd.Variable(input.float(), volatile=True).cuda()
            target_var = torch.autograd.Variable(target.float(), volatile=True)

            # compute output temporal?!!
            # audio_z, bs_z, output = model(input_var, target_var)
            # loss = criterion(output, target_var)
            output = model(input_var) # method2: loss change
            loss = F.l1_loss(output, target_var)

            eval_loss += loss.item()

        eval_loss /= len(val_loader)
        past_time = time.time() - start_time

        print('epoch: {:03} | train_loss: {:.6f} | eval_loss: {:.6f} | {:.4f} sec/epoch'.format(j+1, train_loss, eval_loss, past_time))

        # save best model on val
        is_best = eval_loss < best_loss
        # is_overfit=eval_loss > train_loss
        best_loss = min(eval_loss, best_loss)

        if is_best:
            torch.save(model.state_dict(), os.path.join(result_target, 'model.pth'))
            print('model saved')
        # if is_overfit:
        #     print('best epoch: ',best_epoch)
        #     count+=1
        #     if count>=100:
        #         break
        # else: 
        #     best_epoch=j
        #     count=0

if __name__=='__main__':
    main()