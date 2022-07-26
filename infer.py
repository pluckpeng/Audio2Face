import torch
from dataset import AudioDataset
import numpy as np
import os
from model import A2FNet
import pandas as pd
keys = ['JawForward', 'JawLeft', 'JawRight', 'JawOpen', 'MouthClose', 'MouthFunnel', 'MouthPucker', 'MouthLeft', 'MouthRight', 'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft', 'MouthFrownRight', 'MouthDimpleLeft',
        'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight', 'MouthRollLower', 'MouthRollUpper', 'MouthShrugLower', 'MouthShrugUpper', 'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft', 'MouthLowerDownRight', 'MouthUpperUpLeft',
        'MouthUpperUpRight', 'BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight', 'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight', 'NoseSneerLeft', 'NoseSneerRight']

model_path='/home/pzr/code/aiwin/data/audio2face_data_for_train/train/result_comb/model.pth'
data_path='/home/pzr/code/aiwin/data/audio2face_data_for_train/LPC'
result_path='/home/pzr/code/aiwin/data/audio2face_data_for_train/result_comb'

model=A2FNet(num_blendshapes=37)

def main():
    file_names=os.listdir(data_path)
    for file_name in file_names:
        feature=np.load(os.path.join(data_path,file_name))
        base=np.squeeze(feature[0])
        base=base.T
        base=np.expand_dims(base, axis=0)
        base=np.expand_dims(base, axis=0)
        for f in feature[1:]:
            f=np.squeeze(f)
            f=f.T
            f=np.expand_dims(f, axis=0)
            f=np.expand_dims(f, axis=0)
            base=np.concatenate((base,f),axis=0)
        # print(base.shape)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        result=model(torch.tensor(base).to(torch.float32))
        result=pd.DataFrame(result.detach().numpy(), columns=keys)
        result.to_csv(os.path.join(result_path, file_name.split('.')[0]+'.csv'))
        # np.savetxt(os.path.join(result_path, file_name.split('.')[0]+'.csv'),result.detach().numpy(),delimiter=',')
        print(file_name+' done')

if __name__=='__main__':
    main()