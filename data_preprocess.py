import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats
from tensorflow.keras.utils import to_categorical


def make_pamap2(args):
    TIME_STEP = args.time_step
    OVERLAP_STEP = args.overlap_step
    
    list_of_files = ['Protocol/subject101.dat',
                    'Protocol/subject102.dat',
                    'Protocol/subject103.dat',
                    'Protocol/subject104.dat',
                    'Protocol/subject105.dat',
                    'Protocol/subject106.dat',
                    'Protocol/subject107.dat',
                    'Protocol/subject108.dat',
                    'Protocol/subject109.dat' ]

    colNames = ["timestamp", "activityID","heartrate"]
    IMUhand = ['handTemperature', 
            'handAcc16_1', 'handAcc16_2', 'handAcc16_3', 
            'handAcc6_1', 'handAcc6_2', 'handAcc6_3', 
            'handGyro1', 'handGyro2', 'handGyro3', 
            'handMagne1', 'handMagne2', 'handMagne3',
            'handOrientation1', 'handOrientation2', 'handOrientation3', 'handOrientation4']
    IMUchest = ['chestTemperature', 
            'chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3', 
            'chestAcc6_1', 'chestAcc6_2', 'chestAcc6_3', 
            'chestGyro1', 'chestGyro2', 'chestGyro3', 
            'chestMagne1', 'chestMagne2', 'chestMagne3',
            'chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4']
    IMUankle = ['ankleTemperature', 
            'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3', 
            'ankleAcc6_1', 'ankleAcc6_2', 'ankleAcc6_3', 
            'ankleGyro1', 'ankleGyro2', 'ankleGyro3', 
            'ankleMagne1', 'ankleMagne2', 'ankleMagne3',
            'ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4']

    columns = colNames + IMUhand + IMUchest + IMUankle

    label = {1: 'lying',
            2: 'sitting',
            3: 'standing',
            4: 'walking',
            5: 'running',
            6: 'cycling',
            7: 'Nordic_walking',
            12: 'ascending_stairs',
            13: 'descending_stairs',
            16: 'vacuum_cleaning',
            17: 'ironing',
            24: 'rope_jumping' }.values

    dataCollection = pd.DataFrame()
    for file in list_of_files:
        procData = pd.read_table(file, header=None, sep='\s+')
        procData.columns = columns
        procData['subject_id'] = int(file[-5])
        dataCollection = dataCollection.append(procData, ignore_index=True)
    dataCollection.reset_index(drop=True, inplace=True)
    dataCollection = dataCollection.drop(dataCollection[dataCollection.activityID == 0].index)

    columns_to_use = ['subject_id', "activityID", 'handAcc16_1', 'handAcc16_2', 'handAcc16_3','handGyro1', 'handGyro2', 'handGyro3',
                    'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3', 'ankleGyro1', 'ankleGyro2', 'ankleGyro3',
                    'chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3', 'chestGyro1', 'chestGyro2', 'chestGyro3']
    dataCollection = dataCollection.loc[:, columns_to_use]

    dataCollection["LABEL"] = LabelEncoder().fit_transform(dataCollection['activityID'].values.ravel())
    dataCollection = dataCollection.drop(["activityID"], axis="columns")
    dataCollection = dataCollection.interpolate()


    for columns in dataCollection.columns:
        if columns in ["subject_id", "LABEL"]:
            continue
        data = dataCollection[columns]
        data = np.array(data).reshape(-1, 1)
        dataCollection[columns] = StandardScaler().fit_transform(data)

    use_columns = ['handAcc16_1', 'handAcc16_2', 'handAcc16_3','handGyro1', 'handGyro2', 'handGyro3',
                'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3', 'ankleGyro1', 'ankleGyro2', 'ankleGyro3',
                'chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3', 'chestGyro1', 'chestGyro2', 'chestGyro3']

    x_data = np.empty((1, args.time_step, 18))
    y_data = [None]

    for ID in sorted(dataCollection["subject_id"].unique()):
        user = dataCollection[dataCollection["subject_id"] == ID]
        
        for label in sorted(user["LABEL"].unique()):
            data = user[user["LABEL"] == label]        
            data = data.loc[:, use_columns]
            
            for t in range( 0, int(len(data)), OVERLAP_STEP):
                if len(data)-t < TIME_STEP:
                    break
                
                step_data = np.array(data[t:t+TIME_STEP]).reshape(1, TIME_STEP, 18)
                x_data = np.concatenate((x_data, step_data), axis=0)
                y_data.append(label)
    x_data = x_data[1:, :, :]
    y_data = np.array(y_data)[1:]
    
    return x_data, y_data

    
def make_wisdm(args):
    TIME_STEP = args.time_step
    OVERLAP_STEP = args.overlap_step

    df = pd.read_csv('data/wisdm/WISDM_at_v2.0_raw.txt', header=None, names=['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis'])
    df['z-axis'].replace(regex=True, inplace=True, to_replace=r';', value=r'')
    df = df.astype({'z-axis': 'float64'})    
    df.dropna(axis=0, how='any', inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    le = LabelEncoder()
    df['ActivityEncoded'] = le.fit_transform(df['activity'].values.ravel())
    num_classes = le.classes_.size
                
    df = df.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})
    
    segments = []
    labels = []
    id_user = []
    label_length = dict()
    label_data = [[] for _ in range(num_classes)]
        
    for i in range(0, len(df) - TIME_STEP, OVERLAP_STEP):
        xs = df['x-axis'].values[i: i + TIME_STEP]
        ys = df['y-axis'].values[i: i + TIME_STEP]
        zs = df['z-axis'].values[i: i + TIME_STEP]
        label = stats.mode(df['ActivityEncoded'][i: i + TIME_STEP])
        user = stats.mode(df['user-id'][i: i + TIME_STEP])

        if(len(df['ActivityEncoded'][i: i + TIME_STEP]) == label[1][0]):
            extended_label = np.full((TIME_STEP, 1), label[0][0])
            extended_user = np.full((TIME_STEP, 1), user[0][0])
            segment = np.column_stack([xs, ys, zs, extended_label, extended_user])
            label_data[label[0][0]].append(segment)

            if label[0][0] not in label_length:
                label_length[label[0][0]] = 1
            else:
                label_length[label[0][0]] += 1

    extracted_label_data = label_data[0]
    for i in range(1, num_classes):
        extracted_label_data = np.concatenate((extracted_label_data, label_data[i]), axis=0)
    
    segments = extracted_label_data[:, :, :3]
    extended_labels = extracted_label_data[:, :, 3]
    extended_id_user = extracted_label_data[:, :, 4]

    for label, user in zip(extended_labels, extended_id_user):
        if (len(set(label)) != 1) or (len(set(user)) != 1):
            continue            
        labels.append(label[0])
        id_user.append(user[0])
    
    x_data = np.array(segments, dtype=np.float32).reshape(-1, TIME_STEP, 3)
    x_data = x_data / 20
    
    y_data = np.asarray(labels, dtype=np.float32)
    #y_data = to_categorical(y_data, num_classes)
    id_user = np.asarray(id_user, dtype=np.int32)
    return x_data, y_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_step", default=100, type=int) # pamap2: 100, wisdm: 200
    parser.add_argument("--overlap_step", default=50, type=int) # pamap2: 50, wisdm: 100
    parser.add_argument("--dataset", default="pamap2", choices=["pamap2", "wisdm"])
    args = parser.parse_args()
    if not os.path.exists('data/'):
        os.mkdir('data/')
        os.mkdir('data/pamap2')
        os.mkdir('data/wisdm')

    if not os.path.exists(f"data/{args.dataset}/{args.dataset}_{args.overlap_step}_x_data.npy"):
        if args.dataset == 'pamap2':
            x_data, y_data = make_pamap2(args)
        elif args.dataset == 'wisdm':
            if not os.path.exists("data/wisdm/WISDM_at_v2.0_raw.txt"):
                import requests
                print("Downloading wisdm raw dataset...")
                response = requests.get('https://gitlab.venta.lv/s5_linde_o/model-maker/-/raw/15b8657e78449555490abce9066b7eb418bf7f73/project_files/data/oldtxt/WISDM_at_v2.0_raw.txt')
                open("data/wisdm/WISDM_at_v2.0_raw.txt", "wb").write(response.content)
                
            x_data, y_data = make_wisdm(args)

        np.save(f"data/{args.dataset}/{args.dataset}_x_data.npy", x_data)
        np.save(f"data/{args.dataset}/{args.dataset}_y_data.npy", y_data)
    else:
        print(f"{args.dataset} data already exists")
        