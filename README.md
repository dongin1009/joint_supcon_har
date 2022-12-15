# Joint Contrastive and Supervised Learning in Human Activity Recognition

## How to use
1. Clone the repository
```bash
$ git clone https://github.com/dongin1009/joint_supcon_har
$ cd joint_supcon_har
```

2. Install tensorflow 2.9, numpy 1.23.4, sklearn

3. Get [PAMAP2 dataset](https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring) and [WISDM dataset](https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_at_latest.tar.gz) (Actitracker version) and put them in the `data` directory.

4. Preprocess the data
```python
$ python data_preprocess.py --dataset pamap2 --time_step 100 --overlap_step 50
$ python data_preprocess.py --dataset wisdm --time_step 200 --overlap_step 100
```

5. Train the model
```python
$ python train.py --dataset pamap2 --model deepconvlstm --train_type joint_supcon --epochs 100 --lr 0.005 --alpha 0.2 --batch_size 64
# or other options
$ python train.py --dataset {pamap2, wisdm} --model {deepconvlstm, self_attention, multibranch} --train_type {supervised, supcon, joint_supcon} --epochs EPOCH --lr LR --alpha ALPHA --batch_size BATCH_SIZE
```


## Results
These results are averaged results on 5 random seeds.

### PAMAP2 dataset

|PAMAP2           |  supervised   |||    supcon     |||  joint supcon   |||
| --------------- | --------------- | :---------------: | :---------------: | --------------- | :---------------: | :---------------: | --------------- | :---------------: | :---------------: |
|    **Model**    |**lr \|\| alpha**|    **acc**      |   **f1-score**  |**lr \|\| alpha**|    **acc**      |   **f1-score**  |**lr \|\| alpha**|    **acc**      |   **f1-score**  |
|  *DeepConvLSTM* |  0.005 \|\| -  |  0.9116  |  0.9095  |  0.005 \|\| 0.2  |***0.9202***|  0.9175  |  0.005 \|\| 0.2  |  0.9180  |***0.9185***|
|*Self-Attention*|  0.001 \|\| -  |  0.9565  |  0.9545  |  0.001 \|\| 0.2  |  0.9612  |  0.9602  |  0.001 \|\| 0.2  |***0.9636***|***0.9616***|
| *MultiBranch* |  0.001 \|\| -  |  0.9722  |  0.9712  |  0.003 \|\| 0.8  |  0.9751  |***0.9733***|     0.001 \|\| 0.5  |***0.9780***|  0.9711  |


### WISDM dataset
|WISDM            |  supervised   |||    supcon     |||  joint supcon   |||
| --------------- | --------------- | :---------------: | :---------------: | --------------- | :---------------: | :---------------: | --------------- | :---------------: | :---------------: |
|    **Model**    |**lr \|\| alpha**|    **acc**      |   **f1-score**  |**lr \|\| alpha**|    **acc**      |   **f1-score**  |**lr \|\| alpha**|    **acc**      |   **f1-score**  |
|  *DeepConvLSTM* |  0.003 \|\| -  |  ***0.9363***  |  ***0.9135***  |  0.003 \|\| 0.5  |  0.9132  |  0.8805  |  0.003 \|\| 0.5  |  0.9362  |  0.9111  |
|*Self-Attention*|  0.001 \|\| -  |  0.9097  |  0.8740 |  0.001 \|\| 0.5  |  0.9133  |  0.8837  |  0.001 \|\| 0.5  |***0.9246***|***0.8978***|
| *MultiBranch* |  0.003 \|\| -  |  0.9565  |  0.9421  |  0.005 \|\| 0.2  |  0.9705  |  0.9608  |     0.003 \|\| 0.2  |***0.9723***|***0.9644***|

## Citation
TBD

## Contact
If you have any questions, please contact us via email: [dongin1009@gmail.com](mailto:dongin1009@gmail.com)