import numpy as np
import os
import random

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from models.make_model import create_encoder, create_classifier, add_contrastive_head, joint_supcon
from utils import HARDataset, SupervisedContrastiveLoss, model_evaluation

import argparse

def main(args):
	x_train, y_train, x_val, y_val, x_test, y_test = HARDataset(args.dataset)
	input_shape = x_train.shape[1:]
	encoder = create_encoder(args.model, input_shape=input_shape)
	reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=4, min_lr=0.0001)
	filepath = f"saved/{args.train_type}_{args.dataset}_{args.model}_{str(args.seed)}_lr{str(args.lr)}_alp{str(args.alpha)}.h5"

	if args.train_type == 'supervised':
		model = create_classifier(args.model, encoder, input_shape, num_class=len(np.unique(y_train)))
		model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), metrics=["accuracy"])

	elif args.train_type == 'supcon':
		con_filepath = f"saved/con_{args.dataset}_{str(args.seed)}_lr{str(args.lr)}_alp{str(args.alpha)}_{args.model}.h5"
		contrast = add_contrastive_head(args.model, encoder, input_shape, args.contrastive_features)
		contrast.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr*(1-args.alpha)), loss=SupervisedContrastiveLoss())
		con_history = contrast.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=args.verbose, validation_data=(x_val, y_val), callbacks=[ModelCheckpoint(con_filepath, verbose = args.verbose, monitor='val_loss', mode="min", save_best_only=True, save_weights_only = True), reduce_lr])
		# print(con_history)
		contrast.load_weights(con_filepath)
		print("[Finished Prtraining]")

		model = create_classifier(args.model, encoder, input_shape, num_class=len(np.unique(y_train)), trainable=False)
		model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr*args.alpha), metrics=["accuracy"])

	elif args.train_type == 'joint_supcon':
		model = joint_supcon(args.model, encoder, input_shape, num_class=len(np.unique(y_train)), contrastive_shape=args.contrastive_features)
#		cls = create_classifier(args.model, encoder, input_shape, num_class=len(np.unique(y_train)), trainable=True)
#		con = add_contrastive_head(args.model, encoder, input_shape, args.contrastive_features)
#		model = tf.keras.Model(inputs=tf.keras.Input(shape=input_shape), outputs=[cls.outputs, con.outputs])
		model.compile(loss=["sparse_categorical_crossentropy", SupervisedContrastiveLoss()], loss_weights=[args.alpha, 1-args.alpha], optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), metrics=["accuracy", None])
	
	checkpoint = ModelCheckpoint(filepath, verbose = args.verbose, monitor='val_loss', mode="min", save_best_only=True, save_weights_only = True)
	#earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=30, mode='min', restore_best_weights=True)
	print(model.summary())
	history = model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=args.verbose, validation_data=(x_val, y_val), callbacks=[checkpoint, reduce_lr])
	print(history)
	model.load_weights(filepath)
	
	model_evaluation(model, history, x_test, y_test, y_contrastive=(y_test if args.train_type == 'joint_supcon' else None))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--gpu", default="0", help="gpu id")
	parser.add_argument("--seed", default=333, type=int, help="random seed")
	parser.add_argument("--dataset", default="wisdm", choices=['pamap2', 'wisdm'], help="dataset name")
	parser.add_argument("--model", required=True, choices=['deepconvlstm', 'self_attention', 'multibranch'], help="model name")
	parser.add_argument("--train_type", default="joint_supcon", choices=['supervised', 'supcon', 'joint_supcon'])
	parser.add_argument("--epochs", type=int, default=100, help="epochs")
	parser.add_argument("--batch_size", type=int, default=64, help="batch size")
	parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
	parser.add_argument("--contrastive_features", type=int, default=64, help="contrastive features")
	parser.add_argument("--alpha", type=float, default=0.0, help="alpha")
	parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2])
	args = parser.parse_args()

	if args.gpu != '0':
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
		os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
	os.environ['PYTHONHASHSEED'] = str(args.seed)
	os.environ['TF_DETERMINISTIC_OPS'] = '1'
	tf.random.set_seed(args.seed)
	np.random.seed(args.seed)
	if not os.path.exists('saved/'):
		os.mkdir('saved/')
	main(args)