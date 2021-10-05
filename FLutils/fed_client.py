import os
import FLutils
from keras.callbacks import ModelCheckpoint, EarlyStopping

class Client:
    def __init__(self, client_id: int):
        self.client_id = client_id
        self.model = None
        self.train_data = None
        self.evaluate_data = None

    def _init_model(self, model_fn, model_weights):
        model_fn.set_weights(model_weights)
        self.model = model_fn

    def receive_data(self, train_data, evaluate_data):
        self.train_data = train_data
        self.evaluate_data = evaluate_data

    def receive_and_init_model(self, model_fn, model_weights):
        self._init_model(model_fn, model_weights)

    def edge_train(self, client_train_dict: dict, RootPath: str, Round: int, SAVE_CLIENT_MODEL=False):
        if self.model is None:
            raise ValueError("Model is not created for client: {0}".format(self.client_id))
        callbackList = []
        if SAVE_CLIENT_MODEL:
            WEIGHT_PATH = RootPath+"/Client_"+str(self.client_id)
            if not os.path.exists(WEIGHT_PATH):
                os.makedirs(WEIGHT_PATH)
            checkpoint = ModelCheckpoint(WEIGHT_PATH + f'/Weights-round:{str(Round).zfill(3)}'+r'-epoch:{epoch:03d}-acc:{acc:.4f}-loss:{loss:.4f}.h5',
                                         monitor='acc',
                                         save_best_only=False,
                                         save_weights_only=True,
                                         verbose=0)
            callbackList.append(checkpoint)
        # earlystop = EarlyStopping(monitor='val_acc', patience=200, restore_best_weights=True)
        trn_generator = FLutils.generator(client_train_dict, self.train_data)
        # val_generator = FLutils.generator(client_train_dict, self.evaluate_data)
        hist = self.model.fit_generator(trn_generator,
                                        steps_per_epoch=len(self.train_data) // client_train_dict["batch_size"],
                                        epochs=client_train_dict["epochs"][self.client_id],
                                        callbacks = callbackList,
                                        verbose=1)
                                        # validation_data=val_generator,
                                        # validation_steps=len(self.evaluate_data) // client_train_dict["batch_size"],
        FLutils.get_rid_of_the_models(self.model)
        return hist

    def edge_test(self, client_train_dict: dict):
        if self.model is None:
            raise ValueError("Model is not created for client: {0}".format(self.client_id))
        evaluate_generator = FLutils.generator(client_train_dict, self.evaluate_data)
        results = self.model.evaluate_generator(evaluate_generator,
                                                steps=len(self.evaluate_data) // client_train_dict["val_batch_size"],
                                                verbose=0)
        results_dict = dict(zip(self.model.metrics_names, results))
        FLutils.get_rid_of_the_models(self.model)
        return results_dict
