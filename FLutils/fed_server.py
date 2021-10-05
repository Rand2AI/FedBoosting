import numpy as np
from keras import models
import json, base64, cv2
import FLutils
from FLutils.weight_summarizer import WeightSummarizer

class Server:
    def __init__(self, model_fn,
                 weight_summarizer: WeightSummarizer,
                 nb_clients: int = 100):
        self.nb_clients = nb_clients
        self.weight_summarizer = weight_summarizer

        # Initialize the global model's weights
        self.model_fn = model_fn
        self.global_test_metrics_dict = {k: [] for k in model_fn.metrics_names}
        self.global_model_weights = model_fn.get_weights()
        FLutils.get_rid_of_the_models(model_fn)

        self.global_train_losses = None
        self.round_losses = []

        self.client_model_weights = []

        self.client_test_accuracy = []
        self.client_test_loss = []
        self.client_test_density_distribution = []
        self.client_history = []

        # Training parameters used by the clients
        self.client_train_params_dict = {"batch_size": 32,
                                         "val_batch_size": 64,
                                         "epochs": [1,1,1],
                                         "max_label_length":32,
                                         "verbose": 1,
                                         "image_size": [32,100],
                                         "char_file": ""}

    def _create_model_with_updated_weights(self, model=None) -> models.Model:
        if model is None:
            model = self.model_fn
        model.set_weights(self.global_model_weights)
        return model

    def send_model(self, client):
        client.receive_and_init_model(self.model_fn, self.global_model_weights)

    def init_for_new_round(self):
        # Reset the collected weights
        self.client_model_weights.clear()
        # Reset epoch losses
        self.round_losses.clear()
        self.client_test_accuracy.clear()
        self.client_test_loss.clear()
        self.client_test_density_distribution.clear()
        self.client_history.clear()

    def process_client_test_result(self, fl_strategy_metrics):
        client_test_result = []
        if fl_strategy_metrics=="acc":
            np_client_test_accuracy = np.array(self.client_test_accuracy).transpose() # transpose data from dataset-wise to model-wise
            client_history_acc = [his[fl_strategy_metrics][0] for his in self.client_history]
            for ind, result in enumerate(np_client_test_accuracy):
                temp_value = sum(result)
                client_test_result.append(temp_value*client_history_acc[ind]/sum(client_history_acc))
        else:
            np_client_test_loss = np.array(self.client_test_loss).transpose()
            client_history_loss = [1/np.mean(his[fl_strategy_metrics]) for his in self.client_history] # calculate average loss if epoch more than 1.
            for ind, result in enumerate(np_client_test_loss):
                temp_value = sum(result)
                client_test_result.append(1/temp_value * client_history_loss[ind]/sum(client_history_loss))
        for value in client_test_result:
            self.client_test_density_distribution.append(value / sum(client_test_result))

    def summarize_weights(self):
        new_weights = self.weight_summarizer.process(client_weight_list=self.client_model_weights,
                                                     density_distribution=self.client_test_density_distribution)
        self.global_model_weights = new_weights

    def test_global_model(self, testModel, test_data, char_to_id):
        model = self._create_model_with_updated_weights(testModel)
        id_to_char = {v: k for k, v in char_to_id.items()}
        cur = 0
        tol = 0
        for data in test_data:
            temp = json.loads(data.strip('\r\n'))
            label = temp['label'].upper()
            ori_img = temp['img'].encode('utf-8')
            ori_img = cv2.imdecode(np.frombuffer(base64.b64decode(ori_img), np.uint8), 1)
            if len(ori_img.shape) < 3 or ori_img.shape[2] == 1:
                ori_img = cv2.merge([ori_img, ori_img, ori_img])
            img_processed = cv2.resize(ori_img, (int(ori_img.shape[1] * (32 / ori_img.shape[0])), 32))
            try: _ = [char_to_id[j] for j in label]
            except: continue
            if img_processed.shape[1] < 100: continue
            if len(label) < 3: continue
            if len(label) > img_processed.shape[1] // 4: continue
            img_processed = (np.array(img_processed, 'f') - 127.5) / 127.5
            x = np.zeros((1, 32, img_processed.shape[1], 3), dtype=np.float32)
            x[0] = img_processed
            tol+=1
            pred_num = model.predict(x, verbose=0)
            pred_list = FLutils.fast_ctc_decode(pred_num, 0)
            pred_label = u''.join([id_to_char[x] for [x, _, _] in pred_list])
            if pred_label.upper() == label.upper(): cur += 1
        results = [0, cur/tol]
        results_dict = dict(zip(self.model_fn.metrics_names, results))
        for metric_name, value in results_dict.items():
            self.global_test_metrics_dict[metric_name].append(value)
        FLutils.get_rid_of_the_models(model)
        return results_dict

    def evaluate_global_model(self, client_train_dict, test_data: np.ndarray):
        model = self._create_model_with_updated_weights()
        data_generator = FLutils.generator(client_train_dict, test_data, "test")
        batch_size = min(client_train_dict["batch_size"], len(test_data))
        hist = model.evaluate_generator(data_generator,
                                           steps=len(test_data) // batch_size,
                                           verbose=0)
        results_dict = dict(zip(model.metrics_names, hist))
        for metric_name, value in results_dict.items():
            self.global_test_metrics_dict[metric_name].append(value)
        FLutils.get_rid_of_the_models(model)
        return results_dict

    def save_model_weights(self, path: str):
        model = self._create_model_with_updated_weights()
        model.save_weights(str(path), overwrite=True)
        FLutils.get_rid_of_the_models(model)

    def load_model_weights(self, path: str, by_name: bool = False):
        model = self._create_model_with_updated_weights()
        model.load_weights(str(path), by_name=by_name)
        self.global_model_weights = model.get_weights()
        FLutils.get_rid_of_the_models(model)
