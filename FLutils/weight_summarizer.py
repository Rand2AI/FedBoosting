from tqdm import tqdm
import datetime

class WeightSummarizer:
    def __init__(self):
        pass

    def process(self,
                client_gradient_dict,
                density_distribution = None,
                slice_num = 1000):
        raise NotImplementedError()
    def linear(self, client_gradient_dict, slice_num = 1000):
        raise NotImplementedError()

class FedAvg(WeightSummarizer):
    def __init__(self):
        super().__init__()

    def process(self,
                client_gradient_dict,
                density_distribution = None,
                slice_num = 1000):
        nb_clients = len(client_gradient_dict)
        gradient_average = {}
        process_bar_obj = tqdm(range(len(client_gradient_dict[0])),
                               ncols=120,
                               desc=f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]  Generate global model")
        for layer_index in process_bar_obj:
            gradient_average[layer_index] = {}
            gradient_average[layer_index]["shape"] = client_gradient_dict[0][layer_index]["shape"]
            w = [0 for _ in range(len(client_gradient_dict[0][layer_index]["ciphertext"]))]
            for client_index in range(nb_clients):
                client_gradient_mtx = client_gradient_dict[client_index][layer_index]["ciphertext"]
                w = list(map(lambda x: x[0] + int(round(slice_num/nb_clients))*x[1], zip(w, client_gradient_mtx))) # element-wise addition
            gradient_average[layer_index]["ciphertext"] = w
        return gradient_average

    def linear(self, client_gradient_list, slice_num = 1000):
        pass

class FedBoost(WeightSummarizer):
    def __init__(self):
        super().__init__()

    def process(self,
                client_gradient_dict,
                density_distribution = None,
                slice_num = 1000):
        nb_clients = len(client_gradient_dict)
        gradient_boost = {}
        process_bar_obj = tqdm(range(len(client_gradient_dict[0])),
                               ncols=120,
                               desc=f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generate global model")
        for layer_index in process_bar_obj:
            gradient_boost[layer_index] = {}
            gradient_boost[layer_index]["shape"] = client_gradient_dict[0][layer_index]["shape"]
            w = [0 for _ in range(len(client_gradient_dict[0][layer_index]["ciphertext"]))]
            for client_index in range(nb_clients):
                client_gradient_mtx = client_gradient_dict[client_index][layer_index]["ciphertext"]
                w = list(map(lambda x: x[0] + int(round(density_distribution[client_index]*slice_num)) * x[1], zip(w, client_gradient_mtx))) # element-wise addition
            gradient_boost[layer_index]["ciphertext"] = w
        return gradient_boost

    def linear(self, client_gradient_dict, slice_num = 1000):
        linear_client_list = []
        nb_clients = len(client_gradient_dict)
        process_bar_obj = tqdm(range(nb_clients),
                               ncols=120,
                               desc=f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Encrypt local model")
        for major_client_index in process_bar_obj:
            gradient = {}
            for layer_index in range(len(client_gradient_dict[major_client_index])):
                gradient[layer_index] = {}
                gradient[layer_index]["shape"] = client_gradient_dict[major_client_index][layer_index]["shape"]
                process_bar_obj.set_postfix(layer=f"{layer_index + 1}/{len(client_gradient_dict[major_client_index])} {round((layer_index + 1) / len(client_gradient_dict[major_client_index]) * 100, 2)}%")
                w = [0 for _ in range(len(client_gradient_dict[0][layer_index]["ciphertext"]))]
                for minor_client_index in range(nb_clients):
                    client_gradient_mtx = client_gradient_dict[minor_client_index][layer_index]["ciphertext"]
                    if major_client_index==minor_client_index:
                        w = list(map(lambda x: x[0] + int(round(0.9*slice_num)) * x[1], zip(w, client_gradient_mtx)))  # element-wise addition
                    else:
                        w = list(map(lambda x: x[0] + int(round(0.1/(nb_clients-1)*slice_num)) * x[1], zip(w, client_gradient_mtx)))  # element-wise addition
                gradient[layer_index]["ciphertext"] = w
            linear_client_list.append(gradient)
        process_bar_obj.close()
        return linear_client_list

if __name__=="__main__":
    import FLutils, os

    weights_list = []
    CRNNconfig = FLutils.get_config(os.path.dirname(os.path.realpath(__file__)), 'Inference')
    net = FLutils.Network(CRNNconfig)
    net.deviceModel.load_weights("/home/hans/WorkSpace/Data/Text/_@Models/FL_CRNN/FL_Model/FedBoost/20200204-114127-1E800B_FedBoost/Client_0/Weights-round:093-epoch:001-acc:0.9471-loss:0.3086.h5")
    weights_list.append(net.deviceModel.get_weights())
    net.deviceModel.load_weights("/home/hans/WorkSpace/Data/Text/_@Models/FL_CRNN/FL_Model/FedBoost/20200204-114127-1E800B_FedBoost/Client_1/Weights-round:093-epoch:001-acc:0.8944-loss:0.6310.h5")
    weights_list.append(net.deviceModel.get_weights())
    weight_summarizer = FedBoost()
    new_weights = weight_summarizer.process(client_weight_list=[net.deviceModel.get_weights(),net.deviceModel.get_weights(),net.deviceModel.get_weights()])
    net.deviceModel.set_weights(new_weights)
    net.deviceModel.save_weights("/home/hans/WorkSpace/Data/Text/_@Models/FL_CRNN/FL_Model/FedBoost/20200204-114127-1E800B_FedBoost/EL.h5", overwrite=True)