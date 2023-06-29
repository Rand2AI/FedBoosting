import os, datetime, time, sys, pickle
import tensorflow as tf
import numpy as np
import FLutils
import phe as paillier
from tqdm import tqdm

Client_id = int(sys.argv[1])

CRNNconfig = FLutils.get_config(os.path.dirname(os.path.realpath(__file__)), 'Estimation')
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in CRNNconfig['DEVICE']["DEVICE_GPUID"]])
GPUconfig = tf.ConfigProto()
GPUconfig.allow_soft_placement = True
# GPUconfig.gpu_options.allow_growth=True # if GPU issue, uncomment this sentence may help
GPUconfig.gpu_options.per_process_gpu_memory_fraction = CRNNconfig['DEVICE']['CLIENT_GPU_FRACTION']

client_train_flag = 0
client_done_flag = 1
client_wait_flag = 2
client_val_flag = 3

last_round_required_decryption = 0

client_train_params = {"epochs": CRNNconfig["ESTIMATION"]["CLIENT_EPOCH"],
                       "batch_size": CRNNconfig["ESTIMATION"]["BATCH_SIZE"],
                       "max_label_length": CRNNconfig["NETWORK"]["MAX_LABEL_LENGTH"],
                       "char_file": CRNNconfig["NETWORK"]["DICTIONARY_PATH"],
                       "image_size": CRNNconfig["ESTIMATION"]["IMAGE_SIZE"],
                       "client_number": CRNNconfig["ESTIMATION"]["CLIENTS"]}

def EncryptWeights(key, plaintext, global_w, desc=""):
    encrypted_gradient_dict = {}
    process_bar_obj = tqdm(plaintext,
                           ncols = 130,
                           desc=f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Encrypt {desc}")
    for layer_idx,layer_weight in enumerate(process_bar_obj):
        encrypted_gradient_dict[layer_idx] = {}
        layer_gradient = (global_w[layer_idx] - layer_weight) / CRNNconfig["ESTIMATION"]["SMALL_SLICE"]
        encrypted_gradient_dict[layer_idx]["shape"] = layer_gradient.shape
        layer_gradient = layer_gradient.reshape(-1)
        encrypted_layer_gradient = []
        for sub_idx, single_gradient in enumerate(layer_gradient):
            integer_value = int(round(single_gradient*1e32))
            assert integer_value<key.max_int, "Please set a larger n_length in paillier.generate_paillier_keypair."
            encrypted_layer_gradient.append(key.encrypt(integer_value))
            process_bar_obj.set_postfix(sub_progress=f"{sub_idx+1}/{len(layer_gradient)} {round((sub_idx+1)/len(layer_gradient)*100, 2)}%")
        encrypted_gradient_dict[layer_idx]["ciphertext"] = encrypted_layer_gradient
    process_bar_obj.close()
    return encrypted_gradient_dict

def DecryptWeights(key, ciphertext, weights, desc=""):
    decrypted_weights = []
    process_bar_obj = tqdm(ciphertext.items(),
                           ncols=130,
                           desc=f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Decrypt {desc}")
    for layer_idx, ciphertext_dict in process_bar_obj:
        layer_gradient_list = []
        for sub_idx, single_gradient in enumerate(ciphertext_dict["ciphertext"]):
            layer_gradient_list.append(float(key.decrypt(single_gradient))/1e32)
            process_bar_obj.set_postfix(sub_progress=f'{sub_idx + 1}/{len(ciphertext_dict["ciphertext"])} {round((sub_idx + 1) / len(ciphertext_dict["ciphertext"]) * 100, 2)}%')
        layer_gradient_array = np.array(layer_gradient_list)
        layer_gradient_array = layer_gradient_array.reshape(ciphertext_dict["shape"])
        decrypted_weights.append(weights[layer_idx]-layer_gradient_array)
    process_bar_obj.close()
    return decrypted_weights

if __name__=="__main__":
    print(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # set HE
    try:
        manager_client_client = FLutils.ManagerClient(CRNNconfig["CLIENT"]["CLIENT_DOMAIN"],
                                                      CRNNconfig["CLIENT"]["CLIENT_PORT"],
                                                      CRNNconfig["CLIENT"]["CLIENT_AUTH_KEY"].encode())
        client_dict_item = manager_client_client.get_dict()
        secret_key = client_dict_item.get("secret_key")
        cloud_key = client_dict_item.get("cloud_key")
        key_gen_client_id = client_dict_item.get("key_generator")
        print(f"Client {key_gen_client_id} is key generator.")
    except:
        cloud_key, secret_key = paillier.generate_paillier_keypair(n_length=CRNNconfig["CLIENT"]["KEY_LENGTH"])
        manager_client_client = FLutils.ManagerServer(CRNNconfig["CLIENT"]["CLIENT_DOMAIN"],
                                                       CRNNconfig["CLIENT"]["CLIENT_PORT"],
                                                       CRNNconfig["CLIENT"]["CLIENT_AUTH_KEY"].encode())
        manager_client_client.run()
        client_dict_item = manager_client_client.get_dict()
        client_dict_item.set(key="secret_key", value=secret_key)
        client_dict_item.set(key="cloud_key", value=cloud_key)
        client_dict_item.set(key="key_generator", value=Client_id)
        print("This client is key generator.")
    # data
    char_to_id = FLutils.gen_character(CRNNconfig["NETWORK"]["DICTIONARY_PATH"])
    data_handler = FLutils.DataHandler(CRNNconfig["ESTIMATION"]["DEBUG"])
    client = FLutils.Client(Client_id)
    data_name = CRNNconfig["ESTIMATION"]["TRAIN_NAME_LIST"][Client_id]
    print(f'Start to process training data: {data_name}')
    data_handler.process_train_data(CRNNconfig["ESTIMATION"]["TRAIN_ROOT_PATH"] + data_name)
    # handle evaluate data
    if CRNNconfig["ESTIMATION"]["FL_STRATEGY"]=="FedBoost":
        if CRNNconfig["ESTIMATION"]["TRN_RATIO"]<1:
            data_handler.split_data(CRNNconfig["ESTIMATION"]["TRN_RATIO"])
        else:
            raise Exception("If use FedBoost, please set TRN_RATIO to less than 1.0")
    data_handler.assign_data_to_clients([client])
    manager_server_client = FLutils.ManagerClient(CRNNconfig["SERVER"]["SERVER_DOMAIN"],
                                                  CRNNconfig["SERVER"]["SERVER_PORT"]+Client_id,
                                                  (CRNNconfig["SERVER"]["SERVER_AUTH_KEY"]+f'client{Client_id}').encode())
    server_dict_item = manager_server_client.get_dict()
    client_number = client_train_params["client_number"]
    graph = tf.get_default_graph()
    with tf.Session(config=GPUconfig) as sess, graph.as_default():
        sess.run(tf.global_variables_initializer())
        net = FLutils.Network(CRNNconfig)
        WEIGHT_PATH = server_dict_item.get("DirName") + "/Client_" + str(Client_id)
        n_clients = server_dict_item.get("client_number")
        while 1:
            flag = server_dict_item.get("flag_client")
            Round = server_dict_item.get("round")
            if flag == client_wait_flag:
                if server_dict_item.get("decryotion_require")==1 and (Round+1) % CRNNconfig["CLIENT"]["ROUNDS_SERVER_GET_MODEL"] == 0:
                    global_encryped_gradient = pickle.loads(server_dict_item.get("global_gradient")) # deserialization
                    global_weights = DecryptWeights(secret_key, global_encryped_gradient, global_weights, "global gradient") # decryption and serialization
                    server_dict_item.set(key="decrypted_global_weights", value=pickle.dumps(global_weights))
                    server_dict_item.set(key="decryotion_require", value=0)
                    last_round_required_decryption = 1
                time.sleep(5)
            elif flag == client_val_flag:
                test_accuracy_list = server_dict_item.get("test_accuracy_on_client")
                test_loss_list = server_dict_item.get("test_loss_on_client")
                for ind in range(client_number): # loop clients' model for validation
                    if CRNNconfig["ESTIMATION"]["DP"]:
                        encryped_gradient = pickle.loads(server_dict_item.get(f"linear_client_{ind}")) # deserialization
                        decryped_weights = DecryptWeights(secret_key, encryped_gradient, global_weights, f"model {ind}") # decryption
                    elif ind==Client_id:
                        decryped_weights = weights
                    else:
                        decryped_weights = pickle.loads(client_dict_item.get(f"client_{ind}"))
                    client.receive_and_init_model(net.deviceModel, decryped_weights)
                    results_dict = client.edge_test(client_train_params)
                    if results_dict["acc"]==0: results_dict["acc"]=1e-8 # avoid divided by 0
                    test_accuracy_list[ind] = results_dict["acc"]
                    test_loss_list[ind] = results_dict["loss"]
                    print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Finished evaluation on client {ind} model: accuracy {test_accuracy_list[ind]}, loss {test_loss_list[ind]}')
                server_dict_item.set(key="test_accuracy_on_client", value=test_accuracy_list)
                server_dict_item.set(key="test_loss_on_client", value=test_loss_list)
                server_dict_item.set(key="flag_client", value=client_done_flag)
                if not os.path.exists(WEIGHT_PATH): os.makedirs(WEIGHT_PATH)
                record_path = WEIGHT_PATH + f'/test_result_round:{str(Round).zfill(3)}.txt'
                if os.path.exists(record_path): os.remove(record_path)
                test_result_record = open(record_path, "w")
                test_result_record.write(f"accuracy: {test_accuracy_list}\n")
                test_result_record.write(f"loss: {test_loss_list}\n")
                test_result_record.write(f"train_his: {hist.history}\n")
                test_result_record.close()
                print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  Final evaluation accuracy {test_accuracy_list}, loss {test_loss_list}')
                if Round + 1 == CRNNconfig["ESTIMATION"]["ROUNDS"]: break
            elif flag == client_train_flag:
                print('-'*100)
                print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] >>>>> Round {Round}')
                if Round==0: # initial weights do not need to decrypt
                    global_weights = pickle.loads(server_dict_item.get("initial_global_weights"))  # deserialization
                else:
                    if not last_round_required_decryption:
                        global_encryped_gradient = pickle.loads(server_dict_item.get("global_gradient"))  # deserialization
                        global_weights = DecryptWeights(secret_key, global_encryped_gradient, global_weights, "gradient") # decryption
                    else:
                        last_round_required_decryption=0
                client.receive_and_init_model(net.deviceModel, global_weights)
                hist = client.edge_train(client_train_params, server_dict_item.get("DirName"), Round, CRNNconfig["CLIENT"]["SAVE_CLIENT_MODEL"])
                weights=client.model.get_weights()
                encryped_gradient_dict = EncryptWeights(cloud_key, weights, global_weights, "gradient") # encryption
                encryped_gradient_dict = pickle.dumps(encryped_gradient_dict) # serialization
                server_dict_item.set(key=f"gradient_client", value=encryped_gradient_dict)
                server_dict_item.set(key="history", value=hist.history)
                server_dict_item.set(key="flag_client", value=client_done_flag)
                if not CRNNconfig["ESTIMATION"]["DP"]:
                    client_dict_item.set(key=f"client_{Client_id}", value=pickle.dumps(weights))
                print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] client_{Client_id} training done...')
    manager_client_client.stop()
    print('_' * 30)