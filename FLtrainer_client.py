import os, datetime, time, sys
import tensorflow as tf
import FLutils

Client_id = int(sys.argv[1])

CRNNconfig = FLutils.get_config(os.path.dirname(os.path.realpath(__file__)), 'Estimation')
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in CRNNconfig['DEVICE']["DEVICE_GPUID"]])
GPUconfig = tf.ConfigProto()
GPUconfig.allow_soft_placement = True
# GPUconfig.gpu_options.allow_growth=True # if GPU issue, uncomment this sentence may help
GPUconfig.gpu_options.per_process_gpu_memory_fraction = CRNNconfig['DEVICE']['CLIENT_GPU_FRACTION']

client_train_flag = 0
client_wait_flag = 1
client_done_flag = -1

client_train_params = {"epochs": CRNNconfig["ESTIMATION"]["CLIENT_EPOCH"],
                       "batch_size": CRNNconfig["ESTIMATION"]["BATCH_SIZE"],
                       "val_batch_size": CRNNconfig["ESTIMATION"]["VAL_BITCH_SIZE"],
                       "max_label_length": CRNNconfig["NETWORK"]["MAX_LABEL_LENGTH"],
                       "char_file": CRNNconfig["NETWORK"]["DICTIONARY_PATH"],
                       "image_size": CRNNconfig["ESTIMATION"]["IMAGE_SIZE"],
                       "client_number": CRNNconfig["ESTIMATION"]["CLIENTS"]}

if __name__=="__main__":
    if not CRNNconfig['WORKMODE'] == 'Estimation':
        CRNNconfig['WORKMODE'] = 'Estimation'
    model_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(model_id)
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
    manager_client = FLutils.ManagerClient(CRNNconfig["ESTIMATION"]["MANAGER_DOMAIN"],
                                           CRNNconfig["ESTIMATION"]["MANAGER_PORT"],
                                           CRNNconfig["ESTIMATION"]["MANAGER_AUTH_KEY"].encode())
    dict_item = manager_client.get_dict()
    client_number = client_train_params["client_number"]
    graph = tf.get_default_graph()
    with tf.Session(config=GPUconfig) as sess, graph.as_default():
        sess.run(tf.global_variables_initializer())
        net = FLutils.Network(CRNNconfig)
        WEIGHT_PATH = dict_item.get(f"DirName") + "/Client_" + str(Client_id)
        while 1:
            flag = dict_item.get(f"flag_client_{Client_id}")
            if flag == client_wait_flag:
                if CRNNconfig["ESTIMATION"]["FL_STRATEGY"]=="FedBoost":
                    for ind_other in clients_list: # loop other clients' model for testing
                        flag_other = dict_item.get(f"flag_client_{ind_other}")
                        if flag_other == client_wait_flag: # wait flag means the client is done training
                            client.receive_and_init_model(net.deviceModel, dict_item.get(f"weights_client_{ind_other}"))
                            results_dict = client.edge_test(client_train_params)
                            if results_dict["acc"]==0: results_dict["acc"]=1e-8 # avoid divided by 0
                            test_accuracy_list[ind_other] = results_dict["acc"]
                            test_loss_list[ind_other] = results_dict["loss"]
                            clients_list.remove(ind_other)
                            print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Finished evaluation client {ind_other} model: accuracy {test_accuracy_list[ind_other]}, loss {test_loss_list[ind_other]}')
                    if len(clients_list)==0:
                        dict_item.set(key=f"test_accuracy_on_client_{Client_id}", value=test_accuracy_list)
                        dict_item.set(key=f"test_loss_on_client_{Client_id}", value=test_loss_list)
                        if not os.path.exists(WEIGHT_PATH): os.makedirs(WEIGHT_PATH)
                        record_path = WEIGHT_PATH + f'/test_result_round:{str(dict_item.get(f"round")).zfill(3)}.txt'
                        if not os.path.exists(record_path):
                            test_result_record = open(record_path, "w")
                            test_result_record.write(f"accuracy: {test_accuracy_list}\n")
                            test_result_record.write(f"loss: {test_loss_list}\n")
                            test_result_record.close()
                            print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  Final evaluation accuracy {test_accuracy_list}, loss {test_loss_list}')
                            if dict_item.get(f"round") + 1 == CRNNconfig["ESTIMATION"]["ROUNDS"]: break
                time.sleep(10)
            elif flag == client_train_flag:
                print('-'*100)
                print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} >>>>> Round {dict_item.get("round")}')
                model_weights = dict_item.get(f"weights_client_{Client_id}")
                client.receive_and_init_model(net.deviceModel, model_weights)
                hist = client.edge_train(client_train_params, dict_item.get(f"DirName"), dict_item.get(f"round"), CRNNconfig["ESTIMATION"]["SAVE_CLIENT_MODEL"])
                dict_item.set(key=f"flag_client_{Client_id}", value=client_done_flag)
                dict_item.set(key=f"weights_client_{Client_id}", value=client.model.get_weights())
                dict_item.set(key=f"history_{Client_id}", value=hist.history)
                print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} client_{Client_id} training done...')
                if CRNNconfig["ESTIMATION"]["FL_STRATEGY"] == "FedBoost":
                    clients_list = [i for i in range(client_number)]
                    clients_list.remove(Client_id)  # delete current client id
                    test_accuracy_list = dict_item.get(f"test_accuracy_on_client_{Client_id}")
                    test_loss_list = dict_item.get(f"test_loss_on_client_{Client_id}")
                    client.receive_and_init_model(net.deviceModel, client.model.get_weights())
                    results_dict = client.edge_test(client_train_params)
                    if results_dict["acc"] == 0: results_dict["acc"] = 1e-8  # avoid divided by 0
                    test_accuracy_list[Client_id] = results_dict["acc"]
                    test_loss_list[Client_id] = results_dict["loss"]
                    print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Finished evaluation client {Client_id} model: accuracy {test_accuracy_list[Client_id]}, loss {test_loss_list[Client_id]}')
    print('_' * 30)
