import os, json, datetime, time, random, pickle
import numpy as np
from swiss_army_tensorboard import tfboard_loggers
import tensorflow as tf
import FLutils
CRNNconfig = FLutils.get_config(os.path.dirname(os.path.realpath(__file__)), 'Estimation')
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in CRNNconfig['DEVICE']["DEVICE_GPUID"]])
GPUconfig = tf.ConfigProto()
GPUconfig.allow_soft_placement = True
# GPUconfig.gpu_options.allow_growth=True # if GPU issue, uncomment this sentence may help
GPUconfig.gpu_options.per_process_gpu_memory_fraction = CRNNconfig['DEVICE']['SERVER_GPU_FRACTION']

client_train_flag = 0
client_done_flag = 1
client_wait_flag = 2
client_val_flag = 3

model_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print(model_id)
dir_name = model_id+"-"+str(CRNNconfig["ESTIMATION"]["CLIENT_EPOCH"][0])+"E"+str(CRNNconfig["ESTIMATION"]["BATCH_SIZE"])+f'B_{CRNNconfig["ESTIMATION"]["FL_STRATEGY"]}_HE/'
root_path = CRNNconfig["ESTIMATION"]["MODEL_ROOT_PATH"]+ "/" + CRNNconfig["ESTIMATION"]["FL_STRATEGY"] + "_HE/" + dir_name + "/"

def test_global_weights():
    test_loss = []
    test_acc = []
    for test_data in test_data_list:
        print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {test_data["DataName"].center(10)} global test', end=" ")
        global_test_results = server.evaluate_global_model(params, test_data["test_data"])
        test_loss.append(global_test_results["loss"])
        print(f'loss: {global_test_results["loss"]}', end=" ")
        test_result_record.write(f'loss: {global_test_results["loss"]}' + "\n")
        tf_scalar_logger.log_scalar(f'test_loss/{test_data["DataName"]}_global_loss', global_test_results["loss"], Round)
        test_acc.append(global_test_results["acc"])
        print(f'accuracy: {global_test_results["acc"]}')
        test_result_record.write(f'accuracy: {global_test_results["acc"]}' + "\n")
        tf_scalar_logger.log_scalar(f'test_acc/{test_data["DataName"]}_global_acc', global_test_results["acc"], Round)
    print(f'>>>>>>> Round {Round} test loss (client mean): {np.mean(test_loss)}')
    test_result_record.write(f'>>>>>>> Round {Round} test loss: {np.mean(test_loss)}' + "\n")
    print(f'>>>>>>> Round {Round} test accuracy (client mean): {np.mean(test_acc)}')
    test_result_record.write(f'>>>>>>> Round {Round} test accuracy: {np.mean(test_acc)}' + "\n")
    if CRNNconfig["ESTIMATION"]["FL_STRATEGY"] == "FedBoost":
        print(f'>>>>>>> Round {Round} clients model proportion: {server.client_test_density_distribution}')
        test_result_record.write(f'>>>>>>> Round {Round} clients model proportion: {server.client_test_density_distribution}' + "\n")
    test_result_record.write('-' * 100 + "\n")
    global_weight_path = root_path + "/global/"
    if not os.path.exists(global_weight_path): os.makedirs(global_weight_path)
    server.save_model_weights(global_weight_path + f"/Round_{str(Round).zfill(4)}-global_weights-test_acc:{round(float(np.mean(test_acc)) * 100, 2)}.h5")

if __name__=="__main__":
    if not os.path.exists(root_path):
        print(root_path)
        os.makedirs(root_path)
    manager_server_client = {}
    server_dict_item = {}
    n_clients = CRNNconfig["ESTIMATION"]["CLIENTS"]
    for ind in range(n_clients):
        manager_server_client[ind] = FLutils.ManagerServer(CRNNconfig["SERVER"]["SERVER_DOMAIN"],
                                                          CRNNconfig["SERVER"]["SERVER_PORT"]+ind,
                                                          (CRNNconfig["SERVER"]["SERVER_AUTH_KEY"]+f'client{ind}').encode())
        manager_server_client[ind].run()
        server_dict_item[ind] = manager_server_client[ind].get_dict()
        server_dict_item[ind].set(key="DirName", value=root_path)
        server_dict_item[ind].set(key="decryotion_require", value=0)
        server_dict_item[ind].set(key="client_number", value=n_clients)
    args_json_path = root_path + "/args.json"
    FLutils.save_args_as_json(CRNNconfig, args_json_path)
    tfboard_loggers.TFBoardTextLogger(root_path).log_markdown("args", f"```\n{json.dumps(CRNNconfig, indent=4, sort_keys=True)}\n```", -1)
    tf_scalar_logger = tfboard_loggers.TFBoardScalarLogger(root_path)
    params = {"epochs": CRNNconfig["ESTIMATION"]["CLIENT_EPOCH"],
               "batch_size": CRNNconfig["ESTIMATION"]["BATCH_SIZE"],
               "max_label_length": CRNNconfig["NETWORK"]["MAX_LABEL_LENGTH"],
               "char_file": CRNNconfig["NETWORK"]["DICTIONARY_PATH"],
               "image_size": CRNNconfig["ESTIMATION"]["IMAGE_SIZE"],
               "client_number": n_clients}
    if CRNNconfig["ESTIMATION"]["FL_STRATEGY"]=="FedAvg":
        weight_summarizer = FLutils.FedAvg()
    elif CRNNconfig["ESTIMATION"]["FL_STRATEGY"]=="FedBoost":
        weight_summarizer = FLutils.FedBoost()
    else:
        raise Exception('<<<<<< Incorrect FL_STRATEGY >>>>>>')
    char_to_id = FLutils.gen_character(CRNNconfig["NETWORK"]["DICTIONARY_PATH"])
    data_handler = FLutils.DataHandler(CRNNconfig["ESTIMATION"]["DEBUG"])
    test_data_list = []
    for DataName in CRNNconfig["ESTIMATION"]["TEST_NAME_LIST"]:
        print(f'Start to process testing data: {DataName}')
        data_handler.process_test_data(CRNNconfig["ESTIMATION"]["TRAIN_ROOT_PATH"] + DataName)
        test_data_list.append(dict(DataName = DataName,
                                   test_data = data_handler.test_data))
    graph = tf.get_default_graph()
    with tf.Session(config=GPUconfig) as sess, graph.as_default():
        sess.run(tf.global_variables_initializer())
        net = FLutils.Network(CRNNconfig)
        server = FLutils.Server(net.deviceModel,
                                weight_summarizer,
                                nb_clients=n_clients,
                                slice_num=CRNNconfig["ESTIMATION"]["SMALL_SLICE"])
        if CRNNconfig["ESTIMATION"]["FINETUNE"]=="YES" and CRNNconfig["ESTIMATION"]["WEIGHT_TOLOAD"] != "":
            print(f'\n\n Finetune from {CRNNconfig["ESTIMATION"]["WEIGHT_TOLOAD"]} \n\n')
            server.load_model_weights(CRNNconfig["ESTIMATION"]["WEIGHT_TOLOAD"], by_name=True)
        global_weight_path = root_path + "/initial-global_weights.h5"
        server.save_model_weights(global_weight_path) # save initial global weights
        test_result_record = open(root_path + "/test_result.txt", "a")
        train_hist_path = open(str(root_path + "/fl_global_test_results.json"), 'a')
        for ind in range(n_clients):
            server_dict_item[ind].set(key="initial_global_weights", value=pickle.dumps(server.decrypted_model_weights))
            # server_dict_item[ind].set(key="gradient_client", value={})
        for Round in range(CRNNconfig["ESTIMATION"]["ROUNDS"]):
            print("-"*50)
            server.init_for_new_round()
            print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Round {Round}/{CRNNconfig["ESTIMATION"]["ROUNDS"]} is starting')
            for ind in range(n_clients):
                server_dict_item[ind].set(key="round", value=Round)
                server_dict_item[ind].set(key="flag_client", value=client_train_flag)
                if CRNNconfig["ESTIMATION"]["FL_STRATEGY"] == "FedBoost":
                    server_dict_item[ind].set(key="test_accuracy_on_client", value=[0 for i in range(params["client_number"])])  # dataset-wise result
                    server_dict_item[ind].set(key="test_loss_on_client", value=[0 for i in range(params["client_number"])])
            clients_list = list(range(n_clients))
            while len(clients_list)>0: # train
                ind = random.choice(clients_list)
                flag = server_dict_item[ind].get("flag_client")
                if flag == client_done_flag:
                    print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] client_{ind} done training...')
                    client_gradient = pickle.loads(server_dict_item[ind].get(f"gradient_client")) # decerialization
                    server.client_gradients[ind] = client_gradient
                    server_dict_item[ind].set(key="flag_client", value=client_wait_flag)
                    clients_list.remove(ind)
                time.sleep(5)
            if CRNNconfig["ESTIMATION"]["FL_STRATEGY"] == "FedBoost": # FedBoost validation
                server.linear_client_gradients()
                clients_list = list(range(n_clients))
                for major_ind in clients_list: # distribute linear encrypted gradients
                    for minor_ind in clients_list:
                        if CRNNconfig["ESTIMATION"]["DP"]:
                            server_dict_item[major_ind].set(key=f"linear_client_{minor_ind}", value=pickle.dumps(server.linear_clients_list[minor_ind]))
                        else:
                            server_dict_item[major_ind].set(key=f"linear_client_{minor_ind}", value=pickle.dumps(server.client_gradients[minor_ind]))
                    server_dict_item[major_ind].set(key="flag_client", value=client_val_flag)
                while len(clients_list) > 0:
                    ind = random.choice(clients_list)
                    flag = server_dict_item[ind].get("flag_client")
                    if flag == client_done_flag:
                        clients_list.remove(ind)
                        server_dict_item[ind].set(key="flag_client", value=client_wait_flag)
                        print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] client_{ind} done validation...')
                for ind in range(n_clients):
                    server.client_test_accuracy.append(server_dict_item[ind].get("test_accuracy_on_client")) # dataset-wise result
                    server.client_test_loss.append(server_dict_item[ind].get("test_loss_on_client"))
                    server.client_history.append(server_dict_item[ind].get("history"))
                server.process_client_test_result(CRNNconfig["ESTIMATION"]["FL_STRATEGY_METRICS"])
            server.summarize_weights() # global gradient
            for ind in range(n_clients):
                server_dict_item[ind].set(key="global_gradient", value=pickle.dumps(server.global_model_gradient))
            if (Round+1) % CRNNconfig["CLIENT"]["ROUNDS_SERVER_GET_MODEL"] == 0:
                for ind in range(n_clients):
                    server_dict_item[ind].set(key="decryotion_require", value=1)
                print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Wait for global weights decryption.')
                decryotion_bool = 1
                while decryotion_bool:
                    rdm_client_ind = random.choice(list(range(n_clients)))
                    decryotion_bool = server_dict_item[rdm_client_ind].get("decryotion_require")
                    time.sleep(5)
                server.updata_decrypted_global_model(pickle.loads(server_dict_item[rdm_client_ind].get("decrypted_global_weights"))) # deserialization
                test_global_weights()
            train_hist_path.write(json.dumps(server.global_test_metrics_dict))
            train_hist_path.write('\n')
        test_result_record.close()
        train_hist_path.close()
    for ind in range(n_clients):
        manager_server_client[ind].stop()