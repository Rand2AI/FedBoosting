import os, json, datetime, time, random
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
client_wait_flag = 1
client_done_flag = -1

model_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print(model_id)
dir_name = model_id+"-"+str(CRNNconfig["ESTIMATION"]["CLIENT_EPOCH"][0])+"E"+str(CRNNconfig["ESTIMATION"]["BATCH_SIZE"])+f'B_{CRNNconfig["ESTIMATION"]["FL_STRATEGY"]}/'
root_path = CRNNconfig["ESTIMATION"]["MODEL_ROOT_PATH"]+ "/" + CRNNconfig["ESTIMATION"]["FL_STRATEGY"] + "/" + dir_name + "/"

if __name__=="__main__":
    if not CRNNconfig['WORKMODE'] == 'Estimation':
        CRNNconfig['WORKMODE'] = 'Estimation'
    if not os.path.exists(root_path):
        print(root_path)
        os.makedirs(root_path)
    manager_server = FLutils.ManagerServer(CRNNconfig["ESTIMATION"]["MANAGER_DOMAIN"],
                                           CRNNconfig["ESTIMATION"]["MANAGER_PORT"],
                                           CRNNconfig["ESTIMATION"]["MANAGER_AUTH_KEY"].encode(),
                                           CRNNconfig["ESTIMATION"]["CLIENTS"])
    manager_server.run()
    dict_item = manager_server.get_dict()
    dict_item.set(key="DirName", value=root_path)
    args_json_path = root_path + "/args.json"
    FLutils.save_args_as_json(CRNNconfig, args_json_path)
    tfboard_loggers.TFBoardTextLogger(root_path).log_markdown("args", f"```\n{json.dumps(CRNNconfig, indent=4, sort_keys=True)}\n```", -1)
    tf_scalar_logger = tfboard_loggers.TFBoardScalarLogger(root_path)
    params = {"epochs": CRNNconfig["ESTIMATION"]["CLIENT_EPOCH"],
               "batch_size": CRNNconfig["ESTIMATION"]["BATCH_SIZE"],
               "val_batch_size": CRNNconfig["ESTIMATION"]["VAL_BITCH_SIZE"],
               "max_label_length": CRNNconfig["NETWORK"]["MAX_LABEL_LENGTH"],
               "char_file": CRNNconfig["NETWORK"]["DICTIONARY_PATH"],
               "image_size": CRNNconfig["ESTIMATION"]["IMAGE_SIZE"],
               "client_number": CRNNconfig["ESTIMATION"]["CLIENTS"]}
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
                                CRNNconfig["ESTIMATION"]["CLIENTS"])
        if CRNNconfig["ESTIMATION"]["FINETUNE"] and CRNNconfig["ESTIMATION"]["WEIGHT_TOLOAD"] != "":
            print(f'\n\n Finetune from {CRNNconfig["ESTIMATION"]["WEIGHT_TOLOAD"]} \n\n')
            server.load_model_weights(CRNNconfig["ESTIMATION"]["WEIGHT_TOLOAD"], by_name=True)
        global_weight_path = root_path + "/initial-global_weights.h5"
        server.save_model_weights(global_weight_path)
        test_result_record = open(root_path + "/test_result.txt", "a")
        for Round in range(CRNNconfig["ESTIMATION"]["ROUNDS"]):
            dict_item.set(key=f"round", value=Round)
            server.init_for_new_round()
            print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} >>>>> Round {Round}/{CRNNconfig["ESTIMATION"]["ROUNDS"]} is starting')
            for ind in range(CRNNconfig["ESTIMATION"]["CLIENTS"]):
                dict_item.set(key=f"flag_client_{ind}", value=client_train_flag)
                dict_item.set(key=f"weights_client_{ind}", value=server.global_model_weights)
                dict_item.set(key=f"test_accuracy_on_client_{ind}", value=[0 for i in range(params["client_number"])]) # dataset-wise result
                if CRNNconfig["ESTIMATION"]["FL_STRATEGY"] == "FedBoost":
                    dict_item.set(key=f"test_loss_on_client_{ind}", value=[0 for i in range(params["client_number"])])
                else:
                    dict_item.set(key=f"test_loss_on_client_{ind}", value=[1 for i in range(params["client_number"])])
            dict_item.set(key=f"weights_global", value=server.global_model_weights)
            clients_list = [i for i in range(CRNNconfig["ESTIMATION"]["CLIENTS"])]
            while len(clients_list)>0:
                ind = random.choice(clients_list)
                flag = dict_item.get(f"flag_client_{ind}")
                if flag == client_done_flag:
                    print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} client_{ind} done training...')
                    client_model_weights = dict_item.get(f"weights_client_{ind}")
                    server.client_model_weights.append(client_model_weights)
                    dict_item.set(key=f"flag_client_{ind}", value=client_wait_flag)
                    dict_item.set(key=f"weights_client_{ind}", value=client_model_weights)
                if flag == client_wait_flag and dict_item.get(f"test_loss_on_client_{ind}").count(0)==0:
                    clients_list.remove(ind)
                    if CRNNconfig["ESTIMATION"]["FL_STRATEGY"] == "FedBoost":
                        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} client_{ind} done evaluating...')
                time.sleep(10)
            if CRNNconfig["ESTIMATION"]["FL_STRATEGY"] == "FedBoost":
                for ind in range(CRNNconfig["ESTIMATION"]["CLIENTS"]):
                    server.client_test_accuracy.append(dict_item.get(f"test_accuracy_on_client_{ind}")) # dataset-wise result
                    server.client_test_loss.append(dict_item.get(f"test_loss_on_client_{ind}"))
                    server.client_history.append(dict_item.get(f"history_{ind}"))
                server.process_client_test_result(CRNNconfig["ESTIMATION"]["FL_STRATEGY_METRICS"])
            server.summarize_weights()
            test_loss = []
            test_acc = []
            for test_data in test_data_list:
                print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {test_data["DataName"].center(10)} Global test', end=" ")
                global_test_results = server.evaluate_global_model(params,test_data["test_data"])
                test_loss.append(global_test_results["loss"])
                print(f'Loss: {global_test_results["loss"]}', end=" ")
                test_result_record.write(f'Loss: {global_test_results["loss"]}' + "\n")
                tf_scalar_logger.log_scalar(f'test_loss/{test_data["DataName"]}_global_loss',global_test_results["loss"], Round)
                test_acc.append(global_test_results["acc"])
                print(f'Accuracy: {global_test_results["acc"]}')
                test_result_record.write(f'Accuracy: {global_test_results["acc"]}' + "\n")
                tf_scalar_logger.log_scalar('test_acc/{test_data["DataName"]}_global_acc', global_test_results["acc"], Round)
            print(f'>>>>>>> Round {Round} test loss (client mean): {np.mean(test_loss)}')
            test_result_record.write(f'>>>>>>> Round {Round} test loss: {np.mean(test_loss)}' + "\n")
            print(f'>>>>>>> Round {Round} test accuracy (client mean): {np.mean(test_acc)}')
            test_result_record.write(f'>>>>>>> Round {Round} test accuracy: {np.mean(test_acc)}' + "\n")
            if CRNNconfig["ESTIMATION"]["FL_STRATEGY"] == "FedBoost":
                print(f'>>>>>>> Round {Round} clients model proportion: {server.client_test_density_distribution}')
                test_result_record.write(f'>>>>>>> Round {Round} clients model proportion: {server.client_test_density_distribution}' + "\n")
            print('-' * 100)
            test_result_record.write('-' * 100 + "\n")
            global_weight_path = root_path + "/global/"
            if not os.path.exists(global_weight_path): os.makedirs(global_weight_path)
            server.save_model_weights(global_weight_path+f"/Round_{str(Round).zfill(4)}-global_weights-test_acc:{round(float(np.mean(test_acc))*100, 2)}.h5")
        test_result_record.close()
        train_hist_path = root_path + "/fed_learn_global_test_results.json"
        with open(str(train_hist_path), 'a') as f:
            f.write(json.dumps(server.global_test_metrics_dict))
            f.write('\n')
    manager_server.stop()