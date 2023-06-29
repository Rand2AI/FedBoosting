import numpy as np
import json, cv2, base64, os, datetime
import tensorflow as tf
import FLutils
from tqdm import tqdm
CRNNconfig = FLutils.get_config(os.path.dirname(os.path.realpath(__file__)), 'Inference')

if __name__=="__main__":
    test_list = ["IIIT.5K/IIIT5K_test_20190829.json", "SVT/svt_test_20190828_cropped.json", "ICDAR2015/ICDAR2015_test_20190829.json"]
    # test_list = [test_list[2]]
    char_to_id = FLutils.gen_character(CRNNconfig["NETWORK"]["DICTIONARY_PATH"])
    id_to_char = {v: k for k, v in char_to_id.items()}
    root_path = CRNNconfig["ESTIMATION"]["TRAIN_ROOT_PATH"]
    model_root_path = CRNNconfig["ESTIMATION"]["MODEL_ROOT_PATH"]
    save_path = model_root_path+CRNNconfig["INFERENCE"]["WEIGHT_TOLOAD"].split('global')[0] + "/TestResult/"
    with tf.device("/gpu:0"):
        net = FLutils.Network(CRNNconfig)
        print(CRNNconfig["INFERENCE"]["WEIGHT_TOLOAD"])
        instant_avg = 0.0
        for data in test_list:
            test_path = root_path + data
            with open(test_path, 'r', encoding='utf-8') as imgf:
                images = imgf.readlines()
            cur = 0
            total = 0
            process_bar_obj = tqdm(images,
                                   ncols=130,
                                   desc=f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {data.split('/')[0]}")
            for line in process_bar_obj:
                temp = json.loads(line.strip('\r\n'))
                y_test = temp['label'].upper()

                # if "2" in y_test or "Z" in y_test: # TODO

                ori_img = temp['img'].encode('utf-8')
                ori_img = cv2.imdecode(np.frombuffer(base64.b64decode(ori_img), np.uint8), 1)
                if len(ori_img.shape) < 3 or ori_img.shape[2] == 1:
                    ori_img = cv2.merge([ori_img, ori_img, ori_img])
                imgName = temp['imageName']
                img_processed = cv2.resize(ori_img, (int(ori_img.shape[1] * (32 / ori_img.shape[0])), 32))
                try: _ = [char_to_id[j] for j in y_test]
                except: continue
                if img_processed.shape[1] < 100: continue
                if len(y_test) < 3: continue
                if len(y_test)>img_processed.shape[1]//4: continue
                img_processed = (np.array(img_processed, 'f') - 127.5) / 127.5
                x_test = np.zeros((1, 32, img_processed.shape[1], 3), dtype=np.float32)
                x_test[0] = img_processed
                pred_num = net.deviceModel.predict(x_test)
                pred_list = FLutils.fast_ctc_decode(pred_num, 0)
                pred_label = u''.join([id_to_char[x] for [x, _, _] in pred_list])
                if pred_label.upper() == y_test:
                    cur+=1
                    if CRNNconfig["INFERENCE"]["SAVE_IMG"]:
                        right_path = save_path+data+"/Right/"
                        if not os.path.exists(right_path):
                            os.makedirs(right_path)
                        cv2.imwrite(right_path+y_test+".jpg", ori_img)
                        with open(right_path+"results.txt", "a") as f:
                            f.write(f"{pred_label}\n")
                else:
                    # print(f"pred: {pred_label.upper()}, gt: {y_test}")
                    if CRNNconfig["INFERENCE"]["SAVE_IMG"]:
                        wrong_path = save_path+data+"/Wrong/"
                        if not os.path.exists(wrong_path):
                            os.makedirs(wrong_path)
                        cv2.imwrite(wrong_path+y_test+".jpg", ori_img)
                        with open(wrong_path+"results.txt", "a") as f:
                            f.write(f"P: {pred_label}, L: {y_test}\n")
                total += 1
                process_bar_obj.set_postfix(correct=cur, accuracy=round(cur/total*100, 4))
            print(round(cur/total*100, 4))
            instant_avg += round(cur/total*25, 2)
        print(instant_avg)