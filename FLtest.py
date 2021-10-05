import numpy as np
import json, cv2, base64, os
import tensorflow as tf
import FLutils
CRNNconfig = FLutils.get_config(os.path.dirname(os.path.realpath(__file__)), 'Inference')

if __name__=="__main__":
    if not CRNNconfig['WORKMODE'] == 'Inference':
        CRNNconfig['WORKMODE'] = 'Inference'
    test_list = ['IIIT.5K', 'SVT', "SCUT", "ICDAR2015"]
    char_to_id = FLutils.gen_character(CRNNconfig["NETWORK"]["DICTIONARY_PATH"])
    id_to_char = {v: k for k, v in char_to_id.items()}
    root_path = CRNNconfig["ESTIMATION"]["TRAIN_ROOT_PATH"]
    save_path = CRNNconfig["INFERENCE"]["WEIGHT_TOLOAD"].split('global')[0] + "/TestResult/"
    with tf.device("/gpu:0"):
        net = FLutils.Network(CRNNconfig)
        print(CRNNconfig["INFERENCE"]["WEIGHT_TOLOAD"])
        instant_avg = 0.0
        for data in test_list:
            test_path = root_path +f"{data}/test_FL.json"
            with open(test_path, 'r', encoding='utf-8') as imgf:
                images = imgf.readlines()
            cur = 0
            total = 0
            for line in images:
                temp = json.loads(line.strip('\r\n'))
                y_test = temp['label'].upper()
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
                    if CRNNconfig["INFERENCE"]["SAVE_IMG"]:
                        wrong_path = save_path+data+"/Wrong/"
                        if not os.path.exists(wrong_path):
                            os.makedirs(wrong_path)
                        cv2.imwrite(wrong_path+y_test+".jpg", ori_img)
                        with open(wrong_path+"results.txt", "a") as f:
                            f.write(f"P: {pred_label}, L: {y_test}\n")
                total+=1
            print("{0}: Total {1} images, correct {2}, accuracy: {3}".format(data, total, cur, round(cur/total*100, 2)))
            instant_avg += round(cur/total*25, 2)
        print(instant_avg)