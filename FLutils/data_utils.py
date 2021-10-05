import numpy as np
import base64, json, cv2

class DataHandler:
    def __init__(self, debug = False):
        self.debug = debug
        self.train_data = None
        self.evaluate_data = None
        self.test_data = None

    def split_data(self, ratio):
        thres = int(len(self.train_data)*ratio)
        self.evaluate_data = self.train_data[thres:len(self.train_data)]
        self.train_data = self.train_data[0:thres]
        print(f"Number of training images is {len(self.train_data)}")
        print(f"Number of evaluating images is {len(self.evaluate_data)}")

    def process_train_data(self, Datapath):
        TrainFile = Datapath + '/train_FL.json'
        if self.debug:
            train_data = []
            with open(TrainFile, 'r', encoding='utf-8') as imgf:
                train_data_line = imgf.readline()
                while train_data_line:
                    train_data.append(train_data_line)
                    if len(train_data)==50000: break
                    train_data_line = imgf.readline()

        else:
            with open(TrainFile, 'r', encoding='utf-8') as imgf:
                train_data = imgf.readlines()
        self.train_data = train_data
        print(f"Number of total images is {len(self.train_data)}")

    def process_test_data(self, Datapath):
        TestFile = Datapath + '/test_FL.json'
        if self.debug:
            test_data = []
            with open(TestFile, 'r', encoding='utf-8') as imgf:
                test_data_line = imgf.readline()
                while test_data_line:
                    test_data.append(test_data_line)
                    if len(test_data) == 10000: break
                    test_data_line = imgf.readline()
        else:
            with open(TestFile, 'r', encoding='utf-8') as imgf:
                test_data = imgf.readlines()
        self.test_data = test_data

    def assign_data_to_clients(self, clients):
        for client in clients:
            client.receive_data(self.train_data, self.evaluate_data)

def gen_character(filepath):
    char = ''
    with open(filepath, encoding='utf-8') as fid:
        for ch in fid.readlines():
            ch = ch.strip('\r\n')
            char += ch
    char_to_id = {j: i for i, j in enumerate(char)}
    return char_to_id

class sequence_order_num:
    def __init__(self, total, batchsize=64):
        self.total = total
        self.range = [i for i in range(total)]
        self.index = 0
        max_index = int(total / batchsize)
        self.index_list = [i for i in range(max_index)]
        np.random.shuffle(self.index_list)

    def get(self, batchsize):
        s_o = []
        if self.index + batchsize > self.total:
            s_o_1 = self.range[self.index:self.total]
            self.index = (self.index + batchsize) - self.total
            s_o_2 = self.range[0:self.index]
            s_o.extend(s_o_1)
            s_o.extend(s_o_2)
        else:
            s_o = self.range[self.index:self.index + batchsize]
            self.index = self.index + batchsize
        return s_o

    def shuffle_batch(self, batchsize):
        if self.index== len(self.index_list): self.index=0
        start_index = self.index_list[self.index]*batchsize
        end_index = start_index + batchsize
        s_o = self.range[start_index:end_index]
        self.index += 1
        return s_o

def generator(client_train_dict: dict, data, mode="train"):
    if mode=="train":
        batchsize = client_train_dict["batch_size"]
    else:
        batchsize = client_train_dict["val_batch_size"]
    batchsize = min(batchsize, len(data))
    char_to_id = gen_character(client_train_dict["char_file"])
    idlist = sequence_order_num(total=len(data), batchsize=batchsize)
    while True:
        index = idlist.get(batchsize)
        x_generator = np.zeros((len(index), client_train_dict["image_size"][0], client_train_dict["image_size"][1], 3), dtype=np.float32)
        y_generator = np.ones([len(index), client_train_dict["max_label_length"]]) * 10000
        input_length_generator = np.zeros([len(index), 1])
        label_length_generator = np.zeros([len(index), 1])
        for ind, i in enumerate(index):
            temp = json.loads(data[i].strip('\r\n'))
            IdNumber = temp['label'].upper()
            labelL = len(IdNumber)
            Img = temp['img'].encode('utf-8')
            Img = cv2.imdecode(np.frombuffer(base64.b64decode(Img), np.uint8), 1)
            if len(Img.shape) < 3 or Img.shape[2] == 1:
                Img = cv2.merge([Img, Img, Img])
            img1 = cv2.resize(Img, (100, 32))
            inputL = img1.shape[1] // 4
            img1 = (np.array(img1, 'f') - 127.5) / 127.5
            x_generator[ind] = img1
            y_generator[ind, :labelL] = [char_to_id[i] for i in IdNumber]
            input_length_generator[ind] = inputL
            label_length_generator[ind] = labelL
        inputs = {'the_image': x_generator,
                  'the_label_text': y_generator,
                  'the_length_image': input_length_generator,
                  'the_length_texts': label_length_generator
                  }
        outputs = {'loss_ctc': np.zeros([len(index)])}
        yield inputs, outputs