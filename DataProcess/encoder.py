import  xml.dom.minidom
import cv2, os, base64, json, re
import numpy as np
from PIL import Image,ImageDraw, ImageFont
import scipy.io as sio

def svt():
    ImgList = './Data/SVT/svt/svt1/'
    for mode in ['test', 'train']:
        ID = 'svt_'+mode
        print('>>>>> Start to process:', ID)
        fileout = './Data/SVT/'+ID+'.json'
        dom = xml.dom.minidom.parse(ImgList+mode+'.xml')
        root = dom.documentElement
        imageList = root.getElementsByTagName("image")
        if mode == 'train':
            num = 100
        else:
            num = 249
        with open(fileout, 'w', encoding='utf-8') as fout:
            dat = dict(ID=ID,
                       number=num,
                       structure=['imageName', 'address', 'lex', 'Resolution', 'taggedRectangles', 'img'],
                       introduction='taggedRectangles is a dictionary whose structure is: {label:[x,y,h,w]}; while Resolution is a list: [Resolution_x,Resolution_y]')
            fout.write(json.dumps(dat))
            fout.write('\n')
            for image in imageList:
                recDic = {}
                imageName = image.getElementsByTagName('imageName')[0].childNodes[0].data
                address = image.getElementsByTagName('address')[0].childNodes[0].data
                lex = image.getElementsByTagName('lex')[0].childNodes[0].data
                Resolution_x = int(image.getElementsByTagName('Resolution')[0].getAttribute('x'))
                Resolution_y = int(image.getElementsByTagName('Resolution')[0].getAttribute('y'))
                taggedRectangle = image.getElementsByTagName('taggedRectangle')
                for rec in taggedRectangle:
                    tag = rec.getElementsByTagName('tag')[0].childNodes[0].data
                    rec_h = int(rec.getAttribute('height'))
                    rec_w = int(rec.getAttribute('width'))
                    rec_x = int(rec.getAttribute('x'))
                    rec_y = int(rec.getAttribute('y'))
                    recDic[tag] = [rec_x,rec_y,rec_h,rec_w]
                img = np.array(Image.open(ImgList+imageName))
                dat = dict(imageName=imageName.split('/')[1],
                           address=address,
                           lex=lex,
                           Resolution=[Resolution_x,Resolution_y],
                           taggedRectangles=recDic,
                           img=base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8'))
                fout.write(json.dumps(dat))
                fout.write('\n')

def svt_crop():
    rootPath = './Data/SVT/'
    tfrecords_file = rootPath + '/svt_train.json'
    fileout = rootPath + '/svt_train_cropped.json'
    with open(fileout, 'w', encoding='utf-8') as fout:
        with open(tfrecords_file, 'r', encoding='utf-8') as imgf:
            image = imgf.readline()
            for key, value in json.loads(image.strip('\r\n')).items():
                print(key, ':', value)
            image = imgf.readline()
            num = 0
            while image:
                temp = json.loads(image.strip('\r\n'))
                imageName = temp['imageName']
                IdNumbers = list(temp['taggedRectangles'].keys())
                Boxes = list(temp['taggedRectangles'].values())
                img = temp['img'].encode('utf-8')
                img = cv2.imdecode(np.frombuffer(base64.b64decode(img), np.uint8), cv2.IMREAD_COLOR)
                for index in range(len(IdNumbers)):
                    try:
                        Img = img[Boxes[index][1]:Boxes[index][1]+Boxes[index][2], Boxes[index][0]:Boxes[index][0]+Boxes[index][3]].copy()
                        # cv2.imshow('asd', Img)
                        # cv2.waitKey(0)
                        imageName = imageName.split('.')[0]+'_'+str(index)+'.'+imageName.split('.')[1]
                        label = IdNumbers[index]
                        dat = dict(imageName=imageName,
                                   label=label,
                                   img=base64.b64encode(cv2.imencode('.jpg', Img)[1]).decode('utf-8'))
                        fout.write(json.dumps(dat))
                        fout.write('\n')
                        num += 1
                        if num % 1000 == 0: print("processed: {0}".format(num))
                    except:
                        pass
                image = imgf.readline()

def ustb():
    ImgList = './Data/USTB-SV1K/USTB-SV1K_V1/'
    for mode in ['test', 'train']:
        ID = 'ustb_' + mode
        print('>>>>> Start to process:', ID)
        fileout = './Data/USTB-SV1K/' + ID + '.json'
        with open(fileout, 'w', encoding='utf-8') as fout:
            img = np.array(Image.open('./Data/USTB-SV1K/example.png').convert('L'))
            dat = dict(ID=ID,
                       number=500,
                       structure=['imageName', 'taggedRectangles', 'img'],
                       introduction='taggedRectangles is a dictionary whose structure is: {"label":[x,y,h,w,aoi,HV]}; aoi is the shortage of angle of inclination; The HV is “1” if the rectangle is in the near horizontal direction and if the rectangle is in the near vertical direction it will be “2”; More details please visualise img.',
                       img=base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8'))
            fout.write(json.dumps(dat))
            fout.write('\n')
            for root, dirs, file in os.walk(ImgList+mode):
                for name in file:
                    if '.gt' in name:
                        recDic = {}
                        img = np.array(Image.open(root+'/'+name.replace('.gt','.jpg')).convert('L'))
                        with open(root+'/'+name, 'rb') as f:
                            infos = f.readlines()
                        for info in infos:
                            info = info.decode('utf-8')
                            x = int(info.split('"')[0].split()[2])
                            y = int(info.split('"')[0].split()[3])
                            w = int(info.split('"')[0].split()[4])
                            h = int(info.split('"')[0].split()[5])
                            aoi = float(info.split('"')[0].split()[6])
                            HV = int(info.split('"')[2].split()[0])
                            label = info.split('"')[1]
                            if not label=='':
                                recDic[label] = [x,y,h,w,aoi,HV]
                        dat = dict(imageName=name.replace('.gt','.jpg'),
                                   taggedRectangles=recDic,
                                   img=base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8'))
                        fout.write(json.dumps(dat))
                        fout.write('\n')

def ICDAR2015():
    ImgList = './Data/ICDAR/2015/Word Recognition/'
    for mode in ['train', 'test']:
        ID = 'ICDAR2015_' + mode
        print('>>>>> Start to process:', ID)
        fileout = ImgList + ID + '.json'
        with open(fileout, 'w', encoding='utf-8') as fout:
            dat = dict(ID=ID,
                       number= 4468 if mode=='train' else 2077,
                       lex = "!#$%&'()+,-./0123456789:;=?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]abcdefghijklmnopqrstuvwxyz",
                       structure=['imageName', 'label', 'img'],
                       introduction='ICDAR/2015/Word Recognition/')
            fout.write(json.dumps(dat))
            fout.write('\n')
            with open(ImgList + mode + '/gt.txt', 'rb') as f:
                infos = f.readlines()
            for num, info in enumerate(infos):
                if num%1000==0: print(num+1,'/',len(infos))
                info = info.decode('utf-8')
                try:
                    imageName=info.split(',')[0]
                    img = np.array(Image.open(ImgList + mode + '/' + info.split(',')[0]))
                except:
                    print(info)
                    imageName = 'word_1.png'
                    img = np.array(Image.open(ImgList + mode + '/' + 'word_1.png'))
                label = info.split('"')[1].replace("`", "'").replace("´", "'").replace("É", "E").replace("é", "e").replace(" ", "")
                dat = dict(imageName=imageName,
                           label=label,
                           img=base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8'))
                fout.write(json.dumps(dat))
                fout.write('\n')

def SCUT_Eng():
    ImgList = './Data/SCUT-FORU/SCUT_FORU_DB_Release/English2k/'

    txtFiles = os.listdir(ImgList+'word_annotation/')
    txtFiles = (lambda x: (x.sort(), x)[1])(txtFiles)
    characters = ''
    for file in txtFiles:
        with open(ImgList+'word_annotation/'+file, 'r', encoding='utf-8') as imgf:
            infos = imgf.readlines()
        for info in infos:
            label = info.split('"')[1].replace(" ", "").replace("`", "'").replace("´", "'")
            characters += label
            characters = ''.join(x for i, x in enumerate(characters) if characters.index(x) == i)
    characters = "".join((lambda x: (x.sort(), x)[1])(list(characters)))

    for mode in ['test','train']:
        ID = 'SCUT_Eng_word_' + mode
        print('>>>>> Start to process:', ID)
        fileout = ImgList + ID + '.json'
        with open(fileout, 'w', encoding='utf-8') as fout:
            dat = dict(ID=ID,
                       number= 1000 if mode=='train' else 715,
                       lex = characters,
                       structure=['imageName', 'taggedRectangles', 'img'],
                       introduction='SCUT-FORU/SCUT_FORU_DB_Release/English2k/word_img; taggedRectangles is a dictionary whose structure is: {"label":[x,y,h,w]}')
            fout.write(json.dumps(dat))
            fout.write('\n')
            if mode=='train':
                L = range(1000)
            else:
                L = range(1000, 1715)
            for ind in L:
                try:
                    img = np.array(Image.open(ImgList + 'word_img/' + txtFiles[ind].replace('.txt', '.jpg')))
                except FileNotFoundError:
                    img = np.array(Image.open(ImgList + 'word_img/' + txtFiles[ind].replace('.txt', '.JPG')))
                except OSError:
                    print(txtFiles[ind])
                    continue
                with open(ImgList + 'word_annotation/' + txtFiles[ind], 'r', encoding='utf-8') as imgf:
                    infos = imgf.readlines()
                recDic = {}
                for info in infos:
                    label = info.split('"')[1].replace(" ", "").replace("`", "'").replace("´", "'")
                    x = int(info.split(',')[0])
                    y = int(info.split(',')[1])
                    w = int(info.split(',')[2])
                    h = int(info.split(',')[3])
                    recDic[label] = [x, y, h, w]
                try:
                    dat = dict(imageName=txtFiles[ind].replace('.txt', '.jpg'),
                               taggedRectangles=recDic,
                               img=base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8'))
                    fout.write(json.dumps(dat))
                    fout.write('\n')
                except:
                    pass

def SCUT_Eng_crop():
    mode = 1

    rootPath = './WorkSpace/Data/_@Models/FL_CRNN/SCUT-Client3/'
    if mode==0:
        tfrecords_file = rootPath+'/SCUT_Eng_word_train.json'
        fileout = rootPath+'/SCUT_Eng_word_train_cropped.json'
    else:
        tfrecords_file = rootPath+'/SCUT_Eng_word_test.json'
        fileout = rootPath + '/SCUT_Eng_word_test_cropped.json'
    with open(tfrecords_file, 'r', encoding='utf-8') as imgf:
        image = imgf.readlines()
    for key, value in json.loads(image[0].strip('\r\n')).items():
        print(key, ':', value)
    image.pop(0)
    labels = []
    cropped_images = []
    imageNames = []
    for ind, i in enumerate(image):
        temp = json.loads(i.strip('\r\n'))
        imageName = temp['imageName']
        IdNumbers = list(temp['taggedRectangles'].keys())
        Boxes = list(temp['taggedRectangles'].values())
        img = temp['img'].encode('utf-8')
        img = cv2.imdecode(np.frombuffer(base64.b64decode(img), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imshow('asd', img)
        # cv2.waitKey(0)
        for index in range(len(IdNumbers)):
            labels.append(IdNumbers[index])
            cropped_images.append(img[Boxes[index][1]:Boxes[index][1]+Boxes[index][2], Boxes[index][0]:Boxes[index][0]+Boxes[index][3]].copy())
            imageNames.append(imageName.split('.')[0]+'_'+str(index)+'.'+imageName.split('.')[1])
    print('Total', len(labels), 'text lines')
    with open(fileout, 'w', encoding='utf-8') as fout:
        dat = dict(ID=fileout.split('.')[0].split('/')[-1],
                   number=len(labels),
                   lex="!$%&'()-./0123456789?@ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz~",
                   structure=['imageName', 'label', 'img'],
                   introduction='cropped small images from SCUT-FORU/SCUT_FORU_DB_Release/English2k/word_img.')
        fout.write(json.dumps(dat))
        fout.write('\n')
        for ind in range(len(labels)):
            img = cropped_images[ind]
            imageName = imageNames[ind]
            label = labels[ind]
            dat = dict(imageName=imageName,
                       label=label,
                       img=base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8'))
            fout.write(json.dumps(dat))
            fout.write('\n')

def IIIT5K():
    ImgList = './Data/IIIT.5K-Words/IIIT5K-Word_V3.0/IIIT5K/'
    for mode in ['test', 'train']:
        ID = 'IIIT5K_' + mode
        print('>>>>> Start to process:', ID)
        fileout = ImgList + ID + '.json'
        subName = 'trainCharBound' if mode=='train' else 'testCharBound'
        matFile = ImgList+subName+'.mat' if mode=='train' else ImgList+subName+'.mat'
        infos = sio.loadmat(matFile)
        infos = infos[subName][0]
        characters=''
        for info in infos:
            label = ''.join(info[1]).replace(" ", "").replace("`", "'").replace("´", "'")
            characters += label
            characters = ''.join(x for i, x in enumerate(characters) if characters.index(x) == i)
        characters = "".join((lambda x: (x.sort(), x)[1])(list(characters)))
        with open(fileout, 'w', encoding='utf-8') as fout:
            dat = dict(ID=ID,
                       number=2000 if mode == 'train' else 3000,
                       lex=characters,
                       structure=['imageName', 'label', 'img'],
                       introduction='IIIT5K-Word_V3.0')
            fout.write(json.dumps(dat))
            fout.write('\n')
            for info in infos:
                imageName = ''.join(info[0]).split('/')[1]
                img = np.array(Image.open(ImgList + ''.join(info[0])))
                label = ''.join(info[1])
                try:
                    dat = dict(imageName=imageName,
                               label=label,
                               img=base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8'))
                    fout.write(json.dumps(dat))
                    fout.write('\n')
                except:
                    pass

def SynthText90K():
    ImgList = './Data/SynthText/'
    for mode in ['train', 'test']:
        ID = 'SynthText90K_' + mode
        print('>>>>> Start to process:', ID)
        fileout = ImgList + ID + '.json'
        with open(fileout, 'w', encoding='utf-8') as fout:
            dat = dict(ID=ID,
                       number=7224612 if mode == 'train' else 891927,
                       lex="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
                       structure=['imageName', 'label', 'img'],
                       introduction='SynthText90K')
            fout.write(json.dumps(dat))
            fout.write('\n')
            with open(ImgList + 'Synth90k-mjsynth/annotation_' + mode + '.txt', 'rb') as f:
                infos = f.readlines()
            number = 0
            for num, info in enumerate(infos):
                if num%10000==0: print(num,'/',len(infos))
                info = info.decode('utf-8')
                imageName = info.split(' ')[0].split('./')[1]
                try:
                    img = np.array(Image.open(ImgList + 'Synth90k-mjsynth/' + imageName))
                    label = imageName.split('_')[1]
                    dat = dict(imageName=imageName,
                               label=label,
                               img=base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8'))
                    fout.write(json.dumps(dat))
                    fout.write('\n')
                    number+=1
                except:
                    print(imageName)
        print("Total:", number)

def SynthText90K_preprocess():
    ImgList = './Data/SynthText/'
    for mode in ['train', 'test']:
        ID = 'preprocessed_SynthText90K_' + mode
        print('>>>>> Start to process:', ID)
        fileout = "./Data/_@Models/FL_CRNN/SynthText90K/" + ID + '.json'
        with open(fileout, 'w', encoding='utf-8') as fout:
            dat = dict(ID=ID,
                       number=7224600 if mode == 'train' else 891924,
                       lex="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
                       structure=['imageName', 'label', 'img', 'input_length'],
                       introduction='preprocessed SynthText90K dataset. Resize to (32,100)')
            fout.write(json.dumps(dat))
            fout.write('\n')
            with open(ImgList + 'Synth90k-mjsynth/annotation_' + mode + '.txt', 'rb') as f:
                infos = f.readlines()
            number = 0
            for num, info in enumerate(infos):
                if num%10000==0: print(num,'/',len(infos))
                info = info.decode('utf-8')
                imageName = info.split(' ')[0].split('./')[1]
                try:
                    label = imageName.split('_')[1]
                    BluredImg = np.array(Image.open(ImgList + 'Synth90k-mjsynth/' + imageName).convert('L'))
                    if len(BluredImg.shape) < 3 or BluredImg.shape[2] == 1:
                        BluredImg = cv2.merge([BluredImg, BluredImg, BluredImg])
                    img1 = cv2.resize(BluredImg,(100,32))
                    inputL = 25
                    dat = dict(imageName=imageName,
                               label=label,
                               img=base64.b64encode(cv2.imencode('.jpg', img1)[1]).decode('utf-8'),
                               input_length=inputL)
                    fout.write(json.dumps(dat))
                    fout.write('\n')
                    number+=1
                except:
                    print(imageName)
        print("Total:", number)

def SynthTest80K():
    def arr2list(x):
        x[x > 1] = 1
        x[x < 0] = 0
        return list(x)
    ImgList = './Data/SynthText/'
    ID = 'SynthText80K'
    print('>>>>> Start to process:', ID)
    infos = sio.loadmat(ImgList+"SynthText/SynthText/gt.mat")
    wordBB = infos["wordBB"][0]
    imgName = infos["imnames"][0]
    text = infos['txt'][0]
    fileout = ImgList + ID + '.json'
    with open(fileout, 'w', encoding='utf-8') as fout:
        dat = dict(ID=ID,
                   number=imgName.size,
                   lex='',
                   structure=['imageName', 'taggedRectangles', 'img'],
                   introduction='SynthText80K; taggedRectangles is a dictionary whose structure is: {"label":[minx,miny,maxx,maxy]}')
        fout.write(json.dumps(dat))
        fout.write('\n')
        for i in range(imgName.size):
            if i%1000==0:
                print("{0}/{1}".format(i, imgName.size))
            image_dir = ImgList+"SynthText/SynthText/" + imgName[i][0]
            img = Image.open(image_dir)
            img_size = img.size
            if len(wordBB[i][0].shape) > 1:
                minx = np.amin(wordBB[i][0], axis=0) / img_size[0]
                miny = np.amin(wordBB[i][1], axis=0) / img_size[1]
                maxx = np.amax(wordBB[i][0], axis=0) / img_size[0]
                maxy = np.amax(wordBB[i][1], axis=0) / img_size[1]
            else:
                minx = [np.amin(wordBB[i][0]) / img_size[0]]
                miny = [np.amin(wordBB[i][1]) / img_size[1]]
                maxx = [np.amax(wordBB[i][0]) / img_size[0]]
                maxy = [np.amax(wordBB[i][1]) / img_size[1]]
            minx = arr2list(np.array(minx))
            miny = arr2list(np.array(miny))
            maxy = arr2list(np.array(maxy))
            maxx = arr2list(np.array(maxx))
            bboxes = [minx, miny, maxx, maxy]
            labels = []
            img = np.array(img)
            for val in text[i]:
                v = [x.encode('ascii').decode('utf-8') for x in re.split("[ \n]", val.strip()) if x]
                labels.extend(v)
            recDic = {}
            for ind in range(len(labels)):
                x1 = int(bboxes[0][ind] * img_size[0])
                y1 = int(bboxes[1][ind] * img_size[1])
                x2 = int(bboxes[2][ind] * img_size[0])
                y2 = int(bboxes[3][ind] * img_size[1])
                recDic[labels[ind]] = [x1,y1,x2,y2]
            #     cv2.rectangle(img,(x1,y1), (x2,y2), (255,0,0), 1)
            #     font = cv2.FONT_HERSHEY_SIMPLEX
            #     cv2.putText(img, str(labels[ind]),(int(bboxes[0][ind]*img_size[0]),int(bboxes[1][ind]*img_size[1])), font, 1, (255, 255, 255), 2)
            # cv2.imshow('asd', img)
            # cv2.waitKey(0)
            dat = dict(imageName=imgName[i][0],
                       taggedRectangles=recDic,
                       img=base64.b64encode(cv2.imencode('.jpg', np.array(img))[1]).decode('utf-8'))
            fout.write(json.dumps(dat))
            fout.write('\n')

    print('asd')

def SynthTest80K_crop():
    rootPath = '.Data/Text/SynthText/'
    tfrecords_file = rootPath + '/SynthText80K.json'
    # fileout = rootPath + '/SynthText80K_20190923_cropped.json'
    # with open(fileout, 'w', encoding='utf-8') as fout:
    with open(tfrecords_file, 'r', encoding='utf-8') as imgf:
        image = imgf.readline()
        for key, value in json.loads(image.strip('\r\n')).items():
            print(key, ':', value)
        image = imgf.readline()
        num = 0
        while image:
            temp = json.loads(image.strip('\r\n'))
            imageName = temp['imageName']
            IdNumbers = list(temp['taggedRectangles'].keys())
            Boxes = list(temp['taggedRectangles'].values())
            img = temp['img'].encode('utf-8')
            img = cv2.imdecode(np.frombuffer(base64.b64decode(img), np.uint8), cv2.IMREAD_COLOR)
            for index in range(len(IdNumbers)):
                try:
                    Img = img[Boxes[index][1]:Boxes[index][3], Boxes[index][0]:Boxes[index][2]].copy()
                    imageName = imageName.split('.')[0]+'_'+str(index)+'.'+imageName.split('.')[1]
                    label = IdNumbers[index]
                    cv2.imshow(label, Img)
                    k=cv2.waitKey(0)
                    if k==27: exit(0)
                    cv2.destroyAllWindows()
                    dat = dict(imageName=imageName,
                               label=label,
                               img=base64.b64encode(cv2.imencode('.jpg', Img)[1]).decode('utf-8'))
                    # fout.write(json.dumps(dat))
                    # fout.write('\n')
                    num += 1
                    if num % 1000 == 0: print("processed: {0}".format(num))
                except:
                    pass
            image = imgf.readline()

def cutoff_symbol():
    # svt_test_20190828_cropped
    train_file_root_path = "./Data/Text/SynthText/"
    char = ''
    with open("./Data/Text/_@Models/FL_CRNN/character_mix_case.txt", encoding='utf-8') as fid:
        for ch in fid.readlines():
            ch = ch.strip('\r\n')
            char += ch
    id_to_char = {j: i for i, j in enumerate(char)}
    for mode in ['train', 'test']:
        # try:
        fileout = train_file_root_path+'/'+mode+"_FL.json"
        # filein = train_file_root_path+'/'+mode+".json"
        filein = train_file_root_path + "/SynthText80K_cropped.json"
        with open(filein, 'r', encoding='utf-8') as imgf:
            image = imgf.readlines()
        # for key, value in json.loads(image[0].strip('\r\n')).items():
        #     print(key, ':', value)
        image.pop(0)
        # with open(fileout, 'w', encoding='utf-8') as fout:
        number = 0
        for i, line in enumerate(image):
            if i%1000==0:
                print("{0}/{1}".format(i, len(image)))
            temp = json.loads(line.strip('\r\n'))
            IdNumber = temp['label']
            if len(IdNumber)>=15:
                continue
            try:
                _ = [id_to_char[j] for j in IdNumber]
            except:
                continue
            else:
                imageName = temp['imageName']
                Img = temp['img'].encode('utf-8')
                Img = cv2.imdecode(np.frombuffer(base64.b64decode(Img), np.uint8), cv2.IMREAD_COLOR)
                cv2.imshow(IdNumber, Img)
                k=cv2.waitKey(0)
                if k==27: exit(0)
                cv2.destroyAllWindows()
                # if len(Img.shape) < 3 or Img.shape[2] == 1:
                #     Img = cv2.merge([Img, Img, Img])
                # dat = dict(imageName=imageName,
                #            label=IdNumber,
                #            img=base64.b64encode(cv2.imencode('.jpg', Img)[1]).decode('utf-8'))
                # fout.write(json.dumps(dat))
                # fout.write('\n')
                number+=1
        print(mode, number)
        # except:
        #     pass

if __name__=='__main__':
    import time
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    cutoff_symbol()