import gc, json

def get_rid_of_the_models(model=None):
    if model is not None:
        del model
    gc.collect()

def save_args_as_json(FLconfig,path):
    with open(str(path), "w") as f:
        json.dump(FLconfig, f,indent=4)

def fast_ctc_decode(char_num,ind):
    def MaxProbability(row, MaxIndex, curPro):
        maxPro = curPro
        maxRow = row
        if row != 0:
            num = 0
            while 1:
                num += 1
                LaMaxIndex = char_num[ind, row - num, :].tolist().index(max(char_num[ind, row - num, :].tolist())) # go up to find max probability in row-num
                if MaxIndex != LaMaxIndex: break # if find another label, break
                if MaxIndex==LaMaxIndex and max(char_num[ind, row-num, :].tolist())>maxPro: # upgrade max probability and its row
                    maxPro = max(char_num[ind, row-num, :].tolist())
                    maxRow = row-num
        return maxPro, maxRow
    ResultList = []
    for row in range(0, char_num[ind, :, :].shape[0]):
        MaxIndex = char_num[ind, row, :].tolist().index(max(char_num[ind, row, :].tolist())) # max probability index in current row
        if row == char_num[ind, :, :].shape[0] - 1: # last row
            if MaxIndex != (char_num[ind, :, :].shape[1]-1): # not blank label
                maxPro, maxRow = MaxProbability(row, MaxIndex, max(char_num[ind, row, :].tolist())) # find max probability and its row
                ResultList.append([MaxIndex, maxPro, maxRow])
            continue
        NeMaxIndex = char_num[ind, row + 1, :].tolist().index(max(char_num[ind, row + 1, :].tolist())) # next row
        if NeMaxIndex != MaxIndex and MaxIndex != (char_num[ind, :, :].shape[1]-1): # current lable not equals to next lable and neither a blank
            maxPro, maxRow = MaxProbability(row, MaxIndex, max(char_num[ind, row, :].tolist()))
            ResultList.append([MaxIndex, maxPro, maxRow])
    return ResultList