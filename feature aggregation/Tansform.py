import json
import math

class Selection(object):
    def __init__(self,data,start_id=300):
        """
        :param data: 形式为[[]]，每个外层list包含所有 plabel内层list为6维，[query_id，gallery_id1,gallery_id2,gallery_id3,gallery_id4,gallery_id5...]
        :return :形式为一个id到category的映射，存为json格式
        """
        self.data = data
        self.start_id = start_id
        self.selected_set = []

        pass
    def select(self,percent):
        idTocategory = {}
        # for item in self.data[:len(self.data)*0.7]:
        #     self.selected_set = self.selected_set + item
        # self.selected_set = set(self.selected_set)
        #print(len(self.data))
        for item in self.data[:int(1505*percent)]:
            #print(item)
            if item[0] not in self.selected_set:
                #检查是否单类别数量大于等于4张
                flag_count = 0

                for idx in item[1:6]:
                    if idx not in self.selected_set:
                        flag_count += 1
                #赋予合格图片类别id
                if flag_count >=4:
                    for idx in item[0:6]:
                        if idx not in self.selected_set:
                            idTocategory[idx] = self.start_id
                            self.selected_set.append(idx)
                    self.start_id += 1
                else:
                    continue
            else:
                continue
        # json.dump(idTocategory,('test.json', 'w'))
        with open("test.json", "w") as f:
            json.dump(idTocategory, f)
        print(self.start_id)
        return idTocategory

    # def save_result(self):
    #     with open(epoch_json + '/' + title + '.json', 'w', encoding='utf-8') as json_file:
    #         json.dump(dd, json_file, ensure_ascii=False)
    #     print('json finished')

def convert(result,data):
    new_retrun_dict = {}
    for item in result.keys():
        new_retrun_dict[str(data[item]).zfill(6)+".jpg"] = result[item]
    with open('nameToCate.json', "w") as f:
        json.dump(new_retrun_dict, f)
    return new_retrun_dict


if __name__ == '__main__':
    data_all = json.load(open('metric_label\metric_label\plabel.json'))
    data = data_all['all']

    solver = Selection(data)
    result = solver.select(0.3)
    print(len(result))

    data_query = json.load(open('metric_label\metric_label\q.json'))
    data_q = data_query['q']
    new_dict= convert(result,data_q)
    print(len(new_dict))
