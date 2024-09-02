import numpy as np
import operator
import pandas as pd

class Data_Process():
    def __init__(self,data_addr):
        self.data = pd.read_excel(data_addr, "Sheet1", usecols=range(14))
        #修改索引名称
        self.data.columns={'CryoSleep':self.data.columns[0],'Age':self.data.columns[1],'VIP':self.data.columns[2],'RoomService':self.data.columns[3],'FoodCourt':self.data.columns[4],
              'ShoppingMall':self.data.columns[5],'Spa':self.data.columns[6],'VRDeck':self.data.columns[7],'Cabin_num':self.data.columns[8],'Side':self.data.columns[9],
                'HomePlanet_num':self.data.columns[10],'Destination_num':self.data.columns[11],'Deck_num':self.data.columns[12],'Transported':self.data.columns[13]	}
        #数据去重
        self.data.drop_duplicates()

        #存储归一化后的参数
        self.new_data=[[0.0 for _ in range(self.data.shape[1])] for __ in range(self.data.shape[0])]
        #new_data[i]即为第i个数据的x,y,z

        i=0
        while i<self.data.shape[1]-1:
            arr=self.data.iloc[:,i]
            j=0
            while j<self.data.shape[0]:
                self.new_data[j][i]=arr[j]
                j=j+1
            i=i+1

        i=0
        while i<self.data.shape[0]:
            self.new_data[i][self.data.shape[1]-1]=self.data.iloc[i,self.data.shape[1]-1]
            i=i+1

    #这两个是返回数据规模的函数
    #get_data_num()返回样本数量（未区分训练集和测试集）
    #get_column_num()返回x和y的总个数，x参数个数=get_column_num()-1, y参数个数=1
    def get_data_num(self):
        return self.data.shape[0]
    def get_column_num(self):
        return self.data.shape[1]

    #注意以下三个函数形参范围是1:get_data_num() , 不是从0开始！
    #以下函数返回的列表是原来的拷贝，可以放心修改
    #这个函数用来返回原始数据，包括x和y，一维列表
    def get_ori_data(self,num):
        return np.array(self.data.iloc[num-1])

    #这个函数用来返回原始数据的x，一维列表
    def get_ori_x(self,num):
        return np.array(self.data.iloc[num-1,0:self.data.shape[1]-1])

    #这个函数用来返回原始数据的y
    def get_ori_y(self,num):
        return self.data.iloc[num-1,self.data.shape[1]-1]

    #注意以下三个函数形参范围是1:get_data_num() , 不是从0开始！
    #以下两个函数返回的列表是原来的拷贝，可以放心修改
    #这个函数用来返回归一化后数据，包括x和y，一维列表
    def get_normal_data(self,num):
        return np.array(self.new_data[num-1])

    #这个函数用来返回归一化后数据的x，一维列表
    def get_normal_x(self,num):
        return np.array(self.new_data[num-1][0:self.get_column_num()-1])

    #这个函数用来返回归一化后数据的y
    def get_normal_y(self,num):
        return self.get_ori_y(num)

    #这个函数用来给一个数据进行归一化，传入值是一个一维列表，请和原数据data中的x长度保持一致！
    def normalization(self,x):
        tem_arr=[0.0 for _ in range(self.get_column_num()-1)]
        i=0
        while i<self.get_column_num()-1:
            tem_arr[i]=(x[i]-self.mmm[i][0])/(self.mmm[i][1]-self.mmm[i][2])
            i=i+1
        return tem_arr

    #注意这个函数形参index_pos范围是1:get_column_num()-1 , 不是从0开始！
    #这个函数用来给一个数据进行归一化，传入值是一个需要归一化的数据，和其在x中的位置
    def normalization_single(self,x_num,index_pos):
        return (x_num-self.mmm[index_pos-1][0])/(self.mmm[index_pos-1][1]-self.mmm[index_pos-1][2])

    #获取大量归一化后的x，返回一个矩阵(np.array)，start_pos和end_pos范围是从1:get_data_num(),注意左闭右开，所以全取是(1,get_data_num()+1)
    def get_x_in_array(self,start_pos,end_pos):
        return np.array(self.new_data)[start_pos-1:end_pos-1,0:self.get_column_num()-1]

    #获取大量y，返回一个矩阵(np.array)，start_pos和end_pos范围是从1:get_data_num(),注意左闭右开，所以全取是(1,get_data_num()+1)
    def get_y_in_array(self,start_pos,end_pos):
        return np.array(self.new_data)[start_pos-1:end_pos-1,self.get_column_num()-1].astype(int)




class DecisionTree():
    def __init__(self, criterion='gini', total_Features=None, id_to_Feature=None, if_plot=True):
        self.criteriion = criterion
        self.Features_list = total_Features
        self.id_to_Feature = id_to_Feature
        self.label_to_id = None
        if id_to_Feature is not None:
            self.label_to_id = {id_to_Feature['label'][l_id]: l_id for l_id in id_to_Feature['label']}
        self.if_plot = if_plot
        self.tree = None
        self.Feature_type = []
        self.epsilon = 1e-7
        self.Features_pairs = None

    def fit(self, x, y):

        if not self.Features_list:
            self.Features_list = [('Feature_%d' % i) for i in range(x.shape[1])]
        if not self.id_to_Feature:
            id_to_Feature = {}
            for j in range(x.shape[1]):
                id_to_Featurevalue = {}
                Feature_name = self.Features_list[j]
                unique_Feature_value = set(list(x[:, j]))
                for v in unique_Feature_value:
                    id_to_Featurevalue[v] = 'value_{}'.format(v)
                id_to_Feature[Feature_name] = id_to_Featurevalue
            label = {}
            unique_label = set(list(y.reshape(-1)))
            for label in unique_label:
                label[label] = 'label_{}'.format(label)
            id_to_Feature['label'] = label
            self.id_to_Feature = id_to_Feature
            self.label_to_id = {id_to_Feature['label'][l_id]: l_id for l_id in id_to_Feature['label']}

        for col in range(x.shape[1]):
            self.check_data_type(x[:, col])
            
        self.Features_pairs = [(Feature, self.Feature_type[i]) for i, Feature in enumerate(self.Features_list)]
        Features_pairs = self.Features_pairs[:]
        y = y.reshape(-1, 1)
        data = np.concatenate([x, y], axis=1)

        self.tree = self.createTree(data, Features_pairs,criterion=self.criteriion,id_to_Feature=self.id_to_Feature)

    def predict(self, test_x):
        if len(test_x.shape) == 1:
            test_x = test_x.reshape(-1, len(self.Features_list))
        pred = []
        for i in range(test_x.shape[0]):
            test_vec = test_x[i]
            pred_label = self.predict_(test_vec, self.tree)
            pred.append(self.label_to_id[pred_label])
        return np.array(pred)

    def predict_(self, test_vec, tree=None):
        Feature_name = next(iter(tree))
        Feature_values = tree[Feature_name]
        FeatureIndex = self.Features_list.index(Feature_name)
        classLabel = None
        Feature_index = self.Features_list.index(Feature_name)
        if self.Feature_type[Feature_index] == 'int':
            cur_value = int(test_vec[FeatureIndex])
            for key in Feature_values.keys():
                if self.id_to_Feature[Feature_name][cur_value] == key:
                    if type(Feature_values[key]).__name__ == 'dict':
                        classLabel = self.predict_(test_vec, Feature_values[key])
                    else:
                        classLabel = Feature_values[key]
            if not classLabel:
                key=list(Feature_values.keys())[0]
                if type(Feature_values[key]).__name__ == 'dict':
                    classLabel = self.predict_(test_vec, Feature_values[key])
                else:
                    classLabel = Feature_values[key]

        elif self.Feature_type[Feature_index] == 'float':
            cur_value = test_vec[FeatureIndex]
            for divide_str in Feature_values.keys():
                oper_str = divide_str[0]
                if oper_str == '<' and cur_value > float(divide_str[-5:]):
                    continue
                if oper_str == '>' and cur_value <= float(divide_str[-5:]):
                    continue
                if type(Feature_values[divide_str]).__name__ == 'dict':
                    classLabel = self.predict_(test_vec, Feature_values[divide_str])
                else:
                    classLabel = Feature_values[divide_str]

        return classLabel

    def createTree(self, dataSet, candFeatures, fatherClass=None, criterion='gini', id_to_Feature=None):

        classList = dataSet[:, -1]
        if (classList == classList[0]).sum() == len(classList):
            return id_to_Feature['label'][classList[0]]
        if len(candFeatures) == 0 or self.isSameData(dataSet):
            major_target = self.majorityCnt(classList)
            return id_to_Feature['label'][major_target] 
        if dataSet.size == 0:
            father_major_target = self.majorityCnt(fatherClass)
            return id_to_Feature['label'][father_major_target]

        bestFeatureType, bestFeatureId, bestValue = self.chooseBestFeatureureByGini(dataSet, candFeatures)
        
        bestFeatureLabel = candFeatures[bestFeatureId][0]
        myTree = {bestFeatureLabel: {}}
        #离散数据
        if bestFeatureType == 'int':
            del (candFeatures[bestFeatureId])
            FeatureValues = dataSet[:, bestFeatureId]
            uniqueVals = set(FeatureValues)
            for value in uniqueVals:
                subCandFeatures = candFeatures[:]
                Feature_value_name = id_to_Feature[bestFeatureLabel][value]
                subData = self.splitDataSetDiscrete(dataSet, bestFeatureId, value)
                myTree[bestFeatureLabel][Feature_value_name] = self.createTree(subData, subCandFeatures,fatherClass=classList,criterion=criterion,id_to_Feature=id_to_Feature)
        #连续数据
        elif bestFeatureType == 'float':
            divideEdges = ['<=', '>']
            for idx, subData in enumerate(self.splitDataSetContinuous(dataSet, bestFeatureId, bestValue)):
                subCandFeatures = candFeatures[:]  # copy
                Feature_value_name = '%s%.3f' % (divideEdges[idx], bestValue)
                myTree[bestFeatureLabel][Feature_value_name] = self.createTree(subData, subCandFeatures,fatherClass=classList,criterion=criterion,id_to_Feature=id_to_Feature)
        

        return myTree


    def calcGini(self, dataSet):
        numEntires = len(dataSet)
        labelCounts = {}
        for FeatureVec in dataSet:
            currentLabel = FeatureVec[-1]
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        gini = 1
        for key in labelCounts:
            prob = float(labelCounts[key]) / numEntires
            gini -= prob ** 2
        return gini
    
    def splitDataSetDiscrete(self, dataSet, axis, value):
        retDataSet = []
        for FeatureVec in dataSet:
            if FeatureVec[axis] == value:
                reducedFeatureVec = FeatureVec[:axis]
                reducedFeatureVec = np.concatenate([reducedFeatureVec, FeatureVec[axis + 1:]], axis=0)
                retDataSet.append(reducedFeatureVec)
        retDataSet = np.array(retDataSet)
        return retDataSet

    def splitDataSetContinuous(self, dataSet, axis, value):
        splitDataSets = []
        index_less = dataSet[:, axis] <= value
        splitDataSets.append(dataSet[index_less])
        index_great = dataSet[:, axis] > value
        splitDataSets.append(dataSet[index_great])
        return splitDataSets
    
    def chooseBestFeatureureByGini(self, dataSet, candFeatures):
        
        numFeatureures = len(dataSet[0]) - 1  
        minGini = float('inf')
        bestFeatureureId = None  
        bestDivideValue = None
        for i in range(numFeatureures):
            Feature_type = candFeatures[i][1]
            FeatureList = dataSet[:, i] 
            uniqueVals = set(FeatureList)
            if Feature_type == 'int':
                gini = 0.0
                for value in uniqueVals:
                    subDataSet = self.splitDataSetDiscrete(dataSet, i, value)  # split data
                    prob = len(subDataSet) / float(len(dataSet))
                    gini += prob * self.calcGini(subDataSet)
                if gini < minGini:
                    minGini = gini
                    bestFeatureureId = i
            else:
                sortedUniquevals = sorted(list(uniqueVals))
                for idx in range(len(uniqueVals) - 1):
                    gini = 0.0
                    divide_value = (sortedUniquevals[idx] + sortedUniquevals[idx + 1]) / 2
                    for subData in self.splitDataSetContinuous(dataSet, i, divide_value):
                        prob = len(subData) / float(len(dataSet))
                        gini += prob * self.calcGini(subData)
                    if gini < minGini:
                        minGini = gini
                        bestFeatureureId = i
                        bestDivideValue = divide_value
        return candFeatures[bestFeatureureId][1], bestFeatureureId, bestDivideValue


    def majorityCnt(self, classList):
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

        return sortedClassCount[0][0]

    def isSameData(self, dataSet):
        data_num = len(dataSet)
        for i in range(1, data_num):
            if np.any(dataSet[i - 1, :-1] != dataSet[i, :-1]):
                return False
        return True

    def check_data_type(self, Feature_values):
        Feature_values_int = Feature_values.astype('int')
        dis = abs(Feature_values_int.astype('float64') - Feature_values).sum()
        if dis < self.epsilon:
            self.Feature_type.append('int')
            return 'int'
        self.Feature_type.append('float')
        return 'float'

        


if __name__ == '__main__':
    DP=Data_Process("train.xls")
    x=DP.get_x_in_array(1,DP.get_data_num()+1)
    y=DP.get_y_in_array(1,DP.get_data_num()+1).astype('int')
    total_Features=['CryoSleep','Age','VIP','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Cabin_num','Side','HomePlanet_num','Destination_num','Deck_num']
    id_to_Feature={
        'VIP':{0:'0',1:'1'},
        'Side':{0:'0',1:'1'},
        'Destination_num':{0:'0',1:'1',2:'2'},
        'label': {0: '0', 1: '1'}
    }

        
    train_test_prop=0.78
    x_train = x[:int(len(x) * train_test_prop)]
    x_test = x[int(len(x) * train_test_prop):]
    y_train = y[:int(len(y) * train_test_prop)]
    y_test = y[int(len(y) * train_test_prop):]
    

    #'''
    model = DecisionTree(criterion='gini', total_Features=total_Features, id_to_Feature=id_to_Feature)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    
    #print(pred)
    
	#实际值为0
    common_0=0
    different_0=0
    common_1=0
    different_1=0
    
    # 遍历数组并比对相同索引的元素
    for i in range(len(pred)):
        if pred[i] == y_test[i]:
            if 1==y_test[i]: common_1 += 1
            else: common_0 += 1
        else:
            if 1==y_test[i]: different_1 += 1
            else: different_0 += 1

    # 输出结果
    print("预测值\\实际值:     0     1")
    print("0            :{:6}{:6}".format(common_0,different_0))
    print("1            :{:6}{:6}".format(different_1,common_1))
    print("正确比例=",(common_0+common_1)/len(pred))

	

