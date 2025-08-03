
from collections import Counter
import sklearn as sk
import math


def mul_F1_Auc_Gmean(y_test, result,):
	'''
	acc_i = (TP+TN)/(TP+TN+FP+FN)
	ACC= np.sum(acc_i)/set(y)
	准确率等于微F1值，同时等于微GM1。
	Accuracy = micro precision = micro recall = micro F1-score
	macro：从类计算
	MacroR=(1/n)(R1+R2+R3+....+Rn)
	MicroR=（TP1+TP2+TP3+....TPn）/(（TP1+TP2+TP3+....TPn）+（FN1+FN2+FN3+....FNn）)
	'''
	f1_score = sk.metrics.f1_score(y_test, result, average = 'macro')
	
	# 计算gmean
	
	recall = sk.metrics.recall_score(y_test, result,
	                                 average = 'macro')  # 是先对每一个类统计指标值，然后在对所有类求算术平均值
	precision = sk.metrics.precision_score(y_test, result, average = 'macro')
	# 计算某一个类别的precision和recall，则在评价函数中加上这个参数：pos_label = [4]，这里的4表示索引的第4列
	
	gmean_value = math.sqrt(recall * precision)
	
	return f1_score, gmean_value, recall, precision


def minorityRecall(y, y_pred):
	'''
	计算少数类recall
	:return:
	'''
	# maxlabel = max(Counter(y), key = Counter(y).get)
	classlist = sorted(Counter(y).items(), key = lambda x: x[1])
	maxlabel = classlist[-1][0]
	# print('主类是：',maxlabel,Counter(y),len(y),len(y_pred))
	pos_label = y[y != maxlabel]
	pos_label = set(pos_label)
	recall = sk.metrics.recall_score(y, y_pred, average = 'macro',
	                                 pos_label = pos_label)
	return recall


def MAvA(y_test, y_pre):
	y_r_dict, t = Group(y_pre, y_test)
	p = 0
	for i in y_r_dict:
		r = 0
		for j in y_r_dict[i]:
			if (i == j):
				r += 1
		r /= len(y_r_dict[i])
		p += r
	p /= len(y_r_dict)
	return p


def Group(X, y, w = []):
	'''将X根据y的类别分组，返回按y的label存有X和y的字典'''
	if len(w) == 0:
		X_dict = {}
		y_dict = {}
		for a in (set(y)):
			X_dict[a] = []
			y_dict[a] = []
		for i in range(len(y)):
			X_dict[y[i]].append(X[i])
			y_dict[y[i]].append(y[i])
		return X_dict, y_dict
	else:
		X_dict = {}
		y_dict = {}
		w_dict = {}
		for a in (set(y)):
			X_dict[a] = []
			y_dict[a] = []
			w_dict[a] = []
		for i in range(len(y)):
			X_dict[y[i]].append(X[i])
			y_dict[y[i]].append(y[i])
			w_dict[y[i]].append(w[i])
		return X_dict, y_dict, w_dict


'''
labelList = ['../../Data/eco_label', '../../Data/hay_label', '../../Data/lym_label','../../Data/new_label',
             '../../Data/pag_label', '../../Data/pen_label',
             '../../Data/sat_label', '../../Data/shu_label', '../../Data/tae_label', '../../Data/yea_label', '../../Data/zoo_label',]
featureList = [ '../../Data/eco_feature', '../../Data/hay_feature','../../Data/lym_feature',
'../../Data/new_feature','../../Data/pag_feature', '../../Data/pen_feature', '../../Data/sat_feature', '../../Data/shu_feature', '../../Data/tae_feature', '../../Data/yea_feature', '../../Data/zoo_feature']




labelList = ['../../Data/wqw_label', '../../Data/wqr_label', ]
featureList = [ '../../Data/wqw_feature', '../../Data/wqr_feature',]



labelList = ['../../Data/aut_label', '../../Data/bal_label', '../../Data/car_label','../../Data/cle_label',
             '../../Data/der_label', '../../Data/fla_label',
             '../../Data/gla_label', '../../Data/thy_label', '../../Data/veh_label',]
featureList =['../../Data/aut_feature', '../../Data/bal_feature','../../Data/car_feature',
'../../Data/cle_feature','../../Data/der_feature', '../../Data/fla_feature', '../../Data/gla_feature', '../../Data/thy_feature', '../../Data/veh_feature',]





'''

featureList = ['aut_feature','bal_feature','car_feature','cle_feature','der_feature','eco_feature','fla_feature','gla_feature','hay_feature','lym_feature','new_feature','pag_feature',
'pageblocks_feature','pen_feature','sat_feature','seg_feature','shu_feature','tae_feature','thy_feature','veh_feature','wqr_feature','wqw_feature','yea_feature','zoo_feature',]

labelList  = [ 'aut_label','bal_label','car_label','cle_label','der_label','eco_label','fla_label','gla_label','hay_label','lym_label','new_label','pag_label',
'pageblocks_label','pen_label','sat_label','seg_label','shu_label','tae_label','thy_label','veh_label','wqr_label','wqw_label','yea_label','zoo_label',]
