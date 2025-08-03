'''
输入label一定要连续true_labels = [0, 1, 2, 2, 3,3]
混淆矩阵，macro_gmean, micro_gmean1,micro_gmean2, macro_recall,macro_precision,
f1_score,beta值通常为1，代表均衡F1-score计算。
'''

from MEASUREs import *
import pandas as pd
import numpy as np


def calculate_recall_gmean ( true_labels , predicted_labels ) :
	'''
	gmean_score=math.sqrt(macro_recall * macro_recall)
	'''
	num_classes = len ( set ( true_labels ) )  # 类别总数
	recalls = [ ]  # 每个类别的召回率
	precisions = [ ]  # 每个类别的精确率
	p = 1
	for class_label in range ( num_classes ) :
		tp = sum (
			(true == class_label) and (pred == class_label) for true , pred in
			zip ( true_labels , predicted_labels ) )
		fn = sum (
			(true == class_label) and (pred != class_label) for true , pred in
			zip ( true_labels , predicted_labels ) )
		fp = sum (
			(true != class_label) and (pred == class_label) for true , pred in
			zip ( true_labels , predicted_labels ) )

		recall = tp / (tp + fn) if (tp + fn) > 0 else 0
		# precision = tp / (tp + fp) if (tp + fp) > 0 else 0
		p = p * recall  # recalls.append(recall)

	gmean_score = math.sqrt ( p )
	return gmean_score


# 计算整体的几何平均, 新增实验每个类的f1
def calculate_macro_gmean ( true_labels , predicted_labels ) :
	num_classes = len ( set ( true_labels ) )  # 类别总数
	recalls = [ ]  # 每个类别的召回率
	precisions = [ ]  # 每个类别的精确率
	f1_scores = [ ]  # 每个类别的F1分数
	for class_label in range ( num_classes ) :
		tp = sum (
			(true == class_label) and (pred == class_label) for true , pred in
			zip ( true_labels , predicted_labels ) )
		fn = sum (
			(true == class_label) and (pred != class_label) for true , pred in
			zip ( true_labels , predicted_labels ) )
		fp = sum (
			(true != class_label) and (pred == class_label) for true , pred in
			zip ( true_labels , predicted_labels ) )

		recall = tp / (tp + fn) if (tp + fn) > 0 else 0
		precision = tp / (tp + fp) if (tp + fp) > 0 else 0
		# 计算当前类别的F1分数
		if (precision + recall) > 0 :
			f1 = 2 * (precision * recall) / (precision + recall)
		else :
			f1 = 0
		recalls.append ( recall )
		precisions.append ( precision )
		f1_scores.append ( f1 )  # 存储当前类别的F1分数

	pre_recall = recalls
	macro_recall = sum ( recalls ) / num_classes
	macro_precision = sum ( precisions ) / num_classes

	# gmean_score = scipy.stats.gmean([macro_recall, macro_precision])
	gmean_score = math.sqrt ( macro_recall * macro_precision )
	return pre_recall , macro_recall , macro_precision , gmean_score, f1_scores




# 计算每个类的gmean(recalls*precisions)，再求和
def calculate_micro_gmean1 ( true_labels , predicted_labels ) :
	num_classes = len ( set ( true_labels ) )  # 类别总数
	recalls = [ ]  # 每个类别的召回率
	precisions = [ ]  # 每个类别的精确率
	gmeans = [ ]
	for class_label in range ( num_classes ) :
		tp = sum (
			(true == class_label) and (pred == class_label) for true , pred in
			zip ( true_labels , predicted_labels ) )
		fn = sum (
			(true == class_label) and (pred != class_label) for true , pred in
			zip ( true_labels , predicted_labels ) )
		fp = sum (
			(true != class_label) and (pred == class_label) for true , pred in
			zip ( true_labels , predicted_labels ) )

		recall = tp / (tp + fn) if (tp + fn) > 0 else 0
		precision = tp / (tp + fp) if (tp + fp) > 0 else 0

		recalls.append ( recall )
		precisions.append ( precision )
		# gmeans.append(scipy.stats.gmean([recall, precision]))
		gmeans.append ( math.sqrt ( recall * precision ) )

	macro_recall = sum ( recalls ) / num_classes
	macro_precision = sum ( precisions ) / num_classes
	total_gmean = sum ( gmeans ) / len ( gmeans )

	return macro_recall , macro_precision , total_gmean  # ,gmeans


# 计算每个类的gmean(recalls*specificities)，再求和
def calculate_micro_gmean2 ( true_labels , predicted_labels ) :
	num_classes = len ( set ( true_labels ) )  # 类别总数
	recalls = [ ]  # 每个类别的召回率
	specificities = [ ]  # 每个类别的specificity
	gmeans = [ ]
	for class_label in range ( num_classes ) :
		tp = sum (
			(true == class_label) and (pred == class_label) for true , pred in
			zip ( true_labels , predicted_labels ) )
		fn = sum (
			(true == class_label) and (pred != class_label) for true , pred in
			zip ( true_labels , predicted_labels ) )
		fp = sum (
			(true != class_label) and (pred == class_label) for true , pred in
			zip ( true_labels , predicted_labels ) )
		tn = sum (
			(true != class_label) and (pred != class_label) for true , pred in
			zip ( true_labels , predicted_labels ) )

		recall = tp / (tp + fn) if (tp + fn) > 0 else 0
		specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

		recalls.append ( recall )
		specificities.append ( specificity )
		# gmeans.append(scipy.stats.gmean([recall, specificity]))
		gmeans.append ( math.sqrt ( recall * specificity ) )

	macro_recall = sum ( recalls ) / num_classes
	total_specificity = sum ( specificities ) / num_classes
	total_gmean = sum ( gmeans ) / len ( gmeans )

	return macro_recall , total_specificity , total_gmean  # , gmeans


def calculate_confusion_matrix ( true_labels , predicted_labels ) :
	# 确定类别数
	num_classes = max ( max ( true_labels ) , max ( predicted_labels ) ) + 1

	# 初始化混淆矩阵
	confusion_matrix = np.zeros ( (num_classes , num_classes) )

	# 填充混淆矩阵
	for true_label , predicted_label in zip ( true_labels , predicted_labels ) :
		confusion_matrix [ true_label ] [ predicted_label ] += 1

	return confusion_matrix


def calculate_macro_recall ( true_labels , predicted_labels ) :
	'''
	（召回率，也称为灵敏度或真正例率）：
	计算方法是指定类别的真正例数（True Positives，即模型正确预测为该类别的样本数）
	与该类别所有样本数之比，即 Recall = TP / (TP + FN)。
	'''
	unique_labels = set ( true_labels )  # 获取唯一的标签
	total_recall = 0
	label_count = 0

	for label in unique_labels :
		label_indices = [ i for i in range ( len ( true_labels ) ) if
						  true_labels [ i ] == label ]
		label_true = [ true_labels [ i ] for i in label_indices ]
		label_predicted = [ predicted_labels [ i ] for i in label_indices ]

		label_tp = sum (
			1 for true , pred in zip ( label_true , label_predicted ) if
			true == pred )
		label_fn = len ( label_true ) - label_tp

		if label_tp + label_fn == 0 :
			recall = 0
		else :
			recall = label_tp / (label_tp + label_fn)

		total_recall += recall
		label_count += 1

	macro_recall = total_recall / label_count
	return macro_recall


def calculate_macro_precision ( true_labels , predicted_labels ) :
	def calculate_precision ( tp , fp ) :
		if tp + fp == 0 :
			precision = 0
		else :
			precision = tp / (tp + fp)
		return precision

	label_counts = Counter ( true_labels )
	unique_labels = sorted ( set ( true_labels ) )
	precision_sum = 0

	for label in unique_labels :
		tp = sum (
			1 for true , pred in zip ( true_labels , predicted_labels ) if
			true == pred == label )
		fp = sum (
			1 for true , pred in zip ( true_labels , predicted_labels ) if
			true != label and pred == label )

		precision = calculate_precision ( tp , fp )
		precision_sum += precision

	macro_precision = precision_sum / len ( unique_labels ) if len (
		unique_labels ) != 0 else 0
	return macro_precision


'''
def calculate_f1_score(true_labels, predicted_labels):
    # Get unique class labels
    unique_labels = np.unique(true_labels + predicted_labels)

    # Initialize variables
    num_classes = len(unique_labels)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    precision = np.zeros(num_classes, dtype=np.float64)
    recall = np.zeros(num_classes, dtype=np.float64)

    # Populate confusion matrix
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        print(unique_labels)
        print(true_label)
        #print(np.where(unique_labels == true_label)[0])
        if true_label not in unique_labels or predicted_label not in unique_labels:
	        continue
        true_index = np.where(unique_labels == true_label)[0][0]
        predicted_index = np.where(unique_labels == predicted_label)[0][0]
        confusion_matrix[true_index][predicted_index] += 1

    # Calculate precision, recall, and F1 score for each class
    for class_index in range(num_classes):
        true_positives = confusion_matrix[class_index][class_index]
        false_positives = np.sum(confusion_matrix[:, class_index]) - true_positives
        false_negatives = np.sum(confusion_matrix[class_index, :]) - true_positives

        precision[class_index] = true_positives / (true_positives + false_positives + 1e-10)
        recall[class_index] = true_positives / (true_positives + false_negatives + 1e-10)

    # Calculate F1 score for each class
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    # Calculate macro-average F1 score
    macro_f1_score = np.mean(f1_scores)

    return macro_f1_score
'''


def MAvA ( y_test , y_pre ) :
	y_r_dict , t = Group ( y_pre , y_test )
	p = 0
	for i in y_r_dict :
		r = 0
		for j in y_r_dict [ i ] :
			if (i == j) :
				r += 1
		r /= len ( y_r_dict [ i ] )
		p += r
	p /= len ( y_r_dict )
	return p


if __name__ == '__main__' :
	# 真实标签与预测标签
	true_labels = [ 0 , 1 , 2 , 2 , 3 , 3 ]
	predicted_labels = [ 0 , 0 , 2 , 2 , 1 , 3 ]
	# 计算混淆矩阵
	cm = calculate_confusion_matrix ( true_labels , predicted_labels )
	# 将混淆矩阵展示为DataFrame
	confusion_matrix = pd.DataFrame ( cm , index = range ( cm.shape [ 0 ] ) ,
									  columns = range ( cm.shape [ 1 ] ) )
	print ( "混淆矩阵：" )
	print ( confusion_matrix )

	r = calculate_macro_recall ( true_labels , predicted_labels )
	p = calculate_macro_precision ( true_labels , predicted_labels )
	print ( 'recall、precision 和整体gmean ：' , r , p , math.sqrt ( r * p ) )

	# a,b,c = calculate_macro_gmean(true_labels, predicted_labels)
	a , b , c = calculate_micro_gmean2 ( true_labels , predicted_labels )
	print ( 'recall、specificities 和分类计算gmean2 ：' , a , b , c )
	e , f , g = calculate_micro_gmean1 ( true_labels , predicted_labels )
	print ( 'recall、precision 和分类计算gmean1 ：' , e , f , g )

'''
#"宏平均
def calculate_gmean_recall_precision(true_labels, predicted_labels):
	# Get unique class labels
	unique_labels = np.unique(true_labels + predicted_labels)
	
	# Initialize variables
	num_classes = len(unique_labels)
	confusion_matrix = np.zeros((num_classes, num_classes), dtype = np.int32)
	precision = np.zeros(num_classes, dtype = np.float64)
	recall = np.zeros(num_classes, dtype = np.float64)
	gmean = np.zeros(num_classes, dtype = np.float64)
	
	# Populate confusion matrix
	for true_label, predicted_label in zip(true_labels, predicted_labels):
		true_index = np.where(unique_labels == true_label)[0][0]
		predicted_index = np.where(unique_labels == predicted_label)[0][0]
		confusion_matrix[true_index][predicted_index] += 1
	
	# Calculate recall, precision, and gmean for each class
	for class_index in range(num_classes):
		true_positives = confusion_matrix[class_index][class_index]
		false_positives = np.sum(
			confusion_matrix[:, class_index]) - true_positives
		false_negatives = np.sum(
			confusion_matrix[class_index, :]) - true_positives
		
		precision[class_index] = true_positives / (
				true_positives + false_positives + 1e-10)
		recall[class_index] = true_positives / (
				true_positives + false_negatives + 1e-10)
		
		# Calculate gmean for current class
		gmean[class_index] = np.sqrt(
			recall[class_index] * precision[class_index])
	
	# Calculate average gmean
	average_gmean = np.mean(gmean)
	
	return gmean, average_gmean
#调包
def bao_gmean(true_labels, predicted_labels):
	# 计算宏平均召回率和精确率
	macro_recall = recall_score(true_labels, predicted_labels,
	                            average = 'macro')
	macro_precision = precision_score(true_labels, predicted_labels,
	                                  average = 'macro')
	
	# 计算整体的几何平均
	gmean = np.sqrt(macro_recall * macro_precision)
	
	return macro_recall, macro_precision, gmean
'''
