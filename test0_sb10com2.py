'''
5次结果 求平均值

'''
import csv
import pandas as pd
import MultiPMeasurs
import numpy as np
import sklearn.metrics as sk


def method(clf, name):
	fileList =['eco','shu','yea',]

	with open(
		'Comparison/{}/{}.csv'.format(name, clf),
		'a') as f:  # 方法(记得同步修改写入文件名称)
		csv_writer = csv.writer(f)
		csv_writer.writerow(
			['Data', "F1", 'GMEAN', 'Gmean1', 'Gmean2', 'Mava', 'MAUC',
			 'Precision', 'Specificity', 'Recall', 'preclass_recall',
			 'f1_pre_class'\
			 ])
		for fn in fileList:
			F1 = []
			Recall = []
			Precision = []
			Specificity = []
			GMEAN = []
			Gmean1 = []
			Gmean2 = []
			Mava = []
			MAUC = []
			Recall_pre = []
			Pre_class_f1=[]
			for z in range(1, 6):
				# print(z)
				# for j in range(1,6):
				for i in range(1, 6):
					# print('i=',i)
					df_pred = pd.read_csv(
						'{}_TestData/{}Result/{}_{}/{}/{}_label.csv'.format(
							z, name, z, i, clf, fn),)
					prd = df_pred['0']  # 预测值
					# print('预测值',prd)
					# print('{}_TestData/GaSMOTEBoost2_Result/{}_{}/{}_sigma={}/{}_label.csv'.format(z, z, i,clf,k,fn))
					print(
						'{}_TestData/{}Result/{}_{}/{}/{}_label.csv'.format(
							z, name, z, i, clf, fn))
					prdpro = df_pred.iloc[:, 1:]  # 预测概率
					df_true = pd.read_csv(
						'{}_TestData/Test_{}/{}_label'.format(z, i, fn),
						header = None)
					label = df_true.iloc[:, 0]  # 真实标签
					# print(label)
					# print('{}_TestData/Result/{}_{}/{}_sigma={}/{}_label.csv'.format(z, z, i,clf,k,fn) )
					recall1, precision, gmean1 = MultiPMeasurs.calculate_micro_gmean1(
						label, prd)  # recall1,recall2，macro_recall 一样
					recall2, specificity, gmean2 = MultiPMeasurs.calculate_micro_gmean2(
						label, prd)
					pre_recall,macro_recall, macro_precision, gmean_score,\
                    f1_scores_eachclass = \
						MultiPMeasurs.calculate_macro_gmean(
						label, prd)#

					f1_score = sk.f1_score(label, prd, average = 'macro')
					#print('pre_recall',pre_recall,macro_recall,)
			



					mava = MultiPMeasurs.MAvA(label, prd)

				
					unique_classes = np.unique ( label )
					#print ( "y_true shape:" , label.shape )  # 应为 (n_samples,)
					#print ( "y_pred_proba shape:" ,
					#		prdpro.shape )  # 应为 (n_samples, n_classes)
					#print(prdpro)
					mauc = sk.roc_auc_score(label, prdpro,average = 'macro',multi_class = 'ovo'
					                        ) #average = 'macro',multi_class = 'ovo',labels=unique_classes
					
					F1.append(f1_score)
					Recall.append(recall1)
					Precision.append(precision)
					Specificity.append(specificity)
					GMEAN.append(gmean_score)
					Gmean1.append(gmean1)
					Gmean2.append(gmean2)
					Mava.append(mava)
					MAUC.append(mauc)
					Recall_pre.append ( pre_recall ) #[类别,样本数]
					Pre_class_f1.append(f1_scores_eachclass)

					#print(Recall_pre,len(Recall_pre),len(label))
			column_avgs = [ sum ( column ) / len ( column ) for column in
					zip ( *Recall_pre ) ]
			formatted_avgs = [f"{aa * 100:.4f}" for aa in column_avgs ]

			column_Pre_class_f1 = [ sum ( column ) / len ( column ) for column in
							zip ( *Pre_class_f1 ) ]
			formatted_column_Pre_class_f1 = [ f"{aa * 100:.4f}" for aa in column_Pre_class_f1 ]

			#print ( formatted_column_Pre_class_f1 )
			csv_writer.writerow(
				[fn, str(round(np.average(F1), 4) * 100),
				 str(round(np.average(GMEAN), 4) * 100),
				 str(round(np.average(Gmean1), 4) * 100),
				 str(round(np.average(Gmean2), 4) * 100),
				 str(round(np.average(Mava), 4) * 100),
				 str(round(np.average(MAUC), 4) * 100),
				 str(round(np.average(Precision), 4) * 100),
				 str(round(np.average(Specificity), 4) * 100),
				 str(round(np.average(Recall), 4) * 100), *formatted_avgs,*formatted_column_Pre_class_f1]
			)



if __name__ == '__main__':
	name = 'AMG2_'  # KDN

	list = ['mlp_0.3','CART_0.3']
	for clf in list:
		method(clf, name = name)


