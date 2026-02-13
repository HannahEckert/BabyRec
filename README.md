run by running: python train.py --gpu 0 (if you have a gpu otherwise without the gpu variable)

It will create a model and safe it in the results file as well as the results per user measured in ndcg@10 in the results file as results.pth 

item_feature.pth and user_features.pth has some categorial features (load with weights_only=False)
