# ViT-pytorth
A model of vision transformer for image classification with pre-trained parameters. It runs in pytorch.

Firstly run 'train.py' to fine tune the ViT parameters, with your own training data by modifing the statement 'file_path =' in line 16. Before doing this, make sure that you have pre-trained parameters for ViT (you can find and download it from website, e.g., https://drive.google.com/drive/folders/1azgrD1P413pXLJME0PjRRU-Ez-4GWN-S), and modify the statement 'weight_path = ' in line 17 according to the path where you have saved the pre-trained parameters. The training process allows you to save the fine tuned parameters in a path specified by the statement 'save_path =' in line 18.

Then run 'test.py' to classify objects (e.g., five classes of flowers). It displays classification results in texts on graphics.
