# Speech to Speech Translator

This application converts English speech to French speech. Users can record their voice in English and can convert it to text. The model(trained using a neural network to translate English text to French text, will discuss model training later) on the back-end only accepts text. To translate English speech to French speech we need to convert speech to text(I used google speech API in the backend to convert speech to text) and then converted text will be fed to the trained model to translate english to french. Once we have converted english speech to english text we can feed english text to the trained model and the model will return french text.Translated french text will be sent on the front end. We will get text(French) on the front-end after translation, but as we discussed This application translates English speech to French speech so we will convert translated text(French) to speech. I have used google speech API on the backend to convert text to speech. French Speech converted from the French-text can be played on the front end.

<img src="images/1.jpg" width="100%" align="top-left" alt="" title="RNN" />

<img src="images/2.jpg" width="100%" align="top-left" alt="" title="RNN" />

 <img src="images/3.jpg" width="100%" align="top-left" alt="" title="RNN" />


<img src="images/4.png" width="100%" align="top-left" alt="" title="RNN" />

<img src="images/5.jpg" width="100%" align="top-left" alt="" title="RNN" />



## Install conda

### download the Miniconda installer for Linux:- 

https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Windows-x86_64.exe

### run downloaded file

bash Miniconda3-latest-Linux-x86_64.sh

Press Enter to review the license agreement. Then press and hold Enter to scroll
Enter “yes” to agree to the license agreement.


## Create conda environment 
 
conda create -n myenv python=3.9

## install requirements.txt file

pip install -r requirements.txt
     
## get data set using following link


 
