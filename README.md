# music-genre-classification
Music genre classification from audio spectrograms using deep learning
A convolutional neural network is trained with 7000 sample audios of 10 different music genres. 
  
**GENRES**: blues - classical - country - disco - hiphop - jazz - metal - pop - reggae - rock  
  
**DATA**: [GTZAN Genre Collection](http://marsyasweb.appspot.com/download/data_sets)  
             
## Usage  
$**python3**  get_genre.py  input-audio-path  
  
### Example  
$**cd**  src  
$**python3**  get_genre.py  ../test.mp3  
**--> disco**: 62.50%  
**--> rock**:  35.42%  
**--> reggae**: 2.08	%  
   
Test audio (test.mp3) is a disco song, **Every 1's A Winner** by **Hot Chocolate**. (*https://open.spotify.com/track/5MXXbGYNmRHR7ULMvZYo5R?si=yk6GzvJiS--7hZuGd8awog*)   
