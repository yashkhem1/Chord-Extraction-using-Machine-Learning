# Chord-Extraction-using-Machine-Learning
We aim to implement Machine Learning approach to extract chords from songs. 
We first remove the vocals of a stereo song by adding the inverted right audio.
Then we calculate the PCP vector for each 0.5 sec interval of the song.
The model is then trained using PCP vectors as the input. For that purpose we have used both SVM and Neural Network.

# Running the Code
Run `python3 src/Chord_Sequencer.py` and follow the instructions to find the beats in the given song
