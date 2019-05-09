from pydub import AudioSegment
import filetype
def remove(myAudioFile):
	fmt=filetype.guess(myAudioFile).extension
	sound_stereo = AudioSegment.from_file(myAudioFile, format=fmt)
	mono_list=sound_stereo.split_to_mono()
	if len(mono_list)==1:
		print('File contains Mono channel only. Can\'t operate')
		return
	sound_monoL = mono_list[0]
	sound_monoR = mono_list[1]
	sound_monoR_inv = sound_monoR.invert_phase()
	sound_CentersOut = sound_monoL.overlay(sound_monoR_inv)
	sound_CentersOut.export(myAudioFile.rsplit('.')[0] + '-nolyrics' + '.wav', format='wav')
file=input('File name\n')
remove(file)