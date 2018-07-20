import numpy as np
import scipy.io as sio 

txt_path = 'C:\\Users\\zlucky\\Desktop\\data'
txt_name = 'A00001_wave'
mat_name = 'A00001_wave'
target_path = 'C:\\Users\\zlucky\\Desktop\\data\\txt_to_mat'
lead = 1;

def rdtxt_to_mat(txt_name, txt_path, target_path):
	file = open(txt_path +'\\' + txt_name + '.txt')
	lines = file.readlines()
	data = lines[lead - 1].strip().split(',')

	data =list(map(float, data))

	#print('data',data)

	


	sio.savemat(target_path + '\\' +  mat_name + '.mat', {'vall':data})


if __name__ == "__main__":
	rdtxt_to_mat(txt_name, txt_path, target_path)
	print('success')




