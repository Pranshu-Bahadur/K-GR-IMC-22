exec(open('libs.bat', 'r').read())
exec(open('utils.py', 'r').read())


t_dir = '/home/pranshu-bahadur/Downloads/datasets/image-matching-challenge-2022/train'

print(TrainIMC22Dataset(t_dir)[0])

