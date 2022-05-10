exec(open('libs.bat', 'r').read())


class TrainIMC22Dataset(Dataset):

    def __init__(self, train_dir : str, **kwargs):
        super().__init__()
        

        df = DataFrame(listdir(train_dir), columns = ['classes'])

        csv_files = ['calibrations', 'pair_covisibility']

        calibrations = list(map(lambda x: \
                DataFrame([x], columns = ['class']).join(\
                read_csv(path.abspath(f'{train_dir}/{x}/calibration.csv')), how='cross'), listdir(train_dir)))

        df = concat(calibrations)

        print(df.head())

        """
        df['pair_covisibility'] = df['buildings'].apply(lambda x: \
                read_csv(path.abspath(f'{train_dir}/{x}/pair_covisibility.csv')))
        """

        





#f = lambda x: getattr(path, x)(target)
