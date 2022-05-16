exec(open('libs.bat', 'r').read())

#==============================================================================
#  Utility functions
#==============================================================================
def imc22_train_directory_parser(train_directory : str,
        target_file : str = 'calibration.csv',
        column_name : str = 'basename') -> DataFrame:

    return concat(list(map(lambda x: \
            DataFrame({column_name: [x], 'images': ['images']}, columns = [column_name, 'images']).join(\
                read_csv(path.abspath(f'{train_directory}/{x}/{target_file}')),\
                how='cross'), listdir(train_directory))))


class TrainIMC22Dataset(Dataset):
    def __init__(self, train_directory : str, transforms : Compose = Compose([Resize([128, 128],\
            Image.ANTIALIAS),
            #Grayscale(1),\
            ToTensor()])):

        self.train_directory = train_directory
        self.calibration_df = imc22_train_directory_parser(train_directory)
        self.pair_cov_df = imc22_train_directory_parser(train_directory, \
                'pair_covisibility.csv')
        self.pair_cov_df = self.pair_cov_df.where(lambda x: x['covisibility'] >= 0.1)
        
        self.pair_cov_df = self.pair_cov_df.drop_duplicates(['pair'])
        self.distn = self.pair_cov_df['basename'].value_counts()

        print(self.distn)

        """

        #Oversampling
        #TODO pick a better formula for Balancing...
        lst = [self.pair_cov_df]
        for class_index, group in self.pair_cov_df.groupby('basename'):
            lst.append(group.sample(self.distn.max() - len(group), replace=True))

        self.pair_cov_df = concat(lst)
        self.distn = self.pair_cov_df['basename'].value_counts()

        print(self.distn)

        #TODO USE WEIGHTED SAMPLING...
        self.weights = numpy.vstack(list(map(lambda val: \
                val/self.distn.sum()*numpy.ones((val, 1)),\
                self.distn.values))).flatten().tolist()
        """
        #------------

        self.transforms = transforms

    def __getitem__(self, idx : int) -> Tuple[DataFrame, Tensor]:

        row = self.pair_cov_df.iloc[idx]

        y = Tensor(array(eval(row[-1].replace(' ', ', '))))

        ops = [lambda x: x.get('pair'), lambda x: x.split('-')]

        pairs = reduce(lambda x, op: op(x), ops, row)

        calibrations = self.calibration_df.\
                query(f'{pairs} in image_id')

        dir_converter = lambda y: reduce(lambda x, y: f'{x}/{y}', y, f'{self.train_directory}')

        calibrations['image_dir'] = calibrations['basename'].\
                str.cat(calibrations[['images', 'image_id']].astype(str), sep = '/')

        calibrations['image_dir'] = calibrations['image_dir'].apply(lambda x: f'{self.train_directory}/{x}.jpg')

        calibrations = calibrations.set_index(['image_id'])

        calibrations = calibrations.drop(['basename', 'images'], axis=1)

        image_dirs = calibrations['image_dir'].values.tolist()

        print(image_dirs)

        images = list(map(lambda x: self.transforms(Image.open(x).convert('RGB')), image_dirs))

        print(images)

        _subset = ['camera_intrinsics', 'rotation_matrix', 'translation_vector']

        calis_x = calibrations.get(_subset).applymap(lambda x: Tensor(fromstring(x, sep=' '))).values.tolist()

        print(calis_x)
        
        x = list(zip(images, calis_x))


        x = list(map(lambda data: (torch.stack([
                    data[1][0].view(3, 3), 
                    data[1][1].view(3, 3),
                    data[1][-1]*torch.ones(3, 3)])@(data[0].view(3, -1))).view(3, -1), x))
        return x, y

    def __len__(self):
        return self.pair_cov_df.size


def split_dataset(dataset : Dataset, split_factor : float) -> List[Subset]:
    train_split = int(split_factor * len(dataset))
    eval_split = int(len(dataset) - train_split)
    splits = [train_split, eval_split]
    splits.append(len(dataset) - sum(splits))
    return torch.utils.data.dataset.random_split(dataset, splits)




"""
Let an M be the 2D representation 256x256

m_i =     K   * ((R   *   M_i)        + T')

1x1x3 = 3x3x3 * ((3x3x3)*(1x1x3) + (1x1x3))

//^Due to 3 channels (Note: I need a pen and paper...)

Affine Map:

c_i = R(M_i) + T

//NOT ENOUGH SYMBOLS

"""
