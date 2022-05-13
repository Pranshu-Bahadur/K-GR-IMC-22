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
    def __init__(self, train_directory : str, transforms : Compose = Compose([Resize([256, 256],\
            Image.ANTIALIAS),\
            ToTensor()])):

        self.train_directory = train_directory
        self.train_files_df = imc22_train_directory_parser(train_directory)
        self.distn = self.train_files_df['basename'].value_counts()

        #Oversampling
        #TODO pick a better formula for Balancing...
        lst = [self.train_files_df]
        for class_index, group in self.train_files_df.groupby('basename'):
            lst.append(group.sample(self.distn.max() - len(group), replace=True))

        self.train_files_df = concat(lst)
        self.distn = self.train_files_df['basename'].value_counts()

        # Weights (Might be useful later)
        self.weights = numpy.vstack(list(map(lambda val: \
                val/self.distn.sum()*numpy.ones((val, 1)),\
                self.distn.values))).flatten().tolist()

        #------------

        self.transforms = transforms

    def __getitem__(self, idx : int) -> Tuple[Tensor, List[Tensor]]:
        
        row = self.train_files_df.iloc[idx]

        image_directory = reduce(lambda x, y: f'{x}/{y}', \
                tuple(row[:3].values), \
                self.train_directory) + '.jpg'

        x = self.transforms(Image.open(image_directory).convert('RGB'))
        y = list(map(lambda x: Tensor(list(eval(x.replace(" ", ", ")))), row[3:].values))
        return x, y


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
