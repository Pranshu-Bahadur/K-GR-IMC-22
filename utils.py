exec(open('libs.bat', 'r').read())



#==============================================================================
#  Utility functions
#==============================================================================




def imc22_train_directory_parser(train_directory : str,
        target_file : str = 'calibration.csv',
        column_name : str = 'basename') -> DataFrame:

    return concat(list(map(lambda x: \
                DataFrame([x], columns = [column_name]).join(\
                read_csv(path.abspath(f'{train_directory}/{x}/{target_file}')),\
                how='cross'), listdir(train_directory))))

class TrainIMC22Dataset(Dataset):
    def __init__(self, train_directory : str, transforms : Compose = Compose([ToTensor()])):
        self.train_directory = train_directory
        self.file_df = imc22_train_directory_parser(train_directory)
        self.distn = self.file_df['basename'].value_counts()

    def __getitem__(self, idx : int) -> Tuple[Tensor, List[Tensor]]:
        row = self.file_df.iloc[idx]
        image_directory = reduce(lambda x, y: f'{x}/{y}', tuple(row[:2].values), self.train_directory)
        x = transforms(Image.open(image_directory).convert('RGB'))
        y = list(map(lambda x: Tensor(list(eval(x.replace(" ", ", ")))), row[2:].values))
        return x, y
