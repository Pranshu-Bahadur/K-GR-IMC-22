exec(open('libs.bat', 'r').read())
  

class IMC22Dataset(Dataset):

    def __init__(self, target : str, ext : str='.csv', **kwargs):
        super().__init__()

        self.metadata = {
                "uid" : uuid4(),
                "dir" : target,
                "dir_df" : self._pathfinder(target)
                }
        _exts = self.metadata['dir_df']['abspath'].str.contains(ext)
        self.metadata['flist'] = self.metadata['dir_df'][_exts]
        print(self.metadata['flist']['abspath'].str.rsplit(r'\/.+$'))

    def _dir2dict(self, target : str, \
            ops : List[str] = ['abspath', 'isdir']) \
            -> Dict[str, object]:

        f = lambda x: getattr(path, x)(target)

        return dict(zip(ops, (map(f, ops))))

    def _pathfinder(self, target : str) -> DataFrame:

        row = self._dir2dict(target)
        if row['isdir']:
            return concat(list(map(lambda x: \
                    self._pathfinder(path.join(target, x)), \
                    listdir(row['abspath'])))
                    + [DataFrame([row])])

        return DataFrame([row])

