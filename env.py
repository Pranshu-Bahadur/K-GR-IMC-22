exec(open('utils.py', 'r').read())
exec(open('run_configs/experiment.bat', 'r').read())

t_dir = '/home/pranshu-bahadur/Downloads/datasets/image-matching-challenge-2022/train'

training_dataset = TrainIMC22Dataset(t_dir)

splits = split_dataset(training_dataset, 0.8)

training_args = TrainingArguments(**training_args)

def gen_matching_model(model_name : str):
    pass

train_args = {
        'model': create_model('resnet18', num_classes=1, pretrained=True, in_chans=3),
        'train_dataset': splits[0],
        'eval_dataset': splits[1],
        'args': training_args
}




class IMC22Trainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model.head.fc = torch.nn.Linear(model.classifier.in_features, 1)
        pass

    def compute_loss(self, model, inputs, return_outputs=True):
       """

       Inputs: Batch x Tuple[List[Tuple[Tensor, List[Tensor]]], Tensor]
       """
       x, y = inputs





print(IMC22Trainer(**train_args).model)
